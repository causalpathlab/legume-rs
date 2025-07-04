use std::collections::HashMap;

use crate::common::*;
use crate::data::util_htslib::*;
use crate::data::visitors_htslib::*;

use data_beans::qc::*;
use data_beans::sparse_io::*;

use matrix_util::common_io::remove_file;
use rust_htslib::bam::{self, record::Aux};

#[derive(Args, Debug)]
pub struct GeneCountArgs {
    /// `.bam` files to quantify
    #[arg(value_delimiter = ',', required = true)]
    bam_files: Vec<Box<str>>,

    /// Gene annotation (`GFF`) file
    #[arg(short = 'g', long = "gff", required = true)]
    gff_file: Box<str>,

    /// (10x) cell/sample barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "CB")]
    cell_barcode_tag: Box<str>,

    /// gene barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "GX")]
    gene_barcode_tag: Box<str>,

    /// bam record type (gene, transcript, exon, utr)
    #[arg(long, default_value = "gene")]
    record_type: Box<str>,

    /// gene type (protein_coding, pseudogene, lncRNA)
    #[arg(long, default_value = "protein_coding")]
    gene_type: Box<str>,

    /// number of non-zero cutoff for rows/genes
    #[arg(short, long, default_value_t = 10)]
    row_nnz_cutoff: usize,

    /// number of non-zero cutoff for columns/cells
    #[arg(short, long, default_value_t = 10)]
    column_nnz_cutoff: usize,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// block_size for parallel processing (number of columns)
    #[arg(long, default_value = "100")]
    block_size: usize,

    /// output header for `data-beans` files
    #[arg(short, long, required = true)]
    output: Box<str>,
}

/// simply count the occurence of gene and cell barcode
pub fn run_gene_count(args: &GeneCountArgs) -> anyhow::Result<()> {
    if args.bam_files.is_empty() {
        return Err(anyhow::anyhow!("need bam files"));
    }

    for x in args.bam_files.iter() {
        check_bam_index(x, None)?;
    }

    let backend = args.backend.clone();

    let record_feature_type: FeatureType = args.record_type.as_ref().into();
    info!("parsing GFF file: {}", args.gff_file);

    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref(), Some(&record_feature_type))?;
    gff_map.subset(args.gene_type.clone().into());
    info!("found {} features", gff_map.len(),);

    if gff_map.is_empty() {
        info!("found no feature in {}", args.gff_file.as_ref());
        return Ok(());
    }

    let njobs = gff_map.len() as u64;
    info!("Combining reads over {} blocks", njobs);

    let gene_level_stats = gff_map
        .records()
        .into_iter()
        .par_bridge()
        .progress_count(njobs)
        .map(|x| count_read_per_gene(args, &x))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let (triplets, row_names, col_names) = format_data_triplets(gene_level_stats);

    let mtx_shape = (row_names.len(), col_names.len(), triplets.len());
    let output = args.output.clone();
    let backend_file = match backend {
        SparseIoBackend::HDF5 => format!("{}.h5", &output),
        SparseIoBackend::Zarr => format!("{}.zarr", &output),
    };

    remove_file(&backend_file)?;

    let mut data =
        create_sparse_from_triplets(triplets, mtx_shape, Some(&backend_file), Some(&backend))?;

    data.register_column_names_vec(&col_names);
    data.register_row_names_vec(&row_names);

    if args.row_nnz_cutoff > 0 || args.column_nnz_cutoff > 0 {
        squeeze_by_nnz(
            &data,
            SqueezeCutoffs {
                row: args.row_nnz_cutoff,
                column: args.column_nnz_cutoff,
            },
            args.block_size,
        )?;
    }

    Ok(())
}

fn count_read_per_gene(
    args: &GeneCountArgs,
    rec: &GffRecord,
) -> anyhow::Result<Vec<(CellBarcode, GeneId, f32)>> {
    let gene_id = &rec.gene_id;

    if gene_id == &GeneId::Missing {
        return Ok(vec![]);
    }

    let mut read_counter = ReadCounter::new(&args.cell_barcode_tag);

    for file in &args.bam_files {
        read_counter.visit_bam_by_gene(file, rec, &args.gene_barcode_tag, &ReadCounter::update)?;
    }

    Ok(read_counter
        .to_vec()
        .into_iter()
        .map(|(cb, x)| (cb, gene_id.clone(), x as f32))
        .collect())
}

struct ReadCounter<'a> {
    cell_to_count: HashMap<CellBarcode, usize>,
    cell_barcode_tag: &'a str,
}

impl<'a> VisitWithBamOps for ReadCounter<'a> {}

impl<'a> ReadCounter<'a> {
    fn new(cell_barcode_tag: &'a str) -> Self {
        Self {
            cell_to_count: HashMap::default(),
            cell_barcode_tag,
        }
    }

    fn to_vec(&self) -> Vec<(CellBarcode, usize)> {
        self.cell_to_count
            .iter()
            .map(|(cb, x)| (cb.clone(), *x))
            .collect()
    }

    fn update(&mut self, gff_record: &GffRecord, gene_barcode_tag: &str, bam_record: bam::Record) {
        let gene_id_found = match bam_record.aux(gene_barcode_tag.as_bytes()) {
            Ok(Aux::String(id)) => match parse_ensembl_id(id) {
                Some(id) => GeneId::Ensembl(id.into()),
                _ => GeneId::Missing,
            },
            _ => GeneId::Missing,
        };

        if gene_id_found == gff_record.gene_id {
            let cell_barcode = match bam_record.aux(&self.cell_barcode_tag.as_bytes()) {
                Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
                _ => CellBarcode::Missing,
            };

            let nreads = self.cell_to_count.entry(cell_barcode).or_default();
            *nreads += 1;
        }
    }
}
