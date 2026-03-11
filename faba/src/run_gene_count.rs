use crate::common::*;
use crate::data::bam_io;
use crate::data::util_htslib::*;
use rust_htslib::bam;

use fnv::FnvHashMap as HashMap;

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

    info!("data files:");
    for x in args.bam_files.iter() {
        info!("{}", x);
    }

    let backend = args.backend.clone();
    info!("parsing GFF file: {}", args.gff_file);

    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref())?;
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
        .par_iter()
        .progress_count(njobs)
        .map(|x| count_read_per_gene(args, x))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let output = args.output.clone();
    let backend_file = match backend {
        SparseIoBackend::HDF5 => format!("{}.h5", &output),
        SparseIoBackend::Zarr => format!("{}.zarr", &output),
    };

    format_data_triplets(gene_level_stats)
        .to_backend(&backend_file)?
        .qc(SqueezeCutoffs {
            row: args.row_nnz_cutoff,
            column: args.column_nnz_cutoff,
        })?;

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
        bam_io::for_each_record_in_gene(file, rec, &args.gene_barcode_tag, false, |bam_record| {
            read_counter.count(bam_record);
        })?;
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

    fn count(&mut self, bam_record: bam::Record) {
        let cell_barcode =
            bam_io::extract_cell_barcode(&bam_record, self.cell_barcode_tag.as_bytes());
        *self.cell_to_count.entry(cell_barcode).or_default() += 1;
    }
}
