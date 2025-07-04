use crate::common::*;
use crate::data::bed::*;
use crate::data::util_htslib::*;
use crate::data::visitors_htslib::*;

use data_beans::qc::*;
use data_beans::sparse_io::*;
use matrix_util::common_io::remove_file;
use rust_htslib::bam::{self, record::Aux};

use coitrees::{COITree, Interval, IntervalTree};
use fnv::FnvHashMap as HashMap;

#[derive(Args, Debug)]
pub struct ReadDepthArgs {
    /// `.bam` files to quantify
    #[arg(value_delimiter = ',', required = true)]
    bam_files: Vec<Box<str>>,

    /// resolution (in kb)
    #[arg(short = 'r', long, required = true)]
    resolution_kb: usize,

    /// block size for parallelism (in mb)
    #[arg(short = 'b', long, default_value_t = 1)]
    block_size_mb: usize,

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
    #[arg(short, long, default_value_t = 100)]
    row_nnz_cutoff: usize,

    /// number of non-zero cutoff for columns/cells
    #[arg(short, long, default_value_t = 100)]
    column_nnz_cutoff: usize,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// output header for `data-beans` files
    #[arg(short, long, required = true)]
    output: Box<str>,
}

/// Count read depth
///
pub fn run_read_depth(args: &ReadDepthArgs) -> anyhow::Result<()> {
    if args.resolution_kb * 1000 > args.block_size_mb * 1000_000 {
        return Err(anyhow::anyhow!(
            "resolution should be smaller than the block size"
        ));
    }

    if args.bam_files.is_empty() {
        return Err(anyhow::anyhow!("need bam files"));
    }

    for x in args.bam_files.iter() {
        check_bam_index(x, None)?;
    }

    let backend = args.backend.clone();

    // 1. build a coitree for each chromosome, each cell barcode, and each coitree can keep track of coverage
    let jobs = create_bam_jobs(
        &args.bam_files[0],
        Some(args.block_size_mb * 1000_000),
        Some(0),
    )?;
    let njobs = jobs.len() as u64;
    info!("Combining reads over {} blocks", njobs);

    let segment_stats = jobs
        .par_iter()
        .progress_count(njobs)
        .map(
            |(chr, lb, ub)| -> anyhow::Result<Vec<(CellBarcode, Box<str>, f32)>> {
                let start = *lb;
                let stop = *ub;

                let bed = Bed {
                    chr: chr.clone(),
                    start,
                    stop,
                };

                let mut read_coverage = ReadCoverage::new(&args.cell_barcode_tag);

                for file in &args.bam_files {
                    read_coverage.visit_bam_by_region(file, &bed, &ReadCoverage::update)?;
                }

                let cov_tree = read_coverage.to_coitrees();

                // define segments as specified by the resolution parameter
                let start = *lb as usize;
                let stop = *ub as usize;
                let segment_size = (stop - start).div_ceil(args.resolution_kb * 1000);

                // now count them all
                let mut ret = vec![];
                for (cb, chr_tree) in cov_tree {
                    for (chr, tree) in chr_tree {
                        for lb in (start..stop).step_by(segment_size) {
                            let ub = (lb + segment_size).min(stop);
                            let nn = tree.query_count(lb as i32, ub as i32);
                            if nn > 0 {
                                let feature = format!("{}:{}-{}", chr, lb, ub);
                                ret.push((cb.clone(), feature.into_boxed_str(), nn as f32));
                            }
                        }
                    }
                }

                Ok(ret)
            },
        )
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    info!(
        "constructing backend data with {} segments",
        segment_stats.len()
    );
    let (triplets, row_names, col_names) = format_data_triplets(segment_stats);

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
        info!("final Q/C to remove excessive zeros");

        squeeze_by_nnz(
            data,
            SqueezeCutoffs {
                row: args.row_nnz_cutoff,
                column: args.column_nnz_cutoff,
            },
            1000,
        )?;
    }

    Ok(())
}

struct ReadCoverage<'a> {
    cell_chr_to_intervals: HashMap<CellBarcode, HashMap<Box<str>, Vec<Interval<()>>>>,
    cell_barcode_tag: &'a str,
}

impl<'a> VisitWithBamOps for ReadCoverage<'a> {}

impl<'a> ReadCoverage<'a> {
    fn new(cell_barcode_tag: &'a str) -> Self {
        Self {
            cell_chr_to_intervals: HashMap::default(),
            cell_barcode_tag,
        }
    }

    fn to_coitrees(&self) -> HashMap<CellBarcode, HashMap<Box<str>, COITree<(), u32>>> {
        let mut trees = HashMap::default();

        for (cb, chr_to_intervals) in self.cell_chr_to_intervals.iter() {
            let cb_trees: &mut HashMap<Box<str>, COITree<(), u32>> =
                trees.entry(cb.clone()).or_default();

            for (chr, nodes) in chr_to_intervals.iter() {
                cb_trees.insert(chr.clone(), COITree::new(nodes));
            }
        }

        trees
    }

    fn update(&mut self, chr: &str, bam_record: bam::Record) {
        let cell_barcode = match bam_record.aux(&self.cell_barcode_tag.as_bytes()) {
            Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
            _ => CellBarcode::Missing,
        };
        let first = bam_record.pos() as i32;
        let last = first + bam_record.seq_len() as i32;

        let chr_to_intervals = self.cell_chr_to_intervals.entry(cell_barcode).or_default();

        let intervals = chr_to_intervals.entry(chr.into()).or_default();
        intervals.push(Interval::new(first, last, ()));
    }
}
