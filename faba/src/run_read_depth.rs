use crate::common::*;
use crate::data::util_htslib::*;
use crate::read_coverage::ReadCoverageCollector;
use genomic_data::bed::*;

use coitrees::IntervalTree;

#[derive(Args, Debug)]
pub struct ReadDepthArgs {
    /// `.bam` files to quantify
    #[arg(value_delimiter = ',', required = true)]
    bam_files: Vec<Box<str>>,

    /// resolution (in kb)
    #[arg(short = 'r', long, required = true)]
    resolution_kb: f32,

    /// block size for parallelism (in mb)
    #[arg(short = 'b', long, default_value_t = 1)]
    block_size_mb: usize,

    /// (10x) cell/sample barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "CB")]
    cell_barcode_tag: Box<str>,

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

    /// verbosity
    #[arg(
        long,
        short,
        help = "verbosity",
        long_help = "Enable verbose output `RUST_LOG=info`"
    )]
    verbose: bool,
}

/// Count read depth
///
pub fn run_read_depth(args: &ReadDepthArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    if (args.resolution_kb * 1000.0) as usize > args.block_size_mb * 1000_000 {
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

    // build a coitree for each chromosome, each cell barcode, and
    // each coitree can keep track of coverage
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

                let mut read_coverage = ReadCoverageCollector::new(&args.cell_barcode_tag);

                for file in &args.bam_files {
                    read_coverage.collect_from_bam(file, &bed)?;
                }

                let coverage_interval_tree = read_coverage.to_coitrees();

                // define segments as specified by the resolution parameter
                let start = *lb as usize;
                let stop = *ub as usize;
                let segment_size = (args.resolution_kb * 1000.0) as usize;

                // now count them all
                let mut ret = vec![];
                for (cb, chr_tree) in coverage_interval_tree {
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

    let output = args.output.clone();
    let backend_file = match backend {
        SparseIoBackend::HDF5 => format!("{}.h5", &output),
        SparseIoBackend::Zarr => format!("{}.zarr", &output),
    };

    format_data_triplets(segment_stats)
        .to_backend(&backend_file)?
        .qc(SqueezeCutoffs {
            row: args.row_nnz_cutoff,
            column: args.column_nnz_cutoff,
        })?;

    info!("done");
    Ok(())
}
