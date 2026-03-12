use crate::common::*;
use crate::data::util_htslib::*;
use crate::read_depth_coverage::ReadCoverageCollector;
use genomic_data::bed::*;

use coitrees::IntervalTree;

#[derive(Args, Debug)]
pub struct ReadDepthArgs {
    /// Input BAM file(s), comma-separated
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input BAM file(s)",
        long_help = "Comma-separated list of BAM files to quantify.\n\
                     Each file produces a separate output matrix."
    )]
    bam_files: Vec<Box<str>>,

    /// Bin resolution in kb
    #[arg(
        short = 'r',
        long,
        required = true,
        help = "Bin resolution in kb",
        long_help = "Size of genomic bins in kilobases.\n\
                     The genome is tiled at this resolution and read coverage\n\
                     is counted per bin per cell."
    )]
    resolution_kb: f32,

    /// Block size for parallelism in Mb
    #[arg(
        short = 'b',
        long,
        default_value_t = 1,
        help = "Block size for parallelism (Mb)",
        long_help = "Size of genomic blocks in megabases for parallel processing.\n\
                     Must be larger than the bin resolution."
    )]
    block_size_mb: usize,

    /// Cell barcode BAM tag
    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode BAM tag",
        long_help = "BAM tag for cell/sample barcode identification.\n\
                     Standard 10x Genomics tag is \"CB\"."
    )]
    cell_barcode_tag: Box<str>,

    /// Minimum non-zero entries per row (bin) to keep
    #[arg(
        short,
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per row (bin)",
        long_help = "Bins with fewer than this many non-zero cells are removed\n\
                     from the output matrix."
    )]
    row_nnz_cutoff: usize,

    /// Minimum non-zero entries per column (cell) to keep
    #[arg(
        short,
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per column (cell)",
        long_help = "Cells with fewer than this many non-zero bins are removed\n\
                     from the output matrix."
    )]
    column_nnz_cutoff: usize,

    /// Sparse matrix output backend
    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix output backend",
        long_help = "File format for the output sparse matrix.\n\
                     Supported: zarr, hdf5."
    )]
    backend: SparseIoBackend,

    /// Output directory
    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Directory for output files.\n\
                     One sparse matrix file per input BAM is created here."
    )]
    output: Box<str>,
}

/// Count read depth
///
pub fn run_read_depth(args: &ReadDepthArgs) -> anyhow::Result<()> {
    if (args.resolution_kb * 1000.0) as usize > args.block_size_mb * 1_000_000 {
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

    let batch_names = uniq_batch_names(&args.bam_files)?;
    std::fs::create_dir_all(args.output.as_ref())?;

    let backend = args.backend.clone();
    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

    for (bam_file, batch_name) in args.bam_files.iter().zip(&batch_names) {
        // build a coitree for each chromosome, each cell barcode, and
        // each coitree can keep track of coverage
        let jobs = create_bam_jobs(bam_file, Some(args.block_size_mb * 1_000_000), Some(0))?;
        let njobs = jobs.len() as u64;
        info!("Combining reads for {} over {} blocks", batch_name, njobs);

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
                    read_coverage.collect_from_bam(bam_file, &bed)?;

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

        let backend_file = match &backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &args.output, batch_name),
        };

        format_data_triplets(segment_stats)
            .to_backend(&backend_file)?
            .qc(cutoffs.clone())?;
    }

    info!("done");
    Ok(())
}
