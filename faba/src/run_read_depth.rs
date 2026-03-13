use crate::common::*;
use crate::data::util_htslib::*;

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
    pub(crate) bam_files: Vec<Box<str>>,

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
    pub(crate) resolution_kb: f32,

    /// Block size for parallelism in Mb
    #[arg(
        short = 'b',
        long,
        default_value_t = 1,
        help = "Block size for parallelism (Mb)",
        long_help = "Size of genomic blocks in megabases for parallel processing.\n\
                     Must be larger than the bin resolution."
    )]
    pub(crate) block_size_mb: usize,

    /// Cell barcode BAM tag
    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode BAM tag",
        long_help = "BAM tag for cell/sample barcode identification.\n\
                     Standard 10x Genomics tag is \"CB\"."
    )]
    pub(crate) cell_barcode_tag: Box<str>,

    /// Minimum non-zero entries per row (bin) to keep
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per row (bin)",
        long_help = "Bins with fewer than this many non-zero cells are removed\n\
                     from the output matrix."
    )]
    pub(crate) row_nnz_cutoff: usize,

    /// Minimum non-zero entries per column (cell) to keep
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per column (cell)",
        long_help = "Cells with fewer than this many non-zero bins are removed\n\
                     from the output matrix."
    )]
    pub(crate) column_nnz_cutoff: usize,

    /// Sparse matrix output backend
    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix output backend",
        long_help = "File format for the output sparse matrix.\n\
                     Supported: zarr, hdf5."
    )]
    pub(crate) backend: SparseIoBackend,

    /// Output directory
    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Directory for output files.\n\
                     One sparse matrix file per input BAM is created here."
    )]
    pub(crate) output: Box<str>,
}

/// Count read depth
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

    crate::read_depth::pipeline::run_read_depth_pipeline(args)
}
