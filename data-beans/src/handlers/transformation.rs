use clap::Args;

mod convert;
mod squeeze;
mod subsample;
mod subset;

pub use convert::*;
pub use squeeze::*;
pub use subsample::*;
pub use subset::*;

#[derive(Args, Debug)]
pub struct ReorderRowsArgs {
    /// Data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// Row/feature name file (name per each line; `.tsv.gz` or `.tsv`)
    #[arg(short, long, required = true)]
    pub row_file: Box<str>,

    /// output header
    #[arg(short, long, required = true)]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
pub struct SubsetColumnsArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// column indices to take: e.g., `0,1,2,3`
    #[arg(short = 'i', long, value_delimiter = ',')]
    pub column_indices: Option<Vec<usize>>,

    /// column name file where each line is a column name
    #[arg(short = 'f', long)]
    pub name_file: Option<Box<str>>,

    /// delimiter for base-key extraction (e.g., '@' to match "ACGT-1@batch" with "ACGT-1")
    #[arg(short = 'd', long, default_value = "@")]
    pub delimiter: char,

    /// enable prefix matching (stored name is prefix of query or vice versa)
    #[arg(long, default_value_t = true)]
    pub allow_prefix: bool,

    /// squeeze
    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    /// minimum number of non-zero cutoff for rows
    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    /// minimum number of non-zero cutoff for columns
    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    /// output file
    #[arg(short, long, required = true)]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
pub struct SubsetRowsArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// row indices to take: e.g., `0,1,2,3`
    #[arg(short = 'i', long, value_delimiter = ',')]
    pub row_indices: Option<Vec<usize>>,

    /// row name file where each line is a row name
    #[arg(short = 'f', long)]
    pub name_file: Option<Box<str>>,

    /// delimiter for base-key extraction (e.g., '@' to match "gene@batch" with "gene")
    #[arg(short = 'd', long, default_value = "@")]
    pub delimiter: char,

    /// enable prefix matching (stored name is prefix of query or vice versa)
    #[arg(long, default_value_t = true)]
    pub allow_prefix: bool,

    /// squeeze
    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    /// minimum number of non-zero cutoff for rows
    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    /// minimum number of non-zero cutoff for columns
    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    /// output file
    #[arg(short, long, required = true)]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
#[command(about)]
pub struct RunSqueezeArgs {
    /// data files -- either `.zarr` or `.h5`
    #[arg(required = true, value_delimiter = ',')]
    pub data_files: Vec<Box<str>>,

    /// number of non-zero cutoff for rows
    #[arg(short, long, default_value = "0")]
    pub row_nnz_cutoff: usize,

    /// number of non-zero cutoff for columns
    #[arg(short, long, default_value = "0")]
    pub column_nnz_cutoff: usize,

    /// Cells per rayon job. Omit for auto-scaling by feature count.
    #[arg(long)]
    pub block_size: Option<usize>,

    /// preload data into memory for faster processing
    #[arg(
        long,
        alias = "preload-data",
        default_value_t = true,
        help = "Preload data into memory for faster processing",
        long_help = "Preload all column data into memory before squeezing. \n\
		     This can significantly speed up processing but requires more memory."
    )]
    pub preload: bool,

    /// show nnz histogram before squeezing
    #[arg(
        long,
        default_value_t = false,
        help = "Show ASCII histogram of row/column nnz distributions",
        long_help = "Display log1p-transformed ASCII histograms of row and column \n\
		     non-zero counts before squeezing. Helps determine appropriate cutoff values."
    )]
    pub show_histogram: bool,

    /// save histogram data to files
    #[arg(
        long,
        help = "Output file prefix for saving histogram data",
        long_help = "Save histogram data to {prefix}.row_nnz.txt and {prefix}.col_nnz.txt files. \n\
		     Each file contains nnz counts that can be used for further analysis."
    )]
    pub save_histogram: Option<Box<str>>,

    /// dry run - only show histograms without performing squeeze
    #[arg(
        long,
        default_value_t = false,
        help = "Preview mode - show histograms without squeezing",
        long_help = "Only display histograms and statistics without actually performing the squeeze operation. \n\
		     Useful for determining appropriate cutoff values."
    )]
    pub dry_run: bool,

    /// interactive mode - prompt user after showing histogram
    #[arg(
        short,
        long,
        default_value_t = false,
        help = "Interactive mode - ask for confirmation after showing histogram",
        long_help = "Show histogram and prompt user to proceed, adjust cutoffs, or cancel. \n\
		     Automatically enables --show-histogram."
    )]
    pub interactive: bool,

    /// auto cutoff - apply the k-means-suggested cutoff without prompting
    #[arg(
        long,
        default_value_t = false,
        help = "Apply the k-means-suggested nnz cutoff headlessly (no prompt)",
        long_help = "Resolve row/column cutoffs from a 2-means split of log(1+nnz) and squeeze \n\
		     without prompting. Explicit --row-nnz-cutoff / --column-nnz-cutoff still win \n\
		     per dimension, so you can pin one axis and auto the other. \n\
		     Combine with --dry-run to preview the resolved cutoffs without writing."
    )]
    pub auto_cutoff: bool,

    /// output file for squeezed data
    #[arg(
        short,
        long,
        help = "Output file for squeezed data",
        long_help = "Save squeezed data to a new file instead of modifying in-place. \n\
		     With multiple inputs, all files will be squeezed and merged into {output}.{backend}. \n\
		     If not specified, modifies files in-place (requires confirmation in interactive mode)."
    )]
    pub output: Option<Box<str>>,

    /// row alignment strategy for merging multiple files
    #[arg(
        long,
        value_enum,
        default_value = "common",
        help = "Row alignment strategy when merging multiple files",
        long_help = "How to align rows across files after squeezing:\n\
		     - common: Keep only rows present in ALL files (intersection)\n\
		     - union: Keep rows present in ANY file (union, fills missing with zeros)"
    )]
    pub row_align: RowAlignMode,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum RowAlignMode {
    Common,
    Union,
}
