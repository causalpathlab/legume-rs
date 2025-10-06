use crate::collapse_data::*;
use crate::common::*;

use data_beans_alg::collapse_data::CollapsingOps;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct CollapseArgs {
    /// data files of either `.zarr` or `.h5` format. All the formats
    /// in the given list should be identical. We can convert `.mtx`
    /// to `.zarr` or `.h5` using `data-beans` command.
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// latent topic assignment files (comma-separated file names)
    /// Each line in each file can specify topic name or cell and
    /// topic name pair.
    #[arg(long, short = 't', value_delimiter(','))]
    topic_assignment_files: Option<Vec<Box<str>>>,

    /// latent topic proportion files (comma-separated file names)
    /// Each file contains a full `cell x topic` matrix.
    #[arg(long, short = 'r', value_delimiter(','))]
    topic_proportion_files: Option<Vec<Box<str>>>,

    /// block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn run_collapse(args: CollapseArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    Ok(())
}

struct ArgInputData {
    sparse_data: SparseIoVec,
    cell_to_indv: Vec<Box<str>>,
    cell_topic: Mat,
    // sorted_topic_names: Vec<Box<str>>,
    // indv_to_exposure: HashMap<Box<str>, Box<str>>,
    // exposure_id: HashMap<Box<str>, usize>,
}

fn parse_arg_input_data(args: CollapseArgs) -> anyhow::Result<ArgInputData> {
    use matrix_util::common_io::*;

    // push data files and collect batch membership
    let file = args.data_files[0].as_ref();
    let backend = match extension(file)?.to_string().as_str() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => {
            return Err(anyhow::anyhow!("unknown backend"));
        }
    };

    let data_files = args.data_files;

    let mut sparse_data = SparseIoVec::new();

    for (f, this_data_file) in data_files.iter().enumerate() {


    }

    unimplemented!("");
}
