use crate::collapse_cocoa_data::*;
use crate::common::*;
use crate::input::*;

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

    /// is topic proportion matrix of probability?
    #[arg(long, default_value = "logit")]
    topic_proportion_value: TopicValue,

    /// block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// output directory
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// preload all the columns data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn run_collapse(args: CollapseArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let mut data = read_input_data(InputDataArgs {
        data_files: args.data_files,
        indv_files: None,
        topic_assignment_files: args.topic_assignment_files,
        topic_proportion_files: args.topic_proportion_files,
        exposure_assignment_file: None,
        preload_data: args.preload_data,
        topic_value: args.topic_proportion_value,
    })?;



    Ok(())
}

// struct ArgInputData {
//     sparse_data: SparseIoVec,
//     cell_to_indv: Vec<Box<str>>,
//     cell_topic: Mat,
//     // sorted_topic_names: Vec<Box<str>>,
//     // indv_to_exposure: HashMap<Box<str>, Box<str>>,
//     // exposure_id: HashMap<Box<str>, usize>,
// }

// fn parse_arg_input_data(args: CollapseArgs) -> anyhow::Result<ArgInputData> {
//     use matrix_util::common_io::*;

//     // push data files and collect batch membership
//     let file = args.data_files[0].as_ref();
//     let backend = match extension(file)?.to_string().as_str() {
//         "h5" => SparseIoBackend::HDF5,
//         "zarr" => SparseIoBackend::Zarr,
//         _ => {
//             return Err(anyhow::anyhow!("unknown backend"));
//         }
//     };

//     let data_files = args.data_files;

//     let mut sparse_data = SparseIoVec::new();

//     for (f, this_data_file) in data_files.iter().enumerate() {

//     }

//     unimplemented!("");
// }
