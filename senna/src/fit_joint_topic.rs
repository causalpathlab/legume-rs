use crate::embed_common::*;
use crate::senna_input::*;

use candle_core::Device;
use candle_util::candle_data_loader::*;
use candle_util::candle_decoder_topic::*;
use candle_util::candle_model_traits::*;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

use candle_util::candle_encoder_softmax_iaf::*;
use candle_util::candle_inference::TrainConfig;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_traits::DecoderModuleT;
use candle_util::candle_vae_inference::*;
use indicatif::{ProgressBar, ProgressDrawTarget};

pub struct ReadSharedColumnsArgs {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub preload: bool,
}


pub fn read_data_on_shared_columns(args: ReadSharedColumnsArgs) {

// use matrix_util::common_io::{self, basename, extension, read_lines};

    // let file = args.data_files[0].as_ref();
    // let backend = match extension(file)?.to_string().as_str() {
    //     "h5" => SparseIoBackend::HDF5,
    //     "zarr" => SparseIoBackend::Zarr,
    //     _ => SparseIoBackend::Zarr,
    // };



}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Args, Debug)]
pub struct JointTopicArgs {
    /// Data files - different modalities
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn fit_joint_topic_model(args: &JointTopicArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 1.

    // 2.

    Ok(())
}
