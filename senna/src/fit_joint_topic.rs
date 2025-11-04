use crate::embed_common::*;

use crate::senna_input::*;

use candle_core::Device;
use candle_util::candle_data_loader::*;
use candle_util::candle_decoder_topic::*;
use candle_util::candle_model_traits::*;
use indicatif::ParallelProgressIterator;
use matrix_util::dmatrix_util::concatenate_vertical;
use rayon::prelude::*;

use candle_util::candle_encoder_softmax_iaf::*;
use candle_util::candle_inference::TrainConfig;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_traits::DecoderModuleT;
use candle_util::candle_vae_inference::*;
use indicatif::{ProgressBar, ProgressDrawTarget};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Args, Debug)]
pub struct JointTopicArgs {
    /// Data files
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Data modalities
    #[arg(short = 'm', long, required = true)]
    num_modalities: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// column sum normalization scale (will only affect decoder)
    #[arg(short = 'c', long, default_value_t = 1e4)]
    column_sum_norm: f32,

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// optimization iterations
    #[arg(long, default_value_t = 30)]
    iter_opt: usize,

    /// block_size (# columns) for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of latent topics
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// number of modules of the features in the encoder model.
    /// If not specified, `encoder_layers[0]` will be used.
    #[arg(short = 'f', long)]
    feature_modules: Option<usize>,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    /// number of inverse autoregressive flow transformations
    #[arg(long, default_value_t = 0)]
    iaf_trans: usize,

    /// intensity levels for frequency embedding
    #[arg(long, default_value_t = 10)]
    vocab_size: usize,

    /// intensity embedding dimension
    #[arg(long, default_value_t = 10)]
    vocab_emb: usize,

    /// # training epochs
    #[arg(long, short = 'i', default_value_t = 1000)]
    epochs: usize,

    /// data jitter interval
    #[arg(long, short = 'j', default_value_t = 5)]
    jitter_interval: usize,

    /// Minibatch size
    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

    /// preload all the columns data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn fit_joint_topic_model(args: &JointTopicArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 1. Read the data with batch membership
    let SparseStackWithBatch {
        mut data_stack,
        batch_stack,
        nbatch_stack,
    } = read_data_on_shared_columns(ReadSharedColumnsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        num_types: args.num_modalities,
        preload: args.preload_data,
    })?;

    // 2. Concatenate projections
    let proj_dim = args.proj_dim.max(args.n_latent_topics);
    let proj_out = data_stack.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(batch_stack[0].as_ref()),
    )?;
    let proj_kn = proj_out.proj;

    // 3. Batch-adjusted collapsing (pseudobulk)
    // assign pseudobulk samples by proj_kn
    let nsamp = data_stack.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), None)?;

    for (d, data_vec) in data_stack.stack.iter_mut().enumerate() {
        let nbatch = nbatch_stack[d];
        let batch_membership = &batch_stack[d];

        if nbatch > 1 {
            info!("Registering batch information");
            data_vec.build_hnsw_per_batch(&proj_kn, batch_membership)?;
        }
    }

    info!("Collapsing columns into {} pseudobulk samples ...", nsamp);

    let collapsed_data_vec = data_stack
        .stack
        .iter()
        .map(|x| {
            x.collapse_columns(
                Some(args.knn_batches),
                Some(args.knn_cells),
                None,
                Some(args.iter_opt),
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // 4. Train a topic model on the collapsed data

    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    for epoch in (0..args.epochs).step_by(args.jitter_interval) {}

    Ok(())
}
