use crate::embed_common::*;
use crate::routines_latent_representation::*;
use crate::routines_post_process::*;
use crate::routines_pre_process::*;

use candle_util::candle_decoder_topic::*;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_inference::TrainConfig;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_traits::DecoderModuleT;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Args, Debug)]
pub struct TopicArgs {
    /// Data files
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = false)]
    ignore_batch_effects: bool,

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// reference batch names
    #[arg(long, value_delimiter(','))]
    reference_batches: Option<Vec<Box<str>>>,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value_t = 15)]
    iter_opt: usize,

    /// block_size (# columns) for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of latent topics
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    // /// to reduce row features (#gene modules ~ 2^r)
    // #[arg(short = 'r', long, default_value_t = 10)]
    // n_row_proj_dim: usize,
    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    /// intensity levels for frequency embedding
    #[arg(long, default_value_t = 100)]
    vocab_size: usize,

    /// intensity embedding dimension
    #[arg(long, default_value_t = 3)]
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

pub fn fit_topic_model(args: &TopicArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }

    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
    } = read_sparse_data_with_membership(ReadArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
    })?;

    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;

    let n_features_decoder = data_vec.num_rows()?;
    let n_features_encoder = data_vec.num_rows()?;

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let encoder = LogSoftmaxEncoder::new(
        n_features_encoder,
        n_topics,
        n_vocab,
        d_vocab_emb,
        &args.encoder_layers,
        param_builder.clone(),
    )?;

    let decoder = TopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features_encoder, n_features_decoder
    );

    // 2. Random projection
    let proj_dim = args.proj_dim.max(args.n_latent_topics);

    let proj_out = data_vec.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(&batch_membership),
    )?;

    let proj_kn = proj_out.proj;
    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    let nsamp =
        data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    if !args.ignore_batch_effects {
        info!("Registering batch information");
        data_vec.build_hnsw_per_batch(&proj_kn, &batch_membership)?;
    }

    // 3. Batch-adjusted collapsing (pseudobulk)
    let reference = args.reference_batches.as_ref().map(|x| x.as_slice());

    info!("Collapsing columns into {} pseudobulk samples ...", nsamp);
    let collapsed = data_vec.collapse_columns(
        Some(args.knn_batches),
        Some(args.knn_cells),
        reference,
        Some(args.iter_opt),
    )?;

    let batch_db = collapsed.delta.as_ref();

    if let Some(batch_db) = batch_db {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet(Some(&gene_names), batch_names.as_deref(), &outfile)?;
    }

    // 4. Train embedded topic model on the collapsed data
    let mixed_dn = &collapsed.mu_observed;
    let clean_dn = collapsed.mu_adjusted.as_ref();
    let batch_dn = collapsed.mu_residual.as_ref();

    // let aggregator = build_row_aggregator(&collapsed, n_features_encoder)?;

    let train_config = TrainConfig {
        learning_rate: args.learning_rate,
        batch_size: args.minibatch_size,
        num_epochs: args.epochs,
        num_pretrain_epochs: 0,
        device: dev.clone(),
        verbose: args.verbose,
        show_progress: true,
    };

    let llik = train_encoder_decoder_stochastic(
        mixed_dn,
        batch_dn,
        clean_dn,
        &encoder,
        &decoder,
        &parameters,
        &loss_func::topic_likelihood,
        &train_config,
        args.jitter_interval,
    )?;

    let gene_names = data_vec.row_names()?;

    decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?
        .to_parquet(
            Some(&gene_names),
            None,
            &(args.out.to_string() + ".dictionary.parquet"),
        )?;

    write_types::<f32>(&llik, &(args.out.to_string() + ".log_likelihood.gz"))?;

    /////////////////////////////////////////////////////
    // evaluate latent states while adjusting the bias //
    /////////////////////////////////////////////////////

    let delta_db = batch_db.map(|x| x.posterior_mean());
    let z_nk = evaluate_latent_by_encoder(&data_vec, &encoder, &train_config, delta_db)?;

    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".latent.parquet"),
    )?;

    Ok(())
}
