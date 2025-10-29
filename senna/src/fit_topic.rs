use crate::embed_common::*;
use crate::senna_input::*;

use candle_core::Device;
use candle_util::candle_data_loader::*;
use candle_util::candle_decoder_topic::*;
use candle_util::candle_model_traits::*;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

// use candle_util::candle_encoder_softmax::*;
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

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum AdjMethod {
    Batch,
    Residual,
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

    /// ignore batch adjustment
    #[arg(long, default_value_t = false)]
    ignore_batch_effects: bool,

    /// warm start from the previous projection (`cell x k`)
    #[arg(short = 'w', long = "warm-start")]
    warm_start_proj_file: Option<Box<str>>,

    /// column sum normalization scale (will only affect decoder)
    #[arg(short = 'c', long, default_value_t = 1e4)]
    column_sum_norm: f32,

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// reference batch names
    #[arg(long, value_delimiter(','))]
    reference_batches: Option<Vec<Box<str>>>,

    // /// #downsampling columns per each collapsed sample. If None, no
    // /// downsampling.
    // #[arg(long, short = 's')]
    // down_sample: Option<usize>,
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
    #[arg(short = 'm', long)]
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

    /// adjust by batch or residual
    #[arg(long, value_enum, default_value = "residual")]
    adj_method: AdjMethod,

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
    env_logger::init();

    let reference = args.reference_batches.as_deref();

    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
        nbatch,
    } = read_sparse_data_with_membership(ReadArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;

    // 2. Take projection results by warm start or projecting it again
    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file.as_deref() {
        use matrix_util::common_io::*;
        let ext = extension(proj_file)?;

        let MatWithNames {
            rows: cell_names,
            cols: _,
            mat: proj_nk,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_row_names(&proj_file, Some(0))?,
            _ => Mat::read_data_with_names(&proj_file, &['\t', ',', ' '], Some(0), Some(0))?,
        };

        if data_vec.column_names()? != cell_names {
            return Err(anyhow::anyhow!(
                "warm start projection rows don't match with the data"
            ));
        }

        proj_nk.transpose()
    } else {
        let proj_dim = args.proj_dim.max(args.n_latent_topics);

        let proj_out = data_vec.project_columns_with_batch_correction(
            proj_dim,
            Some(args.block_size),
            Some(&batch_membership),
        )?;

        proj_out.proj
    };

    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    // 3. Batch-adjusted collapsing (pseudobulk)
    // assign pseudobulk samples by proj_kn
    let nsamp = data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), None)?;

    if !args.ignore_batch_effects && nbatch > 1 {
        info!("Registering batch information");
        data_vec.build_hnsw_per_batch(&proj_kn, &batch_membership)?;
    }

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

    // 4. Train a topic model on the collapsed data
    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

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

    let encoder = LogSoftmaxIAFEncoder::new(
        LogSoftmaxIAFEncoderArgs {
            n_features: n_features_encoder,
            n_topics,
            n_modules,
            n_vocab,
            d_vocab_emb,
            layers: &args.encoder_layers,
            n_transforms: args.iaf_trans,
        },
        param_builder.clone(),
    )?;

    let decoder = TopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features_encoder, n_features_decoder
    );

    let mut train_config = TrainConfig {
        learning_rate: args.learning_rate,
        batch_size: args.minibatch_size,
        num_epochs: args.epochs,
        num_pretrain_epochs: 0,
        device: dev.clone(),
        verbose: args.verbose,
        show_progress: true,
    };

    let pb = ProgressBar::new(train_config.num_epochs as u64);

    if !train_config.show_progress || train_config.verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut vae = Vae::build(&encoder, &decoder, &parameters);
    let mut log_likelihoods = Vec::with_capacity(train_config.num_epochs);

    info!("Start training VAE...");

    for epoch in (0..args.epochs).step_by(args.jitter_interval) {
        let mixed_nd = collapsed
            .mu_observed
            .posterior_sample()?
            .sum_to_one_columns()
            .scale(args.column_sum_norm)
            .transpose();

        let clean_nd = collapsed.mu_adjusted.as_ref().map(|x| {
            let mut ret: Mat = x.posterior_sample().unwrap();
            ret.sum_to_one_columns_inplace();
            ret.scale_mut(args.column_sum_norm);
            ret.transpose()
        });

        let batch_nd = collapsed.mu_residual.as_ref().map(|x| {
            let ret: Mat = x.posterior_sample().unwrap();
            ret.transpose()
        });

        let mut data_loader = InMemoryData::from(InMemoryArgs {
            input: &mixed_nd,
            input_null: batch_nd.as_ref(),
            output: clean_nd.as_ref(),
            output_null: None,
        })?;

        train_config.verbose = false;
        train_config.show_progress = args.verbose;
        train_config.num_epochs = args.jitter_interval;

        let llik = vae.train_encoder_decoder(
            &mut data_loader,
            &loss_func::topic_likelihood,
            &train_config,
        )?;

        log_likelihoods.extend(llik);
        pb.inc(args.jitter_interval as u64);

        if args.verbose {
            info!(
                "[{}] log-likelihood: {}",
                epoch + args.jitter_interval,
                log_likelihoods.last().ok_or(anyhow::anyhow!("llik"))?
            );
        }
    }

    pb.finish_and_clear();

    if train_config.verbose {
        info!("Finished {} epochs", train_config.num_epochs);
    }

    info!("Writing down the model parameters");

    let gene_names = data_vec.row_names()?;

    decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?
        .to_parquet(
            Some(&gene_names),
            None,
            &(args.out.to_string() + ".dictionary.parquet"),
        )?;

    write_types::<f32>(
        &log_likelihoods,
        &(args.out.to_string() + ".log_likelihood.gz"),
    )?;

    // encoder
    //     .feature_module_membership()?
    //     .to_device(&candle_core::Device::Cpu)?
    //     .to_parquet(
    //         Some(&gene_names),
    //         None,
    //         &(args.out.to_string() + ".feature_module.parquet"),
    //     )?;

    /////////////////////////////////////////////////////
    // evaluate latent states while adjusting the bias //
    /////////////////////////////////////////////////////

    info!("Writing down the latent states");

    // let delta = match &args.adj_method {
    //     AdjMethod::Batch => collapsed.delta.map(|x| x.posterior_mean().clone()),
    //     AdjMethod::Residual => collapsed.mu_residual.map(|x| x.posterior_mean().clone()),
    // };

    let z_nk = evaluate_latent_by_encoder(
        &data_vec,
        &encoder,
        &train_config,
        &collapsed,
        &args.adj_method,
    )?;

    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".latent.parquet"),
    )?;

    info!("Done");
    Ok(())
}

/// Evaluate latent representation with the trained encoder network
///
/// #Arguments
/// * `data_vec` - full data vector
/// * `encoder` - encoder network
/// * `train_config` - training configuration
/// * `delta_db` - batch effect matrix (feature x batch)
fn evaluate_latent_by_encoder<Enc>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    train_config: &TrainConfig,
    collapsed: &CollapsedOut,
    adj_method: &AdjMethod,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync,
{
    let dev = &train_config.device;
    let ntot = data_vec.num_columns()?;
    let kk = encoder.dim_latent();

    let block_size = train_config.batch_size;

    let jobs = create_jobs(ntot, Some(block_size));
    let njobs = jobs.len() as u64;
    let arc_enc = Arc::new(encoder);

    let delta = match adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref().map(|x| x.posterior_mean().clone()),
        AdjMethod::Residual => collapsed
            .mu_residual
            .as_ref()
            .map(|x| x.posterior_mean().clone()),
    }
    .map(|delta_db| {
        delta_db
            .to_tensor(dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
    });

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&block| match adj_method {
            AdjMethod::Residual => {
                evaluate_with_residuals(block, data_vec, arc_enc.clone(), dev, delta.as_ref())
            }
            AdjMethod::Batch => {
                evalulate_with_batch(block, data_vec, arc_enc.clone(), dev, delta.as_ref())
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    chunks.sort_by_key(|&(lb, _)| lb);
    let chunks = chunks.into_iter().map(|(_, z_nk)| z_nk).collect::<Vec<_>>();

    let mut ret = Mat::zeros(ntot, kk);
    {
        let mut lb = 0;
        for z in chunks {
            let ub = lb + z.nrows();
            ret.rows_range_mut(lb..ub).copy_from(&z);
            lb = ub;
        }
    }
    Ok(ret)
}

fn evalulate_with_batch<Enc>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: Arc<&Enc>,
    dev: &Device,
    delta_bd: Option<&Tensor>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: EncoderModuleT,
{
    let (lb, ub) = block;
    let x0_nd = delta_bd.map(|delta_bm| {
        let batches = data_vec
            .get_batch_membership(lb..ub)
            .into_iter()
            .map(|x| x as u32);
        let batches = Tensor::from_iter(batches.clone(), dev).unwrap();
        delta_bm.index_select(&batches, 0).expect("expand delta")
    });

    let x_nd = data_vec
        .read_columns_dmatrix(lb..ub)?
        .to_tensor(dev)?
        .transpose(0, 1)?;

    let (z_nk, _) = encoder.forward_t(&x_nd, x0_nd.as_ref(), false)?;
    let z_nk = z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk).expect("to mat")))
}

fn evaluate_with_residuals<Enc>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: Arc<&Enc>,
    dev: &Device,
    delta_bp: Option<&Tensor>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: EncoderModuleT,
{
    let (lb, ub) = block;
    let x0_nd = delta_bp.map(|delta_bm| {
        let groups = data_vec
            .get_group_membership(lb..ub)
            .expect("failed to get group membership")
            .into_iter()
            .map(|x| x as u32);
        let groups = Tensor::from_iter(groups.clone(), dev).unwrap();
        delta_bm.index_select(&groups, 0).expect("expand delta")
    });

    let x_nd = data_vec
        .read_columns_dmatrix(lb..ub)?
        .to_tensor(dev)?
        .transpose(0, 1)?;

    let (z_nk, _) = encoder.forward_t(&x_nd, x0_nd.as_ref(), false)?;
    let z_nk = z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk).expect("to mat")))
}
