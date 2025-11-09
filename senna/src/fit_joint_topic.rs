use crate::embed_common::*;
use crate::senna_input::*;

use candle_nn::AdamW;
use candle_nn::Optimizer;

use candle_core::Device;
use candle_util::candle_decoder_multimodal_topic::*;
use candle_util::candle_encoder_multimodal_softmax::*;
use candle_util::candle_joint_data_loader::*;
use candle_util::candle_loss_functions::topic_likelihood;
use candle_util::candle_model_traits::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget};
use rayon::prelude::*;

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

    // 4. Train a joint topic model on the collapsed data
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

    let n_features: Vec<usize> = collapsed_data_vec
        .iter()
        .map(|x| x.mu_observed.nrows())
        .collect();

    let encoder = LogSoftmaxMultimodalEncoder::new(
        LogSoftmaxMultimodalEncoderArgs {
            n_features: n_features.clone(),
            n_topics,
            n_modules,
            n_vocab,
            d_vocab_emb,
            layers: &args.encoder_layers,
        },
        param_builder.clone(),
    )?;

    let decoder = MultimodalTopicDecoder::new(
        &n_features.clone(),
        args.n_latent_topics,
        param_builder.clone(),
    )?;

    let scores =
        train_encoder_decoder(&collapsed_data_vec, &encoder, &decoder, &parameters, &args)?;

    info!("Writing down the model parameters");

    let gene_names = data_stack.row_names()?;

    let dictionaries = decoder
        .get_dictionary()?
        .into_iter()
        .map(|x| x.to_device(&candle_core::Device::Cpu))
        .collect::<candle_core::Result<Vec<_>>>()?;

    candle_core::Tensor::cat(&dictionaries, 0)?.to_parquet(
        Some(&gene_names),
        None,
        &(args.out.to_string() + ".dictionary.parquet"),
    )?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    /////////////////////////////////////////////////////
    // evaluate latent states while adjusting the bias //
    /////////////////////////////////////////////////////

    info!("Writing down the latent states");

    let z_nk = evaluate_latent_by_encoder(&data_stack, &encoder, &collapsed_data_vec, &args)?;
    let cell_names = data_stack.column_names()?;
    z_nk.to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".latent.parquet"),
    )?;

    info!("Done");
    Ok(())
}

fn evaluate_latent_by_encoder<Enc>(
    data_stack: &SparseIoStack,
    encoder: &Enc,
    collapsed_vec: &[CollapsedOut],
    args: &JointTopicArgs,
) -> anyhow::Result<Mat>
where
    Enc: MultimodalEncoderModuleT + Send + Sync,
{
    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let ntot = data_stack.num_columns()?;
    let kk = encoder.dim_latent();

    let block_size = args.minibatch_size;

    let jobs = create_jobs(ntot, Some(block_size));
    let njobs = jobs.len() as u64;
    let arc_enc = Arc::new(encoder);

    // potential batch effects
    let delta = collapsed_vec
        .iter()
        .map(|x| {
            match args.adj_method {
                AdjMethod::Residual => x.mu_residual.as_ref(),
                AdjMethod::Batch => x.delta.as_ref(),
            }
            .map(|delta| -> anyhow::Result<Tensor> {
                Ok(delta
                    .posterior_mean()
                    .clone()
                    .to_tensor(&dev)?
                    .transpose(0, 1)?)
            })
            .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&block| match args.adj_method {
            AdjMethod::Residual => {
                evalulate_with_residuals(block, data_stack, arc_enc.clone(), &dev, delta.as_ref())
            }
            AdjMethod::Batch => {
                evalulate_with_batch(block, data_stack, arc_enc.clone(), &dev, delta.as_ref())
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    chunks.par_sort_by_key(|&(lb, _)| lb);
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
    data_stack: &SparseIoStack,
    encoder: Arc<&Enc>,
    dev: &Device,
    delta_bd_vec: &[Option<Tensor>],
) -> anyhow::Result<(usize, Mat)>
where
    Enc: MultimodalEncoderModuleT,
{
    let (lb, ub) = block;

    let x_vec = data_stack
        .stack
        .iter()
        .map(|dv| -> anyhow::Result<Tensor> {
            Ok(dv
                .read_columns_tensor(lb..ub)?
                .to_device(dev)?
                .transpose(0, 1)?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let x0_vec = data_stack
        .stack
        .iter()
        .zip(delta_bd_vec)
        .map(|(dv, delta)| {
            delta
                .as_ref()
                .map(|delta| -> anyhow::Result<Tensor> {
                    let batches = dv
                        .get_batch_membership(lb..ub)
                        .into_iter()
                        .map(|j| j as u32);
                    let batches = Tensor::from_iter(batches, dev)?;
                    Ok(delta.index_select(&batches, 0)?)
                })
                .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let (z_nk, _) = encoder.forward_t(&x_vec, &x0_vec, false)?;
    let z_nk = z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

fn evalulate_with_residuals<Enc>(
    block: (usize, usize),
    data_stack: &SparseIoStack,
    encoder: Arc<&Enc>,
    dev: &Device,
    delta_bd_vec: &[Option<Tensor>],
) -> anyhow::Result<(usize, Mat)>
where
    Enc: MultimodalEncoderModuleT,
{
    let (lb, ub) = block;

    let x_vec = data_stack
        .stack
        .iter()
        .map(|dv| -> anyhow::Result<Tensor> {
            Ok(dv
                .read_columns_tensor(lb..ub)?
                .to_device(dev)?
                .transpose(0, 1)?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let x0_vec = data_stack
        .stack
        .iter()
        .zip(delta_bd_vec)
        .map(|(dv, delta)| {
            delta
                .as_ref()
                .map(|delta| -> anyhow::Result<Tensor> {
                    let groups = dv
                        .get_group_membership(lb..ub)?
                        .into_iter()
                        .map(|j| j as u32);
                    let groups = Tensor::from_iter(groups, dev)?;
                    Ok(delta.index_select(&groups, 0)?)
                })
                .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let (z_nk, _) = encoder.forward_t(&x_vec, &x0_vec, false)?;
    let z_nk = z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

///////////////////////
// training routines //
///////////////////////

struct TrainScores {
    llik: Vec<f32>,
    kl: Vec<f32>,
}

impl TrainScores {
    fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        let mat = Mat::from_columns(&[
            DVec::from_vec(self.llik.clone()),
            DVec::from_vec(self.kl.clone()),
        ]);

        let score_types = vec![
            "log_likelihood".to_string().into_boxed_str(),
            "kl_divergence".to_string().into_boxed_str(),
        ];

        let epochs: Vec<Box<str>> = (0..mat.nrows())
            .into_iter()
            .map(|x| (x + 1).to_string().into_boxed_str())
            .collect();

        mat.to_parquet(Some(&epochs), Some(&score_types), file_path)
    }
}

fn train_encoder_decoder<Enc, Dec>(
    collapsed_data_vec: &[CollapsedOut],
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    args: &JointTopicArgs,
) -> anyhow::Result<TrainScores>
where
    Enc: MultimodalEncoderModuleT,
    Dec: MultimodalDecoderModuleT,
{
    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let mut adam = AdamW::new_lr(parameters.all_vars(), args.learning_rate as f64)?;

    let pb = ProgressBar::new(args.epochs as u64);

    if args.verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut llik_trace = Vec::with_capacity(args.epochs);
    let mut kl_trace = Vec::with_capacity(args.epochs);

    info!("Start training VAE...");

    for epoch in (0..args.epochs).step_by(args.jitter_interval) {
        //////////////////////////////////////////
        // every jitter interval, resample data //
        //////////////////////////////////////////

        let input = collapsed_data_vec
            .iter()
            .map(|x| -> anyhow::Result<Mat> {
                Ok(x.mu_observed
                    .posterior_sample()?
                    .sum_to_one_columns()
                    .scale(args.column_sum_norm)
                    .transpose())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let input_null = collapsed_data_vec
            .iter()
            .map(|x| -> anyhow::Result<Option<Mat>> {
                x.mu_residual
                    .as_ref()
                    .map(|y| Ok(y.posterior_sample()?.transpose()))
                    .transpose()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let output = collapsed_data_vec
            .iter()
            .map(|x| -> anyhow::Result<Option<Mat>> {
                Ok(x.mu_adjusted
                    .as_ref()
                    .map(|y| y.posterior_sample())
                    .transpose()?
                    .map(|y| {
                        y.sum_to_one_columns()
                            .scale(args.column_sum_norm)
                            .transpose()
                    }))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut data_loader = JointInMemoryData::from(JointInMemoryArgs {
            input: &input,
            input_null: &input_null,
            output: &output,
            output_null: &vec![None; input.len()],
        })?;

        data_loader.shuffle_minibatch(args.minibatch_size)?;

        for jitter in 0..args.jitter_interval {
            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;

            for b in 0..data_loader.num_minibatch() {
                let mb = data_loader.minibatch_shuffled(b, &dev)?;

                let (z_nk, kl) = encoder.forward_t(&mb.input, &mb.input_null, true)?;

                let y_vec = mb
                    .output
                    .into_iter()
                    .zip(mb.input)
                    .map(|(y, x)| y.unwrap_or(x))
                    .collect::<Vec<_>>();

                let (_, llik) = decoder.forward_with_llik(&z_nk, &y_vec, &topic_likelihood)?;

                let loss = (&kl - &llik)?.mean_all()?;
                adam.backward_step(&loss)?;

                let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                let kl_val = kl.sum_all()?.to_scalar::<f32>()?;
                llik_tot += llik_val;
                kl_tot += kl_val;
            }

            kl_trace.push(kl_tot / data_loader.num_minibatch() as f32);
            llik_trace.push(llik_tot / data_loader.num_minibatch() as f32);

            pb.inc(1);

            if args.verbose {
                info!("[{}][{}] {} {}", epoch, jitter, llik_tot, kl_tot);
            }
        }
    }
    pb.finish_and_clear();

    info!("done model training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}
