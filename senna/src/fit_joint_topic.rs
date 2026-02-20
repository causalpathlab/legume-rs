use crate::embed_common::*;
use crate::feature_selection::*;
use crate::senna_input::*;

use candle_nn::AdamW;
use candle_nn::Optimizer;

use candle_core::Device;
use candle_util::candle_decoder_multimodal_topic::*;
use candle_util::candle_encoder_multimodal_softmax::*;
use candle_util::candle_joint_data_loader::*;
use candle_util::candle_loss_functions::topic_likelihood;
use candle_util::candle_model_traits::*;
use indicatif::{ParallelProgressIterator, ProgressBar};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

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
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided (space or comma separated)."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        short = 'm',
        long = "modalities",
        help = "Data modalities",
        long_help = "We will treat the provided data files as\n\
		     a table of data sets in a row-major order.\n\
		     This number of modalities will determine \n\
		     how many different data types are assumed,\n\
		     or the number of rows in the data table.",
        required = true
    )]
    num_modalities: usize,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header for results.\n\
		     Specify the output file or prefix for generated files:\n\
		     - {out}.delta.parquet\n\
		     - {out}.dictionary.parquet\n\
		     - {out}.latent.parquet\n"
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension.",
        long_help = "Random projection dimension to project the data.\n\
		     Controls the dimensionality of the random projection step."
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection.",
        long_help = "Use top {d} components of projection.\n\
		     Number of samples will be less than `2^{d}+1`."
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files.",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.csv,batch2.csv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column sum normalization scale.",
        long_help = "Column sum normalization scale (affects decoder only).\n\
		     Adjusts normalization of columns in the decoder."
    )]
    column_sum_norm: f32,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of k-nearest neighbour batches.",
        long_help = "Number of k-nearest neighbour batches.\n\
		     Controls the number of batches considered \n\
		     for nearest neighbour search."
    )]
    knn_batches: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch.",
        long_help = "Number of k-nearest neighbours within each batch.\n\
		     Controls the number of cells considered \n\
		     for nearest neighbour search within each batch."
    )]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Optimization iterations.",
        long_help = "Number of optimization iterations.\n\
		     Controls the number of steps for model optimization."
    )]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing.",
        long_help = "Block size (number of columns) for parallel processing.\n\
		     Controls the granularity of parallel computation."
    )]
    block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics.",
        long_help = "Number of latent topics.\n\
		     Controls the dimensionality of the latent topic space."
    )]
    n_latent_topics: usize,

    #[arg(
        short = 'f',
        long,
        help = "Number of feature modules.",
        long_help = "Number of modules of the features in the encoder model.\n\
		     If not specified, encoder_layers[0] will be used. \n\
		     Giving the number of features modules smaller than that of features,\n\
		     we can expedite model training while not loosing too much of accuracy,\n\
		     as many features are redundant and frequently dropped out.\n\n\
		     We will assume the same number of feature modules for all data types.\n"
    )]
    feature_modules: Option<usize>,

    #[arg(
        long,
        help = "Maximum number of highly variable features per modality.",
        long_help = "Select top N features by log-variance for each modality.\n\
		     If not specified, all features are used.\n\
		     Applied independently to each data modality."
    )]
    max_features: Option<usize>,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder layers.",
        long_help = "Encoder layers (comma-separated).\n\
		     Specify the size of each layer in the encoder model.\n\
		     Example: 128,1024,128"
    )]
    encoder_layers: Vec<usize>,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "KL annealing warmup epochs",
        long_help = "Number of epochs for KL weight to warm up from 0 to 1.\n\
		     Standard warm-up: kl_weight = 1 - exp(-epoch / warmup)\n\
		     Larger value = slower warm-up. Set to 0 to disable annealing."
    )]
    kl_warmup_epochs: f64,

    #[arg(
        long,
        default_value_t = 10,
        help = "Intensity levels for frequency embedding.",
        long_help = "Intensity levels for frequency embedding.\n\
		     Controls the vocabulary size for intensity embedding."
    )]
    vocab_size: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Intensity embedding dimension.",
        long_help = "Intensity embedding dimension.\n\
		     Controls the size of the embedding for intensity levels."
    )]
    vocab_emb: usize,

    #[arg(
        long,
        short = 'i',
        default_value_t = 1000,
        help = "Number of training epochs.",
        long_help = "Number of training epochs.\n\
		     Controls how many times the model is trained over the data."
    )]
    epochs: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Data jitter interval.",
        long_help = "Data jitter interval.\n\
		     Controls the interval for adding jitter to the collapsed data\n\
		     by posterior resampling during VAE training."
    )]
    jitter_interval: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Minibatch size.",
        long_help = "Minibatch size for training.\n\
		     Controls the number of samples per training batch."
    )]
    minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        help = "Learning rate.",
        long_help = "Learning rate for optimization.\n\
		     Controls the step size for parameter updates."
    )]
    learning_rate: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Candle device.",
        long_help = "Candle device to use for computation.\n\
		     Options: cpu, cuda, metal."
    )]
    device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "A device for cuda.",
        long_help = "For cuda or meta, we may want to choose a different device."
    )]
    device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Adjustment method.",
        long_help = "Adjust by batch or residual.\n\
		     Choose the method for batch adjustment."
    )]
    adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = 2,
        help = "Number of multi-level collapsing levels.",
        long_help = "Number of multi-level collapsing levels.\n\
		     More levels = coarser-to-finer batch correction.\n\
		     Set to 1 to disable multi-level."
    )]
    num_levels: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all columns data.",
        long_help = "Preload all the columns data into memory.\n\
		     Improves performance for large datasets."
    )]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Use sparsemax instead of softmax",
        long_help = "Use sparsemax activation instead of softmax.\n\
		     Sparsemax can output exact zeros for sparse topic assignments.\n\
		     May help with more decisive cell type annotations.\n\
		     Note: Mutually exclusive with --iaf-trans (IAF will be disabled)."
    )]
    use_sparsemax: bool,
}

pub fn fit_joint_topic_model(args: &JointTopicArgs) -> anyhow::Result<()> {
    if args.use_sparsemax {
        info!("Using sparsemax activation for sparse topic assignments");
    }

    // 1. Read the data with batch membership
    let SparseStackWithBatch {
        mut data_stack,
        batch_stack,
    } = read_data_on_shared_columns(ReadSharedColumnsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        num_types: args.num_modalities,
        preload: args.preload_data,
    })?;

    // 1b. Feature selection per modality (if requested)
    let feature_selections: Vec<Option<FeatureSelection>> =
        if let Some(max_feat) = args.max_features {
            info!(
                "Selecting top {} features per modality by log-variance",
                max_feat
            );
            data_stack
                .stack
                .iter()
                .enumerate()
                .map(|(d, data_vec)| {
                    let sel = select_highly_variable_features(
                        data_vec,
                        Some(max_feat),
                        None,
                        false,
                        &format!("{}_{}", args.out, d),
                        args.block_size,
                        None,
                    )?;
                    info!(
                        "Modality {}: selected {} features",
                        d,
                        sel.selected_indices.len()
                    );
                    Ok(Some(sel))
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            data_stack.stack.iter().map(|_| None).collect()
        };

    // 2. Concatenate projections
    let proj_dim = args.proj_dim.max(args.n_latent_topics);
    let proj_out = data_stack.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(batch_stack[0].as_ref()),
    )?;
    let proj_kn = proj_out.proj;

    // 3. Batch-adjusted multilevel collapsing (pseudobulk)
    info!("Multi-level collapsing across {} modalities ...", data_stack.num_types());

    let collapsed_levels: Vec<Vec<CollapsedOut>> = data_stack.collapse_columns_multilevel_vec(
        &proj_kn,
        batch_stack[0].as_ref(),
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
        },
    )?;

    // For delta output, n_features, and latent evaluation, use the finest level
    let collapsed_data_vec = collapsed_levels.last().unwrap();

    // 4. output batch effect information
    for (d, collapsed) in collapsed_data_vec.iter().enumerate() {
        if let Some(batch_db) = &collapsed.delta {
            let outfile = format!("{}_{}.delta.parquet", args.out, d);
            let data_vec = &data_stack.stack[d];
            let batch_names = data_vec.batch_names();
            let gene_names = data_vec.row_names()?;
            batch_db.to_parquet_with_names(
                &outfile,
                (Some(&gene_names), Some("gene")),
                batch_names.as_deref(),
            )?;
        }
    }

    // 5. Train a joint topic model on the collapsed data (progressive)
    let n_topics = args.n_latent_topics;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(args.device_no)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(args.device_no)?,
        _ => candle_core::Device::Cpu,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let n_features: Vec<usize> = collapsed_data_vec
        .iter()
        .zip(&feature_selections)
        .map(|(x, sel)| {
            sel.as_ref()
                .map(|s| s.selected_indices.len())
                .unwrap_or_else(|| x.mu_observed.nrows())
        })
        .collect();

    let encoder = LogSoftmaxMultimodalEncoder::new(
        LogSoftmaxMultimodalEncoderArgs {
            n_features: n_features.clone(),
            n_topics,
            n_modules,
            layers: &args.encoder_layers,
            use_sparsemax: args.use_sparsemax,
        },
        param_builder.clone(),
    )?;

    let decoder = MultimodalTopicDecoder::new(
        &n_features.clone(),
        args.n_latent_topics,
        param_builder.clone(),
    )?;

    // Set up graceful stop flag for SIGINT/SIGTERM
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            info!("Interrupt received — stopping training early and saving results...");
            stop.store(true, Ordering::SeqCst);
        })
        .expect("failed to set signal handler");
    }

    let scores = train_encoder_decoder_progressive(
        &collapsed_levels,
        &encoder,
        &decoder,
        &parameters,
        &dev,
        args,
        &feature_selections,
        &stop,
    )?;

    info!("Writing down the model parameters");

    // Get gene names - use selected names if feature selection was applied
    let gene_names: Vec<Box<str>> = data_stack
        .stack
        .iter()
        .zip(&feature_selections)
        .map(|(dv, sel)| -> anyhow::Result<Vec<Box<str>>> {
            if let Some(sel) = sel {
                Ok(sel.selected_names.clone())
            } else {
                dv.row_names()
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    let dictionaries = decoder
        .get_dictionary()?
        .into_iter()
        .map(|x| x.to_device(&candle_core::Device::Cpu))
        .collect::<candle_core::Result<Vec<_>>>()?;

    candle_core::Tensor::cat(&dictionaries, 0)?.to_parquet_with_names(
        &(args.out.to_string() + ".dictionary.parquet"),
        (Some(&gene_names), Some("gene")),
        None,
    )?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    /////////////////////////////////////////////////////
    // evaluate latent states while adjusting the bias //
    /////////////////////////////////////////////////////

    info!("Writing down the latent states");

    let z_nk = evaluate_latent_by_encoder(
        &data_stack,
        &encoder,
        &collapsed_data_vec,
        &dev,
        args,
        &feature_selections,
    )?;
    let cell_names = data_stack.column_names()?;
    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    info!("Done");
    Ok(())
}

fn evaluate_latent_by_encoder<Enc>(
    data_stack: &SparseIoStack,
    encoder: &Enc,
    collapsed_vec: &[CollapsedOut],
    dev: &candle_core::Device,
    args: &JointTopicArgs,
    feature_selections: &[Option<FeatureSelection>],
) -> anyhow::Result<Mat>
where
    Enc: MultimodalEncoderModuleT + Send + Sync,
{

    let ntot = data_stack.num_columns()?;
    let kk = encoder.dim_latent();

    let block_size = args.minibatch_size;

    let jobs = create_jobs(ntot, Some(block_size));
    let njobs = jobs.len() as u64;
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
                    .to_tensor(dev)?
                    .transpose(0, 1)?)
            })
            .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let eval_block = |&block: &(usize, usize)| match args.adj_method {
        AdjMethod::Residual => evaluate_with_residuals(
            block,
            data_stack,
            encoder,
            dev,
            delta.as_ref(),
            feature_selections,
        ),
        AdjMethod::Batch => evaluate_with_batch(
            block,
            data_stack,
            encoder,
            dev,
            delta.as_ref(),
            feature_selections,
        ),
    };

    // Metal/CUDA don't support parallel dispatch to the same device
    let use_sequential = !dev.is_cpu();

    let mut chunks: Vec<(usize, Mat)> = if use_sequential {
        let pb = ProgressBar::new(njobs);
        let result = jobs
            .iter()
            .map(|block| {
                let r = eval_block(block);
                pb.inc(1);
                r
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        pb.finish_and_clear();
        result
    } else {
        jobs.par_iter()
            .progress_count(njobs)
            .map(eval_block)
            .collect::<anyhow::Result<Vec<_>>>()?
    };

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

fn evaluate_with_batch<Enc>(
    block: (usize, usize),
    data_stack: &SparseIoStack,
    encoder: &Enc,
    dev: &Device,
    delta_bd_vec: &[Option<Tensor>],
    feature_selections: &[Option<FeatureSelection>],
) -> anyhow::Result<(usize, Mat)>
where
    Enc: MultimodalEncoderModuleT,
{
    let (lb, ub) = block;

    let x_vec = data_stack
        .stack
        .iter()
        .zip(feature_selections.iter())
        .map(|(dv, sel)| -> anyhow::Result<Tensor> {
            let x = dv.read_columns_tensor(lb..ub)?;
            let x = if let Some(sel) = sel {
                let indices =
                    Tensor::from_iter(sel.selected_indices.iter().map(|&i| i as u32), dev)?;
                x.index_select(&indices, 0)?
            } else {
                x
            };
            Ok(x.to_device(dev)?.transpose(0, 1)?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let x0_vec = data_stack
        .stack
        .iter()
        .zip(delta_bd_vec)
        .zip(feature_selections.iter())
        .map(|((dv, delta), sel)| {
            delta
                .as_ref()
                .map(|delta| -> anyhow::Result<Tensor> {
                    let batches = dv
                        .get_batch_membership(lb..ub)
                        .into_iter()
                        .map(|j| j as u32);
                    let batches = Tensor::from_iter(batches, dev)?;
                    let selected_delta = delta.index_select(&batches, 0)?;
                    // Also select features from delta if feature selection is active
                    if let Some(sel) = sel {
                        let indices =
                            Tensor::from_iter(sel.selected_indices.iter().map(|&i| i as u32), dev)?;
                        Ok(selected_delta.index_select(&indices, 1)?)
                    } else {
                        Ok(selected_delta)
                    }
                })
                .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let (z_nk, _) = encoder.forward_t(&x_vec, &x0_vec, false)?;
    let z_nk = z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

fn evaluate_with_residuals<Enc>(
    block: (usize, usize),
    data_stack: &SparseIoStack,
    encoder: &Enc,
    dev: &Device,
    delta_bd_vec: &[Option<Tensor>],
    feature_selections: &[Option<FeatureSelection>],
) -> anyhow::Result<(usize, Mat)>
where
    Enc: MultimodalEncoderModuleT,
{
    let (lb, ub) = block;

    let x_vec = data_stack
        .stack
        .iter()
        .zip(feature_selections.iter())
        .map(|(dv, sel)| -> anyhow::Result<Tensor> {
            let x = dv.read_columns_tensor(lb..ub)?;
            let x = if let Some(sel) = sel {
                let indices =
                    Tensor::from_iter(sel.selected_indices.iter().map(|&i| i as u32), dev)?;
                x.index_select(&indices, 0)?
            } else {
                x
            };
            Ok(x.to_device(dev)?.transpose(0, 1)?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let x0_vec = data_stack
        .stack
        .iter()
        .zip(delta_bd_vec)
        .zip(feature_selections.iter())
        .map(|((dv, delta), sel)| {
            delta
                .as_ref()
                .map(|delta| -> anyhow::Result<Tensor> {
                    let groups = dv
                        .get_group_membership(lb..ub)?
                        .into_iter()
                        .map(|j| j as u32);
                    let groups = Tensor::from_iter(groups, dev)?;
                    let selected_delta = delta.index_select(&groups, 0)?;
                    // Also select features from delta if feature selection is active
                    if let Some(sel) = sel {
                        let indices =
                            Tensor::from_iter(sel.selected_indices.iter().map(|&i| i as u32), dev)?;
                        Ok(selected_delta.index_select(&indices, 1)?)
                    } else {
                        Ok(selected_delta)
                    }
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
            .map(|x| (x + 1).to_string().into_boxed_str())
            .collect();

        mat.to_parquet_with_names(
            file_path,
            (Some(&epochs), Some("epoch")),
            Some(&score_types),
        )
    }
}

/// Progressive multi-level training: coarse levels get more epochs for
/// warm start, finer levels for fine-tuning.
/// Epoch allocation: level `i` gets `total_epochs * (num_levels - i) / sum(1..=num_levels)`.
fn train_encoder_decoder_progressive<Enc, Dec>(
    collapsed_levels: &[Vec<CollapsedOut>],
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    dev: &candle_core::Device,
    args: &JointTopicArgs,
    feature_selections: &[Option<FeatureSelection>],
    stop: &AtomicBool,
) -> anyhow::Result<TrainScores>
where
    Enc: MultimodalEncoderModuleT,
    Dec: MultimodalDecoderModuleT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = args.epochs;

    // Compute per-level epoch allocation: w[i] = num_levels - i
    let total_weight: usize = (1..=num_levels).sum();
    let level_epochs: Vec<usize> = (0..num_levels)
        .map(|i| {
            let w = num_levels - i;
            (total_epochs * w / total_weight).max(1)
        })
        .collect();

    info!(
        "Progressive training: {} levels, epoch allocation: {:?} (total {})",
        num_levels,
        level_epochs,
        level_epochs.iter().sum::<usize>()
    );

    let mut adam = AdamW::new_lr(parameters.all_vars(), args.learning_rate as f64)?;

    let total_actual_epochs: usize = level_epochs.iter().sum();
    let pb = ProgressBar::new(total_actual_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_actual_epochs);
    let mut kl_trace = Vec::with_capacity(total_actual_epochs);
    let mut global_epoch: usize = 0;

    for (level, (collapsed_data_vec, &level_ep)) in
        collapsed_levels.iter().zip(level_epochs.iter()).enumerate()
    {
        info!(
            "Level {}/{}: {} epochs, {} samples",
            level + 1,
            num_levels,
            level_ep,
            collapsed_data_vec[0].mu_observed.ncols(),
        );

        for epoch in (0..level_ep).step_by(args.jitter_interval) {
            //////////////////////////////////////////
            // every jitter interval, resample data //
            //////////////////////////////////////////

            let input = collapsed_data_vec
                .iter()
                .zip(feature_selections)
                .map(|(x, sel)| -> anyhow::Result<Mat> {
                    let mat = x
                        .mu_observed
                        .posterior_sample()?
                        .sum_to_one_columns()
                        .scale(args.column_sum_norm);
                    let mat = if let Some(sel) = sel {
                        mat.select_rows(&sel.selected_indices)
                    } else {
                        mat
                    };
                    Ok(mat.transpose())
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let input_null = collapsed_data_vec
                .iter()
                .zip(feature_selections)
                .map(|(x, sel)| -> anyhow::Result<Option<Mat>> {
                    x.mu_residual
                        .as_ref()
                        .map(|y| {
                            let mat = y.posterior_sample()?;
                            let mat = if let Some(sel) = sel {
                                mat.select_rows(&sel.selected_indices)
                            } else {
                                mat
                            };
                            Ok(mat.transpose())
                        })
                        .transpose()
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let output = collapsed_data_vec
                .iter()
                .zip(feature_selections)
                .map(|(x, sel)| -> anyhow::Result<Option<Mat>> {
                    Ok(x.mu_adjusted
                        .as_ref()
                        .map(|y| y.posterior_sample())
                        .transpose()?
                        .map(|y| {
                            let mat = y.sum_to_one_columns().scale(args.column_sum_norm);
                            let mat = if let Some(sel) = sel {
                                mat.select_rows(&sel.selected_indices)
                            } else {
                                mat
                            };
                            mat.transpose()
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

            let kl_weight = if args.kl_warmup_epochs > 0.0 {
                1.0 - (-(global_epoch as f64) / args.kl_warmup_epochs).exp()
            } else {
                1.0
            };

            let jitter_end = args.jitter_interval.min(level_ep - epoch);
            for jitter in 0..jitter_end {
                let mut llik_tot = 0f32;
                let mut kl_tot = 0f32;

                for b in 0..data_loader.num_minibatch() {
                    let mb = data_loader.minibatch_shuffled(b, dev)?;

                    let (z_nk, kl) = encoder.forward_t(&mb.input, &mb.input_null, true)?;

                    let y_vec = mb
                        .output
                        .into_iter()
                        .zip(mb.input)
                        .map(|(y, x)| y.unwrap_or(x))
                        .collect::<Vec<_>>();

                    let (_, llik) =
                        decoder.forward_with_llik(&z_nk, &y_vec, &topic_likelihood)?;

                    let loss = ((&kl * kl_weight)? - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;

                    let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                    let kl_val = kl.sum_all()?.to_scalar::<f32>()?;
                    llik_tot += llik_val;
                    kl_tot += kl_val;
                }

                let n_mb = data_loader.num_minibatch() as f32;
                kl_trace.push(kl_tot / n_mb);
                llik_trace.push(llik_tot / n_mb);

                pb.inc(1);
                global_epoch += 1;

                info!(
                    "[level {}/{}][{}][{}] {} {}",
                    level + 1,
                    num_levels,
                    epoch,
                    jitter,
                    llik_tot / n_mb,
                    kl_tot / n_mb
                );

                if stop.load(Ordering::SeqCst) {
                    pb.finish_and_clear();
                    info!(
                        "Stopping training early at level {}/{}, epoch {}",
                        level + 1,
                        num_levels,
                        epoch
                    );
                    return Ok(TrainScores {
                        llik: llik_trace,
                        kl: kl_trace,
                    });
                }
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
