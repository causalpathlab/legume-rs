use crate::embed_common::*;
use crate::senna_input::*;

use candle_core::{Device, Tensor};
use candle_nn::AdamW;
use candle_nn::Optimizer;
use candle_util::candle_data_loader::*;
use candle_util::candle_decoder_topic::*;
use candle_util::candle_loss_functions::topic_likelihood;
use candle_util::candle_model_traits::*;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use candle_util::candle_encoder_softmax::*;
use candle_util::candle_model_traits::DecoderModuleT;
use candle_util::candle_topic_refinement::*;
use indicatif::ProgressBar;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum DecoderType {
    /// Softmax dictionary with multinomial likelihood
    Multinom,
    /// Negative binomial with per-gene dispersion and library size
    Nb,
}

#[derive(Args, Debug)]
pub struct TopicArgs {
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
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header for results.\n\
		     Specify the output file or prefix for generated files:\n\
		     - {out}.delta.parquet\n\
		     - {out}.dictionary.parquet\n\
		     - {out}.latent.parquet (log-softmax topic proportions)\n"
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Random projection dimension to project the data.\n\
		     Controls the dimensionality of the random projection step."
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection",
        long_help = "Use top {d} components of projection.\n\
		     Number of samples will be less than `2^{d}+1`."
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.csv,batch2.csv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'w',
        long = "warm-start",
        help = "Warm start projection file",
        long_help = "Warm start from the previous projection (cell x k).\n\
		     Provide a file to initialize the projection."
    )]
    warm_start_proj_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch",
        long_help = "Number of k-nearest neighbours within each batch.\n\
		     Controls the number of neighbours for \n\
		     nearest neighbour search."
    )]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of multi-level coarsening levels",
        long_help = "Number of multi-level coarsening levels for batch correction.\n\
		     Higher values add intermediate refinement steps.\n\
		     Level sort dimensions are linearly spaced from 4 to sort_dim."
    )]
    num_levels: usize,

    #[arg(
        long,
        value_enum,
        default_value = "mixed",
        help = "Multi-level training schedule"
    )]
    level_schedule: LevelSchedule,

    #[arg(
        long,
        default_value_t = 30,
        help = "Optimization iterations",
        long_help = "Number of optimization iterations.\n\
		     Controls the number of steps for model optimization."
    )]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing",
        long_help = "Block size (number of columns) for parallel processing.\n\
		     Controls the granularity of parallel computation."
    )]
    block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics",
        long_help = "Number of latent topics.\n\
		     Controls the dimensionality of the latent topic space."
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder layers",
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
        short = 'i',
        default_value_t = 1000,
        help = "Number of training epochs",
        long_help = "Number of training epochs.\n\
		     Controls how many times the model is trained over the data."
    )]
    epochs: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Data jitter interval",
        long_help = "Data jitter interval.\n\
		     Controls the interval for adding jitter to the collapsed data\n\
		     by posterior resampling during VAE training."
    )]
    jitter_interval: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Minibatch size",
        long_help = "Minibatch size for training.\n\
		     Controls the number of samples per training batch."
    )]
    minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        help = "Learning rate",
        long_help = "Learning rate for optimization.\n\
		     Controls the step size for parameter updates."
    )]
    learning_rate: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Candle device",
        long_help = "Candle device to use for computation.\n\
		     Options: cpu, cuda, metal."
    )]
    device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "A device for cuda",
        long_help = "For cuda or meta, we may want to choose a different device."
    )]
    device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Adjustment method",
        long_help = "Adjust by batch or residual.\n\
		     Choose the method for batch adjustment."
    )]
    adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all columns data",
        long_help = "Preload all the columns data into memory.\n\
		     Improves performance for large datasets."
    )]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = 5000,
        help = "Cap feature dimension by coarsening",
        long_help = "Cap the feature dimension by grouping co-expressed features into\n\
		     meta-features. The model trains at this reduced resolution.\n\
		     On output, the dictionary is expanded back to full resolution.\n\
		     Set to 0 to disable. Default: 5000.\n\
		     Applied after --max-features selection if both are specified."
    )]
    max_coarse_features: usize,

    #[arg(
        long,
        value_enum,
        default_value = "multinom",
        help = "Decoder type",
        long_help = "Topic decoder type:\n\
		     multinom: softmax dictionary with multinomial likelihood\n\
		     nb: negative binomial with per-gene dispersion and library size"
    )]
    decoder: DecoderType,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Topic smoothing during training",
        long_help = "Mix encoder topic proportions with uniform distribution during training:\n\
		     z_smooth = (1 - α) * z_nk + α / K\n\
		     Ensures every topic receives gradient signal through the decoder,\n\
		     preventing dead topics. Only applied during training.\n\
		     Typical values: 0.05-0.2. Set to 0 to disable."
    )]
    topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-cell refinement steps at inference",
        long_help = "Number of gradient steps for per-cell topic refinement at inference time.\n\
		     Optimizes topic logits against the frozen decoder likelihood,\n\
		     anchored to the encoder output via L2 regularization.\n\
		     Set to 0 to disable (default)."
    )]
    refine_steps: usize,

    #[arg(long, default_value_t = 0.01, help = "Learning rate for refinement")]
    refine_lr: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 regularization strength for refinement"
    )]
    refine_reg: f64,
}

pub fn fit_topic_model(args: &TopicArgs) -> anyhow::Result<()> {
    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
        ..
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;

    // 2. Take projection results by warm start or projecting it again
    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file.as_deref() {
        use matrix_util::common_io::*;
        let ext = file_ext(proj_file)?;

        let MatWithNames {
            rows: cell_names,
            cols: _,
            mat: proj_nk,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_row_names(proj_file, Some(0))?,
            _ => Mat::read_data_with_names(proj_file, &['\t', ',', ' '], Some(0), Some(0))?,
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

    // 3. Multi-level collapsing (pseudobulk)
    info!("Multi-level collapsing with super-cells ...");
    let mut collapsed_levels: Vec<CollapsedOut> = data_vec.collapse_columns_multilevel_vec(
        &proj_kn,
        &batch_membership,
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
        },
    )?;
    // Reverse so training goes coarse→fine: coarsest (fewest samples)
    // gets the most epochs for a warm start, finest gets brief refinement.
    collapsed_levels.reverse();

    // For delta output and latent evaluation, use the finest level.
    // After reversing, the finest level (most groups) is the last element,
    // matching the group assignments stored in data_vec by assign_groups().
    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();

    let batch_db = finest_collapsed.delta.as_ref();

    if let Some(batch_db) = batch_db {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_melted_parquet(
            &outfile,
            (Some(&gene_names), Some("gene")),
            (batch_names.as_deref(), Some("batch")),
        )?;
    }

    // 4. Per-level feature coarsenings for decoders.
    //    Dense encoder operates at D_coarse (finest level's coarsening).
    //    Decoders at coarser levels use fewer feature groups.
    let n_features_full = data_vec.num_rows();
    let num_levels = collapsed_levels.len();

    let level_coarsenings: Vec<Option<FeatureCoarsening>> = if args.max_coarse_features > 0
        && n_features_full > args.max_coarse_features
    {
        let sketch_ds = finest_collapsed.mu_observed.posterior_mean().clone();
        let finest_target = args.max_coarse_features;
        let min_target = (finest_target / num_levels).max(50);
        (0..num_levels)
            .map(|i| {
                let frac = if num_levels > 1 {
                    i as f64 / (num_levels - 1) as f64
                } else {
                    1.0
                };
                let log_min = (min_target as f64).ln();
                let log_max = (finest_target as f64).ln();
                let target = (log_min + frac * (log_max - log_min)).exp().round() as usize;
                let target = target.clamp(min_target, finest_target);
                Some(compute_feature_coarsening(&sketch_ds, target).expect("feature coarsening"))
            })
            .collect()
    } else {
        vec![None; num_levels]
    };

    // Finest-level coarsening (used for encoder, evaluation, dictionary output)
    let finest_coarsening: Option<&FeatureCoarsening> =
        level_coarsenings.last().and_then(|c| c.as_ref());

    // 5. Train a topic model on the collapsed data
    let n_topics = args.n_latent_topics;

    // Encoder at finest level's D_coarse (dense ops are O(D))
    let n_features_encoder = finest_coarsening
        .map(|c| c.num_coarse)
        .unwrap_or(n_features_full);

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(args.device_no)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(args.device_no)?,
        _ => candle_core::Device::Cpu,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let mut encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_features_encoder,
            n_topics,
            layers: &args.encoder_layers,
        },
        param_builder.clone(),
    )?;

    let level_decoder_dims: Vec<usize> = level_coarsenings
        .iter()
        .map(|fc| fc.as_ref().map(|c| c.num_coarse).unwrap_or(n_features_full))
        .collect();

    info!(
        "input: {} -> encoder -> {:?} decoder(s) (dims {:?}) -> finest: {}",
        n_features_encoder, args.decoder, level_decoder_dims, n_features_encoder,
    );

    let gene_names = data_vec.row_names()?;
    let output_gene_names = gene_names.clone();

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

    // Build refinement config (None if disabled)
    let refine_config = if args.refine_steps > 0 {
        Some(TopicRefinementConfig {
            num_steps: args.refine_steps,
            learning_rate: args.refine_lr,
            regularization: args.refine_reg,
        })
    } else {
        None
    };

    // Build per-level decoders, train, save dictionary, and evaluate.
    let (scores, z_nk) = match args.decoder {
        DecoderType::Nb => {
            let decoders: Vec<NbTopicDecoder> = level_decoder_dims
                .iter()
                .enumerate()
                .map(|(i, &d_l)| {
                    NbTopicDecoder::new(d_l, n_topics, param_builder.pp(format!("dec_{i}")))
                        .expect("decoder creation")
                })
                .collect();

            let train_config = ProgressiveTrainConfig {
                parameters: &parameters,
                dev: &dev,
                args,
                stop: &stop,
            };
            let scores = match args.level_schedule {
                LevelSchedule::Progressive => train_progressive(
                    &collapsed_levels,
                    &mut encoder,
                    &decoders,
                    &level_coarsenings,
                    &train_config,
                )?,
                LevelSchedule::Mixed => train_mixed(
                    &collapsed_levels,
                    &mut encoder,
                    &decoders,
                    &level_coarsenings,
                    &train_config,
                )?,
            };

            info!("Writing down the model parameters");

            let finest_decoder = decoders.last().unwrap();
            write_dictionary_expanded(
                finest_decoder,
                finest_coarsening,
                n_features_full,
                &output_gene_names,
                &args.out,
            )?;

            // Save per-gene dispersion φ_g
            let log_phi = finest_decoder
                .log_phi()
                .to_device(&candle_core::Device::Cpu)?;
            let phi_vec: Vec<f32> = log_phi.exp()?.flatten_all()?.to_vec1()?;
            let phi_expanded: Vec<f32> = if let Some(fc) = finest_coarsening {
                let mut full = vec![0.0f32; n_features_full];
                for (c, group) in fc.coarse_to_fine.iter().enumerate() {
                    for &f in group {
                        full[f] = phi_vec[c];
                    }
                }
                full
            } else {
                phi_vec
            };
            let phi_mat = Mat::from_column_slice(phi_expanded.len(), 1, &phi_expanded);
            let col_names = vec!["dispersion_phi".to_string().into_boxed_str()];
            phi_mat.to_parquet_with_names(
                &(args.out.to_string() + ".dispersion.parquet"),
                (Some(&output_gene_names), Some("gene")),
                Some(&col_names),
            )?;
            info!(
                "Saved dispersion parameters to {}.dispersion.parquet",
                &args.out
            );

            info!("Writing down the latent states");
            let eval_config = EvaluateLatentConfig {
                dev: &dev,
                adj_method: &args.adj_method,
                minibatch_size: args.minibatch_size,
                feature_coarsening: finest_coarsening,
                decoder: Some(finest_decoder),
                refine_config: refine_config.as_ref(),
            };
            let z_nk =
                evaluate_latent_by_encoder(&data_vec, &encoder, finest_collapsed, &eval_config)?;

            (scores, z_nk)
        }
        DecoderType::Multinom => {
            let decoders: Vec<TopicDecoder> = level_decoder_dims
                .iter()
                .enumerate()
                .map(|(i, &d_l)| {
                    TopicDecoder::new(d_l, n_topics, param_builder.pp(format!("dec_{i}")))
                        .expect("decoder creation")
                })
                .collect();

            let train_config = ProgressiveTrainConfig {
                parameters: &parameters,
                dev: &dev,
                args,
                stop: &stop,
            };
            let scores = match args.level_schedule {
                LevelSchedule::Progressive => train_progressive(
                    &collapsed_levels,
                    &mut encoder,
                    &decoders,
                    &level_coarsenings,
                    &train_config,
                )?,
                LevelSchedule::Mixed => train_mixed(
                    &collapsed_levels,
                    &mut encoder,
                    &decoders,
                    &level_coarsenings,
                    &train_config,
                )?,
            };

            info!("Writing down the model parameters");

            let finest_decoder = decoders.last().unwrap();
            write_dictionary_expanded(
                finest_decoder,
                finest_coarsening,
                n_features_full,
                &output_gene_names,
                &args.out,
            )?;

            info!("Writing down the latent states");
            let eval_config = EvaluateLatentConfig {
                dev: &dev,
                adj_method: &args.adj_method,
                minibatch_size: args.minibatch_size,
                feature_coarsening: finest_coarsening,
                decoder: Some(finest_decoder),
                refine_config: refine_config.as_ref(),
            };
            let z_nk =
                evaluate_latent_by_encoder(&data_vec, &encoder, finest_collapsed, &eval_config)?;

            (scores, z_nk)
        }
    };

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    info!("Done");
    Ok(())
}

/// Write dictionary with optional expansion from coarse to fine resolution.
fn write_dictionary_expanded<Dec: DecoderModuleT>(
    decoder: &Dec,
    coarsening: Option<&FeatureCoarsening>,
    n_features_full: usize,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let dict_tensor = decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?;

    if let Some(fc) = coarsening {
        // Dictionary is [d, K] in log-probability space; expand to [D', K]
        let dict_dk: Mat = Mat::from_tensor(&dict_tensor)?;
        let expanded_dk = fc.expand_log_dict_dk(&dict_dk, n_features_full);
        expanded_dk.to_parquet_with_names(
            &(out_prefix.to_string() + ".dictionary.parquet"),
            (Some(gene_names), Some("gene")),
            None,
        )?;
        info!(
            "Expanded dictionary from {} to {} features",
            fc.num_coarse, n_features_full
        );
    } else {
        dict_tensor.to_parquet_with_names(
            &(out_prefix.to_string() + ".dictionary.parquet"),
            (Some(gene_names), Some("gene")),
            None,
        )?;
    }
    Ok(())
}

///////////////////////
// training routines //
///////////////////////

/// Configuration for progressive training
struct ProgressiveTrainConfig<'a> {
    parameters: &'a candle_nn::VarMap,
    dev: &'a Device,
    args: &'a TopicArgs,
    stop: &'a AtomicBool,
}

/// Mixed multi-level VAE training.
///
/// Encoder operates at D_coarse (finest level's feature coarsening).
/// Per-level decoders operate at D_l. All levels are trained simultaneously
/// each epoch — the shared encoder sees data from all levels, while each
/// decoder handles its own feature resolution.
fn train_mixed<Enc, Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &mut Enc,
    decoders: &[Dec],
    level_coarsenings: &[Option<FeatureCoarsening>],
    config: &ProgressiveTrainConfig,
) -> anyhow::Result<TrainScores>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.args.epochs;

    // Encoder coarsening = finest level's coarsening
    let enc_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());

    for (level, (collapsed, decoder)) in collapsed_levels.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} samples, decoder dim {}",
            level + 1,
            num_levels,
            collapsed.mu_observed.ncols(),
            decoder.dim_obs(),
        );
    }

    info!(
        "Mixed multi-level training: {} levels, {} epochs",
        num_levels, total_epochs
    );

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        config.args.learning_rate as f64,
    )?;

    let pb = ProgressBar::new(total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    // Budget: 2^sort_dim total samples per epoch, weighted inversely by level size.
    // Coarser levels (fewer, higher-quality samples) get more budget.
    let target_total = 1usize << config.args.sort_dim;
    let level_sizes: Vec<usize> = collapsed_levels
        .iter()
        .map(|c| c.mu_observed.ncols())
        .collect();
    let level_budgets = compute_level_budgets(&level_sizes, target_total);
    info!(
        "Sample budget per epoch: {} total, per level: {:?} (from {:?})",
        target_total, level_budgets, level_sizes
    );

    let mut rng = rand::rng();

    for epoch in (0..total_epochs).step_by(config.args.jitter_interval) {
        // Sample at full D, subsample rows BEFORE coarsening (avoids wasted work)
        let level_data: Vec<(Mat, Option<Mat>, Mat)> = collapsed_levels
            .iter()
            .zip(level_coarsenings.iter())
            .zip(level_budgets.iter())
            .map(|((collapsed, dec_fc), &budget)| {
                let full_mixed = collapsed
                    .mu_observed
                    .posterior_sample()
                    .unwrap()
                    .transpose();

                let full_batch = collapsed.mu_residual.as_ref().map(|x| {
                    let ret: Mat = x.posterior_sample().unwrap();
                    ret.transpose()
                });

                let target_full = if let Some(adj) = &collapsed.mu_adjusted {
                    adj.posterior_sample().unwrap().transpose()
                } else {
                    full_mixed.clone()
                };

                // Subsample rows BEFORE coarsening
                let (sub_mixed, sub_batch, sub_target) =
                    subsample_rows((full_mixed, full_batch, target_full), budget, &mut rng);

                // Coarsen encoder input to D_coarse (finest level)
                let enc_nd = if let Some(fc) = enc_coarsening {
                    fc.aggregate_columns_nd(&sub_mixed)
                } else {
                    sub_mixed
                };

                let batch_nd = sub_batch.map(|b| {
                    if let Some(fc) = enc_coarsening {
                        fc.aggregate_columns_nd(&b)
                    } else {
                        b
                    }
                });

                // Coarsen decoder target to D_l
                let dec_target = if let Some(fc) = dec_fc.as_ref() {
                    fc.aggregate_columns_nd(&sub_target)
                } else {
                    sub_target
                };

                (enc_nd, batch_nd, dec_target)
            })
            .collect();

        // Build per-level data loaders
        let data_loaders: Vec<InMemoryData> = level_data
            .iter()
            .map(|(enc, batch, target)| {
                let mut loader = InMemoryData::from(InMemoryArgs {
                    input: enc,
                    input_null: batch.as_ref(),
                    output: Some(target),
                    output_null: None,
                })
                .expect("data loader creation");
                loader
                    .shuffle_minibatch(config.args.minibatch_size)
                    .expect("shuffle");
                loader
            })
            .collect();

        let kl_weight = if config.args.kl_warmup_epochs > 0.0 {
            1.0 - (-(epoch as f64) / config.args.kl_warmup_epochs).exp()
        } else {
            1.0
        };

        let jitter_end = config.args.jitter_interval.min(total_epochs - epoch);
        for _jitter in 0..jitter_end {
            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;
            let mut count_tot = 0f32;
            let mut n_tot = 0usize;

            // Train on all levels each epoch
            for (level, loader) in data_loaders.iter().enumerate() {
                let decoder = &decoders[level];
                n_tot += loader.num_data();

                for b in 0..loader.num_minibatch() {
                    let mb = loader.minibatch_shuffled(b, config.dev)?;
                    let (log_z_nk, kl) =
                        encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;

                    let log_z_nk = smooth_topics(log_z_nk, config.args.topic_smoothing)?;

                    let y_nd = mb.output.unwrap_or(mb.input);
                    let (_, llik) =
                        decoder.forward_with_llik(&log_z_nk, &y_nd, &topic_likelihood)?;

                    let loss = ((&kl * kl_weight)? - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;

                    llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    count_tot += y_nd.sum_all()?.to_scalar::<f32>()?;
                }
            }

            let llik_avg = llik_tot / count_tot;
            let kl_avg = kl_tot / n_tot as f32;
            llik_trace.push(llik_avg);
            kl_trace.push(kl_avg);

            pb.inc(1);

            info!("[epoch {}] llik={} kl={}", epoch, llik_avg, kl_avg);

            if config.stop.load(Ordering::SeqCst) {
                pb.finish_and_clear();
                info!("Stopping early at epoch {}", epoch);
                return Ok(TrainScores {
                    llik: llik_trace,
                    kl: kl_trace,
                });
            }
        }
    }

    pb.finish_and_clear();
    info!("done mixed multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

/// Progressive training: coarse→fine, more epochs for coarser levels.
fn train_progressive<Enc, Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &mut Enc,
    decoders: &[Dec],
    level_coarsenings: &[Option<FeatureCoarsening>],
    config: &ProgressiveTrainConfig,
) -> anyhow::Result<TrainScores>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.args.epochs;

    // Encoder coarsening = finest level's coarsening
    let enc_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());

    let total_weight: usize = (1..=num_levels).sum();
    let level_epochs: Vec<usize> = (0..num_levels)
        .map(|i| {
            let w = num_levels - i;
            (total_epochs * w / total_weight).max(1)
        })
        .collect();

    for (level, (collapsed, decoder)) in collapsed_levels.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} epochs, {} samples, decoder dim {}",
            level + 1,
            num_levels,
            level_epochs[level],
            collapsed.mu_observed.ncols(),
            decoder.dim_obs(),
        );
    }

    info!(
        "Progressive training: {} levels, epoch allocation: {:?} (total {})",
        num_levels,
        level_epochs,
        level_epochs.iter().sum::<usize>()
    );

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        config.args.learning_rate as f64,
    )?;

    let total_actual_epochs: usize = level_epochs.iter().sum();
    let pb = ProgressBar::new(total_actual_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_actual_epochs);
    let mut kl_trace = Vec::with_capacity(total_actual_epochs);
    let mut global_epoch: usize = 0;

    for (level, (collapsed, &level_ep)) in
        collapsed_levels.iter().zip(level_epochs.iter()).enumerate()
    {
        let decoder = &decoders[level];
        let dec_coarsening = level_coarsenings[level].as_ref();

        for epoch in (0..level_ep).step_by(config.args.jitter_interval) {
            let full_mixed = collapsed.mu_observed.posterior_sample()?.transpose();

            let enc_nd = if let Some(fc) = enc_coarsening {
                fc.aggregate_columns_nd(&full_mixed)
            } else {
                full_mixed.clone()
            };

            let batch_nd = collapsed.mu_residual.as_ref().map(|x| {
                let mut ret: Mat = x.posterior_sample().unwrap();
                ret = ret.transpose();
                if let Some(fc) = enc_coarsening {
                    ret = fc.aggregate_columns_nd(&ret);
                }
                ret
            });

            let target_full = if let Some(adj) = &collapsed.mu_adjusted {
                adj.posterior_sample()?.transpose()
            } else {
                full_mixed.clone()
            };

            let dec_target = if let Some(fc) = dec_coarsening {
                fc.aggregate_columns_nd(&target_full)
            } else {
                target_full
            };

            let mut data_loader = InMemoryData::from(InMemoryArgs {
                input: &enc_nd,
                input_null: batch_nd.as_ref(),
                output: Some(&dec_target),
                output_null: None,
            })?;

            data_loader.shuffle_minibatch(config.args.minibatch_size)?;

            let kl_weight = if config.args.kl_warmup_epochs > 0.0 {
                1.0 - (-(global_epoch as f64) / config.args.kl_warmup_epochs).exp()
            } else {
                1.0
            };

            let jitter_end = config.args.jitter_interval.min(level_ep - epoch);
            for _jitter in 0..jitter_end {
                let mut llik_tot = 0f32;
                let mut kl_tot = 0f32;
                let mut count_tot = 0f32;

                for b in 0..data_loader.num_minibatch() {
                    let mb = data_loader.minibatch_shuffled(b, config.dev)?;
                    let (log_z_nk, kl) =
                        encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;

                    let log_z_nk = smooth_topics(log_z_nk, config.args.topic_smoothing)?;

                    let y_nd = mb.output.unwrap_or(mb.input);
                    let (_, llik) =
                        decoder.forward_with_llik(&log_z_nk, &y_nd, &topic_likelihood)?;

                    let loss = ((&kl * kl_weight)? - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;

                    llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    count_tot += y_nd.sum_all()?.to_scalar::<f32>()?;
                }

                let n = data_loader.num_data() as f32;
                llik_trace.push(llik_tot / count_tot);
                kl_trace.push(kl_tot / n);

                pb.inc(1);
                global_epoch += 1;

                info!(
                    "[level {}/{}][epoch {}] llik={} kl={}",
                    level + 1,
                    num_levels,
                    epoch,
                    llik_trace.last().unwrap(),
                    kl_trace.last().unwrap()
                );

                if config.stop.load(Ordering::SeqCst) {
                    pb.finish_and_clear();
                    info!(
                        "Stopping early at level {}/{}, epoch {}",
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
    info!("done progressive multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

/// Configuration for latent evaluation by encoder
pub(crate) struct EvaluateLatentConfig<'a, Dec> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub feature_coarsening: Option<&'a FeatureCoarsening>,
    pub decoder: Option<&'a Dec>,
    pub refine_config: Option<&'a TopicRefinementConfig>,
}

pub(crate) fn evaluate_latent_by_encoder<Enc, Dec>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    collapsed: &CollapsedOut,
    config: &EvaluateLatentConfig<Dec>,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync,
    Dec: DecoderModuleT + Send + Sync,
{
    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    let block_size = config.minibatch_size;

    let jobs = create_jobs(ntot, Some(block_size));
    let njobs = jobs.len() as u64;
    // Delta coarsened to D_coarse — encoder operates at D_coarse
    let delta = match config.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| x.posterior_mean().clone())
    .map(|mut delta_db| {
        if let Some(fc) = config.feature_coarsening {
            delta_db = fc.aggregate_rows_ds(&delta_db);
        }
        delta_db
            .to_tensor(config.dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
    });

    let block_config = EvaluateBlockConfig {
        dev: config.dev,
        delta: delta.as_ref(),
        feature_coarsening: config.feature_coarsening,
        decoder: config.decoder,
        refine_config: config.refine_config,
        adj_method: config.adj_method.clone(),
    };

    let eval_block = |block| -> anyhow::Result<(usize, Mat)> {
        evaluate_block(block, data_vec, encoder, &block_config)
    };

    // GPU forward passes are not thread-safe — run sequentially on Metal/CUDA,
    // parallel on CPU
    let mut chunks: Vec<(usize, Mat)> = if config.dev.is_cpu() {
        jobs.par_iter()
            .progress_count(njobs)
            .map(|&block| eval_block(block))
            .collect::<anyhow::Result<Vec<_>>>()?
    } else {
        let pb = ProgressBar::new(njobs);
        let result = jobs
            .iter()
            .map(|&block| {
                let r = eval_block(block);
                pb.inc(1);
                r
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        pb.finish_and_clear();
        result
    };

    chunks.sort_by_key(|&(lb, _)| lb);

    let mut ret = Mat::zeros(ntot, kk);
    let mut lb = 0;
    for (_, z) in chunks {
        let ub = lb + z.nrows();
        ret.rows_range_mut(lb..ub).copy_from(&z);
        lb = ub;
    }
    Ok(ret)
}

/// Configuration for block-wise evaluation
pub(crate) struct EvaluateBlockConfig<'a, Dec> {
    pub dev: &'a Device,
    pub delta: Option<&'a Tensor>,
    pub feature_coarsening: Option<&'a FeatureCoarsening>,
    pub decoder: Option<&'a Dec>,
    pub refine_config: Option<&'a TopicRefinementConfig>,
    pub adj_method: AdjMethod,
}

pub(crate) fn evaluate_block<Enc, Dec>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &Enc,
    config: &EvaluateBlockConfig<Dec>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let (lb, ub) = block;
    let x0_nd = config
        .delta
        .map(|delta_bm| -> anyhow::Result<Tensor> {
            let membership: Vec<u32> = match config.adj_method {
                AdjMethod::Batch => data_vec
                    .get_batch_membership(lb..ub)
                    .into_iter()
                    .map(|x| x as u32)
                    .collect(),
                AdjMethod::Residual => data_vec
                    .get_group_membership(lb..ub)?
                    .into_iter()
                    .map(|x| x as u32)
                    .collect(),
            };
            let indices = Tensor::from_iter(membership.into_iter(), config.dev)?;
            Ok(delta_bm.index_select(&indices, 0)?)
        })
        .transpose()?;

    let x_dn = data_vec.read_columns_csc(lb..ub)?;

    // Coarsen to D_coarse for encoder (same resolution as decoder)
    let x_enc_nd = if let Some(fc) = config.feature_coarsening {
        fc.aggregate_sparse_csc(&x_dn)
            .to_tensor(config.dev)?
            .transpose(0, 1)?
    } else {
        x_dn.to_tensor(config.dev)?.transpose(0, 1)?
    };

    let (log_z_nk, _) = encoder.forward_t(&x_enc_nd, x0_nd.as_ref(), false)?;

    // Apply per-cell refinement (data already at D_coarse)
    let log_z_nk = if let (Some(dec), Some(cfg)) = (config.decoder, config.refine_config) {
        refine_topic_proportions(&log_z_nk, &x_enc_nd, dec, cfg)?
    } else {
        log_z_nk
    };

    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}
