use crate::embed_common::*;
use crate::feature_selection::*;
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
pub(crate) enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub(crate) enum AdjMethod {
    Batch,
    Residual,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum DecoderType {
    /// Flat topic decoder with softmax dictionary
    Flat,
    /// Sparse topic decoder with sparsemax dictionary
    Sparse,
    /// Zero-inflated topic decoder with per-feature dropout
    ZeroInflated,
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
        long,
        default_value_t = false,
        help = "Ignore batch adjustment",
        long_help = "Ignore batch adjustment.\n\
		     Disables batch effect correction during processing."
    )]
    ignore_batch_effects: bool,

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
        default_value_t = 3,
        help = "Number of k-nearest neighbour batches",
        long_help = "Number of k-nearest neighbour batches.\n\
		     Controls the number of batches considered \n\
		     for nearest neighbour search (cell-level mode)."
    )]
    knn_batches: usize,

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
        value_delimiter(','),
        help = "Reference batch names",
        long_help = "Reference batch names (comma-separated).\n\
		     Specify batches to be used as reference during adjustment.\n\
		     Forces cell-level matching (disables super-cell mode)."
    )]
    reference_batches: Option<Vec<Box<str>>>,

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
        default_value_t = false,
        help = "Disable super-cell matching",
        long_help = "Disable super-cell matching and use cell-level KNN instead.\n\
		     The cell-level approach is slower but may be more accurate\n\
		     for small datasets."
    )]
    no_supercell: bool,

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
        short = 'f',
        long,
        help = "Number of feature modules",
        long_help = "Number of modules of the features in the encoder model.\n\
		     If not specified, encoder_layers[0] will be used. \n\
		     Giving the number of features modules smaller than that of features,\n\
		     we can expedite model training while not loosing too much of accuracy,\n\
		     as many features are redundant and frequently dropped out.\n"
    )]
    feature_modules: Option<usize>,

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
        help = "Maximum number of highly variable features",
        long_help = "Select top N features by log-variance.\n\
		     If not specified, all features are used.\n\
		     Skipped if --warm-start is provided."
    )]
    max_features: Option<usize>,

    #[arg(
        long,
        help = "Pre-computed feature selection file",
        long_help = "Path to file with pre-selected feature names (one per line).\n\
		     Takes precedence over --max-features.\n\
		     Skipped if --warm-start is provided."
    )]
    feature_list_file: Option<Box<str>>,

    #[arg(
        long,
        alias = "high-sd",
        help = "Exclude highly expressed features (SD threshold)",
        long_help = "Exclude features with high mean expression before feature selection.\n\
		     Features with log1p(mean) > mean + threshold*SD are excluded.\n\
		     Typical values: 4 or 5.\n\
		     If not specified, no features are excluded based on expression level."
    )]
    exclude_high_expression_sd: Option<f32>,

    #[arg(
        long,
        help = "Save feature variance statistics",
        long_help = "Save computed log-variance for all features to {out}.feature_variance.parquet"
    )]
    save_feature_variance: bool,

    #[arg(
        long,
        value_enum,
        default_value = "flat",
        help = "Decoder type",
        long_help = "Topic decoder type:\n\
		     flat: standard softmax dictionary\n\
		     sparse: sparsemax dictionary (exact zeros)\n\
		     zero-inflated: per-feature dropout gate for structural zeros"
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
    let reference = args.reference_batches.as_deref();

    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
        nbatch,
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;

    // Feature selection (if requested)
    let selected_features: Option<FeatureSelection> = if args.warm_start_proj_file.is_some() {
        if args.max_features.is_some() || args.feature_list_file.is_some() {
            info!("Warm-start provided: skipping feature selection (may be incompatible)");
        }
        None
    } else if args.max_features.is_some() || args.feature_list_file.is_some() {
        Some(select_highly_variable_features(
            &data_vec,
            args.max_features,
            args.feature_list_file.as_deref(),
            args.save_feature_variance,
            &args.out,
            args.block_size,
            args.exclude_high_expression_sd,
        )?)
    } else {
        None
    };

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
    let use_multilevel = args.num_levels > 1 && !args.no_supercell;

    let collapsed_levels: Vec<CollapsedOut> = if use_multilevel {
        info!("Multi-level collapsing with super-cells ...");
        let mut levels = data_vec.collapse_columns_multilevel_vec(
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
        levels.reverse();
        levels
    } else {
        let nsamp = data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), None)?;

        if !args.ignore_batch_effects && nbatch > 1 {
            info!("Registering batch information");
            data_vec.build_hnsw_per_batch(&proj_kn, &batch_membership)?;
        }

        info!("Collapsing columns into {} pseudobulk samples ...", nsamp);
        vec![data_vec.collapse_columns(
            Some(args.knn_batches),
            Some(args.knn_cells),
            reference,
            Some(args.iter_opt),
        )?]
    };

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

    // 4. Train a topic model on the collapsed data
    let n_topics = args.n_latent_topics;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

    let n_features_decoder = selected_features
        .as_ref()
        .map(|sel| sel.selected_indices.len())
        .unwrap_or_else(|| data_vec.num_rows());
    let n_features_encoder = n_features_decoder;

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
            n_modules,
            layers: &args.encoder_layers,
        },
        param_builder.clone(),
    )?;

    info!(
        "input: {} -> encoder -> decoder ({:?}) -> output: {}",
        n_features_encoder, args.decoder, n_features_decoder
    );

    let gene_names = data_vec.row_names()?;

    // Use selected feature names for dictionary if feature selection was applied
    let output_gene_names = selected_features
        .as_ref()
        .map(|sel| sel.selected_names.clone())
        .unwrap_or_else(|| gene_names.clone());

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

    // Build decoder, train, save dictionary, and evaluate — all inside each arm
    // so the decoder is in scope for refinement during evaluation.
    let (scores, z_nk) = match args.decoder {
        DecoderType::ZeroInflated => {
            let decoder = ZITopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

            let train_config = ProgressiveTrainConfig {
                parameters: &parameters,
                dev: &dev,
                args,
                feature_selection: selected_features.as_ref(),
                stop: &stop,
            };
            let scores = train_encoder_decoder_progressive(
                &collapsed_levels,
                &mut encoder,
                &decoder,
                &train_config,
            )?;

            info!("Writing down the model parameters");

            decoder
                .get_dictionary()?
                .to_device(&candle_core::Device::Cpu)?
                .to_parquet_with_names(
                    &(args.out.to_string() + ".dictionary.parquet"),
                    (Some(&output_gene_names), Some("gene")),
                    None,
                )?;

            // Save per-feature dropout probabilities
            let dropout_d = decoder
                .dropout_prob()?
                .to_device(&candle_core::Device::Cpu)?;
            let dropout_vec: Vec<f32> = dropout_d.flatten_all()?.to_vec1()?;
            let dropout_mat = Mat::from_column_slice(dropout_vec.len(), 1, &dropout_vec);
            let col_names = vec!["dropout_prob".to_string().into_boxed_str()];
            dropout_mat.to_parquet_with_names(
                &(args.out.to_string() + ".dropout.parquet"),
                (Some(&output_gene_names), Some("gene")),
                Some(&col_names),
            )?;
            info!(
                "Saved dropout probabilities to {}.dropout.parquet",
                &args.out
            );

            info!("Writing down the latent states");
            let eval_config = EvaluateLatentConfig {
                dev: &dev,
                adj_method: &args.adj_method,
                minibatch_size: args.minibatch_size,
                feature_selection: selected_features.as_ref(),
                decoder: Some(&decoder),
                refine_config: refine_config.as_ref(),
            };
            let z_nk =
                evaluate_latent_by_encoder(&data_vec, &encoder, finest_collapsed, &eval_config)?;

            (scores, z_nk)
        }
        DecoderType::Sparse => {
            let decoder =
                SparseTopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

            let train_config = ProgressiveTrainConfig {
                parameters: &parameters,
                dev: &dev,
                args,
                feature_selection: selected_features.as_ref(),
                stop: &stop,
            };
            let scores = train_encoder_decoder_progressive(
                &collapsed_levels,
                &mut encoder,
                &decoder,
                &train_config,
            )?;

            info!("Writing down the model parameters");

            decoder
                .get_dictionary()?
                .to_device(&candle_core::Device::Cpu)?
                .to_parquet_with_names(
                    &(args.out.to_string() + ".dictionary.parquet"),
                    (Some(&output_gene_names), Some("gene")),
                    None,
                )?;

            info!("Writing down the latent states");
            let eval_config = EvaluateLatentConfig {
                dev: &dev,
                adj_method: &args.adj_method,
                minibatch_size: args.minibatch_size,
                feature_selection: selected_features.as_ref(),
                decoder: Some(&decoder),
                refine_config: refine_config.as_ref(),
            };
            let z_nk =
                evaluate_latent_by_encoder(&data_vec, &encoder, finest_collapsed, &eval_config)?;

            (scores, z_nk)
        }
        DecoderType::Flat => {
            let decoder = TopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

            let train_config = ProgressiveTrainConfig {
                parameters: &parameters,
                dev: &dev,
                args,
                feature_selection: selected_features.as_ref(),
                stop: &stop,
            };
            let scores = train_encoder_decoder_progressive(
                &collapsed_levels,
                &mut encoder,
                &decoder,
                &train_config,
            )?;

            info!("Writing down the model parameters");

            decoder
                .get_dictionary()?
                .to_device(&candle_core::Device::Cpu)?
                .to_parquet_with_names(
                    &(args.out.to_string() + ".dictionary.parquet"),
                    (Some(&output_gene_names), Some("gene")),
                    None,
                )?;

            info!("Writing down the latent states");
            let eval_config = EvaluateLatentConfig {
                dev: &dev,
                adj_method: &args.adj_method,
                minibatch_size: args.minibatch_size,
                feature_selection: selected_features.as_ref(),
                decoder: Some(&decoder),
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

    // Save selected feature list if feature selection was applied
    if let Some(sel) = &selected_features {
        use matrix_util::common_io::write_lines;
        let feature_file = args.out.to_string() + ".selected_features.txt";
        write_lines(&sel.selected_names, &feature_file)?;
        info!(
            "Saved {} selected features to {}",
            sel.selected_names.len(),
            feature_file
        );
    }

    info!("Done");
    Ok(())
}

///////////////////////
// training routines //
///////////////////////

pub(crate) struct TrainScores {
    pub(crate) llik: Vec<f32>,
    pub(crate) kl: Vec<f32>,
}

impl TrainScores {
    pub(crate) fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
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

/// Configuration for progressive training
struct ProgressiveTrainConfig<'a> {
    parameters: &'a candle_nn::VarMap,
    dev: &'a Device,
    args: &'a TopicArgs,
    feature_selection: Option<&'a FeatureSelection>,
    stop: &'a AtomicBool,
}

/// Progressive multi-level VAE training.
///
/// Allocates epochs across levels with weights inversely proportional to
/// level index: level `i` gets `total_epochs * (num_levels - i) / sum(1..=num_levels)`.
/// Coarse levels have fewer samples so each epoch is cheap — train longer
/// there for a solid warm start, then fine-tune briefly on finer data.
/// A single optimizer is created once and carries state across levels.
fn train_encoder_decoder_progressive<Enc, Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &mut Enc,
    decoder: &Dec,
    config: &ProgressiveTrainConfig,
) -> anyhow::Result<TrainScores>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.args.epochs;

    // Compute per-level epoch allocation: w[i] = num_levels - i
    // Coarse levels are cheap per epoch, so train longer there for
    // a solid warm start, then fine-tune briefly on finer data.
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
        let label = if level == 0 {
            "coarsest"
        } else if level + 1 == num_levels {
            "finest"
        } else {
            ""
        };
        info!(
            "Level {}/{}: {} epochs, {} samples {}",
            level + 1,
            num_levels,
            level_ep,
            collapsed.mu_observed.ncols(),
            label,
        );

        for epoch in (0..level_ep).step_by(config.args.jitter_interval) {
            let mut mixed_nd = collapsed.mu_observed.posterior_sample()?.transpose();

            if let Some(sel) = config.feature_selection {
                mixed_nd = mixed_nd.select_columns(&sel.selected_indices);
            }

            let clean_nd = collapsed.mu_adjusted.as_ref().map(|x| {
                let mut ret: Mat = x.posterior_sample().unwrap();
                ret = ret.transpose();
                if let Some(sel) = config.feature_selection {
                    ret = ret.select_columns(&sel.selected_indices);
                }
                ret
            });

            let batch_nd = collapsed.mu_residual.as_ref().map(|x| {
                let mut ret: Mat = x.posterior_sample().unwrap();
                ret = ret.transpose();
                if let Some(sel) = config.feature_selection {
                    ret = ret.select_columns(&sel.selected_indices);
                }
                ret
            });

            if let Some(ref batch) = batch_nd {
                if batch.ncols() != mixed_nd.ncols() {
                    return Err(anyhow::anyhow!(
                        "mixed_nd and batch_nd have different feature dimensions: {} vs {}",
                        mixed_nd.ncols(),
                        batch.ncols()
                    ));
                }
            }

            let mut data_loader = InMemoryData::from(InMemoryArgs {
                input: &mixed_nd,
                input_null: batch_nd.as_ref(),
                output: clean_nd.as_ref(),
                output_null: None,
            })?;

            data_loader.shuffle_minibatch(config.args.minibatch_size)?;

            let kl_weight = if config.args.kl_warmup_epochs > 0.0 {
                1.0 - (-(global_epoch as f64) / config.args.kl_warmup_epochs).exp()
            } else {
                1.0
            };

            let jitter_end = config.args.jitter_interval.min(level_ep - epoch);
            for jitter in 0..jitter_end {
                let mut llik_tot = 0f32;
                let mut kl_tot = 0f32;
                let mut count_tot = 0f32;

                for b in 0..data_loader.num_minibatch() {
                    let mb = data_loader.minibatch_shuffled(b, config.dev)?;
                    let (log_z_nk, kl) =
                        encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;

                    // Smoothing in log-space: exp→mix→log
                    let log_z_nk = if config.args.topic_smoothing > 0.0 {
                        let alpha = config.args.topic_smoothing;
                        let kk = log_z_nk.dim(1)? as f64;
                        ((log_z_nk.exp()? * (1.0 - alpha))? + alpha / kk)?.log()?
                    } else {
                        log_z_nk
                    };

                    let y_nd = mb.output.unwrap_or(mb.input);
                    let (_, llik) =
                        decoder.forward_with_llik(&log_z_nk, &y_nd, &topic_likelihood)?;

                    let loss = ((&kl * kl_weight)? - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;

                    let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                    let kl_val = kl.sum_all()?.to_scalar::<f32>()?;
                    let count_val = y_nd.sum_all()?.to_scalar::<f32>()?;
                    llik_tot += llik_val;
                    kl_tot += kl_val;
                    count_tot += count_val;
                }

                let n = data_loader.num_data() as f32;
                let llik_avg = llik_tot / count_tot;
                let kl_avg = kl_tot / n;
                llik_trace.push(llik_avg);
                kl_trace.push(kl_avg);

                pb.inc(1);
                global_epoch += 1;

                info!(
                    "[level {}/{}][{}][{}] {} {}",
                    level + 1,
                    num_levels,
                    epoch,
                    jitter,
                    llik_avg,
                    kl_avg
                );

                if config.stop.load(Ordering::SeqCst) {
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
    info!("done progressive model training");
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
    pub feature_selection: Option<&'a FeatureSelection>,
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
    let delta = match config.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| {
        let mut delta_db = x.posterior_mean().clone();
        if let Some(sel) = config.feature_selection {
            delta_db = delta_db.select_rows(&sel.selected_indices);
        }
        delta_db
    })
    .map(|delta_db| {
        delta_db
            .to_tensor(config.dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
    });

    let block_config = EvaluateBlockConfig {
        dev: config.dev,
        delta: delta.as_ref(),
        feature_selection: config.feature_selection,
        decoder: config.decoder,
        refine_config: config.refine_config,
    };

    let eval_block = |block| -> anyhow::Result<(usize, Mat)> {
        match config.adj_method {
            AdjMethod::Residual => evaluate_with_residuals(block, data_vec, encoder, &block_config),
            AdjMethod::Batch => evaluate_with_batch(block, data_vec, encoder, &block_config),
        }
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
    pub feature_selection: Option<&'a FeatureSelection>,
    pub decoder: Option<&'a Dec>,
    pub refine_config: Option<&'a TopicRefinementConfig>,
}

pub(crate) fn evaluate_with_batch<Enc, Dec>(
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
    let x0_nd = config.delta.map(|delta_bm| {
        let batches = data_vec
            .get_batch_membership(lb..ub)
            .into_iter()
            .map(|x| x as u32);
        let batches = Tensor::from_iter(batches, config.dev).unwrap();
        delta_bm.index_select(&batches, 0).expect("expand delta")
    });

    // Read as CSC sparse matrix first, apply feature selection, then convert to tensor
    let mut x_dn = data_vec.read_columns_csc(lb..ub)?;

    // Apply feature selection on sparse matrix
    if let Some(sel) = config.feature_selection {
        x_dn = filter_csc_by_rows(&sel.selection_matrix, &x_dn);
    }

    let x_nd = x_dn.to_tensor(config.dev)?.transpose(0, 1)?;

    let (log_z_nk, _) = encoder.forward_t(&x_nd, x0_nd.as_ref(), false)?;

    // Apply per-cell refinement if configured
    let log_z_nk = if let (Some(dec), Some(cfg)) = (config.decoder, config.refine_config) {
        refine_topic_proportions(&log_z_nk, &x_nd, dec, cfg)?
    } else {
        log_z_nk
    };

    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

pub(crate) fn evaluate_with_residuals<Enc, Dec>(
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
    let x0_nd = config.delta.map(|delta_bm| {
        let groups = data_vec
            .get_group_membership(lb..ub)
            .expect("failed to get group membership")
            .into_iter()
            .map(|x| x as u32);
        let groups = Tensor::from_iter(groups, config.dev).unwrap();
        delta_bm.index_select(&groups, 0).expect("expand delta")
    });

    // Read as CSC sparse matrix first, apply feature selection, then convert to tensor
    let mut x_dn = data_vec.read_columns_csc(lb..ub)?;

    // Apply feature selection on sparse matrix
    if let Some(sel) = config.feature_selection {
        x_dn = filter_csc_by_rows(&sel.selection_matrix, &x_dn);
    }

    let x_nd = x_dn.to_tensor(config.dev)?.transpose(0, 1)?;

    let (log_z_nk, _) = encoder.forward_t(&x_nd, x0_nd.as_ref(), false)?;

    // Apply per-cell refinement if configured
    let log_z_nk = if let (Some(dec), Some(cfg)) = (config.decoder, config.refine_config) {
        refine_topic_proportions(&log_z_nk, &x_nd, dec, cfg)?
    } else {
        log_z_nk
    };

    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}
