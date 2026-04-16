use crate::embed_common::*;
use crate::senna_input::*;
use crate::topic::common::*;
use crate::topic::train_joint::*;

use candle_util::candle_decoder_delta_topic::*;
use candle_util::candle_decoder_joint_topic::*;
use candle_util::candle_encoder_joint_softmax::*;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
pub enum JointDecoderType {
    /// Each modality has its own topic-to-feature dictionary
    Independent,
    /// Shared base dictionary + cumulative deltas between consecutive modalities
    Delta,
}

#[derive(Args, Debug)]
pub struct JointTopicArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5), row-major (modality × batch)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Files are arranged as a row-major (modality × batch) table;\n\
                     use -m to set the number of modality rows."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        short = 'm',
        long = "modalities",
        required = true,
        help = "Number of modalities (rows of the data-file table)",
        long_help = "The input files are interpreted row-major as modality × batch.\n\
                     This value sets the number of modality rows."
    )]
    pub(crate) num_modalities: usize,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet       effective topic dictionary\n  \
                     {out}.latent.parquet           cell × topic log-softmax proportions\n  \
                     {out}.log_likelihood.parquet   training loss trace\n  \
                     {out}_{d}.delta.parquet        per-batch effects for modality d\n\n\
                     With --decoder-type delta, additionally:\n  \
                     {out}.base_dictionary.parquet  shared base dictionary\n  \
                     {out}_{m}.delta_logits.parquet delta logits for modality m"
    )]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Target rank of the initial random sketch used to seed\n\
                     batch correction and multi-level pseudobulk collapsing."
    )]
    pub(crate) proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Partition depth: ≤ 2^d + 1 pseudobulk groups",
        long_help = "Binary-tree partitioning over the top d projection components.\n\
                     Produces at most 2^d + 1 pseudobulk leaves."
    )]
    pub(crate) sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell in the same order as its\n\
                     matching data file. Example: batch1.tsv,batch2.tsv"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column-sum normalization scale",
        long_help = "Target library size after per-cell normalization (decoder only)."
    )]
    pub(crate) column_sum_norm: f32,

    #[arg(
        long,
        default_value_t = 10,
        help = "In-batch k-NN for super-cell merging",
        long_help = "Number of within-batch nearest neighbours used when\n\
                     aggregating cells into pseudobulk super-cells."
    )]
    pub(crate) knn_cells: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Batch-correction optimizer iterations",
        long_help = "Coordinate-descent steps when fitting the per-batch delta."
    )]
    pub(crate) iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Column block size for parallel I/O",
        long_help = "Columns streamed per worker. Trades parallel granularity\n\
                     against per-block memory."
    )]
    pub(crate) block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics (K)"
    )]
    pub(crate) n_latent_topics: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder hidden layer sizes (comma-separated)",
        long_help = "Example: 128,1024,128 (input → 128 → 1024 → 128 → topics)."
    )]
    pub(crate) encoder_layers: Vec<usize>,

    #[arg(long, short = 'i', default_value_t = 1000, help = "Training epochs")]
    pub(crate) epochs: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Posterior resampling interval (epochs)",
        long_help = "How often to jitter the collapsed targets by posterior resampling\n\
                     during VAE training."
    )]
    pub(crate) jitter_interval: usize,

    #[arg(long, default_value_t = 100, help = "Training minibatch size")]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        alias = "lr",
        help = "Adam learning rate"
    )]
    pub(crate) learning_rate: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Compute device (cpu|cuda|metal)"
    )]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "CUDA/Metal device index")]
    pub(crate) device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Batch adjustment (batch|residual)",
        long_help = "batch    — subtract per-batch pseudobulk mean.\n\
                     residual — divide by fitted delta per pseudobulk group."
    )]
    pub(crate) adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = 2,
        help = "Multi-level coarsening levels",
        long_help = "Hierarchical pseudobulk refinement passes. Set to 1 to disable."
    )]
    pub(crate) num_levels: usize,

    #[arg(
        long,
        default_value_t = 5000,
        help = "Cap feature dim by meta-feature coarsening (0 to disable)",
        long_help = "Groups co-expressed features into ≤N meta-features so the model\n\
                     trains at reduced resolution. The dictionary is expanded back to\n\
                     full resolution on output.\n\
                     Independent mode: computed per modality.\n\
                     Delta mode: computed on the reference modality and shared."
    )]
    pub(crate) max_coarse_features: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training"
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        value_enum,
        default_value = "independent",
        help = "Joint decoder (independent|delta)",
        long_help = "independent — each modality has its own topic dictionary; features\n\
                                   may differ across modalities.\n\
                     delta       — shared base dictionary + cumulative chain deltas.\n\
                                   Modality 0 = softmax(z @ W_base)\n\
                                   Modality m = softmax(z @ (W_base + Σ δ_1..m))\n\
                                   Requires shared features; reference is modality 0.\n\
                                   Delta logits start at zero and diverge during training."
    )]
    pub(crate) decoder_type: JointDecoderType,
}

pub fn fit_joint_topic_model(args: &JointTopicArgs) -> anyhow::Result<()> {
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

    // 1a. For delta decoder, validate shared features across modalities
    if args.decoder_type == JointDecoderType::Delta {
        let ref_names = data_stack.stack[0].row_names()?;
        for (d, dv) in data_stack.stack.iter().enumerate().skip(1) {
            let names = dv.row_names()?;
            if names != ref_names {
                return Err(anyhow::anyhow!(
                    "Delta decoder requires shared features across modalities, \
                     but modality 0 and modality {} have different row names. \
                     Consider using `data-beans align` to align data first.",
                    d
                ));
            }
        }
        info!(
            "Delta decoder: all {} modalities share {} features",
            args.num_modalities,
            ref_names.len()
        );
    }

    // 2. Concatenate projections
    let proj_dim = args.proj_dim.max(args.n_latent_topics);
    let proj_out = data_stack.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(batch_stack[0].as_ref()),
    )?;
    let proj_kn = proj_out.proj;

    // 3. Batch-adjusted multilevel collapsing (pseudobulk)
    info!(
        "Multi-level collapsing across {} modalities ...",
        data_stack.num_types()
    );

    let mut collapsed_levels: Vec<Vec<CollapsedOut>> = data_stack.collapse_columns_multilevel_vec(
        &proj_kn,
        batch_stack[0].as_ref(),
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
            oversample: false,
        },
    )?;
    // Reverse so training goes coarse→fine: coarsest (fewest samples)
    // gets the most epochs for a warm start, finest gets brief refinement.
    collapsed_levels.reverse();

    // After reversing, the finest level (most groups) is the last element.
    let collapsed_data_vec = collapsed_levels.last().unwrap();

    // 3b. Feature coarsening per modality (if D > max_coarse_features)
    let n_features_full: Vec<usize> = collapsed_data_vec
        .iter()
        .map(|x| x.mu_observed.nrows())
        .collect();

    let coarsenings: Vec<Option<FeatureCoarsening>> =
        if args.max_coarse_features > 0 && args.decoder_type == JointDecoderType::Delta {
            // Delta mode: shared coarsening from reference modality
            let n_full = n_features_full[0];
            if n_full > args.max_coarse_features {
                let collapsed_ref = &collapsed_data_vec[0];
                let sketch = collapsed_ref.mu_observed.posterior_mean().clone();
                let fc = compute_feature_coarsening(&sketch, args.max_coarse_features)?;
                info!(
                    "Shared coarsening: {} → {} meta-features",
                    n_full, fc.num_coarse
                );
                vec![Some(fc); args.num_modalities]
            } else {
                vec![None; args.num_modalities]
            }
        } else if args.max_coarse_features > 0 {
            // Independent mode: per-modality coarsening
            collapsed_data_vec
                .iter()
                .zip(&n_features_full)
                .enumerate()
                .map(
                    |(d, (collapsed, &n_full))| -> anyhow::Result<Option<FeatureCoarsening>> {
                        if n_full > args.max_coarse_features {
                            let sketch = collapsed.mu_observed.posterior_mean().clone();
                            let fc = compute_feature_coarsening(&sketch, args.max_coarse_features)?;
                            info!(
                                "Modality {}: coarsened {} → {} meta-features",
                                d, n_full, fc.num_coarse
                            );
                            Ok(Some(fc))
                        } else {
                            Ok(None)
                        }
                    },
                )
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            collapsed_data_vec.iter().map(|_| None).collect()
        };

    // 4. output batch effect information
    for (d, collapsed) in collapsed_data_vec.iter().enumerate() {
        if let Some(batch_db) = &collapsed.delta {
            let outfile = format!("{}_{}.delta.parquet", args.out, d);
            let data_vec = &data_stack.stack[d];
            let batch_names = data_vec.batch_names();
            let gene_names = data_vec.row_names()?;
            batch_db.to_melted_parquet(
                &outfile,
                (Some(&gene_names), Some("gene")),
                (batch_names.as_deref(), Some("batch")),
            )?;
        }
    }

    // 5. Train a joint topic model on the collapsed data (progressive)
    let n_topics = args.n_latent_topics;

    let dev = create_device(&args.device, args.device_no)?;

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let n_features: Vec<usize> = n_features_full
        .iter()
        .zip(&coarsenings)
        .map(|(&n, fc)| fc.as_ref().map(|c| c.num_coarse).unwrap_or(n))
        .collect();

    let encoder = LogSoftmaxJointEncoder::new(
        LogSoftmaxJointEncoderArgs {
            n_features: n_features.clone(),
            n_topics,
            layers: &args.encoder_layers,
        },
        param_builder.clone(),
    )?;

    let stop = setup_stop_handler();

    let train_config = ProgressiveTrainConfig {
        parameters: &parameters,
        dev: &dev,
        args,
        coarsenings: &coarsenings,
        stop: &stop,
    };

    let gene_names: Vec<Box<str>> = data_stack
        .stack
        .iter()
        .map(|dv| dv.row_names())
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    let save_ctx = SaveContext {
        collapsed_levels: &collapsed_levels,
        encoder: &encoder,
        train_config: &train_config,
        coarsenings: &coarsenings,
        n_features_full: &n_features_full,
        gene_names: &gene_names,
        data_stack: &data_stack,
        args,
    };

    match args.decoder_type {
        JointDecoderType::Independent => {
            let decoder =
                JointTopicDecoder::new(&n_features, args.n_latent_topics, param_builder.clone())?;
            train_and_save(&decoder, &save_ctx)?;
        }
        JointDecoderType::Delta => {
            let shared_d = n_features[0];
            debug_assert!(
                n_features.iter().all(|&d| d == shared_d),
                "Delta decoder requires uniform feature dimensions across modalities"
            );
            let decoder = DeltaTopicDecoder::new(
                args.num_modalities,
                shared_d,
                args.n_latent_topics,
                param_builder.clone(),
            )?;
            train_and_save(&decoder, &save_ctx)?;

            let base_gene_names: Vec<Box<str>> = data_stack.stack[0].row_names()?;

            // Write base dictionary
            let base_dict = decoder.get_base_dictionary()?;
            let base_dict = base_dict.to_device(&candle_core::Device::Cpu)?;
            let base_mat = Mat::from_tensor(&base_dict)?;
            let base_mat = if let Some(fc) = &coarsenings[0] {
                fc.expand_log_dict_dk(&base_mat, n_features_full[0])
            } else {
                base_mat
            };
            base_mat.to_parquet_with_names(
                &format!("{}.base_dictionary.parquet", args.out),
                (Some(&base_gene_names), Some("gene")),
                None,
            )?;

            // Write per-modality delta logits [K, D] with gene names as columns
            for (i, delta) in decoder.get_deltas().iter().enumerate() {
                let delta = delta.to_device(&candle_core::Device::Cpu)?;
                let delta_mat = Mat::from_tensor(&delta)?;
                delta_mat.to_parquet_with_names(
                    &format!("{}_{}.delta_logits.parquet", args.out, i + 1),
                    (None, Some("topic")),
                    Some(&base_gene_names),
                )?;
            }
        }
    }

    info!("Done");
    Ok(())
}
