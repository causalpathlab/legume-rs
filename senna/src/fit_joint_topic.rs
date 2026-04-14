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
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided (space or comma separated)."
    )]
    pub(crate) data_files: Vec<Box<str>>,

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
    pub(crate) num_modalities: usize,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output file prefix. Generated files:\n\
		     - {out}.dictionary.parquet (effective topic dictionaries)\n\
		     - {out}.latent.parquet (log-softmax topic proportions)\n\
		     - {out}.log_likelihood.parquet (training metrics)\n\
		     - {out}_{d}.delta.parquet (batch effects per modality)\n\n\
		     With --decoder-type delta, additionally:\n\
		     - {out}.base_dictionary.parquet (shared base dictionary)\n\
		     - {out}_{m}.delta_logits.parquet (delta logits per modality)\n"
    )]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension.",
        long_help = "Random projection dimension to project the data.\n\
		     Controls the dimensionality of the random projection step."
    )]
    pub(crate) proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection.",
        long_help = "Use top {d} components of projection.\n\
		     Number of samples will be less than `2^{d}+1`."
    )]
    pub(crate) sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files.",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.csv,batch2.csv"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column sum normalization scale.",
        long_help = "Column sum normalization scale (affects decoder only).\n\
		     Adjusts normalization of columns in the decoder."
    )]
    pub(crate) column_sum_norm: f32,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch.",
        long_help = "Number of k-nearest neighbours within each batch.\n\
		     Controls the number of cells considered \n\
		     for nearest neighbour search within each batch."
    )]
    pub(crate) knn_cells: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Optimization iterations.",
        long_help = "Number of optimization iterations.\n\
		     Controls the number of steps for model optimization."
    )]
    pub(crate) iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing.",
        long_help = "Block size (number of columns) for parallel processing.\n\
		     Controls the granularity of parallel computation."
    )]
    pub(crate) block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics.",
        long_help = "Number of latent topics.\n\
		     Controls the dimensionality of the latent topic space."
    )]
    pub(crate) n_latent_topics: usize,

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
    pub(crate) encoder_layers: Vec<usize>,

    #[arg(
        long,
        short = 'i',
        default_value_t = 1000,
        help = "Number of training epochs.",
        long_help = "Number of training epochs.\n\
		     Controls how many times the model is trained over the data."
    )]
    pub(crate) epochs: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Data jitter interval.",
        long_help = "Data jitter interval.\n\
		     Controls the interval for adding jitter to the collapsed data\n\
		     by posterior resampling during VAE training."
    )]
    pub(crate) jitter_interval: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Minibatch size.",
        long_help = "Minibatch size for training.\n\
		     Controls the number of samples per training batch."
    )]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        alias = "lr",
        help = "Learning rate.",
        long_help = "Learning rate for optimization.\n\
		     Controls the step size for parameter updates."
    )]
    pub(crate) learning_rate: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Candle device.",
        long_help = "Candle device to use for computation.\n\
		     Options: cpu, cuda, metal."
    )]
    pub(crate) device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "A device for cuda.",
        long_help = "For cuda or meta, we may want to choose a different device."
    )]
    pub(crate) device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Adjustment method.",
        long_help = "Adjust by batch or residual.\n\
		     Choose the method for batch adjustment."
    )]
    pub(crate) adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = 2,
        help = "Number of multi-level collapsing levels.",
        long_help = "Number of multi-level collapsing levels.\n\
		     More levels = coarser-to-finer batch correction.\n\
		     Set to 1 to disable multi-level."
    )]
    pub(crate) num_levels: usize,

    #[arg(
        long,
        default_value_t = 5000,
        help = "Cap feature dimension by coarsening",
        long_help = "Cap the feature dimension by grouping co-expressed features into\n\
		     meta-features. The model trains at this reduced resolution.\n\
		     On output, the dictionary is expanded back to full resolution.\n\
		     Set to 0 to disable. Default: 5000.\n\
		     Applied after --max-features selection if both are specified.\n\
		     Independent mode: computed per modality.\n\
		     Delta mode: computed on the reference modality, shared across all."
    )]
    pub(crate) max_coarse_features: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all columns data.",
        long_help = "Preload all the columns data into memory.\n\
		     Improves performance for large datasets."
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        value_enum,
        default_value = "independent",
        help = "Joint decoder type.",
        long_help = "Joint decoder type:\n\
		     - independent: each modality has its own topic dictionary.\n\
		       Modalities can have different features.\n\
		     - delta: shared base dictionary + cumulative chain deltas.\n\
		       Modality 0 = softmax(z @ W_base)\n\
		       Modality m = softmax(z @ (W_base + sum of delta_1..delta_m))\n\
		       All modalities must share the same features (genes).\n\
		       Feature selection and coarsening are shared from the\n\
		       reference modality (first). Delta logits are initialized\n\
		       to zero and diverge during training."
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
