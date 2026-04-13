use crate::embed_common::*;
use crate::topic::common::*;
use crate::topic::eval::{evaluate_latent_by_encoder, EvaluateLatentConfig};
use crate::topic::train::{train_mixed, train_mixed_vcd, train_progressive, TrainConfig};

use candle_util::candle_decoder_topic::*;
use candle_util::candle_decoder_vmf_topic::*;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_model_traits::*;
use candle_util::candle_topic_refinement::*;
use log::warn;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub(crate) enum DecoderType {
    /// Softmax dictionary with multinomial likelihood
    Multinom,
    /// Negative binomial with per-gene dispersion and library size
    Nb,
    /// Von Mises-Fisher mixture on unit hypersphere
    Vmf,
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
    pub(crate) data_files: Vec<Box<str>>,

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
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Random projection dimension to project the data.\n\
		     Controls the dimensionality of the random projection step."
    )]
    pub(crate) proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection",
        long_help = "Use top {d} components of projection.\n\
		     Number of samples will be less than `2^{d}+1`."
    )]
    pub(crate) sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.csv,batch2.csv"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'w',
        long = "warm-start",
        help = "Warm start projection file",
        long_help = "Warm start from the previous projection (cell x k).\n\
		     Provide a file to initialize the projection."
    )]
    pub(crate) warm_start_proj_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch",
        long_help = "Number of k-nearest neighbours within each batch.\n\
		     Controls the number of neighbours for \n\
		     nearest neighbour search."
    )]
    pub(crate) knn_cells: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of multi-level coarsening levels",
        long_help = "Number of multi-level coarsening levels for batch correction.\n\
		     Higher values add intermediate refinement steps.\n\
		     Level sort dimensions are linearly spaced from 4 to sort_dim."
    )]
    pub(crate) num_levels: usize,

    #[arg(
        long,
        value_enum,
        default_value = "mixed",
        help = "Multi-level training schedule"
    )]
    pub(crate) level_schedule: LevelSchedule,

    #[arg(
        long,
        default_value_t = 30,
        help = "Optimization iterations",
        long_help = "Number of optimization iterations.\n\
		     Controls the number of steps for model optimization."
    )]
    pub(crate) iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing",
        long_help = "Block size (number of columns) for parallel processing.\n\
		     Controls the granularity of parallel computation."
    )]
    pub(crate) block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics",
        long_help = "Number of latent topics.\n\
		     Controls the dimensionality of the latent topic space."
    )]
    pub(crate) n_latent_topics: usize,

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
    pub(crate) encoder_layers: Vec<usize>,

    #[arg(
        long,
        short = 'i',
        default_value_t = 1000,
        help = "Number of training epochs",
        long_help = "Number of training epochs.\n\
		     Controls how many times the model is trained over the data."
    )]
    pub(crate) epochs: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Data jitter interval",
        long_help = "Data jitter interval.\n\
		     Controls the interval for adding jitter to the collapsed data\n\
		     by posterior resampling during VAE training."
    )]
    pub(crate) jitter_interval: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Minibatch size",
        long_help = "Minibatch size for training.\n\
		     Controls the number of samples per training batch."
    )]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        alias = "lr",
        help = "Learning rate",
        long_help = "Learning rate for optimization.\n\
		     Controls the step size for parameter updates."
    )]
    pub(crate) learning_rate: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Candle device",
        long_help = "Candle device to use for computation.\n\
		     Options: cpu, cuda, metal."
    )]
    pub(crate) device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "A device for cuda",
        long_help = "For cuda or meta, we may want to choose a different device."
    )]
    pub(crate) device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Adjustment method",
        long_help = "Adjust by batch or residual.\n\
		     Choose the method for batch adjustment."
    )]
    pub(crate) adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all columns data",
        long_help = "Preload all the columns data into memory.\n\
		     Improves performance for large datasets."
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        default_value_t = 1000,
        alias = "max-features",
        help = "Cap feature dimension by coarsening",
        long_help = "Cap the feature dimension by grouping co-expressed features into\n\
		     meta-features. The model trains at this reduced resolution.\n\
		     On output, the dictionary is expanded back to full resolution.\n\
		     Set to 0 to disable. Default: 1000.\n\
		     Applied after --max-features selection if both are specified."
    )]
    pub(crate) max_coarse_features: usize,

    #[arg(
        long,
        value_enum,
        default_value = "multinom",
        help = "Decoder type",
        long_help = "Topic decoder type:\n\
		     multinom: softmax dictionary with multinomial likelihood\n\
		     nb: negative binomial with per-gene dispersion and library size\n\
		     vmf: von Mises-Fisher mixture on unit hypersphere"
    )]
    pub(crate) decoder: DecoderType,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "Topic smoothing during training",
        long_help = "Mix encoder topic proportions with uniform distribution during training:\n\
		     z_smooth = (1 - α) * z_nk + α / K\n\
		     Ensures every topic receives gradient signal through the decoder,\n\
		     preventing dead topics. Only applied during training.\n\
		     Typical values: 0.01-0.2. Set to 0 to disable."
    )]
    pub(crate) topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Epochs of VCD training before switching to SGVB",
        long_help = "Number of initial epochs using variational contrastive divergence (VCD).\n\
		     VCD refines encoder samples via elliptical slice sampling (ESS),\n\
		     then switches to standard SGVB for remaining epochs.\n\
		     Set to 0 to use SGVB only (default)."
    )]
    pub(crate) vcd_epochs: usize,

    #[arg(
        long,
        default_value_t = 5,
        alias = "ess-steps",
        help = "ESS steps per minibatch during VCD epochs"
    )]
    pub(crate) vcd_ess_steps: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Max shrink iterations per ESS step"
    )]
    pub(crate) ess_max_shrink: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-cell refinement steps at inference",
        long_help = "Number of gradient steps for per-cell topic refinement at inference time.\n\
		     Optimizes topic logits against the frozen decoder likelihood,\n\
		     anchored to the encoder output via L2 regularization.\n\
		     Set to 0 to disable (default)."
    )]
    pub(crate) refine_steps: usize,

    #[arg(long, default_value_t = 0.01, help = "Learning rate for refinement")]
    pub(crate) refine_lr: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 regularization strength for refinement"
    )]
    pub(crate) refine_reg: f64,

    // CNV detection args
    #[command(flatten)]
    pub(crate) cnv: CnvArgs,
}

pub fn fit_topic_model(args: &TopicArgs) -> anyhow::Result<()> {
    let PreparedData {
        data_vec,
        collapsed_levels,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &args.data_files,
        batch_files: &args.batch_files,
        preload: args.preload_data,
        warm_start_proj_file: args.warm_start_proj_file.as_deref(),
        proj_dim: args.proj_dim.max(args.n_latent_topics),
        sort_dim: args.sort_dim,
        knn_cells: args.knn_cells,
        num_levels: args.num_levels,
        iter_opt: args.iter_opt,
        block_size: args.block_size,
        out: &args.out,
    })?;

    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();

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

    let dev = create_device(&args.device, args.device_no)?;

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

    let stop = setup_stop_handler();

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

    if args.vcd_epochs > 0 && matches!(args.level_schedule, LevelSchedule::Progressive) {
        warn!("--vcd-epochs is only supported with --level-schedule mixed; ignoring VCD");
    }

    // Build per-level decoders, train, save dictionary, and evaluate.
    let ctx = PipelineCtx {
        level_decoder_dims: &level_decoder_dims,
        n_topics,
        param_builder: &param_builder,
        collapsed_levels: &collapsed_levels,
        level_coarsenings: &level_coarsenings,
        finest_coarsening,
        finest_collapsed,
        n_features_full,
        gene_names: &gene_names,
        data_vec: &data_vec,
        parameters: &parameters,
        dev: &dev,
        args,
        stop: &stop,
        refine_config: refine_config.as_ref(),
    };

    let (scores, z_nk) = match args.decoder {
        DecoderType::Nb => run_topic_pipeline::<NbTopicDecoder>(&ctx, &mut encoder)?,
        DecoderType::Multinom => run_topic_pipeline::<TopicDecoder>(&ctx, &mut encoder)?,
        DecoderType::Vmf => run_topic_pipeline::<VmfTopicDecoder>(&ctx, &mut encoder)?,
    };

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    // CNV detection using topic proportions as cell-type membership
    let gene_names = data_vec.row_names()?;
    let cnv_positions = crate::cnv_pseudobulk::load_gene_positions(&args.cnv, &gene_names)?;

    if let Some(positions) = cnv_positions {
        if let Some(batch_labels) = crate::cnv_pseudobulk::reconstruct_batch_labels(&data_vec) {
            let topic_probs = z_nk.map(|x| x.exp());
            let cnv_config = crate::cnv_pseudobulk::build_cnv_config(&args.cnv);

            let cnv_result = crate::cnv_pseudobulk::detect_cnv_topic_informed(
                data_vec,
                &topic_probs,
                &batch_labels,
                &positions,
                &cnv_config,
            )?;

            crate::cnv_pseudobulk::write_cnv_results(&cnv_result, &args.out, &gene_names)?;
        } else {
            info!("CNV detection: skipped (no batch information)");
        }
    }

    info!("Done");
    Ok(())
}

// ---------------------------------------------------------------------------
// DecoderExtras — decoder-specific post-training I/O
// ---------------------------------------------------------------------------

/// Decoder-specific post-training output (dictionary writing, extra parameters).
trait DecoderExtras {
    /// Write the dictionary to parquet.
    /// Default: log-prob space with optional coarsening expansion.
    fn write_dictionary(
        &self,
        coarsening: Option<&FeatureCoarsening>,
        n_features_full: usize,
        gene_names: &[Box<str>],
        out_prefix: &str,
    ) -> anyhow::Result<()>
    where
        Self: DecoderModuleT,
    {
        write_dictionary_expanded(self, coarsening, n_features_full, gene_names, out_prefix)
    }

    /// Write decoder-specific extra parameters (dispersion, kappa, etc.).
    /// No-op by default.
    fn write_extras(
        &self,
        _coarsening: Option<&FeatureCoarsening>,
        _n_features_full: usize,
        _gene_names: &[Box<str>],
        _out_prefix: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl DecoderExtras for TopicDecoder {}

impl DecoderExtras for NbTopicDecoder {
    fn write_extras(
        &self,
        coarsening: Option<&FeatureCoarsening>,
        n_features_full: usize,
        gene_names: &[Box<str>],
        out_prefix: &str,
    ) -> anyhow::Result<()> {
        let log_phi = self.log_phi().to_device(&candle_core::Device::Cpu)?;
        let phi_vec: Vec<f32> = log_phi.exp()?.flatten_all()?.to_vec1()?;
        let phi_expanded: Vec<f32> = if let Some(fc) = coarsening {
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
            &(out_prefix.to_string() + ".dispersion.parquet"),
            (Some(gene_names), Some("gene")),
            Some(&col_names),
        )?;
        info!(
            "Saved dispersion parameters to {}.dispersion.parquet",
            out_prefix
        );
        Ok(())
    }
}

impl DecoderExtras for VmfTopicDecoder {
    /// vMF dictionary: expand coarse directions to full resolution and re-normalize.
    fn write_dictionary(
        &self,
        coarsening: Option<&FeatureCoarsening>,
        n_features_full: usize,
        gene_names: &[Box<str>],
        out_prefix: &str,
    ) -> anyhow::Result<()> {
        let dict_tensor = self
            .get_dictionary()?
            .to_device(&candle_core::Device::Cpu)?;
        let dict_dk: Mat = Mat::from_tensor(&dict_tensor)?;

        let out_dk = if let Some(fc) = coarsening {
            let k = dict_dk.ncols();
            let mut expanded = Mat::zeros(n_features_full, k);
            for (c, fine_indices) in fc.coarse_to_fine.iter().enumerate() {
                for &f in fine_indices {
                    for kk in 0..k {
                        expanded[(f, kk)] = dict_dk[(c, kk)];
                    }
                }
            }
            // Re-normalize each column to unit length
            for kk in 0..k {
                let col = expanded.column(kk);
                let norm = col.dot(&col).sqrt();
                if norm > 1e-12 {
                    expanded.column_mut(kk).scale_mut(1.0 / norm);
                }
            }
            expanded
        } else {
            dict_dk
        };

        out_dk.to_parquet_with_names(
            &(out_prefix.to_string() + ".dictionary.parquet"),
            (Some(gene_names), Some("gene")),
            None,
        )?;
        Ok(())
    }

    fn write_extras(
        &self,
        _coarsening: Option<&FeatureCoarsening>,
        _n_features_full: usize,
        _gene_names: &[Box<str>],
        _out_prefix: &str,
    ) -> anyhow::Result<()> {
        let kappas = self.kappa_vec()?;
        let kappa_strs: Vec<String> = kappas.iter().map(|k| format!("{:.2}", k)).collect();
        info!("vMF concentration κ = [{}]", kappa_strs.join(", "));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Generic topic pipeline
// ---------------------------------------------------------------------------

struct PipelineCtx<'a> {
    level_decoder_dims: &'a [usize],
    n_topics: usize,
    param_builder: &'a candle_nn::VarBuilder<'a>,
    collapsed_levels: &'a [CollapsedOut],
    level_coarsenings: &'a [Option<FeatureCoarsening>],
    finest_coarsening: Option<&'a FeatureCoarsening>,
    finest_collapsed: &'a CollapsedOut,
    n_features_full: usize,
    gene_names: &'a [Box<str>],
    data_vec: &'a SparseIoVec,
    parameters: &'a candle_nn::VarMap,
    dev: &'a candle_core::Device,
    args: &'a TopicArgs,
    stop: &'a std::sync::atomic::AtomicBool,
    refine_config: Option<&'a TopicRefinementConfig>,
}

fn run_topic_pipeline<Dec>(
    ctx: &PipelineCtx,
    encoder: &mut LogSoftmaxEncoder,
) -> anyhow::Result<(TrainScores, Mat)>
where
    Dec: DecoderModuleT + DecoderExtras + NewDecoder + Send + Sync,
{
    let decoders: Vec<Dec> = ctx
        .level_decoder_dims
        .iter()
        .enumerate()
        .map(|(i, &d_l)| {
            Dec::new(d_l, ctx.n_topics, ctx.param_builder.pp(format!("dec_{i}")))
                .expect("decoder creation")
        })
        .collect();

    let train_config = TrainConfig {
        parameters: ctx.parameters,
        dev: ctx.dev,
        args: ctx.args,
        stop: ctx.stop,
    };
    let scores = match (&ctx.args.level_schedule, ctx.args.vcd_epochs > 0) {
        (LevelSchedule::Progressive, _) => train_progressive(
            ctx.collapsed_levels,
            encoder,
            &decoders,
            ctx.level_coarsenings,
            &train_config,
        )?,
        (LevelSchedule::Mixed, false) => train_mixed(
            ctx.collapsed_levels,
            encoder,
            &decoders,
            ctx.level_coarsenings,
            &train_config,
        )?,
        (LevelSchedule::Mixed, true) => train_mixed_vcd(
            ctx.collapsed_levels,
            encoder,
            &decoders,
            ctx.level_coarsenings,
            &train_config,
        )?,
    };

    info!("Writing down the model parameters");

    let finest_decoder = decoders.last().unwrap();
    finest_decoder.write_dictionary(
        ctx.finest_coarsening,
        ctx.n_features_full,
        ctx.gene_names,
        &ctx.args.out,
    )?;
    finest_decoder.write_extras(
        ctx.finest_coarsening,
        ctx.n_features_full,
        ctx.gene_names,
        &ctx.args.out,
    )?;

    info!("Writing down the latent states");
    let eval_config = EvaluateLatentConfig {
        dev: ctx.dev,
        adj_method: &ctx.args.adj_method,
        minibatch_size: ctx.args.minibatch_size,
        feature_coarsening: ctx.finest_coarsening,
        decoder: Some(finest_decoder),
        refine_config: ctx.refine_config,
    };
    let z_nk =
        evaluate_latent_by_encoder(ctx.data_vec, encoder, ctx.finest_collapsed, &eval_config)?;

    Ok((scores, z_nk))
}

/// Write dictionary with optional expansion from coarse to fine resolution.
fn write_dictionary_expanded<Dec: DecoderModuleT + ?Sized>(
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
