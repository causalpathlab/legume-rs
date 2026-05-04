use crate::embed_common::*;
use crate::topic::common::{
    create_device, load_and_collapse, move_varmap_to_cpu, setup_stop_handler, LoadCollapseArgs,
    PreparedData,
};
use crate::topic::eval::{evaluate_latent_by_encoder, EvaluateLatentConfig};
use crate::topic::train::{train_mixed, TrainConfig};

use candle_util::candle_decoder_nb_mixture::{
    NbMixtureTopicDecoder, DECODER_NAME as NBMIXTURE_NAME,
};
use candle_util::candle_decoder_topic::*;
use candle_util::candle_decoder_vmf_topic::VmfTopicDecoder;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_model_traits::*;
use log::warn;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub(crate) enum DecoderType {
    /// Softmax dictionary with multinomial likelihood
    Multinom,
    /// Negative binomial with per-gene dispersion and library size
    Nb,
    /// Von Mises-Fisher mixture on unit hypersphere
    Vmf,
    /// NB with an ambient-RNA mixture (`α_g`) and per-sample ρ from library size
    NbMixture,
}

impl DecoderType {
    pub fn as_str(self) -> &'static str {
        match self {
            DecoderType::Multinom => "multinom",
            DecoderType::Nb => "nb",
            DecoderType::Vmf => "vmf",
            DecoderType::NbMixture => NBMIXTURE_NAME,
        }
    }
}

#[derive(Args, Debug)]
pub struct TopicArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Multiple files may be passed (comma- or space-separated)\n\
                     and are concatenated column-wise on a shared feature set."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet       gene × topic loadings (log-prob)\n  \
                     {out}.latent.parquet           cell × topic log-softmax proportions\n  \
                     {out}.delta.parquet            per-batch effects (if --batch-files)\n  \
                     {out}.log_likelihood.parquet   training loss trace\n  \
                     {out}.safetensors              encoder+decoder weights\n  \
                     {out}.metadata.json            model metadata (for `senna eval-topic`)\n  \
                     {out}.dispersion.parquet       NB dispersion (nb / nbmixture)\n  \
                     {out}.alpha.parquet            ambient gene profile (nbmixture)\n  \
                     {out}.rho.parquet              ρ sigmoid coefficients (nbmixture)\n  \
                     {out}.cell_proj.parquet        cached random projection (consumed by `senna layout`)\n  \
                     {out}.senna.json               run manifest consumed by `senna viz --from` and `senna plot --from`\n\n\
                     With --decoder a,b,c: per-decoder dictionaries written as {out}.{name}.dictionary.parquet."
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
        long,
        help = "Skip per-batch correction; treat all cells as a single batch",
        long_help = "Collapses batch membership to a single label so the projection,\n\
                     multilevel collapsing, and δ estimation all run as if there were\n\
                     no batch structure. Useful for homogeneous datasets or as a\n\
                     reference baseline."
    )]
    pub(crate) ignore_batch: bool,

    #[arg(
        short = 'w',
        long = "warm-start",
        help = "Warm-start projection file (cell × k)",
        long_help = "Skip random projection and use this matrix instead.\n\
                     Rows must match the concatenated cell order of the inputs."
    )]
    pub(crate) warm_start_proj_file: Option<Box<str>>,

    #[arg(
        long = "init-from",
        help = "Initialize encoder + decoder weights from a previously trained model",
        long_help = "Path prefix of a model saved by `senna topic` (matching\n\
                     {prefix}.model.json + {prefix}.safetensors). Architecture must\n\
                     match: same K, encoder layers, level_decoder_dims, and\n\
                     n_features_full / n_features_encoder. Cross-gene-set\n\
                     warm-start is not supported — train on the same gene set.\n\
                     Independent of `--warm-start` (which seeds the random\n\
                     projection)."
    )]
    pub(crate) init_from: Option<Box<str>>,

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
        default_value_t = 3,
        help = "Multi-level coarsening levels",
        long_help = "Hierarchical pseudobulk refinement passes. Level sort dims are\n\
                     linearly spaced from 4 to --sort-dim. Set to 1 to disable."
    )]
    pub(crate) num_levels: usize,

    #[arg(long, default_value_t = 20, help = "Gibbs sweeps per refinement level")]
    pub(crate) refine_gibbs: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Greedy sweeps per refinement level"
    )]
    pub(crate) refine_greedy: usize,

    #[arg(
        long = "weighting",
        value_enum,
        default_value_t = crate::refine_weighting::WeightingArg::NbFisherInfo,
        help = crate::refine_weighting::WEIGHTING_HELP,
    )]
    pub(crate) refine_weighting: crate::refine_weighting::WeightingArg,

    #[arg(long, default_value_t = 42, help = "Seed for refinement Gibbs sampler")]
    pub(crate) refine_seed: u64,

    #[arg(
        long,
        default_value_t = 30,
        help = "Batch-correction optimizer iterations",
        long_help = "Coordinate-descent steps when fitting the per-batch delta."
    )]
    pub(crate) iter_opt: usize,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    pub(crate) block_size: Option<usize>,

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

    #[arg(long, default_value_t = 100, help = "Training minibatch size")]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        alias = "lr",
        help = "Adam learning rate"
    )]
    pub(crate) learning_rate: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off; typical 0.5–5.0)"
    )]
    pub(crate) grad_clip: f32,

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
        default_value_t = false,
        help = "Load all columns into memory before training"
    )]
    pub(crate) preload_data: bool,

    #[command(flatten)]
    pub(crate) hvg: crate::hvg::HvgCliArgs,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Cap feature dim by meta-feature coarsening (0 to disable)",
        long_help = "Groups co-expressed features into ≤N meta-features so the model\n\
                     trains at reduced resolution. The dictionary is expanded back to\n\
                     full resolution on output."
    )]
    pub(crate) max_coarse_features: usize,

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "nbmixture",
        help = "Decoder type(s) [multinom|nb|vmf|nbmixture], comma-separated",
        long_help = "nbmixture — NB with ambient-RNA mixture α and per-sample ρ (default).\n\
                     nb        — negative binomial with per-gene dispersion.\n\
                     multinom  — softmax dictionary with log1p-weighted likelihood.\n\
                     vmf       — von Mises-Fisher mixture on the unit hypersphere.\n\n\
                     Multiple types (e.g. --decoder multinom,vmf) train jointly with\n\
                     a shared encoder; see --decoder-weights for loss weighting."
    )]
    pub(crate) decoder: Vec<DecoderType>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Per-decoder loss weights (default: equal)"
    )]
    pub(crate) decoder_weights: Option<Vec<f64>>,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "Uniform smoothing α for topic proportions (0 = off)",
        long_help = "θ = (1-α) softmax(z) + α/K. Prevents dead topics by keeping\n\
                     every topic on the gradient path. Set 0 to disable."
    )]
    pub(crate) topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-cell refinement steps at inference (0 = off)",
        long_help = "Gradient steps that optimize topic logits against the frozen\n\
                     decoder likelihood, anchored to the encoder output by L2."
    )]
    pub(crate) refine_steps: usize,

    #[arg(long, default_value_t = 0.01, help = "Refinement learning rate")]
    pub(crate) refine_lr: f64,

    #[arg(long, default_value_t = 1.0, help = "Refinement L2 regularization")]
    pub(crate) refine_reg: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Cross-entropy penalty λ on β toward anchor prior (0 = off)",
        long_help = "Pulls the decoder dictionary toward anchor PB expression\n\
                     profiles during training. β starts from random init;\n\
                     the penalty guides it. Ignored by vMF decoder."
    )]
    pub(crate) anchor_penalty: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Beta(α,β) prior weight on ρ (0 = off; nbmixture only, rarely used)"
    )]
    pub(crate) rho_prior_weight: f32,

    #[arg(
        long,
        default_value_t = 2.0,
        hide = true,
        help = "Beta(α,·) shape on ρ prior"
    )]
    pub(crate) rho_prior_alpha: f32,

    #[arg(
        long,
        default_value_t = 18.0,
        hide = true,
        help = "Beta(·,β) shape on ρ prior"
    )]
    pub(crate) rho_prior_beta: f32,

    #[command(flatten)]
    pub(crate) cnv: CnvArgs,
}

pub fn fit_topic_model(args: &TopicArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
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
        max_features: args.hvg.n_hvg,
        feature_list_file: args.hvg.feature_list_file.as_deref(),
        refine: Some(data_beans_alg::refine_multilevel::RefineParams {
            num_gibbs: args.refine_gibbs,
            num_greedy: args.refine_greedy,
            gene_weighting: args.refine_weighting.into(),
            seed: args.refine_seed,
            ..data_beans_alg::refine_multilevel::RefineParams::default()
        }),
        ignore_batch: args.ignore_batch,
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
    let n_features_encoder = finest_coarsening.map_or(n_features_full, |c| c.num_coarse);

    let dev = create_device(&args.device, args.device_no)?;

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let level_decoder_dims: Vec<usize> = level_coarsenings
        .iter()
        .map(|fc| fc.as_ref().map_or(n_features_full, |c| c.num_coarse))
        .collect();

    info!(
        "input: {} -> encoder -> {:?} decoder(s) (dims {:?}) -> finest: {}",
        n_features_encoder, args.decoder, level_decoder_dims, n_features_encoder,
    );

    let gene_names = data_vec.row_names()?;

    let stop = setup_stop_handler();

    let anchor_prior = crate::topic::anchor_prior::AnchorPrior::from_pseudobulk(
        finest_collapsed,
        n_topics,
        finest_coarsening,
    )?;

    // Per-level [K, D_l] anchor tensors on the training device. Built once
    // here, held alive for the entire fit via the outer scope.
    let anchor_tensors = anchor_prior.per_level_device_tensors(&level_coarsenings, &dev)?;

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
        anchor_prior: Some(&anchor_prior),
        anchor_prior_per_level: Some(&anchor_tensors),
    };

    let mut encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_features_encoder,
            n_topics,
            layers: &args.encoder_layers,
        },
        &parameters,
        param_builder.clone(),
    )?;

    let (scores, z_nk) = if args.decoder.len() == 1 {
        match args.decoder[0] {
            DecoderType::Nb => run_topic_pipeline::<_, NbTopicDecoder>(&ctx, &mut encoder)?,
            DecoderType::Multinom => {
                run_topic_pipeline::<_, MultinomTopicDecoder>(&ctx, &mut encoder)?
            }
            DecoderType::Vmf => run_topic_pipeline::<_, VmfTopicDecoder>(&ctx, &mut encoder)?,
            DecoderType::NbMixture => {
                run_topic_pipeline::<_, NbMixtureTopicDecoder>(&ctx, &mut encoder)?
            }
        }
    } else {
        run_multi_decoder_pipeline(&ctx, &mut encoder)?
    };

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    let cell_names = data_vec.column_names()?;

    crate::output_helpers::save_latent(&args.out, &z_nk, &cell_names)?;

    // CNV detection using topic proportions as cell-type membership
    let gene_names = data_vec.row_names()?;
    let cnv_positions = crate::cnv_pseudobulk::load_gene_positions(&args.cnv, &gene_names)?;

    if let Some(positions) = cnv_positions {
        if let Some(batch_labels) = crate::cnv_pseudobulk::reconstruct_batch_labels(&data_vec) {
            let topic_probs = z_nk.map(f32::exp);
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

    crate::postprocess::viz_prep::write_cell_proj(&args.out, &proj_kn, &cell_names)?;

    write_topic_manifest(
        crate::run_manifest::RunKind::Topic,
        &args.out,
        &args.data_files,
        args.batch_files.as_deref(),
    )?;

    info!("Done");
    Ok(())
}

/// Assemble + save the `{prefix}.senna.json` manifest for a topic /
/// itopic / joint-topic run. Factored out so all three callers stay
/// DRY; SVD runs use a distinct helper because they produce no model.
fn write_topic_manifest(
    kind: crate::run_manifest::RunKind,
    prefix: &str,
    data_files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
) -> anyhow::Result<()> {
    let input: Vec<String> = data_files.iter().map(|s| s.to_string()).collect();
    let batch: Vec<String> = batch_files
        .map(|v| v.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind,
        prefix,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: true,
        has_cell_proj: true,
        pb_gene_suffix: Some("pb_gene.parquet"),
        pb_latent_suffix: Some("pb_latent.parquet"),
        dictionary_empirical_suffix: Some("dictionary_empirical.parquet"),
        default_colour_by: "topic",
    })
}

// Decoder-specific post-training I/O moved to `crate::topic::decoder_output`.
// This file keeps only the per-decoder hyperparameter `ConfigureDecoder`
// trait, since that surface depends on the local `TopicArgs`.

use crate::topic::decoder_output::{write_dictionary_tensor, DecoderExtras};

/// Trait for optional per-run hyperparameter configuration from CLI args.
/// Default is no-op; `NbMixtureTopicDecoder` overrides to set Beta(ρ) prior.
trait ConfigureDecoder {
    fn configure(&mut self, _args: &TopicArgs) {}
}

impl ConfigureDecoder for MultinomTopicDecoder {}
impl ConfigureDecoder for NbTopicDecoder {}
impl ConfigureDecoder for VmfTopicDecoder {}
impl ConfigureDecoder for NbMixtureTopicDecoder {
    fn configure(&mut self, args: &TopicArgs) {
        self.set_rho_prior(
            args.rho_prior_weight,
            args.rho_prior_alpha,
            args.rho_prior_beta,
        );
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
    /// Data-driven β prior built from finest-level pseudobulks. Used for
    /// β init and (when `--anchor-penalty > 0`) as a training-time
    /// cross-entropy penalty.
    anchor_prior: Option<&'a crate::topic::anchor_prior::AnchorPrior>,
    /// Per-level `[D_l, K]` anchor tensors pre-built on `dev`.
    anchor_prior_per_level: Option<&'a [candle_core::Tensor]>,
}

fn run_topic_pipeline<Enc, Dec>(
    ctx: &PipelineCtx,
    encoder: &mut Enc,
) -> anyhow::Result<(TrainScores, Mat)>
where
    Enc: EncoderModuleT + Send + Sync,
    Dec: DecoderModuleT + DecoderExtras + NewDecoder + ConfigureDecoder + Send + Sync,
{
    let mut decoders: Vec<Dec> = ctx
        .level_decoder_dims
        .iter()
        .enumerate()
        .map(|(i, &d_l)| {
            Dec::new(d_l, ctx.n_topics, ctx.param_builder.pp(format!("dec_{i}")))
                .expect("decoder creation")
        })
        .collect();
    for dec in &mut decoders {
        dec.configure(ctx.args);
    }

    // β init from anchor prior is disabled — random (Kaiming) initialisation
    // works well when the anchor penalty (default 1.0) pulls β toward the
    // prior during training. Warm-starting logits with log(anchor) can lock
    // the dictionary too early.

    // Optional model-checkpoint warm-start: load encoder + decoder weights
    // from a previously trained run (must match this run's architecture).
    if let Some(prefix) = ctx.args.init_from.as_deref() {
        use crate::topic::warm_start::{warm_start_load, WarmStartCheck};
        let n_features_encoder = *ctx
            .level_decoder_dims
            .last()
            .unwrap_or(&ctx.n_features_full);
        warm_start_load(
            ctx.parameters,
            prefix,
            &WarmStartCheck {
                model_type_expected: crate::topic::model_metadata::MODEL_TYPE_TOPIC,
                n_topics: ctx.n_topics,
                n_features_full: ctx.n_features_full,
                n_features_encoder,
                encoder_hidden: &ctx.args.encoder_layers,
                level_decoder_dims: ctx.level_decoder_dims,
                embedding_dim: None,
            },
        )?;
    }

    let train_config = TrainConfig {
        parameters: ctx.parameters,
        dev: ctx.dev,
        args: ctx.args,
        stop: ctx.stop,
        anchor_prior_per_level: ctx.anchor_prior_per_level,
        anchor_penalty: ctx.args.anchor_penalty,
    };
    let scores = train_mixed(
        ctx.collapsed_levels,
        encoder,
        &decoders,
        ctx.level_coarsenings,
        &train_config,
    )?;

    // PB-level topic usage + persistence of pb_gene / pb_latent for
    // downstream `senna annotate` (enrichment-based annotation works from
    // PB aggregates, no zarr reopen needed).
    {
        let enc_fc = ctx.level_coarsenings.last().and_then(|c| c.as_ref());
        let (mixed, batch, _) = crate::topic::common::sample_collapsed_data(ctx.finest_collapsed)?;
        let enc_nd = if let Some(fc) = enc_fc {
            fc.aggregate_columns_nd(&mixed)
        } else {
            mixed
        };
        let batch_nd = batch.map(|b| {
            if let Some(fc) = enc_fc {
                fc.aggregate_columns_nd(&b)
            } else {
                b
            }
        });
        let enc_t = enc_nd.to_tensor(ctx.dev)?;
        let batch_t = batch_nd.map(|b| b.to_tensor(ctx.dev)).transpose()?;
        let (log_z, _) = encoder.forward_t(&enc_t, batch_t.as_ref(), false)?;
        let theta: Vec<Vec<f32>> = log_z.exp()?.to_vec2()?;
        let n_pb = theta.len();
        let k = theta[0].len();
        let mean_t: Vec<f32> = (0..k)
            .map(|ki| theta.iter().map(|r| r[ki]).sum::<f32>() / n_pb as f32)
            .collect();
        let active = mean_t.iter().filter(|&&v| v > 0.01).count();
        info!(
            "PB topic usage ({} PBs): mean_θ={:?}, active={}/{}",
            n_pb,
            mean_t.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
            active,
            k,
        );

        let mut pb_latent_pk = Mat::zeros(n_pb, k);
        for (pi, row) in theta.iter().enumerate() {
            for (kj, v) in row.iter().enumerate() {
                pb_latent_pk[(pi, kj)] = *v;
            }
        }
        let pb_names = axis_id_names("PB_", n_pb);
        let topic_names = axis_id_names("T", k);
        pb_latent_pk.to_parquet_with_names(
            &format!("{}.pb_latent.parquet", ctx.args.out),
            (Some(&pb_names), Some("pb")),
            Some(&topic_names),
        )?;

        let pb_gene_gp: Mat = ctx.finest_collapsed.mu_observed.posterior_mean().clone();
        crate::output_helpers::save_pb_gene(ctx.args.out.as_ref(), &pb_gene_gp, ctx.gene_names)?;

        // Empirical NB-Fisher-weighted gene × topic dictionary at full gene
        // resolution. Avoids the lossy expand-from-coarse approximation in
        // `dictionary.parquet` so rare informative genes survive into the
        // annotate-side enrichment ranking.
        info!("Computing NB Fisher gene weights for empirical dictionary");
        let fisher_w =
            crate::empirical_dict::compute_nb_fisher_weights(ctx.data_vec, ctx.args.block_size)?;
        crate::output_helpers::save_fisher_weights(
            ctx.args.out.as_ref(),
            &fisher_w,
            ctx.gene_names,
        )?;
        info!(
            "Wrote {}.fisher_weights.parquet ({} genes)",
            ctx.args.out,
            fisher_w.len()
        );

        let beta_emp = crate::empirical_dict::build_empirical_dictionary(
            &pb_gene_gp,
            &pb_latent_pk,
            &fisher_w,
        );
        beta_emp.to_parquet_with_names(
            &format!("{}.dictionary_empirical.parquet", ctx.args.out),
            (Some(ctx.gene_names), Some("gene")),
            Some(&topic_names),
        )?;
        info!(
            "Wrote empirical dictionary {}×{} (NB-Fisher-weighted, column-simplex)",
            beta_emp.nrows(),
            beta_emp.ncols()
        );
    }

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

    let weights = compute_decoder_weights(&ctx.args.decoder, ctx.args.decoder_weights.as_ref());
    let z_nk = save_metadata_and_evaluate::<Dec>(ctx, &weights)?;
    Ok((scores, z_nk))
}

/// Compute normalized decoder weights. If user provided weights, normalize
/// them to sum to 1. Otherwise, use equal weights.
fn compute_decoder_weights(decoders: &[DecoderType], user_weights: Option<&Vec<f64>>) -> Vec<f64> {
    if let Some(w) = user_weights {
        let sum: f64 = w.iter().sum();
        w.iter().map(|x| x / sum).collect()
    } else {
        let n = decoders.len() as f64;
        vec![1.0 / n; decoders.len()]
    }
}

/// Save model metadata/weights, move parameters to CPU, and evaluate latent states.
///
/// When `--refine-steps > 0`, rebuilds the finest-level decoder on CPU and
/// uses it for per-cell likelihood refinement during evaluation.
fn save_metadata_and_evaluate<Dec>(
    ctx: &PipelineCtx,
    decoder_weights: &[f64],
) -> anyhow::Result<Mat>
where
    Dec: DecoderModuleT + NewDecoder + Send + Sync,
{
    use crate::topic::model_metadata::{save_coarsening, save_parameters, TopicModelMetadata};
    use candle_util::candle_topic_refinement::TopicRefinementConfig;

    save_parameters(ctx.parameters, &ctx.args.out)?;

    let mut metadata = TopicModelMetadata {
        model_type: crate::topic::model_metadata::MODEL_TYPE_TOPIC.into(),
        decoder_types: ctx.args.decoder.iter().map(|d| d.as_str().into()).collect(),
        decoder_weights: decoder_weights.to_vec(),
        n_features_encoder: *ctx
            .level_decoder_dims
            .last()
            .unwrap_or(&ctx.n_features_full),
        n_features_full: ctx.n_features_full,
        n_topics: ctx.n_topics,
        encoder_hidden: ctx.args.encoder_layers.clone(),
        num_levels: ctx.level_decoder_dims.len(),
        level_decoder_dims: ctx.level_decoder_dims.to_vec(),
        adj_method: ctx.args.adj_method.as_str().into(),
        has_coarsening: ctx.finest_coarsening.is_some(),
        embedding_dim: None,
        enc_context_size: None,
        dec_context_size: None,
        theta_mean: None,
    };
    metadata.save(&ctx.args.out)?;

    if let Some(fc) = ctx.finest_coarsening {
        save_coarsening(fc, &ctx.args.out)?;
    }

    // Move VarMap to CPU, then rebuild encoder from CPU Vars.
    // The old encoder still holds Metal/CUDA Vars and must not be reused.
    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    move_varmap_to_cpu(ctx.parameters)?;

    let n_features_encoder = ctx
        .finest_coarsening
        .map_or(ctx.n_features_full, |c| c.num_coarse);
    let cpu_vb =
        candle_nn::VarBuilder::from_varmap(ctx.parameters, candle_core::DType::F32, &cpu_dev);
    let cpu_encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_features_encoder,
            n_topics: ctx.n_topics,
            layers: &ctx.args.encoder_layers,
        },
        ctx.parameters,
        cpu_vb.clone(),
    )?;

    let refine_config = if ctx.args.refine_steps > 0 {
        Some(TopicRefinementConfig {
            num_steps: ctx.args.refine_steps,
            learning_rate: ctx.args.refine_lr,
            regularization: ctx.args.refine_reg,
        })
    } else {
        None
    };

    let finest_dec_dim = *ctx
        .level_decoder_dims
        .last()
        .unwrap_or(&ctx.n_features_full);
    let finest_dec_idx = ctx.level_decoder_dims.len().saturating_sub(1);
    let refine_decoder = if refine_config.is_some() {
        let cpu_vb =
            candle_nn::VarBuilder::from_varmap(ctx.parameters, candle_core::DType::F32, &cpu_dev);
        Some(Dec::new(
            finest_dec_dim,
            ctx.n_topics,
            cpu_vb.pp(format!("dec_{finest_dec_idx}")),
        )?)
    } else {
        None
    };

    let eval_config = EvaluateLatentConfig {
        dev: &cpu_dev,
        adj_method: &ctx.args.adj_method,
        minibatch_size: ctx.args.minibatch_size,
        feature_coarsening: ctx.finest_coarsening,
        decoder: refine_decoder.as_ref(),
        refine_config: refine_config.as_ref(),
    };
    let z_nk = evaluate_latent_by_encoder(
        ctx.data_vec,
        &cpu_encoder,
        ctx.finest_collapsed,
        &eval_config,
    )?;

    // Re-save metadata with θ̄_train populated — initial δ guess at predict
    // time, better than uniform 1/K when training is composition-imbalanced.
    metadata.populate_theta_mean_and_save(&z_nk, &ctx.args.out)?;
    Ok(z_nk)
}

/// Multi-decoder pipeline: builds multiple decoder types per level,
/// trains with weighted multi-decoder loss, saves per-decoder dictionaries.
fn run_multi_decoder_pipeline<Enc: EncoderModuleT + Send + Sync>(
    ctx: &PipelineCtx,
    encoder: &mut Enc,
) -> anyhow::Result<(TrainScores, Mat)> {
    use crate::topic::train::train_mixed_multi_decoder;
    use candle_util::candle_dyn_decoder::{create_dyn_decoder, DynDecoderModuleT};

    let decoder_weights =
        compute_decoder_weights(&ctx.args.decoder, ctx.args.decoder_weights.as_ref());

    // Build per-level × per-decoder-type grid
    let decoders_per_level: Vec<Vec<Box<dyn DynDecoderModuleT>>> = ctx
        .level_decoder_dims
        .iter()
        .enumerate()
        .map(|(level_i, &d_l)| {
            ctx.args
                .decoder
                .iter()
                .map(|dec_type| {
                    let name = dec_type.as_str();
                    let prefix = format!("dec_{level_i}.{name}");
                    create_dyn_decoder(name, d_l, ctx.n_topics, ctx.param_builder.pp(prefix))
                        .expect("decoder creation")
                })
                .collect()
        })
        .collect();

    // Optional model-checkpoint warm-start (multi-decoder variant).
    if let Some(prefix) = ctx.args.init_from.as_deref() {
        use crate::topic::warm_start::{warm_start_load, WarmStartCheck};
        let n_features_encoder = *ctx
            .level_decoder_dims
            .last()
            .unwrap_or(&ctx.n_features_full);
        warm_start_load(
            ctx.parameters,
            prefix,
            &WarmStartCheck {
                model_type_expected: crate::topic::model_metadata::MODEL_TYPE_TOPIC,
                n_topics: ctx.n_topics,
                n_features_full: ctx.n_features_full,
                n_features_encoder,
                encoder_hidden: &ctx.args.encoder_layers,
                level_decoder_dims: ctx.level_decoder_dims,
                embedding_dim: None,
            },
        )?;
    }

    let train_config = TrainConfig {
        parameters: ctx.parameters,
        dev: ctx.dev,
        args: ctx.args,
        stop: ctx.stop,
        anchor_prior_per_level: None,
        anchor_penalty: 0.0,
    };

    if ctx.anchor_prior.is_some() {
        warn!("anchor prior is not applied in multi-decoder mode; β init + penalty skipped");
    }

    let scores = train_mixed_multi_decoder(
        ctx.collapsed_levels,
        encoder,
        &decoders_per_level,
        ctx.level_coarsenings,
        &decoder_weights,
        &train_config,
    )?;

    // Write per-decoder dictionaries at finest level
    info!("Writing down the model parameters");
    for dec in decoders_per_level.last().unwrap() {
        let name = dec.decoder_name();
        let out_prefix = format!("{}.{}", ctx.args.out, name);
        let dict_tensor = dec.get_dictionary()?;
        write_dictionary_tensor(
            &dict_tensor,
            ctx.finest_coarsening,
            ctx.n_features_full,
            ctx.gene_names,
            &out_prefix,
        )?;
    }

    let z_nk = save_metadata_and_evaluate::<MultinomTopicDecoder>(ctx, &decoder_weights)?;
    Ok((scores, z_nk))
}
