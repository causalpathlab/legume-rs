//! Unified prediction subcommand for the dense and masked topic models.
//!
//! Loads a model trained by `senna topic` / `masked-topic`, applies it
//! to a held-out backend file, and writes:
//!   - `{out}.latent.parquet`     [N × K] log θ (per-cell topic proportions)
//!   - `{out}.predictive.parquet` [N × 3] per-cell `[llik, total, llik_per_count]`
//!
//! Three latent-inference modes (mutually exclusive):
//!   - `Encoder` (default): forward pass through the trained encoder only.
//!   - `EncoderRefine` (`--refine-steps > 0`): encoder warm-start, then
//!     decoder gradient on θ anchored to encoder by L2.
//!   - `DecoderOnly` (`--decoder-only`): skip encoder; init θ uniform
//!     `log(1/K)`; optimize purely against the frozen decoder. Useful when
//!     the test feature set is too divergent for the encoder.

use crate::embed_common::*;
use crate::masked_topic::FeatureNameKindArg;
use crate::topic::eval::{build_gene_remap_with, GeneRemap, QueryNameOpts};
use crate::topic::model_metadata::{
    load_coarsening, load_dictionary, load_shortlist_weights, TopicModelMetadata,
};
use crate::topic::predict_common::{
    decoder_only_inference_dense, estimate_delta, predictive_llik_dense, LatentMode,
};

use crate::logging::new_progress_bar;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use candle_core::{Device, Tensor};
use candle_util::decoder::nb_mixture::DECODER_NAME as NBMIXTURE_NAME;
use candle_util::decoder::{MultinomTopicDecoder, NbMixtureTopicDecoder, NbTopicDecoder};
use candle_util::encoder::{GaussianEncoder, GaussianEncoderArgs};
use candle_util::encoder::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};
use candle_util::encoder::{LogSoftmaxEncoder, LogSoftmaxEncoderArgs};
use candle_util::topic_refinement::{refine_topic_proportions, TopicRefinementConfig};
use candle_util::traits::{DecoderModuleT, EncoderModuleT, NewDecoder};
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use indicatif::ParallelProgressIterator;
use log::info;
use rayon::prelude::*;

#[derive(Args, Debug)]
pub struct PredictArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Held-out data files (.zarr or .h5)",
        long_help = "Sparse backends to score with the pre-trained model.\n\
                     Gene sets may differ from training; missing genes are padded\n\
                     and per-batch delta is re-estimated from the frozen dictionary."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        help = "Trained model prefix (output of `senna topic` / `masked-topic` -o)",
        long_help = "Loads:\n  \
                     {model}.dictionary.parquet      gene × topic dictionary\n  \
                     {model}.model.json              model architecture metadata\n  \
                     {model}.safetensors             encoder + decoder weights\n  \
                     {model}.coarsening.json         (dense only) feature coarsening\n  \
                     {model}.shortlist_weights.parquet (indexed) NB-Fisher weights"
    )]
    pub(crate) model: Box<str>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file prefix",
        long_help = "Writes:\n  \
                     {out}.latent.parquet      [N × K] log θ\n  \
                     {out}.predictive.parquet  per-cell [llik, total, llik_per_count]"
    )]
    pub(crate) out: Box<str>,

    #[arg(
        short,
        long,
        value_delimiter = ',',
        help = "Batch membership files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = 500, help = "Evaluation minibatch size")]
    pub(crate) minibatch_size: usize,

    #[arg(long, help = "Cells per delta-estimation block (auto by default)")]
    pub(crate) block_size: Option<usize>,

    #[arg(long, help = "Load all columns into memory before evaluation")]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Decoder-side gradient steps on θ at inference (0 = encoder forward only)",
        long_help = "If --decoder-only is set, this controls iterations of\n\
                     uniform-init optimization. Otherwise, controls per-cell\n\
                     refinement steps anchored to the encoder output."
    )]
    pub(crate) refine_steps: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Learning rate for refinement / decoder-only"
    )]
    pub(crate) refine_lr: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 anchor strength for refinement"
    )]
    pub(crate) refine_reg: f64,

    #[arg(
        long,
        help = "Skip the encoder; init θ uniform and optimize purely against the frozen decoder",
        long_help = "Useful when the held-out feature set is too divergent for the\n\
                     trained encoder. Uses --refine-steps and --refine-lr (defaults\n\
                     bumped to 100 / 0.05 if --refine-steps was left at 0)."
    )]
    pub(crate) decoder_only: bool,

    #[arg(
        long,
        default_value_t = 3,
        help = "Iterative TMLE rounds for held-out batch δ (0 = legacy single-pass plug-in)",
        long_help = "Per iteration: encode all cells with current δ → θ̂; refit\n\
                     δ as Σ_obs / Σ_pred per batch (NB-Fisher-weighted for nb /\n\
                     nbmixture decoders, using the saved {model}.dispersion.parquet\n\
                     when present). Default 3 typically converges; 0 reverts to the\n\
                     legacy 1/K-marginal plug-in."
    )]
    pub(crate) delta_iters: usize,

    #[arg(short, long, help = "Verbose logging")]
    pub(crate) verbose: bool,

    #[arg(
        long,
        help = "Also write residual expression to a sparse backend ({out}.residual.zarr / .h5)",
        long_help = "Regress the reference reconstruction μ ∝ δ?·Σ_k θ_k·exp(β_dk) out of\n\
                     the held-out counts by DIVISION and write the leftover as a NEW\n\
                     sparse backend (gene × cell). Reuses matrix-util's\n\
                     `adjust_by_division_inplace`: per cell, x_d /= μ_d·λ with the\n\
                     self-normalizing column scale λ = Σ_d x / Σ_d μ — so the residual\n\
                     is a per-cell relative fold-change against the reference, the same\n\
                     division semantics `senna svd` uses for batch adjustment. Only\n\
                     entries above --residual-threshold are kept (all are ≥ 0), so the\n\
                     file stays sparse. Backend chosen by extension: .zarr or .h5\n\
                     (needs the `hdf5` feature)."
    )]
    pub(crate) residual_out: Option<Box<str>>,

    #[arg(
        long,
        help = "Fold per-batch δ into μ (removes topics AND batch effect)",
        long_help = "When set, the per-gene denominator is δ_{d,b}·Σ_k θ_k·exp(β_dk) — the\n\
                     residual is harmonized (batch effect divided out too). When unset,\n\
                     μ comes from topics only and the residual still carries batch effects."
    )]
    pub(crate) residual_include_delta: bool,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Drop residual entries ≤ this value (default 0 = keep all nonzeros)"
    )]
    pub(crate) residual_threshold: f32,

    #[arg(
        long,
        value_enum,
        default_value_t = FeatureNameKindArg::Exact,
        help = "Canonicalize query row names: auto|exact|gene|locus|locus-overlap|mixed",
        long_help = "Mirrors the training-side flag. `exact` (default) preserves legacy\n\
                     exact-then-flexible matching. `gene` resolves `ENSG..._TSPAN6` →\n\
                     `TSPAN6` (rsplit on '_') so a symbol-keyed dictionary matches a\n\
                     query keyed by `<ensembl>_<symbol>`. Applied AFTER the suffix trim\n\
                     (see --feature-name-suffix-delim)."
    )]
    pub(crate) feature_name_kind: FeatureNameKindArg,

    #[arg(
        long,
        help = "Split query row names on this char; keep prefix as base key",
        long_help = "e.g. '/' turns `ENSG00000000003_TSPAN6/count/spliced` into base\n\
                     `ENSG00000000003_TSPAN6` (+ suffix `count/spliced`). The suffix is\n\
                     then available to --keep-feature-suffix for filtering, and the base\n\
                     is handed to --feature-name-kind for canonicalization."
    )]
    pub(crate) feature_name_suffix_delim: Option<char>,

    #[arg(
        long,
        help = "Keep only rows whose suffix equals this value",
        long_help = "e.g. `count/spliced` drops the `count/unspliced` rows of a faba\n\
                     genes backend, collapsing the {spliced,unspliced} doubling to one\n\
                     row per gene. Requires --feature-name-suffix-delim. Rows lacking\n\
                     the delimiter are dropped when this is set."
    )]
    pub(crate) keep_feature_suffix: Option<Box<str>>,
}

impl PredictArgs {
    /// Assemble the query-row-name transforms from the CLI flags.
    fn query_name_opts(&self) -> QueryNameOpts {
        QueryNameOpts {
            kind: self.feature_name_kind.clone().resolve_or_gene(),
            suffix_delim: self.feature_name_suffix_delim,
            keep_suffix: self.keep_feature_suffix.clone(),
        }
    }
}

pub fn predict_model(args: &PredictArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let metadata = TopicModelMetadata::load(&args.model)?;
    info!(
        "Loaded model metadata: type={}, K={}, D_full={}, D_enc={}",
        metadata.model_type,
        metadata.n_topics,
        metadata.n_features_full,
        metadata.n_features_encoder,
    );

    // Residual-mode x0 needs per-pseudobulk-group mu_residual, which only
    // exists after a multilevel collapse pass over the held-out data — and
    // the predict path skips that pass. We feed Batch-style x0 instead, so
    // the encoder sees a different-distribution null than it saw at training.
    // Warn loudly: θ̂ may be biased on held-out for residual-trained models.
    if metadata.adj_method.as_ref() == "residual" {
        log::warn!(
            "model was trained with --adj-method residual; predict only supports \
             batch-style x0. θ̂ may be biased — retrain with --adj-method batch \
             for clean held-out semantics."
        );
    }

    use crate::topic::model_metadata::{
        masked_head_from_model_type, MODEL_TYPE_INDEXED_MASKED, MODEL_TYPE_MASKED_SBP,
        MODEL_TYPE_MASKED_VAE, MODEL_TYPE_TOPIC, MODEL_TYPE_VAE,
    };
    // All masked heads (softmax / stick-breaking / Gaussian) share the
    // encoder-only path; `masked_head_from_model_type` recovers which one.
    if let Some(head) = masked_head_from_model_type(&metadata.model_type) {
        return predict_masked(args, &metadata, head);
    }
    match metadata.model_type.as_ref() {
        MODEL_TYPE_TOPIC => predict_dense(args, &metadata),
        MODEL_TYPE_VAE => predict_vae(args, &metadata),
        other => anyhow::bail!(
            "predict: unsupported model_type '{other}' (expected '{MODEL_TYPE_TOPIC}', \
             '{MODEL_TYPE_INDEXED_MASKED}', '{MODEL_TYPE_MASKED_SBP}', \
             '{MODEL_TYPE_MASKED_VAE}', or '{MODEL_TYPE_VAE}')",
        ),
    }
}

fn resolve_mode(args: &PredictArgs) -> LatentMode {
    if args.decoder_only {
        LatentMode::DecoderOnly
    } else if args.refine_steps > 0 {
        LatentMode::EncoderRefine
    } else {
        LatentMode::Encoder
    }
}

fn build_remap(
    training_genes: &[Box<str>],
    new_genes: &[Box<str>],
    opts: &QueryNameOpts,
) -> anyhow::Result<Option<GeneRemap>> {
    let gene_remap = build_gene_remap_with(training_genes, new_genes, opts);
    let min_overlap = (training_genes.len() as f32 * 0.1) as usize;
    anyhow::ensure!(
        gene_remap.n_mapped >= min_overlap,
        "Too few genes overlap: {}/{} mapped (need at least {})",
        gene_remap.n_mapped,
        training_genes.len(),
        min_overlap,
    );

    let needs_remap = gene_remap
        .new_to_train
        .iter()
        .enumerate()
        .any(|(i, opt)| *opt != Some(i))
        || new_genes.len() != training_genes.len();

    if needs_remap {
        info!(
            "Gene remapping enabled ({} → {} features)",
            new_genes.len(),
            training_genes.len()
        );
        Ok(Some(gene_remap))
    } else {
        info!("Genes match training — no remapping needed");
        Ok(None)
    }
}

/// Aggregate a `[D_full]` per-gene mean to `[D_coarse]` so the divisor
/// is on the same per-fine-gene rate scale as the encoder expects.
///
/// The encoder's input `y_coarse` and batch null `x0_coarse` are both
/// **sum**-coarsened (`aggregate_columns_nd`), giving `y ≈ G·batch·<μ>·bio`
/// and `x0 ≈ G·batch`. For `clean = y / (x0·μ_coarse)` to recover `bio`,
/// `μ_coarse` must be at the per-fine-gene rate `<μ>`, not the summed
/// `Σμ`. So we sum-coarsen first (matching the call sites used for
/// data and batch null) then divide each coarse cell by its group size.
pub(crate) fn aggregate_feature_mean_to_coarse(
    full: &[f32],
    coarsening: Option<&FeatureCoarsening>,
) -> Vec<f32> {
    let n_full = full.len();
    let mu_1d = nalgebra::DMatrix::<f32>::from_row_slice(1, n_full, full);
    match coarsening {
        Some(fc) => {
            let mu_summed = fc.aggregate_columns_nd(&mu_1d);
            mu_summed
                .row(0)
                .iter()
                .zip(fc.coarse_to_fine.iter())
                .map(|(&s, fines)| s / fines.len().max(1) as f32)
                .collect()
        }
        None => mu_1d.row(0).iter().copied().collect(),
    }
}

//////////////////////
// Dense prediction //
//////////////////////

fn predict_dense(args: &PredictArgs, metadata: &TopicModelMetadata) -> anyhow::Result<()> {
    let (training_genes, beta_dk) = load_dictionary(&args.model)?;
    let coarsening = if metadata.has_coarsening {
        load_coarsening(&args.model)?
    } else {
        None
    };

    // Reload `μ_d` at D_full and aggregate via the (optional) coarsening
    // matrix to the encoder's D_coarse. Saved by `senna topic` at
    // training time; absent for older models, where the encoder falls
    // back to live per-feature batch centering inside `anscombe_residual`.
    let feature_mean_enc: Option<Vec<f32>> =
        match crate::topic::model_metadata::load_feature_mean(&args.model) {
            Ok((_, full)) => Some(aggregate_feature_mean_to_coarse(&full, coarsening.as_ref())),
            Err(_) => None,
        };

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    let mut data_vec = loaded.data;
    data_vec.register_batch_membership(&loaded.batch);
    info!(
        "Held-out data: {} features × {} cells",
        data_vec.num_rows(),
        data_vec.num_columns()
    );

    let new_genes = data_vec.row_names()?;
    let gene_remap = build_remap(&training_genes, &new_genes, &args.query_name_opts())?;

    let delta_db = estimate_delta(
        &data_vec,
        &beta_dk,
        metadata.theta_mean.as_deref(),
        gene_remap.as_ref(),
        args.block_size,
    )?;

    let cpu_dev = Device::Cpu;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);

    let encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: metadata.n_features_encoder,
            n_topics: metadata.n_topics,
            layers: &metadata.encoder_hidden,
            feature_mean: feature_mean_enc.as_deref(),
        },
        &parameters,
        vb.clone(),
    )?;

    let mode = resolve_mode(args);
    let refine_config = TopicRefinementConfig {
        num_steps: if args.decoder_only && args.refine_steps == 0 {
            100
        } else {
            args.refine_steps
        },
        learning_rate: if args.decoder_only && args.refine_lr <= 0.01 {
            0.05
        } else {
            args.refine_lr
        },
        regularization: args.refine_reg,
    };
    info!("Latent inference mode: {mode:?}");

    let decoder_name = metadata
        .decoder_types
        .first()
        .map_or("multinom", std::convert::AsRef::as_ref);
    info!("Predicting with decoder: {decoder_name}");

    let inputs = DensePredictInputs {
        metadata,
        parameters: &mut parameters,
        vb: &vb,
        model_prefix: &args.model,
        encoder: &encoder,
        data_vec: &data_vec,
        delta_db,
        gene_remap: gene_remap.as_ref(),
        coarsening: coarsening.as_ref(),
        beta_dk: &beta_dk,
        delta_iters: args.delta_iters,
        cpu_dev: &cpu_dev,
        adj_method: AdjMethod::Batch,
        minibatch_size: args.minibatch_size,
        mode,
        refine_config: &refine_config,
    };

    let (z_nk, llik, total, delta_final) = match decoder_name {
        "multinom" => predict_dense_with_decoder::<MultinomTopicDecoder>(inputs)?,
        "nb" => predict_dense_with_decoder::<NbTopicDecoder>(inputs)?,
        name if name == NBMIXTURE_NAME => {
            predict_dense_with_decoder::<NbMixtureTopicDecoder>(inputs)?
        }
        other => anyhow::bail!("unsupported decoder type in metadata: {other}"),
    };

    finalize_predict(FinalizePredict {
        args,
        data_vec: &data_vec,
        z_nk: &z_nk,
        llik: &llik,
        total: &total,
        beta_dk: &beta_dk,
        delta_db: delta_final.as_ref(),
        gene_remap: gene_remap.as_ref(),
    })
}

struct DensePredictInputs<'a> {
    metadata: &'a TopicModelMetadata,
    parameters: &'a mut candle_nn::VarMap,
    vb: &'a candle_nn::VarBuilder<'a>,
    model_prefix: &'a str,
    encoder: &'a LogSoftmaxEncoder,
    data_vec: &'a SparseIoVec,
    delta_db: Option<Mat>,
    gene_remap: Option<&'a GeneRemap>,
    coarsening: Option<&'a FeatureCoarsening>,
    beta_dk: &'a Mat,
    delta_iters: usize,
    cpu_dev: &'a Device,
    adj_method: AdjMethod,
    minibatch_size: usize,
    mode: LatentMode,
    refine_config: &'a TopicRefinementConfig,
}

/// `(log θ [N×K], per-cell llik, per-cell total, finalized per-batch δ)`.
type DensePredictOut = (Mat, Vec<f32>, Vec<f32>, Option<Mat>);

fn predict_dense_with_decoder<Dec>(
    inputs: DensePredictInputs<'_>,
) -> anyhow::Result<DensePredictOut>
where
    Dec: DecoderModuleT + NewDecoder + Send + Sync,
{
    let DensePredictInputs {
        metadata,
        parameters,
        vb,
        model_prefix,
        encoder,
        data_vec,
        delta_db,
        gene_remap,
        coarsening,
        beta_dk,
        delta_iters,
        cpu_dev,
        adj_method,
        minibatch_size,
        mode,
        refine_config,
    } = inputs;

    // Register decoders at every level so safetensors keys match training.
    // Predict only uses the finest-level decoder.
    let mut decoders: Vec<Dec> = Vec::with_capacity(metadata.level_decoder_dims.len());
    for (i, &d_l) in metadata.level_decoder_dims.iter().enumerate() {
        decoders.push(Dec::new(d_l, metadata.n_topics, vb.pp(format!("dec_{i}")))?);
    }

    let safetensors_path = format!("{model_prefix}.safetensors");
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;

    // Attach finest-level NB-Fisher weights to the finest decoder so
    // predictive llik uses the same loss as training. Older models
    // without saved coarse weights fall back to the unweighted form.
    if let Some((_, coarse_w)) =
        data_beans_alg::gene_weighting::load_fisher_weights_coarse(model_prefix)?
    {
        if let Some(finest) = decoders.last_mut() {
            finest.attach_feature_weights(&coarse_w, cpu_dev)?;
        }
    }
    let decoder = decoders.last().expect("at least one decoder level");

    // Iterative TMLE δ refinement: replaces the single-pass plug-in with a
    // per-cell θ̂-aware obs/pred update (NB-Fisher-weighted when φ is saved).
    // `delta_iters == 0` falls through with the plug-in δ unchanged.
    let delta_db = if delta_iters > 0 {
        if let Some(initial) = delta_db {
            let phi_opt = crate::topic::model_metadata::load_dispersion(model_prefix)?;
            let phi_for_iter: Option<&[f32]> = match metadata
                .decoder_types
                .first()
                .map(std::convert::AsRef::as_ref)
            {
                Some("nb" | "nbmixture") => phi_opt.as_deref(),
                _ => None,
            };
            let refined = crate::predict_tmle::iterate_delta_dense(
                delta_iters,
                initial,
                data_vec,
                encoder,
                gene_remap,
                coarsening,
                beta_dk,
                phi_for_iter,
                minibatch_size,
                cpu_dev,
                &adj_method,
            )?;
            Some(refined)
        } else {
            None
        }
    } else {
        delta_db
    };

    // Delta tensor at encoder D (coarsened if applicable). Note: dense
    // refinement uses the encoder's input dim, which equals D_finest here.
    let delta_tensor = delta_db
        .as_ref()
        .map(|db| -> anyhow::Result<Tensor> {
            let mut db = db.clone();
            if let Some(fc) = coarsening {
                db = fc.aggregate_rows_ds(&db);
            }
            let t = db.to_tensor(cpu_dev)?.transpose(0, 1)?.contiguous()?;
            Ok(t)
        })
        .transpose()?;

    let ntot = data_vec.num_columns();
    let kk = metadata.n_topics;

    let (z_nk, llik, total) = run_predict_blocks(ntot, kk, minibatch_size, |(lb, ub)| {
        predict_block_dense::<Dec>(PredictBlockDenseArgs {
            lb,
            ub,
            data_vec,
            encoder,
            decoder,
            delta_tensor: delta_tensor.as_ref(),
            gene_remap,
            coarsening,
            dev: cpu_dev,
            adj_method: &adj_method,
            mode,
            refine_config,
        })
    })?;
    // Return the finalized (TMLE-refined) δ so the caller can regress it
    // out when writing the residual backend.
    Ok((z_nk, llik, total, delta_db))
}

struct PredictBlockDenseArgs<'a, Dec> {
    lb: usize,
    ub: usize,
    data_vec: &'a SparseIoVec,
    encoder: &'a LogSoftmaxEncoder,
    decoder: &'a Dec,
    delta_tensor: Option<&'a Tensor>,
    gene_remap: Option<&'a GeneRemap>,
    coarsening: Option<&'a FeatureCoarsening>,
    dev: &'a Device,
    adj_method: &'a AdjMethod,
    mode: LatentMode,
    refine_config: &'a TopicRefinementConfig,
}

fn predict_block_dense<Dec>(
    a: PredictBlockDenseArgs<'_, Dec>,
) -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)>
where
    Dec: DecoderModuleT,
{
    use crate::topic::common::expand_delta_for_block;

    let PredictBlockDenseArgs {
        lb,
        ub,
        data_vec,
        encoder,
        decoder,
        delta_tensor,
        gene_remap,
        coarsening,
        dev,
        adj_method,
        mode,
        refine_config,
    } = a;

    let x0_nd = delta_tensor
        .map(|delta_bm| expand_delta_for_block(data_vec, delta_bm, adj_method, lb, ub, dev))
        .transpose()?;

    let csc = data_vec.read_columns_csc(lb..ub)?;
    let x_at_dec = remap_and_coarsen_dense(&csc, gene_remap, coarsening, dev)?;

    let log_z_nk = match mode {
        LatentMode::Encoder => {
            let (log_z, _) = encoder.forward_t(&x_at_dec, x0_nd.as_ref(), false)?;
            log_z
        }
        LatentMode::EncoderRefine => {
            let (log_z, _) = encoder.forward_t(&x_at_dec, x0_nd.as_ref(), false)?;
            refine_topic_proportions(&log_z, &x_at_dec, decoder, refine_config)?
        }
        LatentMode::DecoderOnly => decoder_only_inference_dense(
            decoder,
            &x_at_dec,
            decoder.dim_latent(),
            refine_config.learning_rate,
            refine_config.num_steps,
            dev,
        )?,
    };

    let llik_t = predictive_llik_dense(decoder, &log_z_nk, &x_at_dec)?;
    let llik: Vec<f32> = llik_t.to_device(&Device::Cpu)?.to_vec1()?;

    let total: Vec<f32> = {
        let summed = x_at_dec.sum(1)?.to_device(&Device::Cpu)?;
        summed.to_vec1()?
    };

    let z_cpu = log_z_nk.to_device(&Device::Cpu)?;
    let z_mat = Mat::from_tensor(&z_cpu)?;
    Ok((lb, z_mat, llik, total))
}

/// Scatter CSC rows from new-data order to training gene order, then optionally
/// coarsen, returning a `[N, D_dec]` tensor on the requested device.
fn remap_and_coarsen_dense(
    csc: &nalgebra_sparse::CscMatrix<f32>,
    gene_remap: Option<&GeneRemap>,
    coarsening: Option<&FeatureCoarsening>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let nd = if let Some(remap) = gene_remap {
        let ncols = csc.ncols();
        let mut out = Mat::zeros(remap.d_train, ncols);
        for j in 0..ncols {
            let col = csc.col(j);
            for (&row_new, &val) in col.row_indices().iter().zip(col.values().iter()) {
                if let Some(row_train) = remap.new_to_train[row_new] {
                    out[(row_train, j)] += val;
                }
            }
        }
        if let Some(fc) = coarsening {
            fc.aggregate_rows_ds(&out)
                .to_tensor(dev)?
                .transpose(0, 1)?
                .contiguous()?
        } else {
            out.to_tensor(dev)?.transpose(0, 1)?.contiguous()?
        }
    } else if let Some(fc) = coarsening {
        fc.aggregate_sparse_csc(csc)
            .to_tensor(dev)?
            .transpose(0, 1)?
            .contiguous()?
    } else {
        csc.to_tensor(dev)?.transpose(0, 1)?.contiguous()?
    };
    Ok(nd)
}

////////////////////////
// Indexed prediction //
////////////////////////

/// Encoder-only prediction for the masked-topic
/// ([`MODEL_TYPE_INDEXED_MASKED`]). Rebuilds the indexed symbol-embedding
/// encoder, runs the deterministic masked-encoder forward (all genes visible)
/// on the held-out cells, and writes the latent. No decoder/refinement (v1);
/// batch correction at predict is gene-mean only (per-cell residual null is a
/// future refinement).
fn predict_masked(
    args: &PredictArgs,
    metadata: &TopicModelMetadata,
    head: candle_util::vae::masked_topic::LatentHead,
) -> anyhow::Result<()> {
    use crate::topic::eval_indexed::{evaluate_latent_masked, EvaluateLatentMaskedConfig};
    use crate::topic::model_metadata::load_feature_mean;
    use crate::topic::model_metadata::masked_head_label;

    let embedding_dim = metadata
        .embedding_dim
        .ok_or_else(|| anyhow::anyhow!("masked-topic metadata missing embedding_dim"))?;
    let enc_context_size = metadata
        .enc_context_size
        .ok_or_else(|| anyhow::anyhow!("masked-topic metadata missing enc_context_size"))?;

    let (training_genes, _beta_dk) = load_dictionary(&args.model)?;
    let (_sw_genes, shortlist_weights) = load_shortlist_weights(&args.model)?;
    let (_fm_genes, feature_mean) = load_feature_mean(&args.model)?;
    anyhow::ensure!(
        shortlist_weights.len() == training_genes.len(),
        "shortlist_weights gene count ({}) != dictionary gene count ({})",
        shortlist_weights.len(),
        training_genes.len()
    );
    anyhow::ensure!(
        feature_mean.len() == training_genes.len(),
        "feature_mean gene count ({}) != dictionary gene count ({})",
        feature_mean.len(),
        training_genes.len()
    );

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    let mut data_vec = loaded.data;
    data_vec.register_batch_membership(&loaded.batch);
    info!(
        "Held-out data: {} features × {} cells",
        data_vec.num_rows(),
        data_vec.num_columns()
    );

    let new_genes = data_vec.row_names()?;
    let gene_remap = build_remap(&training_genes, &new_genes, &args.query_name_opts())?;

    let cpu_dev = Device::Cpu;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);
    let encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: metadata.n_features_full,
            n_topics: metadata.n_topics,
            embedding_dim,
            layers: &metadata.encoder_hidden,
            use_gcn: false,
            attn_pool: true,
        },
        &parameters,
        vb.pp("enc"),
    )?;
    let safetensors_path = format!("{}.safetensors", args.model);
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;

    let adj_method = AdjMethod::Batch;
    let eval_config = EvaluateLatentMaskedConfig {
        dev: &cpu_dev,
        adj_method: &adj_method,
        minibatch_size: args.minibatch_size,
        enc_context_size,
        shortlist_weights: &shortlist_weights,
        feature_mean: &feature_mean,
        head,
    };
    let z_nk = evaluate_latent_masked(
        &data_vec,
        &encoder,
        &eval_config,
        None,
        gene_remap.as_ref().map(|r| r.new_to_train.as_slice()),
    )?;

    let cell_names = data_vec.column_names()?;
    // Inference: emit a row per query cell (no QC dropping).
    crate::output_helpers::save_latent(&args.out, &z_nk, &cell_names, None)?;
    let model_label = masked_head_label(head);
    info!(
        "Wrote {}.latent.parquet ({model_label}, encoder-only)",
        args.out
    );
    Ok(())
}

/// Held-out latent inference for the Gaussian VAE ([`MODEL_TYPE_VAE`]).
/// Rebuilds the [`GaussianEncoder`] and runs it (encoder-only, eval mode →
/// posterior mean `z`) over the held-out cells. Like `predict_masked`, batch
/// correction is gene-mean only (the per-gene `μ_d` divisor inside
/// `anscombe_residual`); a per-cell residual null is a future refinement. The
/// latent is continuous factors, so there is no decoder refinement to do.
fn predict_vae(args: &PredictArgs, metadata: &TopicModelMetadata) -> anyhow::Result<()> {
    use crate::topic::model_metadata::load_feature_mean;

    let (training_genes, _loadings) = load_dictionary(&args.model)?;
    let (_fm_genes, feature_mean) = load_feature_mean(&args.model)?;
    anyhow::ensure!(
        feature_mean.len() == training_genes.len(),
        "feature_mean gene count ({}) != dictionary gene count ({})",
        feature_mean.len(),
        training_genes.len()
    );

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    let mut data_vec = loaded.data;
    data_vec.register_batch_membership(&loaded.batch);
    info!(
        "Held-out data: {} features × {} cells",
        data_vec.num_rows(),
        data_vec.num_columns()
    );

    let new_genes = data_vec.row_names()?;
    let gene_remap = build_remap(&training_genes, &new_genes, &args.query_name_opts())?;

    let cpu_dev = Device::Cpu;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);
    let encoder = GaussianEncoder::new(
        GaussianEncoderArgs {
            n_features: metadata.n_features_encoder,
            n_latent: metadata.n_topics,
            layers: &metadata.encoder_hidden,
            feature_mean: Some(&feature_mean),
        },
        &parameters,
        vb.clone(),
    )?;
    let safetensors_path = format!("{}.safetensors", args.model);
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;

    let ntot = data_vec.num_columns();
    let (z_nk, _llik, _total) =
        run_predict_blocks(ntot, metadata.n_topics, args.minibatch_size, |(lb, ub)| {
            // Gene-mean null only (x0 = None): the divisive μ_d correction is
            // baked into the encoder via `feature_mean`.
            let csc = data_vec.read_columns_csc(lb..ub)?;
            let x_nd = remap_and_coarsen_dense(&csc, gene_remap.as_ref(), None, &cpu_dev)?;
            let (z, _) = encoder.forward_t(&x_nd, None, false)?;
            let z_mat = Mat::from_tensor(&z.to_device(&Device::Cpu)?)?;
            Ok((lb, z_mat, Vec::new(), Vec::new()))
        })?;

    let cell_names = data_vec.column_names()?;
    // Inference: emit a row per query cell (no QC dropping).
    crate::output_helpers::save_latent(&args.out, &z_nk, &cell_names, None)?;
    info!("Wrote {}.latent.parquet (vae, encoder-only)", args.out);
    Ok(())
}

/// Run `block_fn` over `[0, ntot)` in `minibatch_size` blocks, concatenating
/// results into `(z_nk [ntot, kk], llik [ntot], total [ntot])`. Shared by the
/// dense predict drivers.
fn run_predict_blocks<F>(
    ntot: usize,
    kk: usize,
    minibatch_size: usize,
    block_fn: F,
) -> anyhow::Result<(Mat, Vec<f32>, Vec<f32>)>
where
    F: Fn((usize, usize)) -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)> + Sync,
{
    let jobs = create_jobs(ntot, 0, Some(minibatch_size));
    let njobs = jobs.len() as u64;
    let mut chunks: Vec<(usize, Mat, Vec<f32>, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(new_progress_bar(njobs))
        .map(|&block| block_fn(block))
        .collect::<anyhow::Result<Vec<_>>>()?;
    chunks.sort_by_key(|c| c.0);

    let mut z_nk = Mat::zeros(ntot, kk);
    let mut llik = Vec::with_capacity(ntot);
    let mut total = Vec::with_capacity(ntot);
    let mut row = 0;
    for (_, z_block, lk, tot) in chunks {
        let n = z_block.nrows();
        z_nk.rows_range_mut(row..row + n).copy_from(&z_block);
        llik.extend(lk);
        total.extend(tot);
        row += n;
    }
    Ok((z_nk, llik, total))
}

////////////////////
// Output writers //
////////////////////

/// Shared tail of both predict paths: per-cell latent + predictive scores,
/// then (optionally) the residual-expression backend. Both the dense and
/// indexed drivers have the same artifacts available, so they funnel through
/// here rather than duplicating the two write calls.
struct FinalizePredict<'a> {
    args: &'a PredictArgs,
    data_vec: &'a SparseIoVec,
    z_nk: &'a Mat,
    llik: &'a [f32],
    total: &'a [f32],
    beta_dk: &'a Mat,
    delta_db: Option<&'a Mat>,
    gene_remap: Option<&'a GeneRemap>,
}

fn finalize_predict(f: FinalizePredict<'_>) -> anyhow::Result<()> {
    write_outputs(f.args, f.data_vec, f.z_nk, f.llik, f.total)?;
    write_residual_backend(
        f.args,
        f.data_vec,
        f.z_nk,
        f.beta_dk,
        f.delta_db,
        f.gene_remap,
    )
}

fn write_outputs(
    args: &PredictArgs,
    data_vec: &SparseIoVec,
    z_nk: &Mat,
    llik: &[f32],
    total: &[f32],
) -> anyhow::Result<()> {
    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&axis_id_names("T", z_nk.ncols())),
    )?;
    info!("Wrote {}.latent.parquet", args.out);

    // Per-cell predictive scores: [llik, total, llik_per_count]
    let n = llik.len();
    let mut pred = Mat::zeros(n, 3);
    for i in 0..n {
        pred[(i, 0)] = llik[i];
        pred[(i, 1)] = total[i];
        pred[(i, 2)] = if total[i] > 0.0 {
            llik[i] / total[i]
        } else {
            0.0
        };
    }
    let pred_cols: Vec<Box<str>> = vec!["llik".into(), "total".into(), "llik_per_count".into()];
    pred.to_parquet_with_names(
        &(args.out.to_string() + ".predictive.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&pred_cols),
    )?;
    info!("Wrote {}.predictive.parquet", args.out);
    Ok(())
}

/// Regress the reference reconstruction `μ` out of the held-out counts **by
/// division**, reusing matrix-util's `adjust_by_division_inplace`, and write
/// the leftover ("residual expression") to a NEW sparse backend.
///
/// Blocks of cells run in parallel (rayon, like [`run_predict_blocks`]). Per
/// block we form the expected per-gene rate as one `nalgebra` matmul
/// `pred = exp(β) · θᵀ` (`[D_train, n_block]`) — never an `N × D` dense
/// matrix, peak intermediate is `D × minibatch`. `pred` is scattered onto the
/// held-out gene axis (via `gene_remap`) as the per-cell denominator `μ_dn`
/// (`[D_test, n_block]`), optionally weighted by the per-batch δ when
/// `--residual-include-delta` is set. Then
/// `csc.adjust_by_division_inplace(&μ_dn)` performs, per cell `j`,
///   `x_dj ← x_dj / (μ_dj · λ_j)`,  `λ_j = Σ_d x_dj / Σ_d μ_dj`
/// — the same self-normalizing division `senna svd` uses for batch
/// adjustment (`svd/fit.rs`). Absolute scale of `μ` cancels in `λ`, so `pred`
/// is used directly (no library rescale). Genes absent from the reference
/// model have `μ = 0` and are passed through unchanged. Surviving entries
/// above `--residual-threshold` (all ≥ 0) are written as triplets, mirroring
/// the `svd` backend-write idiom.
fn write_residual_backend(
    args: &PredictArgs,
    data_vec: &SparseIoVec,
    z_nk: &Mat,
    beta_dk: &Mat,
    delta_db: Option<&Mat>,
    gene_remap: Option<&GeneRemap>,
) -> anyhow::Result<()> {
    let Some(path) = args.residual_out.as_deref() else {
        return Ok(());
    };

    let threshold = args.residual_threshold;
    // δ to fold into μ (None ⇒ topics-only denominator).
    let delta = args.residual_include_delta.then_some(delta_db).flatten();

    let kk = beta_dk.ncols();
    anyhow::ensure!(
        z_nk.ncols() == kk,
        "residual: latent topics ({}) != dictionary topics ({kk})",
        z_nk.ncols(),
    );

    // exp(β) once: [D_train, K], shared read-only across blocks.
    let exp_beta_dk = beta_dk.map(f32::exp);

    let ntot = data_vec.num_columns();
    let d_test = data_vec.num_rows();

    info!(
        "Computing residual expression by division (include_delta={}, threshold={threshold}) \
         over {ntot} cells",
        delta.is_some(),
    );

    let jobs = create_jobs(ntot, 0, Some(args.minibatch_size));
    let njobs = jobs.len() as u64;
    let triplets: Vec<(u64, u64, f32)> = jobs
        .par_iter()
        .progress_with(new_progress_bar(njobs))
        .map(|&(lb, ub)| -> anyhow::Result<Vec<(u64, u64, f32)>> {
            let mut csc = data_vec.read_columns_csc(lb..ub)?;
            let n_block = csc.ncols();

            // θ for this block: exp of stored log θ → [K, n_block].
            let theta_kn = z_nk.rows(lb, n_block).map(f32::exp).transpose();
            // pred[d, j] = Σ_k exp(β_dk) θ_jk  → [D_train, n_block].
            let pred_dn = &exp_beta_dk * theta_kn;

            // Scatter pred onto the held-out gene axis as the per-cell
            // denominator μ_dn [D_test, n_block]; optionally weight by δ.
            let batch_ids = delta.map(|_| data_vec.get_batch_membership(lb..ub));
            let mut mu_dn = Mat::zeros(d_test, n_block);
            for jloc in 0..n_block {
                for &row_new in csc.col(jloc).row_indices() {
                    let Some(row_train) = (match gene_remap {
                        Some(rm) => rm.new_to_train[row_new],
                        None => Some(row_new),
                    }) else {
                        continue; // gene absent from the reference model → μ = 0
                    };
                    let mut mu = pred_dn[(row_train, jloc)];
                    if let Some(delta) = delta {
                        mu *= delta[(row_train, batch_ids.as_ref().unwrap()[jloc])];
                    }
                    mu_dn[(row_new, jloc)] = mu;
                }
            }

            // Regress out by division (self-normalizing column scale λ = Σx/Σμ).
            csc.adjust_by_division_inplace(&mu_dn);

            Ok(csc
                .triplet_iter()
                .filter(|&(_, _, &val)| val > threshold)
                .map(|(i, j_local, &val)| (i as u64, (lb + j_local) as u64, val))
                .collect())
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    let backend = match file_ext(path)?.as_ref() {
        "zarr" => SparseIoBackend::Zarr,
        "h5" => SparseIoBackend::HDF5,
        other => anyhow::bail!("residual: unknown backend extension '.{other}' (use .zarr or .h5)"),
    };
    let mtx_shape = (d_test, ntot, triplets.len());
    remove_file(path)?;
    let mut residual =
        create_sparse_from_triplets(&triplets, mtx_shape, Some(path), Some(&backend))?;
    residual.register_row_names_vec(&data_vec.row_names()?);
    residual.register_column_names_vec(&data_vec.column_names()?);

    info!(
        "Wrote residual backend: {path} ({d_test} genes × {ntot} cells, {} nonzeros)",
        triplets.len(),
    );
    Ok(())
}
