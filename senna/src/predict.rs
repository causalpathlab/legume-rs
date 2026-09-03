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
use crate::topic::predict_eval::{
    evaluate_predictions, mean_finite, resolve_eval_genes, EvalArgs, EvalOutcome, Reconstruction,
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

#[derive(Args, Clone, Debug)]
#[command(group(
    clap::ArgGroup::new("input")
        .required(true)
        .multiple(false)
        .args(["data_files", "bulk"])
))]
pub struct PredictArgs {
    #[arg(
        value_delimiter = ',',
        help = "Held-out data files (.zarr or .h5); or --bulk for a dense table",
        long_help = "Sparse backends to score with the pre-trained model.\n\
                     Gene sets may differ from training. Missing genes are padded.\n\
                     Per-batch delta is re-estimated from the frozen dictionary.\n\
                     Give these, or --bulk, not both."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        value_name = "FILE",
        num_args = 1..,
        help = "Dense bulk count matrices (parquet, or tab/comma text), genes × samples",
        long_help = "Dense bulk count matrices instead of sparse backends.\n\
                     Parquet, or tab/comma text; column 0 holds the row names\n\
                     and a header line, when present, names the samples.\n\
                     A samples × genes table is turned on read: the axis whose\n\
                     names match the model's genes is the gene axis\n\
                     (see --bulk-orientation).\n\
                     \n\
                     Each table becomes a temp sparse backend and is then scored\n\
                     exactly as a data file would be, by every model family.\n\
                     Several tables behave as several data files: each is its\n\
                     own batch, and sample names get an @<file> suffix.\n\
                     --batch-files is one label per sample.\n\
                     \n\
                     Values must be counts. Negative values are refused;\n\
                     non-integer values are accepted with a warning.\n\
                     --null-from still takes a backend: the null is the\n\
                     composition of the TRAINING half, which the model was fitted on."
    )]
    pub(crate) bulk: Vec<Box<str>>,

    #[command(flatten)]
    pub(crate) bulk_table: BulkTableArgs,

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
                     {out}.latent.parquet      [N × K] per-cell latent — log θ for the\n                               \
                     topic family, raw Gaussian z for vae /\n                               \
                     Gaussian-head masked models\n  \
                     {out}.predictive.parquet  per-cell [llik, total, llik_per_count]\n\
                     \n\
                     vae inference is encoder-only, so --decoder-only and\n\
                     --refine-* do not apply to it. It is still scored:\n\
                     its decoder is rebuilt to grade the same columns\n\
                     every other family is graded on."
    )]
    pub(crate) out: Box<str>,

    #[arg(
        short,
        long,
        value_delimiter = ',',
        help = "Batch membership files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 500,
        help = "Evaluation minibatch size",
        long_help = "Cells per scored block.\n\
                     \n\
                     IGNORED for vae models, which are scored at the block size\n\
                     they were TRAINED at: that encoder carries a batch norm, so\n\
                     its latent depends on how cells are grouped, and scoring at\n\
                     a different size would move it. The substitution is logged."
    )]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        help = "Cells per delta-estimation block (auto by default)",
        hide = true
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(
        long,
        help = "Load all columns into memory before evaluation",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        default_value_t = ComputeDevice::Cpu,
        value_enum,
        help = "Compute device",
        long_help = "Compute device. `cuda` / `metal` require the matching cargo feature.\n\
                     \n\
                     Matters most for a bge model. Every other family infers with one\n\
                     encoder forward pass, but bge has no encoder — prediction re-runs\n\
                     the same per-cell Poisson SGD the training did, so it belongs on\n\
                     the device that trained it."
    )]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub(crate) device_no: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Decoder-side gradient steps on θ at inference (0 = encoder forward only)",
        long_help = "If --decoder-only is set,\n\
                     this controls iterations of uniform-init optimization. Otherwise,\n\
                     controls per-cell refinement steps anchored to the encoder output."
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
        help = "Skip the encoder;\n\
                init θ uniform and optimize purely against the frozen decoder",
        long_help = "Useful when the held-out feature set is too divergent.\n\
                     The trained encoder cannot handle it.\n\
                     Uses --refine-steps and --refine-lr.\n\
                     Those default to 100 / 0.05 if --refine-steps was left at 0."
    )]
    pub(crate) decoder_only: bool,

    #[arg(
        long,
        default_value_t = 3,
        help = "Iterative TMLE rounds for held-out batch δ (0 = legacy single-pass plug-in)",
        long_help = "Per iteration: encode all cells with current δ → θ̂;\n\
                     refit δ as Σ_obs / Σ_pred per batch.\n\
                     Applies to the dense topic family and to every masked head.\n\
                     NB-Fisher-weighted for nb / nbmixture / masked decoders,\n\
                     using the saved {model}.dispersion.parquet when present.\n\
                     Default 3 typically converges;\n\
                     0 reverts to the legacy 1/K-marginal plug-in."
    )]
    pub(crate) delta_iters: usize,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "FRACTION",
        help = "Refuse to score below this share of the model's genes (0 = no gate)",
        long_help = "Gene coverage is always reported. This turns it into a hard floor.\n\
                     \n\
                     Off by default, because low coverage is not by itself wrong:\n\
                     a targeted panel legitimately measures a small slice of a\n\
                     whole-transcriptome model, and a narrow panel that spans the\n\
                     latent directions can still identify it. Zero mapped genes is\n\
                     always refused — that is a naming failure, not thin coverage."
    )]
    pub(crate) min_gene_overlap: f32,

    #[arg(short, long, help = "Verbose logging")]
    pub(crate) verbose: bool,

    #[arg(
        long,
        help = "Also write residual expression to a sparse backend ({out}.residual.zarr / .h5)",
        long_help = "Regress the reference reconstruction out of the held-out counts.\n\
                     The reference is μ ∝ δ?·Σ_k θ_k·exp(β_dk). Regression is by DIVISION.\n\
                     The leftover is written as a NEW sparse backend, gene × cell.\n\
                     \n\
                     It reuses matrix-util's `adjust_by_division_inplace`. Per cell,\n\
                     x_d /= μ_d·λ. λ = Σ_d x / Σ_d μ is the self-normalizing column scale.\n\
                     So the residual is a per-cell relative fold-change.\n\
                     `senna svd` uses the same division semantics for batches.\n\
                     \n\
                     Only entries above --residual-threshold are kept. All are ≥ 0,\n\
                     so the file stays sparse. The backend follows the extension: .zarr,\n\
                     or .h5 with the `hdf5` feature."
    )]
    pub(crate) residual_out: Option<Box<str>>,

    #[arg(
        long,
        help = "Fold per-batch δ into μ (removes topics AND batch effect)",
        long_help = "When set, the per-gene denominator is δ_{d,b}·Σ_k θ_k·exp(β_dk).\n\
                     The residual is then harmonized, with the batch effect divided out.\n\
                     When unset, μ comes from topics only.\n\
                     The residual then still carries batch effects."
    )]
    pub(crate) residual_include_delta: bool,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Drop residual entries ≤ this value (default 0 = keep all nonzeros)"
    )]
    pub(crate) residual_threshold: f32,

    /// Restrict the agreement metrics to these features (one name per line).
    ///
    /// Without it every gene the model knows is scored, which makes two models
    /// trained on different gene axes incomparable — each is graded on its own
    /// curriculum. Pass the same file to every arm of a benchmark and the
    /// correlations become comparable numbers. It is also what makes the
    /// per-gene table affordable: that table has to hold every scored gene's
    /// values across all cells, so it is written only when this is given.
    #[arg(
        long,
        value_name = "FILE",
        help = "Score the agreement metrics on these features only (one name per line)"
    )]
    pub(crate) eval_features: Option<Box<str>>,

    /// Hide these features from the encoder, then score on exactly them.
    ///
    /// Without it the test cell's latent is fitted from that same cell's counts,
    /// so the score measures reconstruction with K free parameters per cell —
    /// and a model with a larger latent wins on capacity rather than on the
    /// quality of its dictionary. Hiding a slice of the features breaks that:
    /// the scored genes never enter the fit, so extra latent dimensions stop
    /// buying accuracy and the number becomes a genuine prediction. Implies
    /// `--eval-features` on the same file unless that is given separately.
    #[arg(
        long,
        value_name = "FILE",
        help = "Hide these features from the encoder and score on exactly them (one name per line)"
    )]
    pub(crate) ablate_features: Option<Box<str>>,

    /// bge only. Genes in the query that the model never saw are placed through the
    /// learned modules (their membership is the similarity-weighted mean of the k
    /// most similar matched genes' memberships, their bias is moment-matched to the
    /// pass-1 latents) instead of being dropped. `--no-init-genes` restores the drop.
    #[arg(
        long,
        help = "bge: drop genes the model never saw instead of initializing them through the modules"
    )]
    pub(crate) no_init_genes: bool,

    #[arg(
        long,
        default_value_t = graph_embedding_util::transfer::DEFAULT_INIT_NEIGHBOURS,
        value_name = "K",
        help = "bge: matched genes whose memberships are averaged to initialize an unseen gene"
    )]
    pub(crate) init_neighbours: usize,

    #[arg(
        long,
        default_value_t = graph_embedding_util::transfer::DEFAULT_SIMILARITY_FLOOR,
        value_name = "S",
        help = "bge: below this best profile similarity an unseen gene takes the diffuse prior"
    )]
    pub(crate) init_similarity_floor: f32,

    #[arg(
        long,
        help = "bge: re-project every cell with the initialized genes as observations (pass 2)",
        long_help = "bge: after initialization, run a second projection in which the initialized\n\
                     genes' counts are observed. Off, they are scored from the pass-1 latent and\n\
                     never move it. The comparable score still normalizes over the model's genes."
    )]
    pub(crate) init_genes_in_fit: bool,

    #[arg(
        long,
        help = "bge: write {out}.gene_rates.parquet, per-cell predicted rates of the missing and initialized genes"
    )]
    pub(crate) emit_gene_rates: bool,

    /// Training data, read once to build the null every arm is scored against.
    ///
    /// The floor is the count-weighted gene composition of this data — what a
    /// predictor with no cell-specific information would say. It has to come from
    /// the TRAINING half: a null built from the test half knows that half's exact
    /// marginal and is an oracle, not a floor. Without this flag the test half is
    /// used and a warning says so, because a silently different floor makes two
    /// runs quietly incomparable.
    #[arg(
        long,
        value_name = "FILE",
        num_args = 1..,
        help = "Training data whose gene composition is the null (pass the train half)"
    )]
    pub(crate) null_from: Option<Vec<Box<str>>>,

    #[arg(
        long,
        value_enum,
        default_value_t = FeatureNameKindArg::Exact,
        help = "Canonicalize query row names: auto|exact|gene|locus|locus-overlap|mixed",
        long_help = "Mirrors the training-side flag.\n\
                     `exact` (default) preserves legacy exact-then-flexible matching.\n\
                     `gene` resolves `ENSG..._TSPAN6` → `TSPAN6`, rsplit on '_',\n\
                     so a symbol-keyed dictionary matches a query keyed by `<ensembl>_<symbol>`.\n\
                     Applied AFTER the suffix trim (see --feature-name-suffix-delim)."
    )]
    pub(crate) feature_name_kind: FeatureNameKindArg,

    #[arg(
        long,
        help = "Split query row names on this char; keep prefix as base key",
        long_help = "Split query row names on this character.\n\
                     With '/', `ENSG00000000003_TSPAN6/count/spliced` splits in two.\n\
                     The base is `ENSG00000000003_TSPAN6`.\n\
                     The suffix is `count/spliced`.\n\
                     The suffix is then available to --keep-feature-suffix.\n\
                     The base is handed to --feature-name-kind."
    )]
    pub(crate) feature_name_suffix_delim: Option<char>,

    #[arg(
        long,
        help = "Keep only rows whose suffix equals this value",
        long_help = "e.g. `count/spliced` drops the `count/unspliced` rows of a faba genes backend,\n\
                     collapsing the {spliced,unspliced} doubling to one row per gene.\n\
                     Requires --feature-name-suffix-delim.\n\
                     Rows lacking the delimiter are dropped when this is set."
    )]
    pub(crate) keep_feature_suffix: Option<Box<str>>,
}

impl PredictArgs {
    /// The compute device this run asks for, resolved once.
    fn resolve_device(&self) -> anyhow::Result<Device> {
        self.device.to_device(self.device_no)
    }

    /// Resolve every query-axis rule, including `--ablate-features`.
    ///
    /// Fallible now because the ablation list is read here rather than at the
    /// bottom of the score path — which is the point: a mistyped path fails
    /// immediately instead of after the whole backend has been loaded.
    fn query_name_opts(&self) -> anyhow::Result<QueryNameOpts> {
        let hide = match self.ablate_features.as_deref() {
            // The SAME reader `--eval-features` uses. These two flags name the
            // same file by contract, so parsing them differently would hide a
            // different gene set than the one being scored.
            Some(path) => Some(std::sync::Arc::new(
                matrix_util::common_io::read_name_list(path)
                    .map_err(|e| anyhow::anyhow!("reading --ablate-features {path}: {e}"))?
                    .into_iter()
                    .collect::<std::collections::HashSet<Box<str>>>(),
            )),
            None => None,
        };
        Ok(QueryNameOpts {
            kind: self.feature_name_kind.clone().resolve_or_gene(),
            suffix_delim: self.feature_name_suffix_delim,
            keep_suffix: self.keep_feature_suffix.clone(),
            min_overlap: self.min_gene_overlap,
            hide,
        })
    }
}

pub fn predict_model(args: &PredictArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    // Resolve the family BEFORE reaching for `{model}.model.json`. A `senna bge` run never
    // writes one (`has_model: false`), so opening it first turned "predict on a bge run"
    // into a bare file-not-found naming a path the user never typed — the same confusion
    // `BgeEmbedding::open` exists to prevent, reintroduced one level up.
    let kind = crate::topic::model_metadata::resolve_run_kind(&args.model)?;

    // `--bulk`: dense tables become temp backends here, and every family path
    // below reads them as it would any data file. The guard removes them when
    // this function returns, so it has to live to the end of it.
    let mut args = args.clone();
    let _bulk_guard = if args.bulk.is_empty() {
        None
    } else {
        let model_genes = bulk::model_gene_axis(kind, &args.model)?;
        let b = bulk::materialize(&args.bulk, &model_genes, &args.bulk_table.opts())?;
        b.warn_if_split(args.minibatch_size);
        args.data_files = b.paths().to_vec();
        Some(b)
    };
    let args = &args;

    match kind {
        crate::run_manifest::RunKind::Bge => return predict_bge(args),
        // svd writes no checkpoint either; its query side is the Nyström
        // projection onto the frozen dictionary.
        crate::run_manifest::RunKind::Svd => return predict_svd(args),
        crate::run_manifest::RunKind::Simba => anyhow::bail!(
            "predict does not support a `simba` run: it writes no encoder and no frozen projector"
        ),
        _ => {}
    }

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

/// Score a held-out backend against a `senna bge` run.
///
/// bge's whole gene-side model is `(ρ, b_feat)` — there is no checkpoint, no encoder and no
/// decoder — so this is the projection path and nothing else: fit each query cell against
/// the frozen ρ by Poisson MAP, then score it with the profile (multinomial) likelihood.
/// `--decoder-only` and `--refine-steps` have nothing to act on here and are rejected
/// rather than silently ignored, so a command line that asks for refinement does not come
/// back looking like it got it. `--delta-iters` and `--batch-files` are inert here too:
/// bge has no per-batch δ, and its projection is per cell.
///
/// ⚠️ Read `.predictive.parquet` from a bge model as a FLOOR on novelty, not a verdict —
/// see [`crate::bge::score::BgeEmbedding::score`] for the measurement. `--eval-mask-fraction`
/// on the training run, or `senna probe`, answer "has the model seen this biology" better
/// than a per-cell fit with H free parameters can.
fn predict_bge(args: &PredictArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        !args.decoder_only && args.refine_steps == 0,
        "--decoder-only / --refine-steps are for the topic families; {} is a `senna bge` run, \
         whose only inference is the Poisson-MAP projection onto the frozen ρ.",
        args.model
    );
    if args.batch_files.is_some() {
        log::warn!(
            "--batch-files has no effect on a bge run: the projection is per cell against a \
             frozen ρ, with no per-batch δ to estimate"
        );
    }
    if args.residual_out.is_some() {
        log::warn!(
            "--residual-out is not available for a bge run (no count decoder to regress out); \
             ignoring it"
        );
    }

    let model = crate::bge::score::BgeEmbedding::open(&args.model)?;
    let qopts = args.query_name_opts()?;
    let fit = model.score_with_init(
        &args.data_files,
        args.preload_data,
        args.minibatch_size,
        &qopts,
        crate::bge::score::InitOpts {
            align: (!args.no_init_genes).then_some(graph_embedding_util::transfer::AlignKnobs {
                k: args.init_neighbours,
                similarity_floor: args.init_similarity_floor,
            }),
            in_fit: args.init_genes_in_fit,
        },
        &args.resolve_device()?,
    )?;

    // ρ is held row-major `[D, H]` for the projection; the eval pass wants it as
    // a matrix, so it is reshaped once here rather than per block.
    let rho_dh = Mat::from_row_slice(model.gene_names.len(), model.h, &model.rho);
    let agreement = evaluate_agreement(AgreementInputs {
        args,
        training_genes: &model.gene_names,
        data_vec: &fit.data_vec,
        recon: Reconstruction::Embedding {
            rho_dh,
            b_feat: &model.b_feat,
            theta_nh: &fit.latent,
        },
    })?;

    write_outputs(
        args,
        &fit.data_vec,
        &fit.latent,
        &fit.llik,
        &fit.total,
        agreement.as_ref(),
    )?;
    crate::bge::transfer::write_init_outputs(&args.out, &model, &fit, args.emit_gene_rates)?;

    Ok(())
}

/// Score a held-out backend against a `senna svd` run.
///
/// svd writes no checkpoint, so there is no encoder to run; the query side is
/// the Nyström projection onto the frozen dictionary
/// ([`crate::svd::project::project_onto_dictionary`]), the same map `impute`
/// uses.
///
/// **What the columns mean here, because svd is not a count model.**
/// `spearman` / `pearson_log1p` come from the same
/// [`matrix_util::agreement::agreement_from_rate`] every other family is
/// graded by, on predicted COUNTS, so they are directly comparable across
/// families — that is the axis to compare svd on. `llik` is a GAUSSIAN
/// log-likelihood in `log1p` space, which is the loss svd actually minimises,
/// at the per-cell ML σ̂; like every other family's `llik` it is the backend's
/// own and must NOT be compared against another family's. It is normalised as
/// `llik_per_gene`, not `llik_per_count`: it sums over the scored gene axis
/// and does not scale with library size, so it does not belong in a column
/// that means nats-per-count everywhere else. The multinomial
/// `eval_llik_*` columns are deliberately absent rather than filled with a
/// differently-scaled number: svd has no count likelihood to put there.
fn predict_svd(args: &PredictArgs) -> anyhow::Result<()> {
    use matrix_util::agreement::agreement_from_rate;

    anyhow::ensure!(
        !args.decoder_only && args.refine_steps == 0,
        "--decoder-only / --refine-steps are for the topic families; {} is a `senna svd` run, \
         whose only inference is the projection onto the frozen dictionary.",
        args.model
    );
    if args.residual_out.is_some() {
        log::warn!("--residual-out is not available for an svd run (no count decoder); ignoring");
    }
    if args.null_from.is_some() {
        log::warn!(
            "--null-from has no effect on an svd run: the null it builds is a count \
             composition for the multinomial columns, which this path does not write \
             (svd is scored on the correlations and a Gaussian log1p llik)"
        );
    }
    crate::svd::project::warn_if_batch_corrected(&args.model, args.batch_files.is_some());

    let (training_genes, u_dk) = load_dictionary(&args.model)?;
    let column_sum_norm = crate::svd::project::column_sum_norm(&args.model);

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.to_vec(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    let data_vec = loaded.data;
    // The projection sees the caller's query-axis rules, INCLUDING
    // `--ablate-features`: without that the latent is fitted on the genes it
    // is about to be scored on, which is a reconstruction, not a prediction.
    // Only the remap is built here — the projection itself happens inside the
    // scoring loop, off the same block read, so the data is streamed once.
    let proj_remap = crate::svd::project::projection_remap(
        &data_vec,
        &training_genes,
        &args.query_name_opts()?,
        "query",
    )?;

    // Restrict to the scored genes exactly as the other families do, so a
    // benchmark passing one --eval-features file grades every arm on the
    // same curriculum.
    let restrict = args
        .eval_features
        .as_deref()
        .or(args.ablate_features.as_deref());
    let eval_genes = resolve_eval_genes(restrict, &training_genes)?;
    anyhow::ensure!(
        !eval_genes.is_empty(),
        "no evaluation features matched the model's genes"
    );
    // Scoring reads the OBSERVED counts, hidden genes included — that is the
    // point of hiding them from the projection above.
    let mut opts = args.query_name_opts()?;
    opts.hide = None;
    let remap = build_remap(&training_genes, &data_vec.row_names()?, &opts)?;

    let ntot = data_vec.num_columns();
    let d_train = training_genes.len();
    // The scored slice of the dictionary, transposed once: `[K, n_eval]`, so
    // a block's reconstruction is one matmul rather than a strided scalar dot
    // per (cell, gene). `Mat` is column-major, so `u_dk.row(g)` is a strided
    // view and the per-cell form was O(N x n_eval x K) of unvectorisable loads.
    let u_eval_t = Mat::from_fn(u_dk.ncols(), eval_genes.len(), |c, i| {
        u_dk[(eval_genes[i], c)]
    });

    let mut z_nk = Mat::zeros(ntot, u_dk.ncols());
    let mut per_cell: Vec<CellScore> = Vec::with_capacity(ntot);
    for (lb, ub) in create_jobs(ntot, 0, Some(args.minibatch_size)) {
        // ONE read per block: the latent and the observed counts both come off
        // it. Projecting up front and re-reading to score doubled the pass
        // over the backend.
        let csc = data_vec.read_columns_csc(lb..ub)?;
        let z_block = crate::svd::project::project_block(
            &csc,
            &proj_remap.new_to_train,
            &u_dk,
            column_sum_norm,
        )
        .transpose();
        z_nk.rows_range_mut(lb..ub).copy_from(&z_block);
        // `[n, n_eval]` for this block only; the block loop is sequential, so
        // one of these is live at a time.
        let recon_std_block = &z_block * &u_eval_t;
        let block: Vec<CellScore> = (0..csc.ncols())
            .into_par_iter()
            .map(|j| {
                // Observed counts on the model's gene axis.
                let mut obs_full = vec![0f32; d_train];
                let col = csc.col(j);
                for (&row_new, &v) in col.row_indices().iter().zip(col.values()) {
                    let row_train = match remap.as_ref() {
                        Some(rm) => rm.new_to_train[row_new],
                        None => Some(row_new),
                    };
                    if let Some(r) = row_train {
                        obs_full[r] += v;
                    }
                }

                // The training chain's own per-cell view: L2-normalise to the
                // recorded scale, then log1p. This is the space svd fits in,
                // and `u·z` reconstructs its STANDARDISED form.
                let l2 = obs_full.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
                let obs_log1p: Vec<f32> = obs_full
                    .iter()
                    .map(|v| (v / l2 * column_sum_norm).ln_1p())
                    .collect();
                // Mean/sd over the stored (non-zero) entries, matching how the
                // sparse standardisation was computed at training time.
                let nz: Vec<f32> = obs_log1p
                    .iter()
                    .zip(&obs_full)
                    .filter(|(_, &c)| c > 0.0)
                    .map(|(&l, _)| l)
                    .collect();
                let (mu, sd) = mean_sd(&nz);

                // Reconstruct, then undo the standardisation to land back in
                // log1p space; `expm1` gives a predicted count profile, which
                // `agreement_from_rate` renormalises to the observed depth.
                let mut pred_counts = vec![0f32; eval_genes.len()];
                let mut obs_eval = vec![0f32; eval_genes.len()];
                let mut sq_resid = 0f64;
                for (slot, &g) in eval_genes.iter().enumerate() {
                    let pred_log1p = recon_std_block[(j, slot)] * sd + mu;
                    // Clamped before `exp_m1`: one overflowing gene would make
                    // the profile's sum infinite, and `rate_to_counts` divides
                    // by that sum — zeroing the whole cell and turning both
                    // correlations into NaN. `ln(f32::MAX)` is ~88.
                    pred_counts[slot] = pred_log1p.min(80.0).exp_m1().max(0.0);
                    obs_eval[slot] = obs_full[g];
                    let r = f64::from(obs_log1p[g] - pred_log1p);
                    sq_resid += r * r;
                }
                let agree = agreement_from_rate(&obs_eval, &pred_counts);

                // Gaussian log-likelihood in log1p space at the ML σ̂:
                // `-0.5·Σ[r²/σ̂² + ln(2πσ̂²)]`, which at σ̂² = mean(r²) reduces
                // to `-0.5·D·(1 + ln(2πσ̂²))`.
                let d = eval_genes.len() as f64;
                let sigma2 = (sq_resid / d).max(1e-12);
                let llik = -0.5 * d * (1.0 + (std::f64::consts::TAU * sigma2).ln());
                CellScore {
                    llik: llik as f32,
                    total: obs_eval.iter().sum(),
                    spearman: agree.spearman,
                    pearson_log1p: agree.pearson_log1p,
                }
            })
            .collect();
        per_cell.extend(block);
    }

    let cell_names = data_vec.column_names()?;
    crate::output_helpers::save_latent(&args.out, &z_nk, &cell_names, None)?;

    let n_eval = eval_genes.len() as f32;
    let mut pred = Mat::zeros(per_cell.len(), 5);
    for (i, c) in per_cell.iter().enumerate() {
        pred[(i, 0)] = c.llik;
        pred[(i, 1)] = c.total;
        // Per GENE, not per count: this llik is a Gaussian over the scored
        // gene axis and does not scale with library size, so dividing it by
        // counts would put a different estimand under a column name that means
        // nats-per-count for every other family.
        pred[(i, 2)] = c.llik / n_eval;
        pred[(i, 3)] = c.spearman;
        pred[(i, 4)] = c.pearson_log1p;
    }
    let cols: Vec<Box<str>> = vec![
        "llik".into(),
        "total".into(),
        "llik_per_gene".into(),
        "spearman".into(),
        "pearson_log1p".into(),
    ];
    pred.to_parquet_with_names(
        &(args.out.to_string() + ".predictive.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&cols),
    )?;
    info!(
        "Agreement (svd, Gaussian log1p llik — compare families on the correlations): \
         per-cell spearman {:.3}, pearson(log1p) {:.3}; {} genes scored",
        mean_finite(per_cell.iter().map(|c| c.spearman)),
        mean_finite(per_cell.iter().map(|c| c.pearson_log1p)),
        eval_genes.len(),
    );
    info!("Wrote {}.predictive.parquet", args.out);
    Ok(())
}

/// One scored cell on the svd path. Named rather than a positional 4-tuple
/// because the aggregate log reads two of the fields by name.
struct CellScore {
    llik: f32,
    total: f32,
    spearman: f32,
    pearson_log1p: f32,
}

/// Mean and (population) standard deviation, with a floor so a constant slice
/// cannot divide by zero.
fn mean_sd(v: &[f32]) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 1.0);
    }
    let n = v.len() as f32;
    let mu = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mu) * (x - mu)).sum::<f32>() / n;
    (mu, var.sqrt().max(1e-8))
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

pub(crate) mod bulk;

#[cfg(test)]
mod tests;

/// Align query genes onto the model's axis, or `None` when the axes already
/// match. Coverage is logged and gated by
/// [`crate::topic::eval::ensure_gene_coverage`].
fn build_remap(
    training_genes: &[Box<str>],
    new_genes: &[Box<str>],
    opts: &QueryNameOpts,
) -> anyhow::Result<Option<GeneRemap>> {
    let mut gene_remap = build_gene_remap_with(training_genes, new_genes, opts);
    crate::topic::eval::ensure_gene_coverage(&gene_remap, opts.min_overlap, "--feature-name-kind")?;
    // After the coverage gate, never before: hidden features are withheld on
    // purpose, so counting them as missing would refuse every ablated run.
    if let Some(hide) = opts.hide.as_deref() {
        crate::topic::eval::hide_features(&mut gene_remap, new_genes, hide)?;
        return Ok(Some(gene_remap));
    }

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

    let s = score_dense_backend(DenseScoreArgs {
        model: &args.model,
        data_files: &args.data_files,
        batch_files: args.batch_files.as_deref(),
        preload: args.preload_data,
        minibatch_size: args.minibatch_size,
        block_size: args.block_size,
        delta_iters: args.delta_iters,
        query_name_opts: &args.query_name_opts()?,
        metadata,
        mode,
        refine_config: &refine_config,
        dev: &args.resolve_device()?,
    })?;

    finalize_predict(FinalizePredict {
        args,
        training_genes: &s.training_genes,
        data_vec: &s.data_vec,
        z_nk: &s.z_nk,
        llik: &s.llik,
        total: &s.total,
        beta_dk: &s.beta_dk,
        delta_db: s.delta_db.as_ref(),
        gene_remap: s.gene_remap.as_ref(),
    })
}

/// Inputs for [`score_dense_backend`] — the `predict`-independent subset, so `probe` can
/// score a dense model without constructing a whole `PredictArgs`.
pub(crate) struct DenseScoreArgs<'a> {
    pub model: &'a str,
    pub data_files: &'a [Box<str>],
    pub batch_files: Option<&'a [Box<str>]>,
    pub preload: bool,
    pub minibatch_size: usize,
    pub block_size: Option<usize>,
    pub delta_iters: usize,
    pub query_name_opts: &'a QueryNameOpts,
    pub metadata: &'a TopicModelMetadata,
    pub mode: LatentMode,
    pub refine_config: &'a TopicRefinementConfig,
    /// Where the encoder and decoder run.
    pub dev: &'a Device,
}

/// What the dense scoring pass produces.
///
/// `predict` consumes every field; `probe` needs only `data_vec` / `llik` / `total`. The
/// struct exists because the dense path used to return a bare 4-tuple from inside a driver
/// that also wrote files, so there was no way to ask it for a score and nothing else.
///
/// ⚠️ `llik`'s **scale** is decoder-dependent: `multinom` gives `Σ_d w_d·x_d·log p_d` (NB-Fisher
/// weighted, against an unweighted `total`), while `nb`/`nbmixture` give an NB log-density sum.
/// All are monotone in fit, and `probe` calibrates against its own null, so verdicts are
/// unaffected — but these numbers are not comparable to the masked path's nats/count.
pub(crate) struct DenseScored {
    pub data_vec: SparseIoVec,
    /// The model's feature axis — the space `beta_dk` and the agreement pass index in.
    pub training_genes: Vec<Box<str>>,
    pub z_nk: Mat,
    pub llik: Vec<f32>,
    pub total: Vec<f32>,
    pub gene_remap: Option<GeneRemap>,
    pub beta_dk: Mat,
    /// The TMLE-refined δ, so the caller can regress it out when writing a residual backend.
    pub delta_db: Option<Mat>,
}

pub(crate) fn score_dense_backend(a: DenseScoreArgs<'_>) -> anyhow::Result<DenseScored> {
    let metadata = a.metadata;
    let (training_genes, beta_dk) = load_dictionary(a.model)?;
    let coarsening = if metadata.has_coarsening {
        load_coarsening(a.model)?
    } else {
        None
    };

    // Reload `μ_d` at D_full and aggregate via the (optional) coarsening
    // matrix to the encoder's D_coarse. Saved by `senna topic` at
    // training time; absent for older models, where the encoder falls
    // back to live per-feature batch centering inside `anscombe_residual`.
    let feature_mean_enc: Option<Vec<f32>> =
        match crate::topic::model_metadata::load_feature_mean(a.model) {
            Ok((_, full)) => Some(aggregate_feature_mean_to_coarse(&full, coarsening.as_ref())),
            Err(_) => None,
        };

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: a.data_files.to_vec(),
        batch_files: a.batch_files.map(<[_]>::to_vec),
        preload: a.preload,
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
    let gene_remap = build_remap(&training_genes, &new_genes, a.query_name_opts)?;

    let delta_db = estimate_delta(
        &data_vec,
        &beta_dk,
        metadata.theta_mean.as_deref(),
        gene_remap.as_ref(),
        a.block_size,
    )?;

    let dev = a.dev;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);

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

    info!("Latent inference mode: {:?}", a.mode);

    let decoder_name = metadata
        .decoder_types
        .first()
        .map_or("multinom", std::convert::AsRef::as_ref);
    info!("Predicting with decoder: {decoder_name}");

    let inputs = DensePredictInputs {
        metadata,
        parameters: &mut parameters,
        vb: &vb,
        model_prefix: a.model,
        encoder: &encoder,
        data_vec: &data_vec,
        delta_db,
        gene_remap: gene_remap.as_ref(),
        coarsening: coarsening.as_ref(),
        beta_dk: &beta_dk,
        delta_iters: a.delta_iters,
        dev,
        adj_method: AdjMethod::Batch,
        minibatch_size: a.minibatch_size,
        mode: a.mode,
        refine_config: a.refine_config,
    };

    let (z_nk, llik, total, delta_final) = match decoder_name {
        "multinom" => predict_dense_with_decoder::<MultinomTopicDecoder>(inputs)?,
        "nb" => predict_dense_with_decoder::<NbTopicDecoder>(inputs)?,
        name if name == NBMIXTURE_NAME => {
            predict_dense_with_decoder::<NbMixtureTopicDecoder>(inputs)?
        }
        other => anyhow::bail!("unsupported decoder type in metadata: {other}"),
    };

    Ok(DenseScored {
        training_genes,
        data_vec,
        z_nk,
        llik,
        total,
        gene_remap,
        beta_dk,
        delta_db: delta_final,
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
    dev: &'a Device,
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
        dev,
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
            finest.attach_feature_weights(&coarse_w, dev)?;
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
                dev,
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
            let t = db.to_tensor(dev)?.transpose(0, 1)?.contiguous()?;
            Ok(t)
        })
        .transpose()?;

    let ntot = data_vec.num_columns();
    let kk = metadata.n_topics;
    // The finest DECODER's width, not `n_features_full`: the dozen-odd dense
    // temporaries live in the likelihood chain, which runs on the decoder's
    // axis. A coarsened dense model densifies briefly at full width but scores
    // at the coarsened one, so sizing the cap off the full axis would throttle
    // the topic paths that have never had a memory problem.
    let d_dense = metadata
        .level_decoder_dims
        .last()
        .copied()
        .unwrap_or(metadata.n_features_full);
    // Ask the decoder rather than assume: one that slices holds the block's
    // input plus a chunk's temporaries, one that does not still materialises
    // the reconstruction and the whole chain at full decoder width.
    let bytes_per_block = if decoder.llik_is_gene_chunked() {
        dense_bytes(minibatch_size, d_dense, 1)
            + dense_bytes(
                minibatch_size,
                crate::topic::predict_common::SCORE_GENE_CHUNK,
                NB_CHAIN_TENSORS,
            )
    } else {
        dense_bytes(minibatch_size, d_dense, NB_CHAIN_TENSORS)
    };

    let (z_nk, llik, total) =
        run_predict_blocks(ntot, kk, minibatch_size, bytes_per_block, |(lb, ub)| {
            predict_block_dense::<Dec>(PredictBlockDenseArgs {
                lb,
                ub,
                data_vec,
                encoder,
                decoder,
                delta_tensor: delta_tensor.as_ref(),
                gene_remap,
                coarsening,
                dev,
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
/// on the held-out cells, and writes the latent plus the per-cell full-cell
/// predictive log-likelihood (`{out}.predictive.parquet`, via
/// [`predictive_llik_masked`]) — the reconstruction-residual fit score. Encoder
/// forward only (no decoder refinement); batch correction at predict is
/// gene-mean only.
fn predict_masked(
    args: &PredictArgs,
    metadata: &TopicModelMetadata,
    head: candle_util::vae::masked_topic::LatentHead,
) -> anyhow::Result<()> {
    use crate::topic::model_metadata::masked_head_label;

    let scored = score_masked_backend(MaskedScoreArgs {
        model: &args.model,
        data_files: &args.data_files,
        batch_files: args.batch_files.as_deref(),
        preload: args.preload_data,
        minibatch_size: args.minibatch_size,
        query_name_opts: &args.query_name_opts()?,
        metadata,
        head,
        need_llik: true,
        block_size: args.block_size,
        delta_iters: args.delta_iters,
        estimate_batch_delta: true,
        dev: &args.resolve_device()?,
    })?;
    let (masked_training_genes, masked_beta_dk) = load_dictionary(&args.model)?;
    let agreement = topic_agreement(&FinalizePredict {
        args,
        training_genes: &masked_training_genes,
        data_vec: &scored.data_vec,
        z_nk: &scored.z_nk,
        llik: &scored.llik,
        total: &scored.total,
        beta_dk: &masked_beta_dk,
        delta_db: scored.delta_db.as_ref(),
        gene_remap: scored.gene_remap.as_ref(),
    })?;
    write_outputs(
        args,
        &scored.data_vec,
        &scored.z_nk,
        &scored.llik,
        &scored.total,
        agreement.as_ref(),
    )?;
    if let Some(delta) = scored.delta_db.as_ref() {
        // The dense path writes δ into the residual backend; the masked path has no
        // residual output, so this line is the only record of what the encoder was
        // fed. A mean far from 1 says the query batches sit well off the training
        // marginal — read the latent with that in mind.
        info!(
            "held-out δ: {} genes × {} batches, mean {:.3}",
            delta.nrows(),
            delta.ncols(),
            delta.iter().sum::<f32>() / (delta.nrows() * delta.ncols()).max(1) as f32
        );
    }
    info!(
        "predict complete ({}, encoder-only latent + full-cell predictive llik)",
        masked_head_label(head)
    );
    Ok(())
}

/// Encoder + full-cell predictive scores for a masked / indexed model on one
/// backend. Shared by `predict` and `probe` (the latter reuses only the scores,
/// not the file output).
pub(crate) struct MaskedScored {
    pub data_vec: SparseIoVec,
    pub z_nk: Mat,
    /// Empty when the caller passed `need_llik: false` (see `MaskedScoreArgs`).
    pub llik: Vec<f32>,
    /// Empty when the caller passed `need_llik: false`.
    pub total: Vec<f32>,
    /// Query→training gene mapping used for the scores; `probe` reuses it for
    /// the influence/gradient pass.
    pub gene_remap: Option<GeneRemap>,
    /// Per-batch δ `[D_train, B]` the latent was encoded under — `None` for a
    /// single-batch query.
    pub delta_db: Option<Mat>,
}

pub(crate) struct MaskedScoreArgs<'a> {
    pub model: &'a str,
    pub data_files: &'a [Box<str>],
    pub batch_files: Option<&'a [Box<str>]>,
    pub preload: bool,
    pub minibatch_size: usize,
    pub query_name_opts: &'a QueryNameOpts,
    pub metadata: &'a TopicModelMetadata,
    pub head: candle_util::vae::masked_topic::LatentHead,
    /// Compute the per-cell predictive log-likelihood. `false` skips it entirely,
    /// leaving `llik`/`total` empty — worth it for callers that want only the latent,
    /// since the score costs a second full pass over every column plus a dense
    /// `[D, minibatch]` reconstruction per block.
    pub need_llik: bool,
    /// Cells per δ-estimation block (`None` = auto).
    pub block_size: Option<usize>,
    /// TMLE rounds for the held-out δ; `0` keeps the plug-in estimate.
    pub delta_iters: usize,
    /// Estimate a per-batch δ from the query and apply it to the encoder null
    /// and the predictive likelihood.
    ///
    /// **Opt-in, because δ is fitted from the query's own gene marginal.** That
    /// is what `predict` wants — it is the held-out batch effect — and exactly
    /// what `probe` must not have: δ absorbs compositional novelty, which is
    /// the signal probe exists to detect, and a query's batch count depends on
    /// how many files the user happened to pass, so leaving it implicit made
    /// probe's calibration and query arms score under different models.
    pub estimate_batch_delta: bool,
    /// Where the encoder and decoder run.
    pub dev: &'a Device,
}

/// Rebuild the indexed encoder from a trained masked model, run encoder-only
/// latent inference on `data_files`, and compute the per-cell full-cell
/// predictive log-likelihood (the probe fit score). Writes nothing.
pub(crate) fn score_masked_backend(a: MaskedScoreArgs<'_>) -> anyhow::Result<MaskedScored> {
    use crate::topic::eval_indexed::{evaluate_latent_masked, EvaluateLatentMaskedConfig};
    use crate::topic::model_metadata::load_feature_mean;

    let embedding_dim = a
        .metadata
        .embedding_dim
        .ok_or_else(|| anyhow::anyhow!("masked-topic metadata missing embedding_dim"))?;
    let enc_context_size = a
        .metadata
        .enc_context_size
        .ok_or_else(|| anyhow::anyhow!("masked-topic metadata missing enc_context_size"))?;

    let (training_genes, beta_dk) = load_dictionary(a.model)?;
    let (_sw_genes, shortlist_weights) = load_shortlist_weights(a.model)?;
    let (_fm_genes, feature_mean) = load_feature_mean(a.model)?;
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
        data_files: a.data_files.to_vec(),
        batch_files: a.batch_files.map(<[_]>::to_vec),
        preload: a.preload,
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
    let gene_remap = build_remap(&training_genes, &new_genes, a.query_name_opts)?;

    // Held-out batch effect, the same contrast the dense path takes: query pseudobulk
    // per batch against the training-implied gene marginal `Σ_k θ̄_k·exp(β_dk)`. The
    // masked path used to skip this and pass NO null to the encoder — see
    // `iterate_delta_masked` for what that cost.
    let delta_db = if a.estimate_batch_delta {
        estimate_delta(
            &data_vec,
            &beta_dk,
            a.metadata.theta_mean.as_deref(),
            gene_remap.as_ref(),
            a.block_size,
        )?
    } else {
        None
    };

    let dev = a.dev;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);
    let encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: a.metadata.n_features_full,
            n_topics: a.metadata.n_topics,
            embedding_dim,
            layers: &a.metadata.encoder_hidden,
            use_gcn: false,
            attn_pool: true,
            // Must match the checkpoint: M widens the first FC layer, and `VarMap::load`
            // errors on a shape mismatch.
            n_gene_modules: a.metadata.n_gene_modules.unwrap_or(0),
        },
        &parameters,
        vb.pp("enc"),
    )?;
    let safetensors_path = format!("{}.safetensors", a.model);
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;

    let adj_method = AdjMethod::Batch;
    let eval_config = EvaluateLatentMaskedConfig {
        dev,
        adj_method: &adj_method,
        minibatch_size: a.minibatch_size,
        enc_context_size,
        shortlist_weights: &shortlist_weights,
        feature_mean: &feature_mean,
        head: a.head,
    };
    // Refine δ against the encoder, if asked. The masked decoder is NB (its
    // `dispersion.parquet` is always written), so the Fisher weights use the saved φ
    // when it is there; a multinomial-trained masked model carries an untrained φ, and
    // the weighting is then merely uninformative rather than wrong.
    let delta_db = match (delta_db, a.delta_iters) {
        (Some(initial), n) if n > 0 => {
            let phi = crate::topic::model_metadata::load_dispersion(a.model)?;
            Some(crate::predict_tmle::iterate_delta_masked(
                n,
                initial,
                &data_vec,
                &encoder,
                &eval_config,
                a.head,
                gene_remap.as_ref(),
                &beta_dk,
                phi.as_deref(),
            )?)
        }
        (d, _) => d,
    };
    // `[B, D_train]` for the encoder null — the orientation the fit-time latent write
    // uses (`masked_topic.rs`), so predict and fit encode a batch the same way.
    let delta_bd = delta_db
        .as_ref()
        .map(|db| -> anyhow::Result<Tensor> {
            Ok(db.to_tensor(dev)?.transpose(0, 1)?.contiguous()?)
        })
        .transpose()?;

    let z_nk = evaluate_latent_masked(
        &data_vec,
        &encoder,
        &eval_config,
        delta_bd.as_ref(),
        gene_remap.as_ref().map(|r| r.new_to_train.as_slice()),
    )?;

    {
        let (eff, mx) = latent_sharpness(&crate::topic::model_metadata::latent_to_theta(
            &z_nk, a.head,
        ));
        info!(
            "Latent sharpness on the query: {eff:.2} effective topics of {}, mean max θ {mx:.3}",
            z_nk.ncols()
        );
    }

    // The stored latent stays raw (`z` for the Gaussian head); scoring needs
    // proportions, so convert per the head rather than assuming `exp`.
    let (llik, total) = if a.need_llik {
        let theta_nk = crate::topic::model_metadata::latent_to_theta(&z_nk, a.head);
        predictive_llik_masked(
            &data_vec,
            &theta_nk,
            &beta_dk,
            delta_db.as_ref(),
            gene_remap.as_ref(),
            a.minibatch_size,
        )?
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(MaskedScored {
        data_vec,
        z_nk,
        llik,
        total,
        gene_remap,
        delta_db,
    })
}

/// Per-cell predictive log-likelihood for the masked / indexed topic model.
///
/// The masked ETM decoder's dictionary `β` is a gene-simplex per topic
/// (`exp(β)` columns sum to 1), so `recon = exp(β)·θ` is the reconstructed
/// composition over genes (columns sum to 1). Scoring the observed counts under
/// it gives the multinomial predictive log-likelihood
///   `llik(cell) = Σ_g x_gj · log recon_gj`,   `total(cell) = Σ_g x_gj`,
/// and `write_outputs` derives `llik_per_count = llik / total`. Genes absent
/// from the reference model are skipped (μ = 0), mirroring
/// [`write_residual_backend`].
///
/// Takes `theta_nk` — proportions, rows summing to 1 — **not** the raw latent.
/// The score is head-agnostic only once that conversion has happened: the
/// simplex heads reach it with `exp(log θ)` and the Gaussian masked-VAE head
/// with `softmax(z)`. See `crate::topic::model_metadata::latent_to_theta`.
///
/// With a per-batch `delta_db` the composition is `δ_{d,b} · Σ_k θ_k·exp(β_dk)`,
/// renormalized over genes per cell — the multinomial analogue of the NB head's
/// `μ = θβ · residual · lib`, so a batch's held-out score is taken against the same
/// batch-adjusted rate its latent was encoded from.
fn predictive_llik_masked(
    data_vec: &SparseIoVec,
    theta_nk: &Mat,
    beta_dk: &Mat,
    delta_db: Option<&Mat>,
    gene_remap: Option<&GeneRemap>,
    minibatch_size: usize,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let exp_beta_dk = beta_dk.map(f32::exp);
    let ntot = data_vec.num_columns();

    let jobs = create_jobs(ntot, 0, Some(minibatch_size));
    let njobs = jobs.len() as u64;
    let blocks: Vec<(usize, Vec<f32>, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(new_progress_bar(njobs))
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>, Vec<f32>)> {
            let csc = data_vec.read_columns_csc(lb..ub)?;
            let n_block = csc.ncols();
            let theta_kn = theta_nk.rows(lb, n_block).transpose();
            let mut recon_dn = &exp_beta_dk * theta_kn; // [D_train, n_block]
            if let Some(delta) = delta_db {
                let batch_ids = data_vec.get_batch_membership(lb..ub);
                for (j, &b) in batch_ids.iter().enumerate() {
                    let mut col = recon_dn.column_mut(j);
                    col.component_mul_assign(&delta.column(b));
                    let z = col.sum().max(1e-12);
                    col /= z;
                }
            }

            let mut llik = vec![0f32; n_block];
            let mut total = vec![0f32; n_block];
            for jloc in 0..n_block {
                let col = csc.col(jloc);
                for (&row_new, &val) in col.row_indices().iter().zip(col.values().iter()) {
                    let row_train = match gene_remap {
                        Some(rm) => rm.new_to_train[row_new],
                        None => Some(row_new),
                    };
                    let Some(row_train) = row_train else {
                        continue; // gene absent from the reference model
                    };
                    let recon = recon_dn[(row_train, jloc)].max(1e-12);
                    llik[jloc] += val * recon.ln();
                    total[jloc] += val;
                }
            }
            Ok((lb, llik, total))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut llik = vec![0f32; ntot];
    let mut total = vec![0f32; ntot];
    for (lb, ll, tot) in blocks {
        for (k, (&l, &t)) in ll.iter().zip(tot.iter()).enumerate() {
            llik[lb + k] = l;
            total[lb + k] = t;
        }
    }
    Ok((llik, total))
}

/// Held-out latent inference for the Gaussian VAE ([`MODEL_TYPE_VAE`]).
/// Rebuilds the [`GaussianEncoder`] and runs it (encoder-only, eval mode →
/// posterior mean `z`) over the held-out cells. Like `predict_masked`, batch
/// correction is gene-mean only (the per-gene `μ_d` divisor inside
/// `anscombe_residual`); a per-cell residual null is a future refinement. The
/// latent is continuous factors, so there is no decoder refinement to do.
fn predict_vae(args: &PredictArgs, metadata: &TopicModelMetadata) -> anyhow::Result<()> {
    // The vae path is encoder-only: it never calls `resolve_mode`, so the
    // latent-mode flags have no effect here. Say so instead of ignoring them
    // silently — a caller passing --decoder-only otherwise gets encoder-only
    // output with no indication the flag was dropped.
    if args.decoder_only || args.refine_steps > 0 {
        log::warn!(
            "--decoder-only / --refine-* do not apply to vae models (encoder-only \
             inference); ignoring them for this run"
        );
    }

    // The decoder is rebuilt so a vae run is scorable like every other family.
    // It used to be skipped (latent only, no `.predictive.parquet`), which left
    // vae the one family a benchmark could not grade on the shared columns.
    let s = score_vae_backend(VaeScoreArgs {
        model: &args.model,
        data_files: &args.data_files,
        batch_files: args.batch_files.as_deref(),
        preload: args.preload_data,
        minibatch_size: vae_training_minibatch(&args.model, args.minibatch_size),
        query_name_opts: &args.query_name_opts()?,
        metadata,
        need_llik: true,
        dev: &args.resolve_device()?,
    })?;

    // `π = softmax_d(z·W + b)` is `exp(b + ρ·θ)` normalised over genes — the
    // same arithmetic `Reconstruction::Embedding` already grades bge by, so
    // the two families land on one comparable axis. The bare `llik` column
    // stays the backend's own NB density and is NOT comparable across
    // families; `eval_llik_per_count` is.
    let (training_genes, _) = load_dictionary(&args.model)?;
    let agreement = match s.recon.as_ref() {
        Some((rho_dh, b_feat)) => evaluate_agreement(AgreementInputs {
            args,
            training_genes: &training_genes,
            data_vec: &s.data_vec,
            recon: Reconstruction::Embedding {
                rho_dh: rho_dh.clone(),
                b_feat,
                theta_nh: &s.z_nk,
            },
        })?,
        None => {
            log::warn!(
                "no decoder bias recovered from {}.safetensors; writing the latent and \
                 the backend likelihood without the agreement metrics",
                args.model
            );
            None
        }
    };

    write_outputs(
        args,
        &s.data_vec,
        &s.z_nk,
        &s.llik,
        &s.total,
        agreement.as_ref(),
    )?;
    Ok(())
}

/// Inputs for [`score_vae_backend`].
pub(crate) struct VaeScoreArgs<'a> {
    pub model: &'a str,
    pub data_files: &'a [Box<str>],
    pub batch_files: Option<&'a [Box<str>]>,
    pub preload: bool,
    pub minibatch_size: usize,
    pub query_name_opts: &'a QueryNameOpts,
    pub metadata: &'a TopicModelMetadata,
    /// `predict` wants the latent only and passes `false`, which also skips rebuilding the
    /// decoder entirely. `probe` needs the per-cell score and passes `true`.
    pub need_llik: bool,
    /// Where the encoder and decoder run.
    pub dev: &'a Device,
}

pub(crate) struct VaeScored {
    pub data_vec: SparseIoVec,
    pub z_nk: Mat,
    /// Empty when `need_llik` was `false`.
    pub llik: Vec<f32>,
    /// Empty when `need_llik` was `false`.
    pub total: Vec<f32>,
    /// The gene side of the rate, `(loadings [D,K], feature bias [D])`, on the
    /// dictionary's gene axis. `None` when `need_llik` was `false` (no decoder
    /// was rebuilt) or when the checkpoint carries no decoder bias.
    pub recon: Option<(Mat, Vec<f32>)>,
}

/// Encoder pass for a `senna vae` model, optionally with the NB predictive score.
///
/// **This is new capability, not a refactor.** `predict_vae` never built a decoder, so the
/// vae family had no fit score at all while every other family did — which is exactly what
/// `probe` needs. The decoder is rebuilt here only when asked for.
///
/// ⚠️ `dictionary.parquet` for a vae holds **raw `[D,K]` factor loadings**, not log-β: the
/// model is `π = softmax_d(z·W + b)`, a per-cell softmax over the gene axis, not a mixture of
/// per-topic simplices. So the parquet shortcut the masked path uses is invalid here and the
/// score must go through the rebuilt decoder.
/// The block size the run was TRAINED with, which is the one to score it at.
///
/// `GaussianEncoder` carries a `BatchNorm`, so its output depends on the block
/// a cell is scored in — `senna vae`'s own `--minibatch-size` help says as
/// much ("minibatch size is not fit-neutral"). Scoring at predict's default
/// while the model was fitted at vae's puts the encoder on batch statistics it
/// never saw, and the latent moves. Taking the training value removes that
/// systematic mismatch.
///
/// It is the DECLARED value: training may shrink the block further to fit
/// device memory, and that effective size is not recorded, so this narrows the
/// gap rather than closing it. A manifest that cannot be read falls back to
/// what the caller asked for.
fn vae_training_minibatch(model: &str, requested: usize) -> usize {
    /// `VaeArgs::minibatch_size.unwrap_or(100)` — the fit's own default when
    /// the flag was left unset.
    const VAE_TRAIN_DEFAULT: usize = 100;
    let Ok((manifest, _)) = crate::run_manifest::load_for(model) else {
        return requested;
    };
    if manifest.train_args.is_none() {
        return requested;
    }
    match manifest.train_args_as::<crate::vae::VaeArgs>(model) {
        Ok(recorded) => {
            let trained_at = recorded.minibatch_size.unwrap_or(VAE_TRAIN_DEFAULT);
            if trained_at != requested {
                info!(
                    "Scoring at the training block size ({trained_at}, not {requested}): the \
                     encoder's batch norm makes the latent block-dependent"
                );
            }
            trained_at
        }
        Err(e) => {
            log::warn!(
                "{model}: cannot read the recorded fit configuration ({e}); scoring at \
                 {requested}, which may differ from the block size the encoder was fitted at"
            );
            requested
        }
    }
}

pub(crate) fn score_vae_backend(a: VaeScoreArgs<'_>) -> anyhow::Result<VaeScored> {
    use crate::topic::model_metadata::load_feature_mean;

    let metadata = a.metadata;
    let (training_genes, loadings_dk) = load_dictionary(a.model)?;
    let (_fm_genes, feature_mean) = load_feature_mean(a.model)?;
    anyhow::ensure!(
        feature_mean.len() == training_genes.len(),
        "feature_mean gene count ({}) != dictionary gene count ({})",
        feature_mean.len(),
        training_genes.len()
    );

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: a.data_files.to_vec(),
        batch_files: a.batch_files.map(<[_]>::to_vec),
        preload: a.preload,
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
    let gene_remap = build_remap(&training_genes, &new_genes, a.query_name_opts)?;

    let dev = a.dev;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);
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

    // Only the finest level is ever scored, and only it needs registering: `VarMap::load`
    // iterates the varmap's OWN entries and looks each up in the file, so extra tensors in the
    // checkpoint are simply never visited. (`counterfactual::rebuild_model` must register every
    // level for the opposite reason — it *writes* a child checkpoint that `--init-from` would
    // reject if levels were missing. This path writes nothing.)
    let finest = metadata.num_levels.saturating_sub(1);
    let decoder = if a.need_llik {
        Some(candle_util::decoder::GaussianNbDecoder::new(
            *metadata
                .level_decoder_dims
                .last()
                .unwrap_or(&metadata.n_features_full),
            metadata.n_topics,
            vb.pp(format!("dec_{finest}")),
        )?)
    } else {
        None
    };

    let safetensors_path = format!("{}.safetensors", a.model);
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;
    let decoder = decoder.as_ref();

    // Read the gene side AFTER the load, or it is the initialization rather
    // than the fit. The loadings come from `dictionary.parquet` rather than
    // the checkpoint's weight so they are guaranteed to sit on the same gene
    // axis as `training_genes`; only `b` has no parquet of its own.
    let recon = decoder.and_then(|dec| {
        let bias = dec.feature_bias()?;
        let bias: Vec<f32> = bias.flatten_all().ok()?.to_vec1().ok()?;
        if bias.len() != loadings_dk.nrows() {
            log::warn!(
                "vae decoder bias has {} entries but the dictionary has {} genes; \
                 skipping the agreement metrics for this run",
                bias.len(),
                loadings_dk.nrows()
            );
            return None;
        }
        Some((loadings_dk.clone(), bias))
    });

    let ntot = data_vec.num_columns();
    let (z_nk, llik, total) = run_predict_blocks(
        ntot,
        metadata.n_topics,
        a.minibatch_size,
        // The encoder input is the only full-width tensor left; the likelihood
        // works one `crate::topic::predict_common::SCORE_GENE_CHUNK` slice at a time, so the chain's
        // temporaries are counted at slice width, not at D.
        dense_bytes(a.minibatch_size, training_genes.len(), 1)
            + dense_bytes(
                a.minibatch_size,
                crate::topic::predict_common::SCORE_GENE_CHUNK,
                NB_CHAIN_TENSORS,
            ),
        |(lb, ub)| {
            // Gene-mean null only (x0 = None): the divisive μ_d correction is
            // baked into the encoder via `feature_mean`.
            let csc = data_vec.read_columns_csc(lb..ub)?;
            let x_nd = remap_and_coarsen_dense(&csc, gene_remap.as_ref(), None, dev)?;
            let (z, _) = encoder.forward_t(&x_nd, None, false)?;
            let z_mat = Mat::from_tensor(&z.to_device(&Device::Cpu)?)?;
            let Some(dec) = decoder else {
                return Ok((lb, z_mat, Vec::new(), Vec::new()));
            };
            // `GaussianNbDecoder` takes the **raw** `z` — it applies its own softmax over the
            // gene axis — unlike the topic decoders, which are handed log θ.
            //
            // Gene-chunked, not `forward_with_llik`: that returns `(π, llik)`
            // and π is `[N, D]` by construction — training needs it for the
            // gradient, inference throws it away. Paying for a full dense
            // matrix (and the dozen more inside the NB chain) to get one
            // scalar per cell is what made this path an OOM kill at
            // whole-transcriptome width.
            let llik: Vec<f32> = dec
                .llik_gene_chunked(&z, &x_nd, crate::topic::predict_common::SCORE_GENE_CHUNK)?
                .to_device(&Device::Cpu)?
                .to_vec1()?;
            let total: Vec<f32> = x_nd.sum(1)?.to_device(&Device::Cpu)?.to_vec1()?;
            Ok((lb, z_mat, llik, total))
        },
    )?;

    Ok(VaeScored {
        data_vec,
        z_nk,
        llik,
        total,
        recon,
    })
}

/// Dense `[rows × width]` f32 tensors the NB likelihood chain holds at peak —
/// counted off `candle_util::loss::nb_log_likelihood_elem` (phi, mu, phi+mu,
/// three logs, two terms, x+phi, the lgammas) plus the decoder's logits,
/// softmax and mu. Rounded up: under-counting is an OOM, over-counting is
/// only slower.
pub(crate) const NB_CHAIN_TENSORS: usize = 16;

/// Bytes of dense f32 one block holds, as a shape rather than a number the
/// caller multiplied out — the callers and the tests all go through this, so
/// there is one formula instead of three that can drift apart. Mirrors
/// `data_beans::sparse_io::helpers::preload_within_budget`, which likewise
/// takes the raw shape and computes the cost itself.
pub(crate) fn dense_bytes(rows: usize, width: usize, tensors: usize) -> usize {
    rows.saturating_mul(width)
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_mul(tensors)
}

/// How many dense blocks may be in flight at once, given the working set one
/// block costs.
///
/// The dense drivers densify each block to `[minibatch × D]` and then run a
/// decoder over it; the NB likelihood chain alone allocates a dozen-odd
/// tensors of that shape. Left to `par_iter`, EVERY block runs at once, so the
/// peak is `threads × per-block` — which at whole-transcriptome D is tens of
/// gigabytes and was an OOM kill, silently, with only a SIGKILL to show for
/// it. The work per block is unchanged; only how many run together is capped.
///
/// `LEGUME_PREDICT_BUDGET_BYTES` overrides the default, following the
/// `LEGUME_PRELOAD_BUDGET_BYTES` precedent for memory knobs.
fn dense_block_concurrency(bytes_per_block: usize) -> usize {
    let budget = std::env::var("LEGUME_PREDICT_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PREDICT_BUDGET_BYTES);
    block_concurrency(bytes_per_block, budget, rayon::current_num_threads())
}

/// The memory budget the cap works from when `LEGUME_PREDICT_BUDGET_BYTES` is unset.
pub(crate) const DEFAULT_PREDICT_BUDGET_BYTES: usize = 8 << 30;

/// The cap as a pure function of the three things that set it. Kept apart from
/// the env/rayon lookup so a test can state the machine it means instead of
/// inheriting the one it happens to run on: the same block that runs wide
/// open on a 32-thread box is rightly held at ~34 on a 64-thread one.
pub(crate) fn block_concurrency(bytes_per_block: usize, budget: usize, threads: usize) -> usize {
    (budget / bytes_per_block.max(1)).clamp(1, threads.max(1))
}

/// Run `block_fn` over `[0, ntot)` in `minibatch_size` blocks, concatenating
/// results into `(z_nk [ntot, kk], llik [ntot], total [ntot])`. Shared by the
/// dense predict drivers.
fn run_predict_blocks<F>(
    ntot: usize,
    kk: usize,
    minibatch_size: usize,
    // Peak dense bytes ONE block holds, for the concurrency cap.
    bytes_per_block: usize,
    block_fn: F,
) -> anyhow::Result<(Mat, Vec<f32>, Vec<f32>)>
where
    F: Fn((usize, usize)) -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)> + Sync,
{
    let jobs = create_jobs(ntot, 0, Some(minibatch_size));
    let njobs = jobs.len() as u64;
    let max_conc = dense_block_concurrency(bytes_per_block);
    if max_conc < rayon::current_num_threads() {
        info!(
            "Scoring {njobs} blocks at most {max_conc} at a time: one block's ~{} MB dense \
             working set would otherwise be multiplied by every thread \
             (LEGUME_PREDICT_BUDGET_BYTES to raise)",
            bytes_per_block >> 20
        );
    }
    let bar = new_progress_bar(njobs);
    // A pool sized to the cap rather than waves of `par_iter`: the cap has to
    // bound how many blocks hold their dense tensors at once, and a wave stalls
    // on its slowest block before the next starts. Sizing the pool keeps rayon
    // work-stealing continuously under the same ceiling.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_conc)
        .build()?;
    let mut chunks: Vec<(usize, Mat, Vec<f32>, Vec<f32>)> = pool.install(|| {
        jobs.par_iter()
            .progress_with(bar.clone())
            .map(|&block| block_fn(block))
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    bar.finish_and_clear();
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
    training_genes: &'a [Box<str>],
    data_vec: &'a SparseIoVec,
    z_nk: &'a Mat,
    llik: &'a [f32],
    total: &'a [f32],
    beta_dk: &'a Mat,
    delta_db: Option<&'a Mat>,
    gene_remap: Option<&'a GeneRemap>,
}

fn finalize_predict(f: FinalizePredict<'_>) -> anyhow::Result<()> {
    let agreement = topic_agreement(&f)?;
    write_outputs(
        f.args,
        f.data_vec,
        f.z_nk,
        f.llik,
        f.total,
        agreement.as_ref(),
    )?;
    write_residual_backend(
        f.args,
        f.data_vec,
        f.z_nk,
        f.beta_dk,
        f.delta_db,
        f.gene_remap,
    )
}

/// Everything the agreement pass needs, independent of which family produced it.
struct AgreementInputs<'a> {
    args: &'a PredictArgs,
    /// The model's feature axis — what `recon` and the eval indices live on.
    training_genes: &'a [Box<str>],
    data_vec: &'a SparseIoVec,
    recon: Reconstruction<'a>,
}

/// Score the reconstruction against the observed test counts.
///
/// A separate streaming pass rather than a hook inside each backend's likelihood
/// loop. It duplicates one matrix product per block — but it buys one formula in
/// one place for every family, and the test half is the small side of a split by
/// construction.
fn evaluate_agreement(a: AgreementInputs<'_>) -> anyhow::Result<Option<EvalOutcome>> {
    // Training axis, not query axis: the pass densifies the observed counts onto
    // the model's features before scoring, so an index that came from the query's
    // row names would point at the wrong gene whenever the two axes differ.
    let restrict = a
        .args
        .eval_features
        .as_deref()
        .or(a.args.ablate_features.as_deref());
    let eval_genes = resolve_eval_genes(restrict, a.training_genes)?;
    if eval_genes.is_empty() {
        info!("No evaluation features matched; skipping agreement metrics");
        return Ok(None);
    }

    // Built here rather than taken from the caller, and always without the
    // ablation. The backend's remap has the hidden genes pointing at `None`,
    // which is right for the encoder and wrong here: densifying the *observed*
    // counts through it would read the scored genes as zero and grade every model
    // against a blank. Rebuilding costs one name-matching pass and removes the
    // chance of a caller handing over the encoder's view by mistake.
    let mut scoring_opts = a.args.query_name_opts()?;
    scoring_opts.hide = None;
    let score_remap = build_remap(a.training_genes, &a.data_vec.row_names()?, &scoring_opts)?;

    let mut out = evaluate_predictions(EvalArgs {
        data_vec: a.data_vec,
        null_comp: training_marginal(a.args, a.training_genes, &eval_genes)?,
        recon: a.recon,
        gene_remap: score_remap.as_ref(),
        eval_genes,
        minibatch_size: a.args.minibatch_size,
        keep_per_gene: restrict.is_some(),
    })?;
    out.train_gene_names = a.training_genes.to_vec();
    info!("Agreement: {}", out.summary().line());
    Ok(Some(out))
}

/// The training composition over the scored genes — the null every arm shares.
///
/// A count-weighted marginal over the training cells: `Σ_cells x_g`, normalised
/// over the scored genes. Two properties earn it the job. It does not depend on
/// the model, so every arm scored on the same training half and genes is measured
/// against one identical floor. And it is not fitted on the test half, so it is a
/// baseline a model could actually have matched.
///
/// Deliberately NOT `{model}.feature_mean.parquet`, which was the first thing
/// tried here. That file is a per-gene *rate the encoder divides by*, formed as a
/// Gamma posterior mean averaged over pseudobulk groups — so its prior floors
/// every gene unseen in training at the same positive value, and a third of the
/// axis ends up sharing it. As a composition it therefore spends real probability
/// mass on genes that carry almost none, which measured 0.49 nats/count weaker
/// than the count-weighted marginal on the same genes. A weak floor flatters every
/// model, which is the one direction a null must not err in.
fn training_marginal(
    args: &PredictArgs,
    training_genes: &[Box<str>],
    eval_genes: &[usize],
) -> anyhow::Result<Option<Vec<f32>>> {
    let Some(files) = args.null_from.as_deref() else {
        log::warn!(
            "no --null-from: the null falls back to the TEST half's own composition, which \
             is fitted on the data being scored and is an upper reference rather than a \
             floor. Pass the training half to make gains comparable across runs"
        );
        return Ok(None);
    };
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: files.to_vec(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    // The training half is aligned onto the model's axis the same way the query
    // is, so a name-kind difference between the two files cannot silently drop
    // genes from the floor that the model is being scored on.
    let mut opts = args.query_name_opts()?;
    opts.hide = None;
    let remap = build_remap(training_genes, &loaded.data.row_names()?, &opts)?;
    let comp = crate::topic::predict_eval::empirical_composition(
        &loaded.data,
        remap.as_ref(),
        eval_genes,
        training_genes.len(),
        args.minibatch_size,
    )?;
    anyhow::ensure!(
        comp.iter().any(|&v| v > 0.0),
        "--null-from carries no counts on the scored genes; the null would be empty"
    );
    info!(
        "Null: gene composition of {} training cells from {}",
        loaded.data.num_columns(),
        files.join(", ")
    );
    Ok(Some(comp))
}

/// The topic families' reconstruction inputs, in the shape the eval pass wants.
///
/// The latent is stored on the log scale and every consumer exponentiates it;
/// doing that here once keeps the two callers from disagreeing about which scale
/// they hold.
fn topic_agreement(f: &FinalizePredict<'_>) -> anyhow::Result<Option<EvalOutcome>> {
    let theta_nk = f.z_nk.map(f32::exp);
    evaluate_agreement(AgreementInputs {
        args: f.args,
        training_genes: f.training_genes,
        data_vec: f.data_vec,
        recon: Reconstruction::Topic {
            exp_beta_dk: f.beta_dk.map(f32::exp),
            theta_nk: &theta_nk,
            delta_db: f.delta_db,
        },
    })
}

fn write_outputs(
    args: &PredictArgs,
    data_vec: &SparseIoVec,
    z_nk: &Mat,
    llik: &[f32],
    total: &[f32],
    agreement: Option<&EvalOutcome>,
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
        // NaN, not 0: a cell with no counts has no per-count score, and 0 nats
        // per count reads as a PERFECT prediction to anything that averages this
        // column. Ablation makes the case common — a cell can easily carry
        // nothing in the hidden gene set.
        pred[(i, 2)] = if total[i] > 0.0 {
            llik[i] / total[i]
        } else {
            f32::NAN
        };
    }
    let mut pred_cols: Vec<Box<str>> = vec!["llik".into(), "total".into(), "llik_per_count".into()];
    if let Some(a) = agreement {
        anyhow::ensure!(
            a.per_cell.len() == n && a.per_cell_llik.len() == n,
            "agreement pass scored {} cells, likelihood pass scored {n}",
            a.per_cell.len()
        );
        pred = pred.insert_columns(3, 5, 0.0);
        for (i, c) in a.per_cell.iter().enumerate() {
            pred[(i, 3)] = c.spearman;
            pred[(i, 4)] = c.pearson_log1p;
        }
        // The cross-family likelihood. Written per cell, not just summarised in
        // the log, because a benchmark reads this file: the `llik` column above
        // is the backend's own and is not comparable between decoders, so these
        // are the only columns two families may be ranked on.
        for (i, l) in a.per_cell_llik.iter().enumerate() {
            pred[(i, 5)] = l.count;
            let per_count = |v: f32| if l.count > 0.0 { v / l.count } else { f32::NAN };
            pred[(i, 6)] = per_count(l.model);
            pred[(i, 7)] = per_count(l.null);
        }
        pred_cols.push("spearman".into());
        pred_cols.push("pearson_log1p".into());
        pred_cols.push("eval_count".into());
        pred_cols.push("eval_llik_per_count".into());
        pred_cols.push("eval_null_llik_per_count".into());
    }
    pred.to_parquet_with_names(
        &(args.out.to_string() + ".predictive.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&pred_cols),
    )?;

    if let Some(a) = agreement.filter(|a| !a.per_gene.is_empty()) {
        // The other axis: each gene's agreement *across* cells. A model can score
        // well per cell — every cell's genes ranked about right — while getting a
        // gene's variation across cells backwards, which is the direction that
        // matters for anything downstream that compares cells.
        let mut gm = Mat::zeros(a.per_gene.len(), 3);
        let mut rows: Vec<Box<str>> = Vec::with_capacity(a.per_gene.len());
        for (i, &(g, sp, pe, mean_obs)) in a.per_gene.iter().enumerate() {
            gm[(i, 0)] = sp;
            gm[(i, 1)] = pe;
            gm[(i, 2)] = mean_obs;
            rows.push(
                a.train_gene_names
                    .get(g)
                    .cloned()
                    .unwrap_or_else(|| "?".into()),
            );
        }
        let cols: Vec<Box<str>> = vec![
            "spearman".into(),
            "pearson_log1p".into(),
            "mean_observed".into(),
        ];
        gm.to_parquet_with_names(
            &(args.out.to_string() + ".gene_agreement.parquet"),
            (Some(&rows), Some("gene")),
            Some(&cols),
        )?;
        info!("Wrote {}.gene_agreement.parquet", args.out);
    }
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
