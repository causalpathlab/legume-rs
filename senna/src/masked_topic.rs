use crate::embed_common::*;
use crate::topic::common::{
    create_device, load_and_collapse, move_varmap_to_cpu, setup_stop_handler, LoadCollapseArgs,
    PreparedData,
};
use crate::topic::eval_indexed::{evaluate_latent_masked, EvaluateLatentMaskedConfig};
use crate::topic::train_masked::{
    train_masked, write_feature_embedding, write_masked_dictionary, IndexedTrainConfig,
};

use candle_util::decoder::EmbeddedNbTopicDecoder;
use candle_util::encoder::*;
use candle_util::vae::masked_topic::LatentHead;
use log::warn;

/// Mask-rate schedule (CLI surface for `MaskSchedule`).
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaskScheduleArg {
    /// Constant mask fraction (`--mask-fraction`).
    #[default]
    Fixed,
    /// Sample the rate per minibatch in `[--mask-rate-lo, --mask-rate-hi]`.
    Uniform,
}

/// Per-gene likelihood for the masked imputation loss (CLI surface for
/// [`candle_util::vae::masked_topic::MaskedLikelihood`]).
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaskedLikelihoodArg {
    /// Negative binomial — overdispersed counts (library-scaled, learnable φ).
    #[default]
    Nb,
    /// Multinomial / categorical — depth-invariant composition; the same
    /// likelihood the generative ELBO path uses.
    Multinomial,
}

impl MaskedLikelihoodArg {
    /// Map to the candle-util training enum.
    pub fn to_lib(self) -> candle_util::vae::masked_topic::MaskedLikelihood {
        use candle_util::vae::masked_topic::MaskedLikelihood as L;
        match self {
            MaskedLikelihoodArg::Nb => L::Nb,
            MaskedLikelihoodArg::Multinomial => L::Multinomial,
        }
    }
}

#[derive(Args, Debug)]
pub struct MaskedTopicArgs {
    #[arg(
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5; optional when --from is given)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Multiple files may be passed (comma- or space-separated)\n\
                     and are concatenated column-wise on a shared feature set.\n\
                     When `--from <run.senna.json>` is provided and this list is empty,\n\
                     the data paths come from the source manifest."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Chain inputs + warm-start from a prior `senna bge / fne / topic` manifest",
        long_help = "Read a `{run}.senna.json` manifest and pre-fill `data_files`,\n\
                     `--batch-files`, and `--freeze-feature-embedding` from it.\n\
                     Explicit CLI flags override the manifest values.\n\
                     SVD-family manifests are rejected (no feature embedding to inherit).\n\
                     Typical use: bge → masked-topic warm-start.\n\
                     \n\
                     senna bge   data.zarr.zip -b batch.gz -o run-bge ...\n\
                     senna masked-topic --from run-bge.senna.json -o run-topic ..."
    )]
    from: Option<Box<str>>,

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
                     {out}.cell_proj.parquet        cached random projection (for `senna layout`)\n  \
                     {out}.senna.json               run manifest for `senna layout/plot --from`\n\n\
                     With -x bulk files: {out}.bulk_latent.parquet additionally."
    )]
    out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell in the same order as its matching data file.\n\
                     Example: batch1.tsv,batch2.tsv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    collapse: crate::refine_weighting::CollapseArgs,

    #[arg(
        long = "init-from",
        help = "Initialize encoder + decoder weights from a previously trained model",
        long_help = "Path prefix of a model saved by `senna masked-topic`\n\
                     (matching {prefix}.model.json + {prefix}.safetensors).\n\
                     Architecture must match: same K, encoder layers,\n\
                     embedding_dim, and n_features_full.\n\
                     Cross-gene-set warm-start is not supported — train on the same gene set."
    )]
    init_from: Option<Box<str>>,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    block_size: Option<usize>,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics (K)"
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder hidden layer sizes (comma-separated)",
        long_help = "Example: 128,1024,128 (input → 128 → 1024 → 128 → topics)."
    )]
    encoder_layers: Vec<usize>,

    #[arg(long, short = 'i', default_value_t = 1000, help = "Training epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 100, help = "Training minibatch size")]
    minibatch_size: usize,

    #[arg(
        long,
        alias = "lr",
        default_value_t = 0.01,
        help = "Adam learning rate"
    )]
    learning_rate: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off; typical 0.5–5.0)"
    )]
    grad_clip: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 penalty λ on the feature embedding matrix ρ (default 1.0; 0 = off)",
        long_help = "L2 penalty λ on the feature embedding matrix ρ ∈ ℝ^{D×H}:\n\
                     adds λ · mean(ρ²) to the per-minibatch loss (mean-normalized,\n\
                     so λ stays scale-invariant across D·H).\n\
                     Shrinks β dynamic range (β = log_softmax(α·ρᵀ)) and can speed ETM convergence\n\
                     on high-D gene sets.\n\
                     Default 1.0 (mild shrinkage). Set 0.0 to disable. Typical: 0.1–10.0."
    )]
    feature_embedding_l2: f32,

    #[arg(
        long,
        help = "Freeze a pre-trained per-gene embedding ρ; encoder/decoder ρ fixed",
        long_help = "Reuse a pre-trained per-gene embedding ρ from a prior senna run.\n\
                     Loads `{prefix}.feature_embedding.parquet` (topic / cell-embedded-topic layout)\n\
                     or `{prefix}.dictionary.parquet` (gbe layout).\n\
                     Gene names are strict-intersected against this dataset's gene axis\n\
                     under the `--feature-name-kind` rule; unmatched genes are dropped from training.\n\
                     The encoder/decoder ρ stays fixed; everything else\n\
                     (α, FC, BN, value-embedding, decoder topic embeddings) trains as usual.\n\
                     Incompatible with `--feature-network`\n\
                     (its restriction would change the gene axis that the frozen ρ pins);\n\
                     forces `--feature-embedding-l2 0` (frozen ρ doesn't need shrinkage)."
    )]
    freeze_feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        conflicts_with = "freeze_feature_embedding",
        help = "Warm-start ρ from a prior senna run; AdamW continues to update it",
        long_help = "Warm-start ρ from a prior senna run (typically `senna bge`).\n\
                     Same layout resolution as `--freeze-feature-embedding`\n\
                     (gbe `{prefix}.dictionary.parquet`\n\
                     or topic `{prefix}.feature_embedding.parquet`), same strict gene-name intersection.\n\
                     The difference is that AdamW continues to update ρ during training —\n\
                     this just gives a biology-aware starting point\n\
                     instead of random Kaiming-normal init.\n\
                     Pairs well with `senna bge` pre-training:\n\
                     cheap NCE-based gene embedding that's robust to batch effects,\n\
                     used as the warm-start here.\n\
                     Mutually exclusive with `--freeze-feature-embedding`."
    )]
    init_feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Cross-entropy penalty λ on β toward the anchor prior (0 = off)",
        long_help = "Cross-entropy penalty λ on β toward the anchor prior —\n\
                     anchors topic indices to Gram-Schmidt anchor gene sets\n\
                     derived from the finest-level pseudobulks.\n\
                     Breaks the K-way permutation symmetry of the ETM-factorized β\n\
                     and is the load-bearing anti-mode-collapse force for this model.\n\
                     0 disables; default 1.0."
    )]
    anchor_penalty: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay for all parameters (default 0.0 = off)",
        long_help = "AdamW decoupled weight decay applied uniformly to every parameter\n\
                     (encoder ρ + α + FC + BN).\n\
                     Per-step post-update shrinkage; doesn't enter the backward graph.\n\
                     Default 0.0 (off, i.e. plain Adam despite the name).\n\
                     Typical: 1e-5 to 1e-4."
    )]
    weight_decay: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Compute device (cpu|cuda|metal)"
    )]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "CUDA/Metal device index")]
    device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Batch adjustment (batch|residual)",
        long_help = "batch    — subtract per-batch pseudobulk mean.\n\
                     residual — divide by fitted delta per pseudobulk group."
    )]
    adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training"
    )]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Treat input files as modalities of the same cells, glued by raw barcode.",
        long_help = "Patchy multi-modal (multiome) load. Each file keeps its own feature space\n\
                     (no cross-file barcode suffixing); cells are unioned across files by raw barcode —\n\
                     a cell observed only in RNA contributes triplets just to the RNA row block,\n\
                     ATAC-only cells just to the ATAC block, and shared cells get both.\n\
                     Disables `@<basename>` suffixing on cell names.\n\
                     Maps to `ColumnAlignment::Union` in the loader.\n\
                     \n\
                     With --multiome, batch resolution is constrained:\n\
                     a single --batch-files file is allowed (one label per unified cell),\n\
                     or embedded `@batch` tags in raw column names that AGREE across modalities.\n\
                     The default `@<filename>` fallback is disabled —\n\
                     a cell can come from multiple files and cannot carry two labels."
    )]
    multiome: bool,

    #[command(flatten)]
    hvg: crate::hvg::HvgCliArgs,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Uniform smoothing of topic proportions during training",
        long_help = "z_smooth = (1-α) z + α/K. Keeps every topic on the gradient path\n\
                     and prevents dead topics. Typical: 0.01–0.2. Set 0 to disable."
    )]
    topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "Masked-imputation fraction: held-out top-K genes per cell (typical 0.2–0.5)",
        long_help = "Masked-imputation fraction: per cell, this fraction of its top-K genes\n\
                     is held out (masked) and predicted by the NB embedded-topic head;\n\
                     the rest are the encoder's visible input.\n\
                     Typical 0.2–0.5.\n\
                     The masking is the regularizer that replaces the (collapse-prone) ELBO/KL."
    )]
    mask_fraction: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Held-out imputation eval: fraction to hold out after training (0 = off)",
        long_help = "After training, run a held-out masked-imputation evaluation\n\
                     and log the mean log-likelihood per held-out gene.\n\
                     For each cell this fraction of its observed genes is hidden,\n\
                     the rest encode a latent, and the trained decoder imputes the hidden genes.\n\
                     Unlike the per-epoch training likelihood this is not optimized,\n\
                     so it distinguishes real structure from overfitting.\n\
                     Use a fixed --eval-seed to compare heads on the same held-out positions.\n\
                     0 disables (default)."
    )]
    eval_mask_fraction: f64,

    #[arg(
        long,
        default_value_t = 42,
        help = "Seed for the held-out imputation mask (see --eval-mask-fraction)"
    )]
    eval_seed: u64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "KL weight β for the Gaussian latent (masked-vae only; default 1.0)",
        long_help = "KL weight β for the Gaussian latent (masked-vae only; ignored by masked-topic).\n\
                     The masked-NB signal is weaker than a full reconstruction,\n\
                     so β < 1 (e.g. 0.1–0.5) often avoids over-regularizing the posterior toward the prior."
    )]
    kl_weight: f64,

    #[arg(
        long,
        value_enum,
        default_value_t = MaskScheduleArg::Fixed,
        help = "Mask-rate schedule: fixed or uniform per-minibatch sampling",
        long_help = "Mask-rate schedule: fixed (use --mask-fraction)\n\
                     or uniform (sample the rate per minibatch in [--mask-rate-lo, --mask-rate-hi];\n\
                     any-order / absorbing-diffusion style)."
    )]
    mask_schedule: MaskScheduleArg,

    #[arg(
        long,
        value_enum,
        default_value_t = MaskedLikelihoodArg::Nb,
        help = "Masked-loss likelihood: nb (overdispersed counts) or multinomial (depth-invariant)."
    )]
    masked_likelihood: MaskedLikelihoodArg,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Lower bound of the per-minibatch mask rate when --mask-schedule=uniform."
    )]
    mask_rate_lo: f64,

    #[arg(
        long,
        default_value_t = 0.6,
        help = "Upper bound of the per-minibatch mask rate when --mask-schedule=uniform."
    )]
    mask_rate_hi: f64,

    #[arg(
        long,
        default_value_t = 512,
        help = "Encoder context window (top-K features per cell)",
        long_help = "Each cell keeps its top-K features by value;\n\
                     minibatches use the union of selected indices. Smaller K = faster decoder."
    )]
    context_size: usize,

    #[arg(
        long,
        default_value_t = 128,
        help = "Per-feature embedding dimension H (default 128; 0 = auto = 2 × n-latent-topics)",
        long_help = "Dimension H of the per-gene embedding ρ ∈ ℝ^{D×H}.\n\
                     ρ is shared between encoder (value-weighted pool over each cell's top-K features)\n\
                     and decoder (β_kd = log_softmax_d(α_k · ρ_dᵀ), with α ∈ ℝ^{K×H} as topic embeddings).\n\
                     β is rank ≤ H, so H must be ≥ K (--n-latent-topics)\n\
                     for K independent topics to be representable.\n\
                     Default 128; pass 0 to auto-resolve to 2K instead.\n\
                     H < K errors at startup."
    )]
    embedding_dim: usize,

    #[arg(
        long,
        help = "Decoder context window (default: --context-size)",
        long_help = "Top-K features per sample used on the decoder side.\n\
                     Defaults to the encoder's --context-size when not set."
    )]
    decoder_context_size: Option<usize>,

    #[arg(
        short = 'x',
        long,
        value_delimiter = ',',
        help = "Bulk expression files for joint deconvolution",
        long_help = "Accepts .parquet or .tsv.gz; rows are aligned to the SC gene set.\n\
                     Bulk samples are embedded using the trained encoder/decoder."
    )]
    bulk_data_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        help = "Feature-feature edge list (TSV/CSV) to restrict the feature axis",
        long_help = "Optional feature-feature edge list (TSV/CSV) —\n\
                     used to RESTRICT the feature axis to graph-connected genes\n\
                     (see --no-feature-network-restrict).\n\
                     Graph *diffusion* (GCN) is not supported by the masked encoder,\n\
                     so the edges only drive feature selection here.\n\
                     Edges may be intra- or cross-modal\n\
                     (gene-gene PPI, peak-gene ABC, ATAC-derived regulatory links).\n\
                     Edge names are resolved against the loaded gene axis."
    )]
    feature_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Also try forward/reverse prefix matching for feature-network name resolution"
    )]
    feature_network_prefix_match: bool,

    #[arg(
        long,
        help = "Alias-splitting delimiter for feature-network name resolution (e.g. '_')",
        long_help = "Alias-splitting delimiter for feature-network name resolution.\n\
                     When set (e.g. '_'), each row name is registered under its full form\n\
                     AND every split component, so a row like `ENSG00000105329_TGFB1`\n\
                     matches network edges that name *either* `ENSG00000105329` *or* `TGFB1`\n\
                     (via matrix-util's GeneIndexResolver — both aliases point to the same row)."
    )]
    feature_network_delim: Option<char>,

    #[arg(
        long,
        default_value_t = 1,
        help = "Shared-neighbor edge QC threshold (default 1; 0 = keep all edges)",
        long_help = "Shared-neighbor edge QC: drop edges (u,v)\n\
                     where the endpoints share fewer than N neighbors in the feature network.\n\
                     Default 1 drops edges with zero corroboration\n\
                     — standard PPI/topological-overlap denoising.\n\
                     Set 0 to keep every parsed edge."
    )]
    feature_network_min_shared_neighbors: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-node degree cap on the feature network (0 = off)",
        long_help = "Per-node degree cap on the feature network (0 = off).\n\
                     After shared-neighbor QC, for each feature with degree > N,\n\
                     rank its neighbors by shared-neighbor count and keep the top N\n\
                     (union-symmetric: an edge survives iff either endpoint kept it).\n\
                     Caps PPI hubs whose degree would otherwise blow up per-cell sub-adjacency."
    )]
    feature_network_max_degree: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Iterative k-core pruning threshold on the feature network (0 = off)",
        long_help = "Iterative k-core pruning threshold on the feature network (default 0 = off).\n\
                     Drops every feature whose degree falls below N\n\
                     until the surviving subgraph is N-degenerate."
    )]
    feature_network_min_degree: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable feature-network feature restriction (keep full feature axis)",
        long_help = "Disable feature-network feature restriction.\n\
                     By default when --feature-network is supplied,\n\
                     features with zero edges after the QC pipeline\n\
                     (shared-neighbor prune → hub cap → k-core) are dropped from the data axis\n\
                     before projection, collapse, and training.\n\
                     Pass to keep the full feature axis\n\
                     (note: with restriction off the graph has no effect —\n\
                     the masked encoder does not diffuse)."
    )]
    no_feature_network_restrict: bool,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Per-name canonicalization across input backends",
        long_help = "How row names align across `--data-files`.\n\
                     `auto` — sniff sampled row names and pick:\n\
                     locus-overlap if ≥50% parse as `chr:start-end`,\n\
                     gene if ≥50% contain `_`, exact otherwise (default).\n\
                     `exact` — strict string match.\n\
                     `gene` — also register each `_`-split component as an alias\n\
                     (so `ENSG000_TGFB1` and `TGFB1` resolve to the same row).\n\
                     `locus` — normalize `chr1:1000-2000`, `1:1000-2000`, etc. to a canonical form.\n\
                     `locus-overlap` — same as `locus` plus cluster any intervals\n\
                     that overlap on the same chromosome\n\
                     (useful for cross-dataset ATAC peak sets called independently)."
    )]
    feature_name_kind: FeatureNameKindArg,

    #[command(flatten)]
    cnv: CnvArgs,

    #[command(flatten)]
    qc: QcArgs,
}

#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum FeatureNameKindArg {
    #[default]
    Auto,
    Exact,
    Gene,
    Locus,
    LocusOverlap,
    Mixed,
}

impl FeatureNameKindArg {
    /// Resolve to a concrete [`FeatureNameKind`], defaulting `Auto` to
    /// `Gene { delim: '_' }` — the standard for gene-keyed pre-train
    /// inputs (bge / fne / topic-family dictionaries).
    pub fn resolve_or_gene(&self) -> auxiliary_data::feature_names::FeatureNameKind {
        Option::<auxiliary_data::feature_names::FeatureNameKind>::from(self.clone())
            .unwrap_or(auxiliary_data::feature_names::FeatureNameKind::Gene { delim: '_' })
    }
}

impl From<FeatureNameKindArg> for Option<auxiliary_data::feature_names::FeatureNameKind> {
    fn from(arg: FeatureNameKindArg) -> Self {
        use auxiliary_data::feature_names::FeatureNameKind;
        match arg {
            FeatureNameKindArg::Auto => None,
            FeatureNameKindArg::Exact => Some(FeatureNameKind::Exact),
            FeatureNameKindArg::Gene => Some(FeatureNameKind::Gene { delim: '_' }),
            FeatureNameKindArg::Locus => Some(FeatureNameKind::Locus {
                merge_overlapping: false,
            }),
            FeatureNameKindArg::LocusOverlap => Some(FeatureNameKind::Locus {
                merge_overlapping: true,
            }),
            FeatureNameKindArg::Mixed => Some(FeatureNameKind::Mixed),
        }
    }
}

/// `senna masked-topic` — softmax simplex-`θ`, deterministic (no-KL) masked ETM.
pub fn fit_masked_topic_model(args: &MaskedTopicArgs) -> anyhow::Result<()> {
    fit_masked_model(args, LatentHead::Softmax)
}

/// `senna masked-sbp` — **stick-breaking process** simplex-`θ`, deterministic
/// (no-KL) masked ETM. Same pipeline as `masked-topic`; the encoder maps its
/// logits through a stick-breaking simplex instead of softmax, giving ordered,
/// exchangeability-broken topics with a self-pruning tail.
pub fn fit_masked_sbp_model(args: &MaskedTopicArgs) -> anyhow::Result<()> {
    fit_masked_model(args, LatentHead::StickBreaking)
}

/// `senna masked-vae` — Gaussian-latent (reparam + KL) masked VAE. Shares the
/// masked-topic pipeline (PB training, encoder-only cell eval, ETM ρ/α decoder);
/// the encoder emits a reparameterized `z` (no softmax) and the loss gains a KL
/// term. `exp(z)` plays the role of the per-topic intensities in the NB head.
pub fn fit_masked_vae_model(args: &MaskedTopicArgs) -> anyhow::Result<()> {
    fit_masked_model(args, LatentHead::Gaussian)
}

fn fit_masked_model(args: &MaskedTopicArgs, head: LatentHead) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let k = args.n_latent_topics;

    // --from chain-inheritance: explicit CLI flags win per-field; with no
    // explicit init/freeze flag, default to freezing the inherited ρ.
    let inherited = args
        .from
        .as_deref()
        .map(crate::run_manifest::inherit_from)
        .transpose()?;
    if let Some(inh) = inherited.as_ref() {
        info!(
            "--from: inheriting data + batch + ρ-prefix from a '{}' manifest",
            inh.source_kind
        );
    }
    let data_files = crate::run_manifest::InheritedFromManifest::resolve_data(
        inherited.as_ref(),
        &args.data_files,
    )?;
    let batch_files = crate::run_manifest::InheritedFromManifest::resolve_batch(
        inherited.as_ref(),
        args.batch_files.as_deref(),
    );

    let init_feature_embedding: Option<Box<str>> = args.init_feature_embedding.clone();
    let freeze_feature_embedding: Option<Box<str>> =
        args.freeze_feature_embedding.clone().or_else(|| {
            match (init_feature_embedding.is_none(), inherited.as_ref()) {
                (true, Some(inh)) => Some(inh.feature_embedding_prefix.clone()),
                _ => None,
            }
        });
    let pretrained_prefix = freeze_feature_embedding
        .as_deref()
        .or(init_feature_embedding.as_deref());
    let freeze_rho = freeze_feature_embedding.is_some();
    // Resolved before H so the pre-trained dictionary's column count
    // pins the encoder dim.
    let pretrained_spec: Option<crate::topic::freeze::FrozenFeatureSpec> = match pretrained_prefix {
        None => None,
        Some(prefix) => {
            if freeze_rho {
                anyhow::ensure!(
                    args.feature_network.is_none(),
                    "--freeze-feature-embedding is incompatible with --feature-network \
                     (network restriction would change the gene axis that the frozen ρ pins)"
                );
            }
            // Pre-train inputs are gene-keyed; row names aren't available yet.
            let kind = args.feature_name_kind.resolve_or_gene();
            Some(crate::topic::freeze::FrozenFeatureSpec::resolve_from_prefix(prefix, kind)?)
        }
    };

    let pretrained_h: Option<usize> = pretrained_spec
        .as_ref()
        .map(super::topic::freeze::FrozenFeatureSpec::dictionary_h)
        .transpose()?;
    let h = crate::topic::common::resolve_embedding_dim(args.embedding_dim, pretrained_h, k)?;

    let prebuilt_partition = inherited
        .as_ref()
        .map(super::run_manifest::InheritedFromManifest::load_cell_to_pb)
        .transpose()?
        .flatten();

    let effective_hvg =
        crate::hvg::resolve_multiome_with_hvg(args.multiome, data_files.len(), &args.hvg);
    let effective_multiome = effective_hvg.multiome;

    let net_opts = crate::topic::common::FeatureNetworkOpts {
        prefix_match: args.feature_network_prefix_match,
        delim: args.feature_network_delim,
        min_shared_neighbors: args.feature_network_min_shared_neighbors,
        max_degree: args.feature_network_max_degree,
        min_degree: args.feature_network_min_degree,
    };
    let restrict_path = if args.no_feature_network_restrict {
        None
    } else {
        args.feature_network.as_deref()
    };

    let feature_network = crate::topic::common::setup_feature_network(restrict_path, net_opts);
    let pretrained_mask_holder = pretrained_spec.as_ref().map(|s| s.mask_fn());
    let feature_mask_fn: Option<&crate::topic::common::FeatureMaskFn> = match (
        pretrained_mask_holder.as_deref(),
        feature_network.mask_fn.as_deref(),
    ) {
        (Some(fm), _) => Some(fm),
        (None, Some(nm)) => Some(nm),
        (None, None) => None,
    };

    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
        cell_to_pb_per_level,
        output_keep_idx,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &data_files,
        batch_files: &batch_files,
        preload: args.preload_data,
        proj_dim: args.collapse.proj_dim.max(args.n_latent_topics),
        sort_dim: args.collapse.sort_dim,
        knn_cells: args.collapse.knn_cells,
        num_levels: args.collapse.num_levels,
        iter_opt: args.collapse.iter_opt,
        block_size: args.block_size,
        out: &args.out,
        max_features: effective_hvg.n_hvg,
        feature_list_file: effective_hvg.feature_list_file,
        must_train_file: effective_hvg.must_train_file,
        qc: args.qc.to_config(),
        qc_block_size: args.block_size,
        qc_report_out: args.qc.qc_report.as_deref(),
        feature_mask_fn,
        row_alignment: data_beans::sparse_io_vector::RowAlignment::default(),
        column_alignment: if effective_multiome {
            data_beans::sparse_io_vector::ColumnAlignment::Union
        } else {
            data_beans::sparse_io_vector::ColumnAlignment::Disjoint
        },
        feature_kind: args.feature_name_kind.clone().into(),
        refine: Some(args.collapse.pb_refine.to_params()),
        ignore_batch: args.collapse.ignore_batch,
        want_hierarchy: true,
        prebuilt_partition,
    })?;

    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();

    // 4. No feature coarsening for indexed model — both encoder and decoder
    //    use indexed top-K lookup, so D_full is efficient.
    //    Levels differ only in sample coarsening (N).
    let n_features_full = data_vec.num_rows();
    let num_levels = collapsed_levels.len();

    // 5. Train masked topic model on collapsed data
    let n_topics = args.n_latent_topics;

    let dev = create_device(&args.device, args.device_no)?;

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let dec_context_size = args.decoder_context_size.unwrap_or(args.context_size);

    // Gene names — used for output artifacts further down.
    let gene_names = data_vec.row_names()?;

    // `--feature-network` on the masked path is used for feature *restriction*
    // only (applied above when building `feature_network`). GCN graph diffusion
    // is NOT supported by the masked encoder: `forward_indexed_masked` pools the
    // visible top-K by single-query attention and takes no sparse edges, so a
    // GcnBlock here would never enter the forward graph (dead weights, no
    // effect). Diffusion would have to be plumbed through the encoder-only eval
    // path too (which reads CSC directly, with no edge cache) to keep the latent
    // train/eval-consistent — out of scope for v1.
    if args.feature_network.is_some() && args.no_feature_network_restrict {
        warn!(
            "--feature-network with --no-feature-network-restrict has no effect on \
             masked-topic: graph diffusion is not supported by the masked encoder, \
             and restriction is disabled."
        );
    }

    let base_encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: h,
            layers: &args.encoder_layers,
            use_gcn: false,
            attn_pool: true,
        },
        &parameters,
        param_builder.pp("enc"),
    )?;

    // Per-level decoders: all at D_full, levels differ in N (sample coarsening).
    // ETM-factorized — each decoder shares the encoder's feature embeddings ρ,
    // and learns only its own topic embeddings α_{level} [K, H].
    let shared_rho = base_encoder.feature_embeddings().clone();
    let decoders: Vec<EmbeddedNbTopicDecoder> = (0..num_levels)
        .map(|i| {
            EmbeddedNbTopicDecoder::new(
                n_topics,
                shared_rho.clone(),
                param_builder.pp(format!("dec_{i}")),
            )
            .expect("decoder creation")
        })
        .collect();

    // Overwrite ρ in place with the pre-trained values BEFORE warm-start
    // from a prior topic checkpoint. The encoder/decoder both hold a
    // Tensor reference into the same Var, so a single `var.set(...)`
    // updates everywhere. The Var stays in the VarMap (round-trips
    // through safetensors); for freeze mode the optimizer excludes it
    // via `trainable_vars` (see `train_masked.rs`), for init mode it
    // keeps updating.
    if let Some(spec) = pretrained_spec.as_ref() {
        anyhow::ensure!(
            args.init_from.is_none(),
            "ρ pre-training is incompatible with --init-from \
             (warm-start would overwrite the pre-trained ρ from a different checkpoint)"
        );
        let host = spec.materialize(&gene_names)?;
        anyhow::ensure!(
            host.h == h,
            "pre-trained feature embedding has H={} but --embedding-dim={}",
            host.h,
            h
        );
        anyhow::ensure!(
            host.e_feat.nrows() == n_features_full,
            "pre-trained ρ has {} rows but the post-mask gene axis has {} — \
             loader and feature-mask disagree (this is a bug, please report)",
            host.e_feat.nrows(),
            n_features_full
        );
        candle_util::frozen_features::overwrite_var_2d(
            &parameters,
            "enc.feature.embeddings",
            &host.e_feat,
            &dev,
        )?;
        if freeze_rho {
            info!(
                "Freeze mode: ρ seeded from {} (D={}, H={}); encoder/decoders share frozen ρ, \
                 only α + FC + BN train",
                spec.dictionary_path, n_features_full, h
            );
        } else {
            info!(
                "Warm-start: ρ initialised from {} (D={}, H={}); AdamW continues to update it \
                 alongside α + FC + BN",
                spec.dictionary_path, n_features_full, h
            );
        }
    }

    if let Some(prefix) = args.init_from.as_deref() {
        use crate::topic::warm_start::{warm_start_load, WarmStartCheck};
        warm_start_load(
            &parameters,
            prefix,
            &WarmStartCheck {
                // Warm-start from a checkpoint of the same head — the z_mean
                // semantics differ per head, so `--init-from` must match.
                model_type_expected: crate::topic::model_metadata::masked_model_type(head),
                n_topics,
                n_features_full,
                n_features_encoder: n_features_full,
                encoder_hidden: &args.encoder_layers,
                level_decoder_dims: &vec![n_features_full; num_levels],
                embedding_dim: Some(h),
            },
        )?;
    }

    info!(
        "input: {} -> indexed encoder (emb={}, ctx={}) -> {} decoders (D={}, ctx={})",
        n_features_full, h, args.context_size, num_levels, n_features_full, dec_context_size,
    );

    // Bulk deconvolution is not supported on the masked-imputation path (v1).
    if args.bulk_data_files.is_some() {
        warn!("--bulk-data-files is ignored by the masked-topic (not yet supported)");
    }

    let stop = setup_stop_handler();

    info!("Computing NB-Fisher weights for shortlist scoring");
    let shortlist_weights: Vec<f32> =
        crate::empirical_dict::compute_nb_fisher_weights(&data_vec, args.block_size)?;

    // Per-gene mean expression rate `μ_d` from the finest-level pseudobulk
    // posterior. The encoder composes it with the per-cell batch null as a
    // multiplicative count-rate divisor before Anscombe — joint correction
    // for batch effect × gene-typical-rate, leaving the cell's biological
    // deviation. Stored as the raw mean; Anscombe is applied inside the
    // encoder via `anscombe_lite`.
    let feature_mean: Vec<f32> = {
        let mu = finest_collapsed.mu_observed.posterior_mean();
        let n_pb = mu.ncols().max(1) as f32;
        (0..n_features_full)
            .map(|d| mu.row(d).iter().sum::<f32>() / n_pb)
            .collect()
    };
    // Anchor-prior CE penalty: Gram-Schmidt anchors selected on the
    // finest-level pseudobulks give each topic k a per-gene prior simplex
    // (rows of `[K, D]`), which we cross-entropy against
    // `β_kd = log_softmax_d(α · ρᵀ)` at every minibatch. This is the
    // only K-way symmetry-breaking force on the ETM-factorized β —
    // without it the model collapses onto one or two dominant topics.
    let anchor_tensors: Option<Vec<candle_core::Tensor>> = if args.anchor_penalty > 0.0 {
        info!("Building anchor prior (K={n_topics}) from finest pseudobulks");
        let anchor_prior = crate::topic::anchor_prior::AnchorPrior::from_pseudobulk(
            finest_collapsed,
            n_topics,
            None,
        )?;
        // Indexed decoders all run at D_full — no per-level feature coarsening.
        let level_coarsenings_none: Vec<
            Option<data_beans_alg::feature_coarsening::FeatureCoarsening>,
        > = (0..num_levels).map(|_| None).collect();
        Some(anchor_prior.per_level_device_tensors(&level_coarsenings_none, &dev)?)
    } else {
        None
    };

    let train_config = IndexedTrainConfig {
        parameters: &parameters,
        dev: &dev,
        epochs: args.epochs,
        minibatch_size: args.minibatch_size,
        learning_rate: args.learning_rate,
        topic_smoothing: args.topic_smoothing,
        enc_context_size: args.context_size,
        dec_context_size,
        stop: &stop,
        shortlist_weights: &shortlist_weights,
        feature_mean: &feature_mean,
        feature_fisher_weights: &shortlist_weights,
        grad_clip: args.grad_clip,
        feature_graph: None,
        feature_embedding_l2: args.feature_embedding_l2,
        weight_decay: args.weight_decay,
        frozen_feature_var: if freeze_rho {
            Some("enc.feature.embeddings")
        } else {
            None
        },
        anchor_prior_per_level: anchor_tensors.as_deref(),
        anchor_penalty: args.anchor_penalty,
    };

    use candle_util::vae::masked_topic::{MaskSchedule, MaskedTrainOpts};
    let masked_opts = MaskedTrainOpts {
        mask_schedule: match args.mask_schedule {
            MaskScheduleArg::Fixed => MaskSchedule::Fixed,
            MaskScheduleArg::Uniform => MaskSchedule::Uniform {
                lo: args.mask_rate_lo,
                hi: args.mask_rate_hi,
            },
        },
        likelihood: args.masked_likelihood.to_lib(),
        latent: head,
        kl_weight: args.kl_weight,
    };

    let scores = train_masked(
        &collapsed_levels,
        &base_encoder,
        &decoders,
        &train_config,
        args.mask_fraction,
        &masked_opts,
    )?;

    info!("Writing down the model parameters");

    // Use finest-level decoder for output
    let finest_decoder = decoders.last().unwrap();
    write_masked_dictionary(finest_decoder, &gene_names, &args.out)?;
    write_feature_embedding(base_encoder.feature_embeddings(), &gene_names, &args.out)?;

    // Optional held-out masked-imputation evaluation — the un-optimized
    // generalization metric (see `--eval-mask-fraction`). Runs on the training
    // device with the in-memory encoder + finest decoder, before the CPU move.
    if args.eval_mask_fraction > 0.0 {
        use crate::topic::eval_indexed::{evaluate_holdout_imputation, HoldoutEvalConfig};
        let delta_train = match args.adj_method {
            AdjMethod::Batch => finest_collapsed.delta.as_ref(),
            AdjMethod::Residual => finest_collapsed.mu_residual.as_ref(),
        }
        .map(|x| {
            x.posterior_mean()
                .to_tensor(&dev)
                .expect("delta to tensor")
                .transpose(0, 1)
                .expect("transpose")
                .contiguous()
                .expect("contiguous")
        });
        let holdout_cfg = HoldoutEvalConfig {
            dev: &dev,
            adj_method: &args.adj_method,
            minibatch_size: args.minibatch_size,
            enc_context_size: args.context_size,
            shortlist_weights: &shortlist_weights,
            feature_mean: &feature_mean,
            head,
            likelihood: args.masked_likelihood.to_lib(),
            topic_smoothing: args.topic_smoothing,
            mask_fraction: args.eval_mask_fraction,
            seed: args.eval_seed,
        };
        let holdout_llik = evaluate_holdout_imputation(
            &data_vec,
            &base_encoder,
            finest_decoder,
            &holdout_cfg,
            delta_train.as_ref(),
        )?;
        info!(
            "Held-out imputation: mean log-likelihood/gene = {holdout_llik:.4} \
             (mask={}, seed={})",
            args.eval_mask_fraction, args.eval_seed
        );
    }

    // Persist trainable weights, model architecture, and shortlist weights
    // so `senna predict` (and `--init-from` re-runs) can rebuild this model.
    use crate::topic::model_metadata::{
        masked_decoder_type, masked_model_type, save_feature_mean, save_parameters,
        save_shortlist_weights, TopicModelMetadata,
    };

    // No feature graph is persisted on the masked path (GCN diffusion is not
    // wired into the masked encoder — see the encoder construction above).
    save_parameters(&parameters, &args.out)?;
    let mut metadata = TopicModelMetadata {
        model_type: masked_model_type(head).into(),
        decoder_types: vec![masked_decoder_type(head).into()],
        decoder_weights: vec![1.0],
        n_features_encoder: n_features_full,
        n_features_full,
        n_topics,
        encoder_hidden: args.encoder_layers.clone(),
        num_levels,
        level_decoder_dims: vec![n_features_full; num_levels],
        adj_method: args.adj_method.as_str().into(),
        has_coarsening: false,
        embedding_dim: Some(h),
        enc_context_size: Some(args.context_size),
        dec_context_size: Some(dec_context_size),
        theta_mean: None,
        n_train_cells: Some(data_vec.num_columns()),
    };
    metadata.save(&args.out)?;
    save_shortlist_weights(&shortlist_weights, &gene_names, &args.out)?;
    save_feature_mean(&feature_mean, &gene_names, &args.out)?;

    // Move VarMap to CPU, then rebuild the encoder from the CPU Vars. The
    // masked model's inference is encoder-only (no decoder / no refinement).
    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    move_varmap_to_cpu(&parameters)?;

    let cpu_vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);
    let cpu_encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: h,
            layers: &args.encoder_layers,
            use_gcn: false,
            attn_pool: true,
        },
        &parameters,
        cpu_vb.pp("enc"),
    )?;

    info!("Writing down the latent states");
    // Residual/batch correction tensor at full D (encoder operates at D_full).
    let delta = match args.adj_method {
        AdjMethod::Batch => finest_collapsed.delta.as_ref(),
        AdjMethod::Residual => finest_collapsed.mu_residual.as_ref(),
    }
    .map(|x| {
        x.posterior_mean()
            .to_tensor(&cpu_dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
            .contiguous()
            .expect("contiguous")
    });
    let eval_config = EvaluateLatentMaskedConfig {
        dev: &cpu_dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size,
        enc_context_size: args.context_size,
        shortlist_weights: &shortlist_weights,
        feature_mean: &feature_mean,
        head,
    };
    let z_nk = evaluate_latent_masked(&data_vec, &cpu_encoder, &eval_config, delta.as_ref(), None)?;

    metadata.populate_theta_mean_and_save(&z_nk, &args.out)?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    let cell_names = data_vec.column_names()?;

    crate::output_helpers::save_latent(&args.out, &z_nk, &cell_names, output_keep_idx.as_deref())?;

    // PB aggregates + the empirical dictionary, exactly as `senna topic` writes them.
    //
    // `topic` gets θ_PB by running the encoder's forward pass on the collapsed pseudobulk. The
    // indexed encoder's PB-level pass is not wired, and it does not need to be: θ is already known
    // per cell, and the finest collapsing already says which pseudobulk each cell belongs to — so
    // θ_PB is just the within-PB mean of it. That is the same quantity, without a second encoder
    // pass on data the encoder never saw in that shape.
    //
    // The empirical dictionary is the **model-free** counterpart to `dictionary.parquet`. The
    // latter is factorized — β = log_softmax(ρ·αᵀ) — so it can only say what the H-dimensional
    // embedding can express. This one is the plain NB-Fisher-weighted average of each topic's
    // pseudobulk expression, and depends on the factorization only through θ.
    //
    // Neither dominates, and this is not the "sharper" of the two — the softmax over ρ·αᵀ makes the
    // factorized β the more peaked one. It is an *independent* estimate, so having both lets a
    // caller cross-check a topic against something the model did not itself construct. It is also
    // the same object `senna topic` writes, so consumers that already prefer it (`plot-topic` does
    // `dictionary_empirical.or(dictionary)`) behave the same across every topic-family run instead
    // of silently falling back here.
    {
        let pb_gene_gp: Mat = finest_collapsed.mu_observed.posterior_mean().clone();
        crate::output_helpers::save_pb_gene(&args.out, &pb_gene_gp, &gene_names)?;

        if let Some(c2p) = cell_to_pb_per_level.as_ref().and_then(|l| l.last()) {
            let n_pb = pb_gene_gp.ncols();
            let k = z_nk.ncols();

            // θ_PB[p, ·] = mean over the cells collapsed into pseudobulk p. `z_nk` is log θ.
            let mut pb_latent_pk = Mat::zeros(n_pb, k);
            let mut n_in_pb = vec![0f32; n_pb];
            for (n, &p) in c2p.iter().enumerate() {
                if p < n_pb && n < z_nk.nrows() {
                    n_in_pb[p] += 1.0;
                    for kk in 0..k {
                        pb_latent_pk[(p, kk)] += z_nk[(n, kk)].exp();
                    }
                }
            }
            for (p, &cnt) in n_in_pb.iter().enumerate() {
                if cnt > 0.0 {
                    for kk in 0..k {
                        pb_latent_pk[(p, kk)] /= cnt;
                    }
                }
            }

            let pb_names = axis_id_names("PB_", n_pb);
            let topic_names = axis_id_names("T", k);
            pb_latent_pk.to_parquet_with_names(
                &format!("{}.pb_latent.parquet", args.out),
                (Some(&pb_names), Some("pb")),
                Some(&topic_names),
            )?;

            let beta_emp = crate::empirical_dict::build_empirical_dictionary(
                &pb_gene_gp,
                &pb_latent_pk,
                &shortlist_weights,
            );
            beta_emp.to_parquet_with_names(
                &format!("{}.dictionary_empirical.parquet", args.out),
                (Some(&gene_names), Some("gene")),
                Some(&topic_names),
            )?;
            info!(
                "Wrote empirical dictionary {}×{} (NB-Fisher-weighted, column-simplex) + \
                 pb_latent {}×{}",
                beta_emp.nrows(),
                beta_emp.ncols(),
                n_pb,
                k,
            );
        } else {
            warn!(
                "no cell→pseudobulk map, so no θ_PB and no empirical dictionary — \
                 {}.dictionary_empirical.parquet is not written and downstream falls back to the \
                 factorized dictionary.parquet",
                args.out
            );
        }
    }

    // CNV detection using topic proportions
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

    crate::postprocess::viz_prep::write_cell_proj(
        &args.out,
        &proj_kn,
        &cell_names,
        output_keep_idx.as_deref(),
    )?;
    let has_cell_to_pb = if let Some(ref c2p) = cell_to_pb_per_level {
        crate::postprocess::viz_prep::write_cell_to_pb(
            &args.out,
            c2p,
            &cell_names,
            output_keep_idx.as_deref(),
        )?;
        true
    } else {
        false
    };

    let input: Vec<String> = data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch: Vec<String> = batch_files
        .as_ref()
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::run_manifest::RunKind::Itopic,
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: true,
        has_cell_proj: true,
        pb_gene_suffix: Some("pb_gene.parquet"),
        pb_latent_suffix: Some("pb_latent.parquet"),
        // Full-gene-resolution β̂, the same object `senna topic` writes. Preferred over the
        // factorized `dictionary.parquet` by anything that ranks genes (plot-topic already does
        // `dictionary_empirical.or(dictionary)`), because the factorization flattens rare genes.
        dictionary_empirical_suffix: Some("dictionary_empirical.parquet"),
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        cell_embedding_suffix: None,
        default_colour_by: "cluster",
        has_latent: true,
        has_cell_to_pb,
    })?;

    info!("Done");
    Ok(())
}
