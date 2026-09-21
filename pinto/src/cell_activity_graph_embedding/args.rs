//! CLI arguments for `pinto cage`.

use crate::cell_activity_graph_embedding::feature_gating::ActivityNorm;
use clap::{Parser, ValueEnum};
use data_beans::alg::hvg::HvgCliArgs;
use data_beans::aux::feature_names::FeatureNameKind;

use crate::util::device::ComputeDevice;

/// CLI mirror of [`graph_embedding_util::loss::NceObjective`] (that crate keeps
/// its model enums clap-free, so every consumer carries its own wrapper — see
/// `senna/src/bge/mod.rs` and `faba/src/gem/common.rs`).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum NceObjectiveArg {
    /// Per-pair logistic (SGNS): each (positive, negative) pair decided
    /// independently — cage's historical loss.
    Logistic,
    /// Sampled-softmax / InfoNCE: the negatives compete with the positive in one
    /// softmax. The default here and in `senna bge` / `senna gem`.
    Softmax,
}

impl NceObjectiveArg {
    pub fn to_ge(self) -> graph_embedding_util::loss::NceObjective {
        match self {
            Self::Logistic => graph_embedding_util::loss::NceObjective::Logistic,
            Self::Softmax => graph_embedding_util::loss::NceObjective::Softmax,
        }
    }
}

/// Row-name canonicalization strategy for matching the data's feature
/// names against external resources (PPI networks, marker lists,
/// pretrained feature embeddings). `Auto` sniffs the first data file's
/// row names and dispatches to [`FeatureNameKind::auto_detect`] —
/// gene-symbol-style names (`ENSG..._SYMBOL`) get the `Gene` rule
/// applied automatically.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum FeatureNameMode {
    /// Peek the first file's row names and pick `Exact` / `Gene` /
    /// `Locus` / `Mixed` via [`FeatureNameKind::auto_detect`].
    Auto,
    /// Strict string match — no canonicalization. The historical pinto
    /// default for `lc` / `svd`.
    Exact,
    /// `Gene { delim: '_' }`: gene-symbol rows — register every `_`-split
    /// component as an alias of the full row name.
    Gene,
    /// `Locus { merge_overlapping: true }`: normalize chrom-coord names
    /// and collapse overlapping intervals.
    Locus,
    /// Heterogeneous axis: dispatch per row name.
    Mixed,
}

impl FeatureNameMode {
    /// Resolve to a concrete [`FeatureNameKind`]. `peek_names` is only
    /// consulted under `Auto`; other modes ignore it.
    pub fn resolve_kind(self, peek_names: &[Box<str>]) -> FeatureNameKind {
        match self {
            FeatureNameMode::Auto => FeatureNameKind::auto_detect(peek_names),
            FeatureNameMode::Exact => FeatureNameKind::Exact,
            FeatureNameMode::Gene => FeatureNameKind::Gene { delim: '_' },
            FeatureNameMode::Locus => FeatureNameKind::Locus {
                merge_overlapping: true,
            },
            FeatureNameMode::Mixed => FeatureNameKind::Mixed,
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub struct CellActivityGraphEmbeddingArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,

    #[arg(
        long,
        default_value_t = FeatureNameMode::Auto,
        value_enum,
        help = "Feature-name canonicalization for matching external resources",
        long_help = "Row-name canonicalization strategy:\n\
                     auto  — peek row names and auto-detect (default)\n\
                     exact — strict string equality (pinto lc / svd default)\n\
                     gene  — gene symbols; split on '_' so 'ENSG..._TGFB1' and 'TGFB1' alias\n\
                     locus — normalize chrom-coord names; collapse overlaps mixed —\n\
                     per-row dispatch (RNA+ATAC paired axes)"
    )]
    #[arg(alias = "gene-name-mode")]
    pub feature_name_mode: FeatureNameMode,

    #[arg(
        long,
        help = "Skip the degree-corrected Poisson refinement of the coarsening levels",
        long_help = "Each coarsening level gets a second-opinion refinement.\n\
                     It runs on RAW counts, degree-corrected Poisson.\n\
                     It is the same pass `pinto lc` runs. Without it,\n\
                     levels are cut on cosine-of-projection alone,\n\
                     which ignores depth and over-dispersion in the counts.\n\
                     \n\
                     Set this to skip that pass.\n\
                     The context build reads the count matrix once more up front,\n\
                     so this is the lever if that I/O matters.",
        hide = true
    )]
    pub no_dc_poisson: bool,

    #[arg(
        long,
        value_name = "H",
        default_value_t = graph_embedding_util::EmbeddingDim::Auto,
        value_name = "H|auto",
        help = "Cell embedding dimensionality (auto = a pinned dictionary's width, else 16)",
        long_help = "Cell embedding dimensionality. auto is the default.\n\
                     With --feature-embedding under freeze, free or lora the rows are\n\
                     installed verbatim, so auto is the dictionary's width and a\n\
                     given width must agree with it.\n\
                     Otherwise auto is 16; the adapt mode maps the dictionary into\n\
                     this run's own width."
    )]
    pub embedding_dim: graph_embedding_util::EmbeddingDim,

    #[arg(
        long,
        default_value_t = 100,
        help = "Training epochs over the feature axis (early-stops on --convergence-tol)",
        long_help = "Passes over the feature axis.\n\
                     \n\
                     The run early-stops once the loss flattens, per\n\
                     --convergence-tol over --convergence-window.\n\
                     A high value here is a ceiling, not a fixed cost.\n\
                     \n\
                     Pair with --features-per-epoch to cap per-epoch cost."
    )]
    pub epochs: usize,

    #[arg(
        long,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Features per outer parallel sampling chunk",
        long_help = "The outer loop samples this many features in parallel via rayon.\n\
                     Forward and backward then run serially. candle Var is not parallel-safe.\n\
                     \n\
                     This is ALSO the SGD minibatch, not just a parallel width.\n\
                     An epoch takes trainable-features / this many optimizer steps,\n\
                     each paying a forward, a backward, a gradient clip and an\n\
                     AdamW update over the whole feature table.\n\
                     So it sets both wall-clock and how many parameter updates\n\
                     a run performs, and it is not fit-neutral:\n\
                     raising 64 to 2048 cut wall-clock several fold AND raised\n\
                     spatial coherence, at an unchanged sampling budget.\n\
                     \n\
                     Below ~this many trainable features an epoch is ONE step,\n\
                     so a small panel may want a smaller value here\n\
                     or more --epochs.\n\
                     \n\
                     Unset, the default is 2048 on CPU.\n\
                     On CUDA the size is chosen automatically:\n\
                     a short probe measures the memory one step retains,\n\
                     and grows the chunk\n\
                     while it fits --gpu-mem-fraction of free device memory,\n\
                     never past 2048.\n\
                     The coherence result above was validated at 2048;\n\
                     a memory-constrained device that resolves lower\n\
                     trades some of that benefit for fitting at all.\n\
                     Passing a value disables the probe and always wins."
    )]
    #[arg(alias = "gene-batch-size")]
    pub feature_batch_size: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.6,
        help = "Fraction of free GPU memory the training chunk may target",
        long_help = "Ceiling for the automatic chunk sizing on CUDA.\n\
                     The probe grows the chunk\n\
                     while one step's retained memory,\n\
                     with half reserved for the backward pass,\n\
                     fits this fraction of the device memory free at start.\n\
                     Fractions outside 0.05 to 0.95 are clamped to that range.\n\
                     Ignored on CPU and when --feature-batch-size is set."
    )]
    pub gpu_mem_fraction: f32,

    #[arg(
        long,
        default_value_t = 12,
        help = "Positive super-edge draws per (feature, batch) sample",
        long_help = "Every feature draws this many positive SUPER EDGES\n\
                     per experimental batch each epoch, with replacement:\n\
                     a batch's per-feature pool is tens of super edges,\n\
                     so repeated draws are by design.\n\
                     Over-sampling does not merely cost time,\n\
                     it can DEGRADE the fit\n\
                     (measured under the former cell-level trainer:\n\
                     cutting the budget ~20x raised spatial coherence).\n\
                     Raise it only if the fit looks under-trained,\n\
                     and check the coherence rather than the loss.\n\
                     \n\
                     Kept per-feature so the budget tracks the feature axis.\n\
                     --positives-per-epoch overrides it with an absolute total."
    )]
    pub per_feature_batch: usize,

    #[arg(
        long,
        value_name = "N",
        help = "Total positive edges drawn per epoch, across all features (unset = auto)",
        long_help = "The epoch's total SUPER-EDGE sampling budget.\n\
                     \n\
                     Divided evenly: each feature draws\n\
                     N / (trainable features x batches) positives per batch.\n\
                     Unset keeps the historical --per-feature-batch instead.\n\
                     Each positive carries\n\
                     1 + --n-negatives x --chain-levels scores.\n\
                     \n\
                     This is the knob for how much data an epoch sees,\n\
                     and it is the one that moved the fit.\n\
                     Pair it with --feature-batch-size: the budget sets work per\n\
                     step, that sets how many steps an epoch takes.\n\
                     \n\
                     --features-per-epoch is the coarse alternative:\n\
                     it drops features rather than sampling each one less.",
        hide = true
    )]
    pub positives_per_epoch: Option<usize>,

    #[arg(
        long,
        default_value_t = 8,
        help = "Sibling-PB negatives drawn per positive super edge per chain level",
        hide = true
    )]
    pub n_negatives: usize,

    #[arg(
        long,
        default_value_t = 0.75,
        help = "Negative-degree exponent (power-of-degree negative sampling)",
        hide = true
    )]
    pub alpha_neg: f32,

    #[arg(long, default_value_t = 5e-3, help = "AdamW learning rate")]
    pub lr: f32,

    #[arg(
        long,
        default_value_t = ActivityNorm::Log1p,
        value_enum,
        help = "Per-feature activity normalization",
        hide = true,
    )]
    pub activity_norm: ActivityNorm,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Exponent on within-feature positive-edge weights a_g[u]·a_g[v]",
        long_help = "Stage-2 coverage exponent, one axis down from bge's alpha_pb.\n\
                     Positive edges within a feature are drawn with probability ∝ (a_g[u]·a_g[v])^activity-alpha.\n\
                     The default of 1.0 keeps the activity-proportional draw.\n\
                     0.0 makes every active edge of a feature equally likely,\n\
                     so no high-activity hub pair dominates that feature.",
        hide = true
    )]
    pub activity_alpha: f32,

    #[arg(
        long,
        help = "Disable NB-Fisher per-feature precision weighting of the loss",
        long_help = "Each feature's contribution to the loss is down-weighted.\n\
                     The weight is its NB Fisher-info w_g ∈ (0,1]. High-mean,\n\
                     high-dispersion housekeeping features go toward 0,\n\
                     and informative low-mean features go toward 1. This matches `pinto lc` and `senna bge`.\n\
                     Set this flag to train every feature at equal weight.",
        hide = true
    )]
    pub no_fisher_weights: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Features visited per epoch; 0 = the whole axis",
        long_help = "Cost lever. cage walks the feature axis once per epoch.\n\
                     Runtime is therefore linear in the feature count.\n\
                     This caps how many are VISITED per epoch.\n\
                     A fresh random subset is drawn each time.\n\
                     \n\
                     That is stochastic coverage, NOT feature selection.\n\
                     Every feature stays on the trained axis. It keeps its sampled loading.\n\
                     It appears in every output table.\n\
                     A feature left out simply waits for a later epoch. Contrast --n-hvg,\n\
                     which weights the projection. That likewise drops nobody."
    )]
    #[arg(alias = "genes-per-epoch")]
    pub features_per_epoch: usize,

    #[arg(
        long,
        default_value_t = 0.0625,
        help = "L2 penalty λ on the shared cell and feature embeddings; 0 = off",
        long_help = "L2 penalty λ on E_pb ∈ ℝ^{P×D} and E_feature ∈ ℝ^{G×D}.\n\
                     It adds λ · (mean_n ‖e_n‖² + mean_g ‖e_g‖²) to the loss:\n\
                     a sum over the D latent dims, averaged over rows.\n\
                     The row-mean keeps λ scale-invariant across N and G, and\n\
                     summing over D rather than averaging keeps it invariant to\n\
                     --embedding-dim too — see loss::embedding_ridge.\n\
                     The default 0.0625 is the shrinkage cage was tuned at\n\
                     (~40% off the free feature-embedding norm); it is 1/16 only\n\
                     because the penalty used to be divided by D and D defaulted\n\
                     to 16. It now means the same thing at every D.\n\
                     Useful range 0.01-0.25. Do NOT reach for 1.0: measured on\n\
                     a 10.9k-cell 18k-feature sample it drives both embeddings to\n\
                     zero. A dense every-row penalty competes with a SPARSE data\n\
                     gradient under Adam's per-parameter normalization, so it\n\
                     bites far harder than its size against the loss suggests.",
        hide = true
    )]
    pub embedding_l2: f32,

    #[arg(
        long,
        value_delimiter(','),
        help = "Chain levels from the coarsening hierarchy.\nUnset: every level coarser than the finest, up to three.\nExplicit levels must all be coarser than the finest,\nwhich is the trained unit itself.",
        long_help = "Chain levels from the coarsening hierarchy,\n\
                     coarsest first.\n\
                     Unset, the default takes every level coarser than\n\
                     the finest, up to three of them,\n\
                     so it adapts to --num-levels.\n\
                     Explicit levels must all be coarser than the finest\n\
                     level: the finest IS the trained unit,\n\
                     so no positive super edge can share one."
    )]
    pub chain_levels: Option<Vec<usize>>,

    /// HVG selection: senna-style shared CLI (`--n-hvg`,
    /// `--feature-list-file`). cage **weights the random projection** with it,
    /// exactly as `senna bge` and `senna gem` do — non-selected features get
    /// projection weight 0 and so sit out the basis the coarsening hierarchy is
    /// built from, but they stay on the trained axis. The selection shapes
    /// *where the pseudobulks land*, not *which features the model may use*.
    /// `--n-hvg 0` disables. Use `--features-per-epoch` for the cost lever a hard
    /// subset used to provide.
    #[command(flatten)]
    pub hvg: HvgCliArgs,

    #[arg(
        long,
        default_value_t = 0,
        help = "Window (epochs) for convergence check; 0 disables",
        long_help = "After each epoch, look at the recent mean losses.\n\
                     The window is `convergence-window` epochs wide.\n\
                     Stop training when their (max − min) / |mean| falls below\n\
                     --convergence-tol.\n\
                     Pass 0 to run all --epochs unconditionally.",
        hide = true
    )]
    pub convergence_window: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Relative-range threshold over --convergence-window for stopping",
        hide = true
    )]
    pub convergence_tol: f32,

    /// How the pair latent becomes link communities. Shared verbatim with
    /// `dsvd` — the k-means fallback width is `--embedding-dim` here.
    #[command(flatten)]
    pub edge_clustering: crate::util::edge_clustering::EdgeClusterArgs,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Clip the global gradient norm to this before each step; 0 = off",
        long_help = "Global-L2-norm gradient clipping for the training loop.\n\
                     Gradients are scaled to this norm when they exceed it,\n\
                     which bounds the update without turning it.\n\
                     A step whose global norm is not finite is skipped.\n\
                     Pass 0 to disable clipping.",
        hide = true
    )]
    pub grad_clip: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Ridge λ on the per-pair latent; saved with the pair encoder",
        long_help = "Gaussian prior strength λ on the per-pair latent e_uv.\n\
                     The pair encoder is trained under it,\n\
                     the exact per-pair solve it is checked against carries it,\n\
                     and it is saved with the encoder,\n\
                     so `pinto predict` and `pinto impute` use the model's value.\n\
                     The log-partition is summed over every feature,\n\
                     so this is a mild prior, not the only bound on the fit.\n\
                     The per-pair intercept is never penalized."
    )]
    pub pair_ridge: f32,

    #[arg(
        long,
        default_value_t = 8192,
        help = "Cells per read block and cell pairs per placement pass (memory only)",
        hide = true
    )]
    pub pair_block: usize,

    #[arg(
        long = "nce-objective",
        default_value_t = NceObjectiveArg::Softmax,
        value_enum,
        help = "NCE objective: softmax or logistic",
        long_help = "NCE objective. softmax is InfoNCE, where negatives compete.\n\
                     It is sharper on dense data, and is the default.\n\
                     logistic is per-pair SGNS, cage's historical loss.",
        hide = true
    )]
    pub nce_objective: NceObjectiveArg,

    #[arg(
        long,
        help = "Pre-trained feature x H embedding parquet to start the feature side from",
        long_help = "Path to a pre-trained feature x H embedding parquet.\n\
                     Row column 0 holds the feature name; value columns are the dimensions.\n\
                     Feed a RAW dictionary: a run's feature_embedding.parquet.\n\
                     Co-embedding outputs (feature_coembedding.parquet) are not\n\
                     dictionaries and are rejected.\n\
                     Features are matched under --feature-name-mode.\n\
                     A feature with no dictionary row is seeded from the matched feature\n\
                     with the most similar count profile and listed in\n\
                     {out}.feature_embedding_init.parquet. Under freeze, free and lora such\n\
                     rows train; under adapt a seeded feature follows its seed through\n\
                     the shared map until --feature-adapter-residual gives it its own\n\
                     correction."
    )]
    #[arg(alias = "gene-embedding")]
    pub feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        requires = "feature_embedding",
        help = "Optional per-feature bias parquet ([D, 1]) paired with --feature-embedding;\n\
                features without a row get bias 0"
    )]
    #[arg(alias = "gene-embedding-bias")]
    pub feature_embedding_bias: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = FeatureEmbeddingMode::Adapt,
        requires = "feature_embedding",
        help = "What training may do to the pre-trained feature embedding",
        long_help = "What training may do to the pre-trained feature embedding.\n\
                     \n\
                     adapt keeps the dictionary fixed and trains one\n\
                     shared linear map on top of it, so every feature's gradient\n\
                     updates the same few parameters.\n\
                     The dictionary width and --embedding-dim may differ.\n\
                     Add --feature-adapter-residual for a per-feature correction\n\
                     where the shared map is not enough.\n\
                     \n\
                     freeze keeps every dictionary-matched row fixed at its loaded value.\n\
                     Neighbor-seeded rows still train.\n\
                     Requires the dictionary width to equal --embedding-dim.\n\
                     The e_feat ridge is skipped: a fixed table needs no shrinkage.\n\
                     \n\
                     free initializes from the dictionary and then trains every row.\n\
                     Also requires the widths to match. This is the fallback\n\
                     when the shared map underfits.\n\
                     \n\
                     lora keeps every dictionary-matched row fixed, as freeze does,\n\
                     and trains a low-rank residual on top of those rows:\n\
                     row_g = dictionary_g + u_g · V, with u_g per feature (--lora-rank numbers)\n\
                     and V shared by every matched feature, at the LoRA+ rate\n\
                     (--lora-lr-ratio) and under --lora-ridge.\n\
                     Neighbor-seeded rows still train. Requires the widths to match.\n\
                     The written feature embedding carries the residual folded in."
    )]
    #[arg(alias = "gene-embedding-mode")]
    pub feature_embedding_mode: FeatureEmbeddingMode,

    /// `--lora-rank`, `--lora-lr-ratio`, `--lora-ridge`; read under
    /// `--feature-embedding-mode lora` only.
    #[command(flatten)]
    pub lora: graph_embedding_util::LoraArgs,

    #[arg(
        long,
        requires = "feature_embedding",
        help = "adapt only: add a ridge-shrunk per-feature correction\n\
                on top of the shared map"
    )]
    #[arg(alias = "gene-adapter-residual")]
    pub feature_adapter_residual: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = FeatureInitMode::Membership,
        requires = "feature_embedding",
        help = "How a feature with no dictionary row starts",
        long_help = "How a feature with no row in --feature-embedding starts.\n\
                     \n\
                     membership places it through the dictionary's learned modules:\n\
                     its membership is the similarity-weighted mean of the closest\n\
                     matched features' memberships (by count profile), and its row is\n\
                     that membership times the module dictionary, with no residual.\n\
                     Needs {stem}.module_membership.parquet and\n\
                     {stem}.module_dictionary.parquet beside the dictionary; without\n\
                     them it falls back to neighbor and says so in\n\
                     {out}.feature_embedding_init.parquet.\n\
                     \n\
                     neighbor copies the row of the single closest matched feature."
    )]
    #[arg(alias = "gene-init-mode")]
    pub feature_init_mode: FeatureInitMode,

    #[arg(
        long,
        default_value_t = graph_embedding_util::transfer::DEFAULT_INIT_NEIGHBOURS,
        value_name = "K",
        requires = "feature_embedding",
        help = "membership init: matched features whose memberships are averaged"
    )]
    #[arg(alias = "gene-init-neighbours")]
    pub feature_init_neighbours: usize,

    #[arg(
        long,
        default_value_t = graph_embedding_util::transfer::DEFAULT_SIMILARITY_FLOOR,
        value_name = "S",
        requires = "feature_embedding",
        help = "membership init: below this best profile similarity a feature takes the diffuse prior"
    )]
    #[arg(alias = "gene-init-similarity-floor")]
    pub feature_init_similarity_floor: f32,

    /// The `--feature-modules` flag group (see `graph_embedding_util::FeatureModuleArgs`).
    /// Cage has no feature-negative NCE, so the within-module negatives do not apply
    /// here; the composition, the exact pseudobulk–module term, the feature dropout
    /// and the warm start do. The residual takes `--embedding-l2` like every other
    /// feature-side table in cage.
    #[command(flatten)]
    pub modules: graph_embedding_util::FeatureModuleArgs,
}

/// How a feature with no dictionary row is initialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FeatureInitMode {
    /// Through the dictionary's learned modules (falls back to `Neighbor`
    /// without module tables).
    Membership,
    /// The closest matched feature's row.
    Neighbor,
}

/// What training may do to a pre-trained feature embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FeatureEmbeddingMode {
    /// Fixed dictionary + one shared trainable map.
    Adapt,
    /// Dictionary-matched rows stay fixed; neighbor-seeded rows train.
    Freeze,
    /// Initialize from the dictionary, then train every row.
    Free,
    /// Dictionary-matched rows stay fixed under a low-rank residual;
    /// neighbor-seeded rows train.
    Lora,
}

/// The width when neither the flag nor a pinned dictionary decides it.
pub const DEFAULT_EMBEDDING_DIM: usize = 16;

impl CellActivityGraphEmbeddingArgs {
    /// The rules between `--feature-embedding-mode` and the flags only one mode
    /// reads, checked before any data is opened.
    pub fn validate_feature_embedding(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !(self.feature_adapter_residual
                && self.feature_embedding_mode != FeatureEmbeddingMode::Adapt),
            "--feature-adapter-residual is the adapter's per-feature correction and only \
             --feature-embedding-mode adapt trains one; under another mode the flag \
             would be read and ignored. Drop it, or use the adapt mode."
        );
        self.lora.refuse_unless_selected(
            self.feature_embedding_mode == FeatureEmbeddingMode::Lora,
            "--feature-embedding-mode lora",
        )
    }

    /// The embedding width this run trains at: the flag, else a pinned
    /// dictionary's width (`dictionary_width`, when `--feature-embedding` is
    /// given), else the default. A pinned dictionary (every mode but adapt)
    /// installs its rows verbatim, so a flag that disagrees with it is
    /// refused; the adapter maps into its own width. The LoRA rank is checked
    /// against the resolved width here, the one place it is known.
    pub fn resolve_embedding_dim(&self, dictionary_width: Option<usize>) -> anyhow::Result<usize> {
        self.validate_feature_embedding()?;
        let pinned = dictionary_width.filter(|_| {
            self.feature_embedding.is_some()
                && self.feature_embedding_mode != FeatureEmbeddingMode::Adapt
        });
        let dim = self
            .embedding_dim
            .resolve(pinned)?
            .unwrap_or(DEFAULT_EMBEDDING_DIM);
        if self.feature_embedding_mode == FeatureEmbeddingMode::Lora {
            graph_embedding_util::PresetMode::Lora(self.lora.spec()).validate(dim)?;
        }
        Ok(dim)
    }
}
