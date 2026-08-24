//! CLI arguments for `pinto cage`.

use crate::cell_activity_graph_embedding::gene_gating::ActivityNorm;
use auxiliary_data::feature_names::FeatureNameKind;
use clap::{Parser, ValueEnum};
use data_beans_alg::hvg::HvgCliArgs;

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

/// Which selector supplies the per-(gene, dim) inclusion weights.
///
/// Both arms land on the same `GateKind::Identity` slot in geu's model, but by
/// different routes: `learned` installs the variational gate and lets SGD fit
/// `σ(S/τ)`, while `sampled` installs a `pip` from cage's Gibbs Poisson sampler,
/// which then takes over the slot entirely (`model/gate.rs:503-511`).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum GateMode {
    /// Variational spike-and-slab gate trained by SGD alongside the embedding.
    Learned,
    /// Gibbs profiled-Poisson sampler over super-cell counts, installed as a
    /// per-epoch Bernoulli mask.
    Sampled,
}

/// Row-name canonicalization strategy for matching the data's gene
/// names against external resources (PPI networks, marker lists,
/// pretrained gene embeddings). `Auto` sniffs the first data file's
/// row names and dispatches to [`FeatureNameKind::auto_detect`] —
/// gene-symbol-style names (`ENSG..._SYMBOL`) get the `Gene` rule
/// applied automatically.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum GeneNameMode {
    /// Peek the first file's row names and pick `Exact` / `Gene` /
    /// `Locus` / `Mixed` via [`FeatureNameKind::auto_detect`].
    Auto,
    /// Strict string match — no canonicalization. The historical pinto
    /// default for `lc` / `svd`.
    Exact,
    /// `Gene { delim: '_' }`: register every `_`-split component as an
    /// alias of the full row name.
    Gene,
    /// `Locus { merge_overlapping: true }`: normalize chrom-coord names
    /// and collapse overlapping intervals.
    Locus,
    /// Heterogeneous axis: dispatch per row name.
    Mixed,
}

impl GeneNameMode {
    /// Resolve to a concrete [`FeatureNameKind`]. `peek_names` is only
    /// consulted under `Auto`; other modes ignore it.
    pub fn resolve_kind(self, peek_names: &[Box<str>]) -> FeatureNameKind {
        match self {
            GeneNameMode::Auto => FeatureNameKind::auto_detect(peek_names),
            GeneNameMode::Exact => FeatureNameKind::Exact,
            GeneNameMode::Gene => FeatureNameKind::Gene { delim: '_' },
            GeneNameMode::Locus => FeatureNameKind::Locus {
                merge_overlapping: true,
            },
            GeneNameMode::Mixed => FeatureNameKind::Mixed,
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
        default_value_t = GeneNameMode::Auto,
        value_enum,
        help = "Gene-name canonicalization for matching external resources",
        long_help = "Row-name canonicalization strategy:\n\
                     auto  — peek row names and auto-detect (default)\n\
                     exact — strict string equality (pinto lc / svd default)\n\
                     gene  — split on '_'; both 'ENSG..._TGFB1' and 'TGFB1' alias\n\
                     locus — normalize chrom-coord names; collapse overlaps mixed —\n\
                     per-row dispatch (RNA+ATAC paired axes)"
    )]
    pub gene_name_mode: GeneNameMode,

    #[arg(
        long,
        default_value_t = 5,
        help = "Re-estimate pip every N epochs against the current embedding; 0 = never",
        long_help = "How often to re-estimate the gate's keep-probabilities.\n\
                     \n\
                     The per-epoch z draw is dropout-style regularization.\n\
                     It is a hard 0/1 mask per (gene, dim).\n\
                     The mask zeroes the gradient for excluded coordinates,\n\
                     so each epoch trains a different sub-network.\n\
                     This flag sets how often the DROP RATES behind it refresh.\n\
                     \n\
                     They need refreshing.\n\
                     The initial fit sees an SVD of pseudobulk log-counts.\n\
                     That is not the embedding the model ships.\n\
                     So every N epochs the sampler re-runs. It sees the live cell embedding,\n\
                     folded into pseudobulks. Each round warm-starts from the previous one.\n\
                     \n\
                     This is NOT EM. The refresh is a Poisson fit on counts.\n\
                     Training is edge NCE. No single objective improves, so nothing converges.\n\
                     \n\
                     The per-epoch z draw does NOT depend on this flag.\n\
                     It costs one Bernoulli per (gene, dim) on device.\n\
                     Every epoch trains its own sub-network regardless.\n\
                     This flag prices only the sampler.\n\
                     \n\
                     The default of 5 is a budget choice, not a statistical one.\n\
                     Measured on real data, the rates settle by epoch 6 and hold.\n\
                     Refreshing every epoch then re-derives a settled answer.\n\
                     Pass 1 to refresh every epoch anyway.\n\
                     Pass 0 to keep the cold rates for the whole run.\n\
                     That is what `senna bge` does.",
        hide = true
    )]
    pub selection_refresh_epochs: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Sweeps per pip refresh (half are burn-in)",
        long_help = "Gibbs sweeps for each --selection-refresh-epochs round.\n\
                     \n\
                     Deliberately too few to converge a posterior on its own.\n\
                     A refresh is one cheap step in a long sequence.\n\
                     The chain warm-starts from the previous round's state.\n\
                     Many rounds against a moving embedding make it good.\n\
                     Running any single round to convergence does not.\n\
                     \n\
                     10 sweeps means 5 kept.\n\
                     That is only sane because the PIP is Rao-Blackwellized.\n\
                     Each kept sweep contributes an ANALYTIC probability.\n\
                     That beats contributing one 0/1 draw.\n\
                     So 5 sweeps do not pin the estimate to a 1/5 grid.\n\
                     Raising this buys precision per round.\n\
                     Spending the same budget on more epochs is usually better.",
        hide = true
    )]
    pub selection_refresh_sweeps: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Gibbs sweeps for the COLD initial pip (half are burn-in)",
        long_help = "Sweeps for the one-off fit that runs before training.\n\
                     It conditions on an SVD of the pseudobulk log-counts.\n\
                     \n\
                     This round only has to BREAK IN.\n\
                     It gets the chain off its initialization and somewhere sane.\n\
                     It is not asked to converge.\n\
                     So it carries no more sweeps than a refresh does.\n\
                     The SVD basis it sees is not the embedding the model ships.\n\
                     A precise fit here would answer the wrong question precisely.\n\
                     The refreshes against the live embedding do the real work.\n\
                     \n\
                     Raise it if you set --selection-refresh-epochs 0.\n\
                     The cold rates are then the only ones the run ever uses.",
        hide = true
    )]
    pub selection_sweeps: usize,

    #[arg(
        long,
        help = "Let the nascent deviation load on dims the gene's identity does not",
        long_help = "Splice-channelized input only, and off by default.\n\
                     \n\
                     The nascent deviation is a DEVIATION from the identity loading.\n\
                     By default it may only be included where the identity is:\n\
                     a gene moving along a dim its identity does not load\n\
                     is a state the model should not visit.\n\
                     The nesting also breaks a symmetry.\n\
                     Two independent spike-and-slabs can otherwise split\n\
                     inclusion mass between (identity on, deviation off)\n\
                     and (identity off, deviation on)\n\
                     on a gene where only their sum is identified.\n\
                     \n\
                     Set this to sample the two gates independently.\n\
                     It is an A/B arm, not a tuning knob.\n\
                     `senna gem` nests by default for the same reason.",
        hide = true
    )]
    pub independent_delta_gate: bool,

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

    #[arg(long, default_value_t = 16, help = "Cell embedding dimensionality")]
    pub embedding_dim: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Training epochs over the gene axis (early-stops on --convergence-tol)",
        long_help = "Passes over the gene axis.\n\
                     \n\
                     Each epoch draws its own z ~ Bern(pip).\n\
                     Epochs are therefore how the gate's dropout averages out.\n\
                     A handful of them samples the sub-network space too thinly.\n\
                     The run early-stops once the loss flattens, per\n\
                     --convergence-tol over --convergence-window.\n\
                     A high value here is a ceiling, not a fixed cost.\n\
                     \n\
                     Pair with --genes-per-epoch to cap per-epoch cost.\n\
                     Pair with --selection-refresh-epochs as well.\n\
                     That keeps the sampler off the critical path."
    )]
    pub epochs: usize,

    #[arg(
        long,
        default_value_t = 2048,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Genes per outer parallel sampling chunk",
        long_help = "The outer loop samples this many genes in parallel via rayon.\n\
                     Forward and backward then run serially. candle Var is not parallel-safe.\n\
                     \n\
                     This is ALSO the SGD minibatch, not just a parallel width.\n\
                     An epoch takes trainable-genes / this many optimizer steps,\n\
                     each paying a forward, a backward, a gradient clip and an\n\
                     AdamW update over the whole feature table.\n\
                     So it sets both wall-clock and how many parameter updates\n\
                     a run performs, and it is not fit-neutral:\n\
                     raising 64 to 2048 cut wall-clock several fold AND raised\n\
                     spatial coherence, at an unchanged sampling budget.\n\
                     \n\
                     Below ~this many trainable genes an epoch is ONE step,\n\
                     so a small panel may want a smaller value here\n\
                     or more --epochs.\n\
                     Lower it under memory pressure.",
        hide = true
    )]
    pub gene_batch_size: usize,

    #[arg(
        long,
        default_value_t = 12,
        help = "Positive edges drawn per (gene, batch) sample",
        long_help = "Every gene draws this many positives per batch each epoch.\n\
                     \n\
                     The default was 256, and that was far too high.\n\
                     Over-sampling does not merely cost time here,\n\
                     it DEGRADES the fit: cutting the epoch budget ~20x\n\
                     raised spatial coherence several fold on the test set.\n\
                     Raise it only if the fit looks under-trained,\n\
                     and check the coherence rather than the loss.\n\
                     \n\
                     Kept per-gene so the budget tracks the gene axis.\n\
                     --positives-per-epoch overrides it with an absolute total.",
        hide = true
    )]
    pub per_gene_batch: usize,

    #[arg(
        long,
        value_name = "N",
        help = "Total positive edges drawn per epoch, across all genes (unset = auto)",
        long_help = "The epoch's total sampling budget.\n\
                     \n\
                     Divided evenly: each gene draws\n\
                     N / (trainable genes x batches) positives per batch.\n\
                     Unset keeps the historical --per-gene-batch instead.\n\
                     On a 17k-gene run the default is ~4.4M positives an epoch,\n\
                     each carrying 1 + --n-negatives x --chain-levels scores.\n\
                     \n\
                     This is the knob for how much data an epoch sees,\n\
                     and it is the one that moved the fit.\n\
                     Pair it with --gene-batch-size: the budget sets work per\n\
                     step, that sets how many steps an epoch takes.\n\
                     \n\
                     --genes-per-epoch is the coarse alternative:\n\
                     it drops genes rather than sampling each one less.",
        hide = true
    )]
    pub positives_per_epoch: Option<usize>,

    #[arg(
        long,
        default_value_t = 8,
        help = "Sibling negatives drawn per positive edge per chain level",
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
        help = "Per-gene activity normalization",
        hide = true,
    )]
    pub activity_norm: ActivityNorm,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Exponent on within-gene positive-edge weights a_g[u]·a_g[v]",
        long_help = "Stage-2 coverage exponent, one axis down from bge's alpha_pb.\n\
                     Positive edges within a gene are drawn with probability ∝ (a_g[u]·a_g[v])^activity-alpha.\n\
                     The default of 1.0 keeps the activity-proportional draw.\n\
                     0.0 makes every active edge of a gene equally likely,\n\
                     so no high-activity hub pair dominates that gene.",
        hide = true
    )]
    pub activity_alpha: f32,

    #[arg(
        long,
        help = "Disable NB-Fisher per-gene precision weighting of the loss",
        long_help = "Each gene's contribution to the loss is down-weighted.\n\
                     The weight is its NB Fisher-info w_g ∈ (0,1]. High-mean,\n\
                     high-dispersion housekeeping genes go toward 0,\n\
                     and informative low-mean genes go toward 1. This matches `pinto lc` and `senna bge`.\n\
                     Set this flag to train every gene at equal weight.",
        hide = true
    )]
    pub no_fisher_weights: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Genes visited per epoch; 0 = the whole axis",
        long_help = "Cost lever. cage walks the gene axis once per epoch.\n\
                     Runtime is therefore linear in the gene count.\n\
                     This caps how many are VISITED per epoch.\n\
                     A fresh random subset is drawn each time.\n\
                     \n\
                     That is stochastic coverage, NOT feature selection.\n\
                     Every gene stays on the trained axis. It keeps its sampled loading.\n\
                     It appears in every output table.\n\
                     A gene left out simply waits for a later epoch. Contrast --n-hvg,\n\
                     which weights the projection. That likewise drops nobody."
    )]
    pub genes_per_epoch: usize,

    #[arg(
        long,
        default_value_t = 0.0625,
        help = "L2 penalty λ on the shared cell and gene embeddings; 0 = off",
        long_help = "L2 penalty λ on E_cell ∈ ℝ^{N×D} and E_gene ∈ ℝ^{G×D}.\n\
                     It adds λ · (mean_n ‖e_n‖² + mean_g ‖e_g‖²) to the loss:\n\
                     a sum over the D latent dims, averaged over rows.\n\
                     The row-mean keeps λ scale-invariant across N and G, and\n\
                     summing over D rather than averaging keeps it invariant to\n\
                     --embedding-dim too — see loss::embedding_ridge.\n\
                     The default 0.0625 is the shrinkage cage was tuned at\n\
                     (~40% off the free gene-embedding norm); it is 1/16 only\n\
                     because the penalty used to be divided by D and D defaulted\n\
                     to 16. It now means the same thing at every D.\n\
                     Useful range 0.01-0.25. Do NOT reach for 1.0: measured on\n\
                     a 10.9k-cell 18k-gene sample it drives both embeddings to\n\
                     zero. A dense every-row penalty competes with a SPARSE data\n\
                     gradient under Adam's per-parameter normalization, so it\n\
                     bites far harder than its size against the loss suggests.",
        hide = true
    )]
    pub embedding_l2: f32,

    #[arg(
        long,
        value_delimiter(','),
        default_value = "0,1,2",
        help = "Chain levels (coarsest → finest) drawn from the coarsening hierarchy"
    )]
    pub chain_levels: Vec<usize>,

    /// HVG selection: senna-style shared CLI (`--n-hvg`,
    /// `--feature-list-file`). cage **weights the random projection** with it,
    /// exactly as `senna bge` and `senna gem` do — non-selected genes get
    /// projection weight 0 and so sit out the basis the coarsening hierarchy is
    /// built from, but they stay on the trained axis: still fit, still sampled,
    /// still in the PIP table. The selection shapes *where the pseudobulks
    /// land*, not *which genes the model may use*. `--n-hvg 0` disables.
    ///
    /// This used to hard-subset the trained axis. It no longer does, because
    /// the Gibbs-sampled spike-and-slab is the feature selector now and running
    /// a variance cut in front of it would select twice, cruder first and
    /// irreversibly. Use `--genes-per-epoch` for the cost lever the subset used
    /// to provide.
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
        help = "Ridge λ on the per-pair latent in the projection",
        long_help = "Gaussian prior strength on `e_uv` in the pair projection.\n\
                     The log-partition is summed over every gene. So this is a mild prior,\n\
                     not the only bound on the fit. The per-pair intercept is never penalized.",
        hide = true
    )]
    pub pair_ridge: f32,

    #[arg(
        long,
        default_value_t = 300,
        help = "Adam steps per cell pair in the projection",
        hide = true
    )]
    pub pair_steps: usize,

    #[arg(
        long,
        default_value_t = 512,
        help = "Genes sampled per step for the projection log-partition; 0 = all",
        long_help = "The projection's log-partition runs over every gene.\n\
                     That sum is the dominant cost.\n\
                     So each Adam step draws this many genes instead,\n\
                     ∝ their empirical abundance.\n\
                     The importance weights cancel under that proposal.\n\
                     The estimate is therefore unbiased,\n\
                     and exact at e_uv = 0. Pass 0 to sum every gene instead.",
        hide = true
    )]
    pub pair_gene_sample: usize,

    #[arg(
        long,
        default_value_t = 8192,
        help = "Cell pairs per projection read block (bounds the count slab held at once)",
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
                     logistic is per-pair SGNS, cage's historical loss.\n\
                     \n\
                     Under --gate-mode sampled, prefer softmax.\n\
                     The mask is drawn from a profiled Poisson whose normalizer\n\
                     IS the sampled-softmax estimand, so softmax puts the gate\n\
                     and the loss on one estimand.\n\
                     Against logistic that identity does not hold:\n\
                     SGNS is a sum of per-pair decisions, with no logsumexp.\n\
                     geu makes the same pairing a hard error under --posterior.\n\
                     cage permits it, because its pip is a dropout rate rather\n\
                     than a reported posterior, but it is the off-label choice.",
        hide = true
    )]
    pub nce_objective: NceObjectiveArg,

    #[arg(
        long = "gate-mode",
        default_value_t = GateMode::Sampled,
        value_enum,
        help = "How the per-(gene, dim) selection is estimated",
        long_help = "Which selector supplies the feature gate.\n\
                     \n\
                     sampled is the default, and it MEASURED BETTER.\n\
                     It runs cage's Gibbs profiled-Poisson sampler\n\
                     over super-cell counts,\n\
                     installing the result as a per-epoch Bernoulli mask.\n\
                     It gates hard, leaving roughly a twentieth of the\n\
                     (gene, dim) table on, and that selection buys finer\n\
                     communities: 12.0x a neighbour-agreement null against\n\
                     learned's 10.3x, at matched budget.\n\
                     It scans every gene on every sweep,\n\
                     so it costs about a third more wall-clock.\n\
                     Tune it with the --selection-* flags.\n\
                     \n\
                     learned is the arm `senna bge` uses,\n\
                     and the cheaper one.\n\
                     It fits a per-(gene, dim) Bernoulli spike-and-slab by SGD,\n\
                     alongside the embedding, scanning no counts.\n\
                     Its sharpness knob is --feature-gate-temp,\n\
                     and its sparsity knob is --gate-ibp-alpha.\n\
                     It does select (median inclusion ~0.29),\n\
                     so prefer it when the runtime matters more\n\
                     than the last of the resolution.\n\
                     \n\
                     Both arms now draw selection from the same truncated-IBP\n\
                     ladder, so they estimate ONE quantity two ways.\n\
                     An earlier learned arm selected nothing there,\n\
                     with every inclusion probability above 0.95.\n\
                     That was the inclusion KL it used at the time:\n\
                     a penalty with a free coefficient under an NCE objective,\n\
                     needing lambda around 1000 in cage against 1/1024 in geu.\n\
                     The ladder replaced it and has nothing to calibrate.\n\
                     \n\
                     The two arms ship different tables.\n\
                     learned bakes the gate into feature_embedding.parquet,\n\
                     because the gate multiplied the loading during training,\n\
                     and reports the inclusion table as feature_selection.parquet.\n\
                     sampled ships the raw embedding plus\n\
                     feature_posterior_mean.parquet, since there the mask was\n\
                     regularization rather than a coefficient."
    )]
    pub gate_mode: GateMode,

    #[arg(
        long = "gate-ibp-alpha",
        help = "Truncated-IBP concentration for the learned gate's per-dim inclusion\n\
                ladder (--gate-mode learned only); unset = auto",
        long_help = "Concentration alpha of the truncated Indian Buffet Process whose\n\
                     ladder tilts the learned gate: dim h carries a fixed logit\n\
                     offset h * ln(alpha/(alpha+1)), so later dims must earn their\n\
                     inclusion against a steeper prior. This is the model's sparsity\n\
                     knob, and it is CHOSEN, never fitted.\n\
                     \n\
                     Unset (the default) derives alpha from --embedding-dim so the\n\
                     ladder spans 4 logits end to end: dim 0 starts at sigma(4)=0.98\n\
                     and the last dim at sigma(0)=0.5, the most responsive point of\n\
                     the sigmoid. That keeps the tail trainable at any dimension.\n\
                     \n\
                     SMALLER alpha means a steeper ladder and more pressure toward\n\
                     sparsity. Do not copy the sampler's alpha=1 here: on 16 dims it\n\
                     is an ~11-logit drop, which does not make the gate sparse but\n\
                     freezes the tail at init — a gradient cannot recover a zeroed\n\
                     multiplier, where Gibbs resurrects one from a likelihood ratio.\n\
                     \n\
                     Rejected under --gate-mode sampled, where the sampler draws the\n\
                     mask from its own IBP and no learned gate exists to tilt.",
        hide = true
    )]
    pub gate_ibp_alpha: Option<f64>,

    #[arg(
        long = "feature-gate-temp",
        default_value_t = 1.0,
        help = "Learned-gate temperature τ; < 1 sharpens toward 0/1 (--gate-mode learned only)",
        hide = true
    )]
    pub feature_gate_temp: f32,

    #[arg(
        long,
        help = "Pre-trained gene x H embedding parquet to start the gene side from",
        long_help = "Path to a pre-trained gene x H embedding parquet.\n\
                     Row column 0 holds the gene name; value columns are the dimensions.\n\
                     Feed a RAW dictionary: a topic model's feature_embedding.parquet,\n\
                     or an embedding run's feature_loading.parquet.\n\
                     Co-embedding outputs are not dictionaries and are rejected.\n\
                     Genes are matched under --gene-name-mode.\n\
                     A gene with no dictionary row is seeded from the matched gene\n\
                     with the most similar count profile and listed in\n\
                     {out}.gene_embedding_init.parquet. Under freeze and free such\n\
                     rows train; under adapt a seeded gene follows its seed through\n\
                     the shared map until --gene-adapter-residual gives it its own\n\
                     correction."
    )]
    pub gene_embedding: Option<Box<str>>,

    #[arg(
        long,
        requires = "gene_embedding",
        help = "Optional per-gene bias parquet ([D, 1]) paired with --gene-embedding;\n\
                genes without a row get bias 0"
    )]
    pub gene_embedding_bias: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = GeneEmbeddingMode::Adapt,
        requires = "gene_embedding",
        help = "What training may do to the pre-trained gene embedding",
        long_help = "What training may do to the pre-trained gene embedding.\n\
                     \n\
                     adapt keeps the dictionary fixed and trains one\n\
                     shared linear map on top of it, so every gene's gradient\n\
                     updates the same few parameters.\n\
                     The dictionary width and --embedding-dim may differ.\n\
                     Add --gene-adapter-residual for a per-gene correction\n\
                     where the shared map is not enough.\n\
                     \n\
                     freeze keeps every dictionary-matched row fixed at its loaded value.\n\
                     Neighbor-seeded rows still train, and the selection gate stays live,\n\
                     so a gated run still ships a gate-scaled table.\n\
                     Requires the dictionary width to equal --embedding-dim.\n\
                     The e_feat ridge is skipped: a fixed table needs no shrinkage.\n\
                     \n\
                     free initializes from the dictionary and then trains every row.\n\
                     Also requires the widths to match. This is the fallback\n\
                     when the shared map underfits."
    )]
    pub gene_embedding_mode: GeneEmbeddingMode,

    #[arg(
        long,
        requires = "gene_embedding",
        help = "adapt only: add a ridge-shrunk per-gene correction\n\
                on top of the shared map"
    )]
    pub gene_adapter_residual: bool,
}

/// What training may do to a pre-trained gene embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum GeneEmbeddingMode {
    /// Fixed dictionary + one shared trainable map.
    Adapt,
    /// Dictionary-matched rows stay fixed; neighbor-seeded rows train.
    Freeze,
    /// Initialize from the dictionary, then train every row.
    Free,
}
