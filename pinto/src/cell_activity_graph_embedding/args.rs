//! CLI arguments for `pinto cage`.

use crate::cell_activity_graph_embedding::gene_gating::ActivityNorm;
use auxiliary_data::feature_names::FeatureNameKind;
use clap::{Parser, ValueEnum};
use data_beans_alg::hvg::HvgCliArgs;

use crate::util::device::ComputeDevice;

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

/// Clusterer for the per-pair latent. `kmeans` fixes the count;
/// `leiden` discovers it from the graph.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum EdgeClusterMethod {
    Kmeans,
    Leiden,
}

impl std::fmt::Display for EdgeClusterMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kmeans => write!(f, "kmeans"),
            Self::Leiden => write!(f, "leiden"),
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
                     Measured on GBM, the rates settle by epoch 6 and hold.\n\
                     Refreshing every epoch then re-derives a settled answer.\n\
                     Pass 1 to refresh every epoch anyway.\n\
                     Pass 0 to keep the cold rates for the whole run.\n\
                     That is what `senna bge` does."
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
                     Spending the same budget on more epochs is usually better."
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
                     The cold rates are then the only ones the run ever uses."
    )]
    pub selection_sweeps: usize,

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
                     so this is the lever if that I/O matters."
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
        default_value_t = 64,
        help = "Genes per outer parallel sampling chunk",
        long_help = "The outer loop samples this many genes in parallel via rayon.\n\
                     Forward and backward then run serially. candle Var is not parallel-safe.\n\
                     The default is sized for a laptop. Raise it if you have many cores."
    )]
    pub gene_batch_size: usize,

    #[arg(
        long,
        default_value_t = 256,
        help = "Positive edges drawn per (gene, batch) sample"
    )]
    pub per_gene_batch: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Sibling negatives drawn per positive edge per chain level"
    )]
    pub n_negatives: usize,

    #[arg(
        long,
        default_value_t = 0.75,
        help = "Negative-degree exponent (power-of-degree negative sampling)"
    )]
    pub alpha_neg: f32,

    #[arg(long, default_value_t = 5e-3, help = "AdamW learning rate")]
    pub lr: f32,

    #[arg(
        long,
        default_value_t = ActivityNorm::Log1p,
        value_enum,
        help = "Per-gene activity normalization"
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
                     so no high-activity hub pair dominates that gene."
    )]
    pub activity_alpha: f32,

    #[arg(
        long,
        help = "Disable NB-Fisher per-gene precision weighting of the loss",
        long_help = "Each gene's contribution to the loss is down-weighted.\n\
                     The weight is its NB Fisher-info w_g ∈ (0,1]. High-mean,\n\
                     high-dispersion housekeeping genes go toward 0,\n\
                     and informative low-mean genes go toward 1. This matches `pinto lc` and `senna bge`.\n\
                     Set this flag to train every gene at equal weight."
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
        default_value_t = 1.0,
        help = "L2 penalty λ on the shared cell and gene embeddings; 0 = off",
        long_help = "L2 penalty λ on E_cell ∈ ℝ^{N×D} and E_gene ∈ ℝ^{G×D}.\n\
                     It adds λ · (mean(E_cell²) + mean(E_gene²)) to the loss.\n\
                     The means keep λ scale-invariant across N,\n\
                     G and D. The default of 1.0 is mild shrinkage; 0.0 disables it.\n\
                     Typical values run from 0.1 to 10.0."
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
    /// exactly as `senna bge` and `faba gem` do — non-selected genes get
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
                     Pass 0 to run all --epochs unconditionally."
    )]
    pub convergence_window: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Relative-range threshold over --convergence-window for stopping"
    )]
    pub convergence_tol: f32,

    #[arg(
        long,
        default_value_t = EdgeClusterMethod::Kmeans,
        value_enum,
        help = "How to cut the pair latent into link communities",
        long_help = "kmeans fixes the community count at --n-edge-clusters.\n\
                     leiden builds a cosine kNN graph over the pair latent.\n\
                     --leiden-resolution then decides how many communities exist.\n\
                     Under leiden, --n-edge-clusters is only a target.\n\
                     The resolution is steered toward it, not fixed at it."
    )]
    pub edge_cluster_method: EdgeClusterMethod,

    #[arg(
        long,
        default_value_t = 30,
        help = "Neighbours per pair in the Leiden kNN graph over the pair latent"
    )]
    pub leiden_knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution; higher gives more, finer communities"
    )]
    pub leiden_resolution: f64,

    #[arg(
        long,
        help = "Link communities to cut from the pair latent [default: --embedding-dim]",
        long_help = "Number of edge clusters k-means cuts from the pair latent.\n\
                     A cell's propensity is its incident-edge fraction, taken per community.\n\
                     This is the definition `pinto lc` and `pinto dsvd` use.\n\
                     Omit to fall back to --embedding-dim. Writes {prefix}.propensity.parquet,\n\
                     {prefix}.link_community.parquet, and {prefix}.gene_community.parquet."
    )]
    pub n_edge_clusters: Option<usize>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Clip the global gradient norm to this before each step; 0 = off",
        long_help = "Global-L2-norm gradient clipping for the training loop.\n\
                     Gradients are scaled to this norm when they exceed it,\n\
                     which bounds the update without turning it.\n\
                     A step whose global norm is not finite is skipped.\n\
                     Pass 0 to disable clipping."
    )]
    pub grad_clip: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Ridge λ on the per-pair latent in the projection",
        long_help = "Gaussian prior strength on `e_uv` in the pair projection.\n\
                     The log-partition is summed over every gene. So this is a mild prior,\n\
                     not the only bound on the fit. The per-pair intercept is never penalized."
    )]
    pub pair_ridge: f32,

    #[arg(
        long,
        default_value_t = 300,
        help = "Adam steps per cell pair in the projection"
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
                     and exact at e_uv = 0. Pass 0 to sum every gene instead."
    )]
    pub pair_gene_sample: usize,

    #[arg(
        long,
        default_value_t = 8192,
        help = "Cell pairs per projection read block (bounds the count slab held at once)"
    )]
    pub pair_block: usize,
}
