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
                       locus — normalize chrom-coord names; collapse overlaps\n\
                       mixed — per-row dispatch (RNA+ATAC paired axes)"
    )]
    pub gene_name_mode: GeneNameMode,

    #[arg(
        long,
        default_value_t = 5,
        help = "Re-estimate pip every N epochs against the current embedding; 0 = never",
        long_help = "How often to re-estimate the gate's keep-probabilities.\n\
                     \n\
                     The per-epoch z draw is dropout-style regularization: a hard \
                     0/1 mask per (gene, dim) that zeroes the gradient for \
                     excluded coordinates, so each epoch trains a different \
                     sub-network. This flag controls how often the DROP RATES \
                     behind it are refreshed.\n\
                     \n\
                     They need refreshing because the initial fit runs against an \
                     SVD of pseudobulk log-counts, which is not the embedding the \
                     model ships. Every N epochs the live cell embedding is \
                     folded up into the pseudobulks and the sampler re-runs, \
                     warm-started from the previous round.\n\
                     \n\
                     This is NOT EM: the refresh is a Poisson fit on counts while \
                     training is edge NCE, so there is no single objective being \
                     improved and no convergence guarantee.\n\
                     \n\
                     Note what does NOT depend on this flag: the per-epoch z draw \
                     is free (one Bernoulli per (gene, dim) on device), so every \
                     epoch still trains its own sub-network no matter how rarely \
                     the rates behind it are re-estimated. This flag prices only \
                     the sampler.\n\
                     \n\
                     5 (default) is a budget choice, not a statistical one. \
                     Measured on GBM the rates settle by epoch 6 and hold, so \
                     with epochs in the hundreds refreshing every epoch spends \
                     most of its sweeps re-deriving a converged answer. 1 \
                     refreshes every epoch; 0 disables the refresh entirely and \
                     keeps the cold rates for the whole run, which is what \
                     `senna bge` does."
    )]
    pub selection_refresh_epochs: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Sweeps per pip refresh (half are burn-in)",
        long_help = "Gibbs sweeps for each --selection-refresh-epochs round.\n\
                     \n\
                     Deliberately too few to converge a posterior on its own. A \
                     refresh is one cheap step in a long sequence, not a \
                     stand-alone fit: the chain is warm-started from the previous \
                     round's inclusion state, and what makes the selection good \
                     is running many rounds against an embedding that keeps \
                     moving, not running any single round to convergence.\n\
                     \n\
                     10 sweeps means 5 kept. That is only sane because the PIP is \
                     Rao-Blackwellized: each kept sweep contributes the ANALYTIC \
                     inclusion probability rather than one 0/1 draw, so 5 sweeps \
                     no longer pins the estimate to a 1/5 grid the way averaging \
                     indicators would. Raising this buys precision per round; \
                     spending the same budget on more epochs is usually better."
    )]
    pub selection_refresh_sweeps: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Gibbs sweeps for the COLD initial pip (half are burn-in)",
        long_help = "Sweeps for the one-off fit that runs before training, against \
                     an SVD of the pseudobulk log-counts.\n\
                     \n\
                     This round only has to BREAK IN — get the chain off its \
                     initialization and somewhere sane. It is not asked to \
                     converge, which is why it carries no more sweeps than a \
                     refresh does: the SVD basis it conditions on is not the \
                     embedding the model ships, so a precise fit here would be a \
                     precise answer to the wrong question. The refreshes against \
                     the live embedding do the real work.\n\
                     \n\
                     Raise it if you set --selection-refresh-epochs 0, since the \
                     cold rates are then the only ones the run ever uses."
    )]
    pub selection_sweeps: usize,

    #[arg(
        long,
        help = "Skip the degree-corrected Poisson refinement of the coarsening levels",
        long_help = "By default each coarsening level gets a second-opinion \
                     refinement on RAW counts (degree-corrected Poisson), the same \
                     pass `pinto lc` runs. Without it the levels are cut on \
                     cosine-of-projection alone, which ignores depth and \
                     over-disperion in the counts.\n\
                     \n\
                     Set this to skip it: the context build reads the count matrix \
                     once more up front, so this is the lever if that I/O matters."
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
                     Each epoch draws its own z ~ Bern(pip), so epochs are how \
                     the gate's dropout actually averages out — a handful of them \
                     samples the sub-network space too thinly to do that. The \
                     run early-stops once the loss flattens (--convergence-tol \
                     over --convergence-window), so a high value here is a \
                     ceiling rather than a fixed cost.\n\
                     \n\
                     Pair with --genes-per-epoch to cap per-epoch cost, and with \
                     --selection-refresh-epochs to keep the sampler off the \
                     critical path."
    )]
    pub epochs: usize,

    #[arg(
        long,
        default_value_t = 64,
        help = "Genes per outer parallel sampling chunk",
        long_help = "Outer training loop samples this many genes in parallel \
                     via rayon, then runs forward / backward serially \
                     (candle Var is not parallel-safe). Default is sized for \
                     a laptop; raise if you have many cores."
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
        long_help = "Stage-2 coverage exponent (bge's alpha_pb analog, one axis \
                     down): positive edges within a gene are drawn with \
                     probability ∝ (a_g[u]·a_g[v])^activity-alpha. 1.0 (default) \
                     keeps the activity-proportional draw; 0.0 makes every active \
                     edge of a gene equally likely, so one high-activity hub pair \
                     can't dominate that gene's training."
    )]
    pub activity_alpha: f32,

    #[arg(
        long,
        help = "Disable NB-Fisher per-gene precision weighting of the loss",
        long_help = "By default cage down-weights each gene's contribution to the \
                     contrastive loss by its NB Fisher-info weight w_g ∈ (0,1] \
                     (high-mean / high-dispersion housekeeping genes → 0, \
                     informative low-mean genes → 1), matching `pinto lc` and \
                     `senna bge`. Set this to train every gene at equal weight."
    )]
    pub no_fisher_weights: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Genes visited per epoch; 0 = the whole axis",
        long_help = "Cost lever. cage walks the gene axis once per epoch, so runtime is \
                     linear in the number of genes. This caps how many are VISITED per \
                     epoch, drawing a fresh random subset each time.\n\
                     \n\
                     Stochastic coverage, NOT feature selection: every gene stays on the \
                     trained axis, keeps its sampled loading, and appears in every output \
                     table — it simply waits for a later epoch. Contrast --n-hvg, which \
                     weights the projection and likewise drops nobody."
    )]
    pub genes_per_epoch: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 penalty λ on the shared embeddings E_cell ∈ ℝ^{N×D} and \
                E_gene ∈ ℝ^{G×D}: adds λ · (mean(E_cell²) + mean(E_gene²)) \
                to the per-step composite loss (mean-normalized, so λ stays \
                scale-invariant across N, G, D). Default 1.0 (mild shrinkage). \
                Set 0.0 to disable. Typical: 0.1–10.0."
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
        long_help = "After each epoch, look at the last `convergence-window` \
                     mean losses; if their (max − min) / |mean| is below \
                     --convergence-tol, stop training. 0 runs all --epochs \
                     unconditionally."
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
        default_value_t = 0,
        help = "K-means clusters on cell embedding; 0 disables clustering",
        long_help = "After training, run k-means++ (Lloyd's algorithm, via \
                     `matrix-util::clustering`) on the L2-normalized cell \
                     embedding. Writes {prefix}.clusters.parquet, \
                     {prefix}.cluster_propensity.parquet, \
                     {prefix}.feature_dictionary.parquet, and \
                     {prefix}.link_community.parquet."
    )]
    pub n_clusters: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature τ for cell + gene propensity (higher = sharper)"
    )]
    pub propensity_temp: f32,

    #[arg(
        long,
        default_value_t = 100,
        help = "Lloyd's algorithm iteration cap for k-means"
    )]
    pub kmeans_max_iter: usize,
}
