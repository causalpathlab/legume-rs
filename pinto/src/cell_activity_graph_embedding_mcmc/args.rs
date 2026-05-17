//! CLI arguments for `pinto cage-mcmc`.

use crate::cell_activity_graph_embedding::args::GeneNameMode;
use crate::cell_activity_graph_embedding::gene_gating::ActivityNorm;
use clap::Parser;
use data_beans_alg::hvg::HvgCliArgs;

#[derive(Parser, Debug, Clone)]
pub struct CageMcmcArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(
        long,
        default_value_t = GeneNameMode::Auto,
        value_enum,
        help = "Gene-name canonicalization for matching external resources"
    )]
    pub gene_name_mode: GeneNameMode,

    #[arg(long, default_value_t = 16, help = "Cell embedding dimensionality")]
    pub embedding_dim: usize,

    #[arg(
        long,
        value_delimiter(','),
        default_value = "0,1,2",
        help = "Chain levels (coarsest → finest) drawn from the coarsening hierarchy"
    )]
    pub chain_levels: Vec<usize>,

    #[arg(
        long,
        default_value_t = 64,
        help = "Genes per outer rayon-parallelized log-lik chunk"
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

    #[arg(
        long,
        default_value_t = ActivityNorm::Log1p,
        value_enum,
        help = "Per-gene activity normalization"
    )]
    pub activity_norm: ActivityNorm,

    /// HVG selection (subset, not reweighting).
    #[command(flatten)]
    pub hvg: HvgCliArgs,

    //
    // ───── MCMC knobs ─────
    //
    #[arg(long, default_value_t = 200, help = "Warm-up sweeps (discarded)")]
    pub warmup: usize,

    #[arg(long, default_value_t = 500, help = "Post-warmup sweeps to record")]
    pub n_samples: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Thinning: keep every T-th post-warmup sweep"
    )]
    pub thin: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Sweeps between minibatch resamples"
    )]
    pub resample_every: usize,

    #[arg(long, default_value_t = 0.1, help = "Prior SD for cell embeddings")]
    pub prior_sd_cell: f32,

    #[arg(long, default_value_t = 0.1, help = "Prior SD for gene embeddings")]
    pub prior_sd_gene: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Prior SD for level-dim gates γ (pre-softplus)"
    )]
    pub prior_sd_gate: f32,

    #[arg(long, default_value_t = 1.0, help = "Prior SD for cell biases")]
    pub prior_sd_bias: f32,

    #[arg(
        long,
        help = "Suppress thinned per-sample trace parquet (large for big runs)"
    )]
    pub no_trace: bool,

    //
    // ───── Leiden clustering (mirrors `pinto cage`) ─────
    //
    #[arg(
        long,
        default_value_t = 0,
        help = "Leiden kNN on posterior-mean cell embedding; 0 disables",
        long_help = "After MCMC, build a kNN graph on the L2-normalized \
                     posterior-mean cell embedding (cosine ≡ dot-product \
                     space) and run Leiden. Writes {prefix}.clusters.parquet, \
                     {prefix}.cluster_propensity.parquet, and \
                     {prefix}.feature_dictionary.parquet — same artifacts as \
                     `pinto cage --leiden-knn`, so `pinto plot --from \
                     {prefix}.pinto.json` works the same way."
    )]
    pub leiden_knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution (γ in CPM after scaling)"
    )]
    pub leiden_resolution: f64,

    #[arg(
        long,
        help = "If set, binary-search Leiden resolution to approximate this K"
    )]
    pub leiden_target_clusters: Option<usize>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Drop Leiden clusters smaller than this; 0 keeps all"
    )]
    pub leiden_min_cluster_size: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature τ for cell + gene propensity (higher = sharper)"
    )]
    pub propensity_temp: f32,
}
