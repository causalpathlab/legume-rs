//! CLI arguments for the link community model.
//!
//! Split out of `fit.rs` so the orchestrator stays focused on pipeline flow.

use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct SrtLinkCommunityArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(
        long,
        default_value_t = 50,
        help = "Number of spatial link communities to discover",
        long_help = "Number of link communities (K). Each edge in the spatial graph\n\
                       is assigned to one of K communities via collapsed Gibbs sampling.\n\
                       Communities capture distinct spatial gene expression patterns.\n\
                       Cell propensity = fraction of edges per community.\n\
                       Defaulting to 50 and letting the BHC post-pass merge redundant\n\
                       communities is preferred over under-shooting K."
    )]
    pub n_communities: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Gibbs iterations at the coarsest level",
        long_help = "Number of Gibbs iterations at the coarsest coarsening level.\n\
                       Subsequent V-cycle levels use num_gibbs/5 (minimum 10) since\n\
                       they are warm-started from the previous level.\n\
                       Full-resolution EM iterations are controlled by --num-em."
    )]
    pub num_gibbs: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Max greedy refinement sweeps after Gibbs",
        long_help = "Maximum number of greedy (argmax) sweeps after Gibbs sampling.\n\
                       Each sweep deterministically moves edges to their best community.\n\
                       Stops early if no edges move. Typically converges in 2-5 sweeps."
    )]
    pub num_greedy: usize,

    #[arg(
        long,
        help = "EM Gibbs sweeps on full edge set",
        long_help = "Number of EM Gibbs sweeps on full-resolution edges.\n\
                       Set to 0 to skip EM entirely and use only greedy refinement.\n\
                       If omitted, defaults to num_gibbs/4 (minimum 5)."
    )]
    pub num_em: Option<usize>,

    #[arg(
        long,
        help = "Dirichlet concentration for community mixing weights",
        long_help = "Concentration parameter α for the symmetric Dirichlet prior\n\
                       on community mixing weights. Enables variational truncation:\n\
                       communities with few edges are naturally pruned.\n\
                       Set to 0 to disable (uniform prior).\n\
                       If omitted, auto-scaled per level from edge profile sparsity:\n\
                       α = mean_size_factor / K, so sparser data gets a weaker prior."
    )]
    pub alpha: Option<f32>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Min total count to include a gene in the projection basis",
        long_help = "Genes with total count below this threshold are zeroed out\n\
                       in the Gaussian projection basis, effectively removing them\n\
                       from every profile dim. Ignored in gene-pair mode.\n\
                       Set to 0 to include all genes."
    )]
    pub min_gene_count: f32,

    #[arg(
        long,
        help = "External gene-gene network file (two-column TSV: gene1, gene2)",
        long_help = "External gene-gene network file (two-column TSV: gene1, gene2).\n\
                       When provided, edge profiles are built from gene-pair interaction\n\
                       deltas instead of gene modules. Each edge e=(i,j) gets a profile\n\
                       y_e[p] = sum of positive co-expression deltas for gene pair p."
    )]
    pub gene_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching for gene names in external network"
    )]
    pub gene_network_allow_prefix: bool,

    #[arg(
        long,
        default_value = "_",
        help = "Delimiter for splitting compound gene names"
    )]
    pub gene_network_delimiter: Option<char>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Shared-neighbor count to add an SNN edge (0 disables)",
        long_help = "Augment the input gene network with shared-neighbor edges:\n\
                       add a synthetic edge between any gene pair (u, v) that\n\
                       shares at least N neighbors in the input graph but is not\n\
                       already connected. Densifies incomplete networks.\n\
                       Only used with --gene-network. Set to 0 to disable."
    )]
    pub snn_min_shared: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Minimum gene degree to keep before Leiden module resolution",
        long_help = "Iteratively drop genes with current-subgraph degree below\n\
                       this threshold (k-core trim) before running Leiden on the\n\
                       gene graph. Genes trimmed at any round do not contribute\n\
                       to modules or the module-pair basis. Only used with\n\
                       --gene-network."
    )]
    pub gene_trim_min_degree: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution for gene-module clustering",
        long_help = "Modularity γ passed to Leiden on the SNN-augmented, k-core-\n\
                       trimmed gene graph. Higher γ yields more, smaller modules;\n\
                       lower γ yields fewer, larger ones. Only used with\n\
                       --gene-network."
    )]
    pub gene_modules_resolution: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Modularity-gain resolution for the coarsening merge veto",
        long_help = "Resolution γ for the degree-corrected merge veto. A proposed\n\
                       merge (i, j) is rejected when sim(i,j) < γ · deg(i) · deg(j) / (2W),\n\
                       the Louvain/Leiden modularity-gain criterion adapted to\n\
                       cosine-weighted edges. γ = 1.0 is the standard modularity\n\
                       resolution. Set to 0 to disable the veto."
    )]
    pub modularity_gamma: f32,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable the post-hoc BHC merge over the K fitted communities",
        long_help = "By default, after the K link communities are fit, a Bayesian\n\
                       hierarchical clustering (BHC) pass runs over the K communities.\n\
                       Every pair is scored with a log Bayes factor under an empirical-\n\
                       Bayes Dirichlet-Multinomial model centered on the pooled background:\n\
                           log_bf > 0  → data favor merging the two communities\n\
                           log_bf < 0  → data favor keeping them separate\n\
                           log_bf = 0  → indifferent (the natural consensus-cut boundary)\n\
                       Magnitude is BIC-like (~ n · KL of proportions); compare only\n\
                       within a single run. Emits four files under the `.bhc.` prefix:\n\
                         <out>.bhc.merges.parquet — full merge tree (scipy-linkage-style)\n\
                         <out>.bhc.cut.parquet    — consensus id per original community\n\
                         <out>.bhc.link_community.parquet — edges remapped to the cut\n\
                         <out>.bhc.propensity.parquet    — cell×community propensity,\n\
                                                           columns collapsed by the cut\n\
                       Cost is negligible. Pass --no-bhc to skip."
    )]
    pub no_bhc: bool,

    #[arg(
        long,
        help = "Total Dirichlet concentration γ for the BHC empirical-Bayes prior",
        long_help = "Total concentration γ for the empirical-Bayes asymmetric Dirichlet\n\
                       prior Dir(γ · bg), where bg[g] is the pooled per-gene marginal\n\
                       (the \"housekeeping baseline\"). Higher γ → stronger prior pull\n\
                       toward the baseline → smaller |log_bf| per merge. Per-cluster\n\
                       sufficient stats are rescaled so S_eff = edge_count.\n\
                       Node log marginal:\n\
                         f(T, S) = lgamma(γ) − lgamma(γ + S)\n\
                                 + Σ_g [lgamma(γ·bg[g] + T_g) − lgamma(γ·bg[g])]\n\
                       Default γ = 1.0 (one effective prior observation; data dominate)."
    )]
    pub bhc_gamma: Option<f64>,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "log BF cutoff for the BHC consensus cut",
        long_help = "Merges with log_bf ≥ cutoff collapse into one consensus super-\n\
                       community; merges below the cutoff stay separate. Default 0.0 is\n\
                       the natural Bayesian break point (positive BF = data prefers\n\
                       merging). Set higher to be more conservative (only strong-\n\
                       evidence merges collapse) or lower (e.g. −3) to also collapse\n\
                       weakly-distinct pairs. Emitted as <out>.bhc.cut.parquet with\n\
                       columns (community, consensus). Empty communities get\n\
                       consensus = −1."
    )]
    pub bhc_cut: f64,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable per-level cascade outputs (still runs the V-cycle)",
        long_help = "By default the V-cycle emits per-level outputs:\n\
                         <out>.L{l}.link_community.parquet\n\
                         <out>.L{l}.propensity.parquet\n\
                         <out>.L{l}.gene_topic.parquet\n\
                       so the clustering can be inspected at every coarsening\n\
                       resolution. Pass this flag to skip those writes and emit\n\
                       only the final fine-resolution outputs (matches the\n\
                       pre-V-cycle behaviour). The cascade still runs internally."
    )]
    pub no_level_outputs: bool,
}
