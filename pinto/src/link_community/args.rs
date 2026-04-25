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
                       Defaulting to 50 and letting the cosine dictionary-merge\n\
                       post-pass collapse redundant gene programs is preferred\n\
                       over under-shooting K."
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
        help = "Disable the post-hoc dictionary-merge over the K fitted communities",
        long_help = "By default, after the K link communities are fit, an\n\
                       agglomerative cosine-similarity merge runs over the K\n\
                       gene-topic posterior columns (the dictionary). Average linkage\n\
                       (UPGMA) on per-gene-centred log-rates produces a binary merge\n\
                       tree; the tree is cut at `--merge-cut` to collapse near-\n\
                       redundant communities into super-communities representing the\n\
                       same gene program across spatial locations. Emits:\n\
                         <out>.dict_merges.parquet     — full merge tree\n\
                         <out>.dict_merges.cut.parquet — fine_id → super_id mapping\n\
                       and writes the consensus output triple (link_community,\n\
                       propensity, gene_topic) under the bare `<out>.` prefix.\n\
                       The pre-merge fine partition is preserved under `<out>.draft.*`\n\
                       so the location/neighbourhood layer is never lost.\n\
                       Cost is negligible. Pass --no-merge to skip."
    )]
    pub no_merge: bool,

    #[arg(
        long,
        default_value_t = 0.9,
        help = "Cosine similarity cutoff for the dictionary-merge consensus cut",
        long_help = "Merges whose cosine similarity ≥ cutoff collapse into one\n\
                       consensus super-community; merges below the cutoff stay\n\
                       separate. Cosine is computed on per-gene-centred log-rates\n\
                       of the NB-Fisher-weighted gene-topic posterior, so it reads\n\
                       like Pearson on log-fold patterns and is scale-free wrt\n\
                       housekeeping abundance. Default 0.90 is moderately\n\
                       conservative (collapses obviously redundant programs while\n\
                       keeping closely-related cell-state distinctions). Try 0.95\n\
                       for a finer partition, 0.85 for an aggressive collapse.\n\
                       Emitted as <out>.dict_merges.cut.parquet with columns\n\
                       (community, consensus). Empty communities get consensus = −1."
    )]
    pub merge_cut: f64,

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

    #[arg(
        long,
        default_value_t = false,
        help = "Disable the frozen K×K incidence (RBM-style vertex prior) in the final EM/greedy",
        long_help = "By default, after the V-cycle pinto derives a vertex propensity\n\
                       from the cascade-final edge labels and freezes the K×K\n\
                       incidence matrix\n\
                           log B[k, k'] = ψ(a + S[k, k']) − log(b + W[k'])\n\
                       (the variational E_q[log B] under a Gamma(a, b) posterior).\n\
                       The score in the final EM-Gibbs / greedy gains the term\n\
                           Σ_{k'} (θ_L[k'] + θ_R[k']) · log B[k, k']\n\
                       which pulls the labelling toward block-structured solutions\n\
                       the factorised Poisson rate alone cannot see (~+50% MI on\n\
                       Xenium leukemia in our smoke test, ~+9% wall time).\n\
                       Pass --no-incidence to disable."
    )]
    pub no_incidence: bool,

    #[arg(
        long,
        default_value_t = 1.0,
        value_name = "A",
        help = "Gamma prior shape a for the incidence term (no effect with --no-incidence)"
    )]
    pub incidence_a: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        value_name = "B",
        help = "Gamma prior rate b for the incidence term (no effect with --no-incidence)"
    )]
    pub incidence_b: f64,
}
