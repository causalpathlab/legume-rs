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
        long_help = "Number of link communities (K).\n\
                     Collapsed Gibbs sampling assigns each spatial edge to one.\n\
                     Communities capture distinct spatial expression patterns.\n\
                     Cell propensity is the fraction of edges per community.\n\
                     \n\
                     Prefer over-shooting K to under-shooting it.\n\
                     The default of 50 leans on the cosine dictionary-merge pass.\n\
                     That pass collapses redundant gene programs."
    )]
    pub n_communities: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Gibbs iterations at the coarsest level",
        long_help = "Gibbs iterations at the coarsest coarsening level.\n\
                     Later V-cycle levels use num_gibbs/5, at least 10. They can afford fewer:\n\
                     each warm-starts from the level above.\n\
                     --num-em controls the full-resolution iterations.",
        hide = true
    )]
    pub num_gibbs: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Max greedy refinement sweeps after Gibbs",
        long_help = "Maximum greedy (argmax) sweeps after Gibbs sampling.\n\
                     Each sweep moves every edge to its best community.\n\
                     Sweeping stops early once no edge moves.\n\
                     This typically converges in 2-5 sweeps.",
        hide = true
    )]
    pub num_greedy: usize,

    #[arg(
        long,
        help = "EM Gibbs sweeps on full edge set",
        long_help = "EM Gibbs sweeps over the full-resolution edges.\n\
                     Pass 0 to skip EM and refine greedily only. If omitted,\n\
                     this defaults to num_gibbs/4, at least 5.",
        hide = true
    )]
    pub num_em: Option<usize>,

    #[arg(
        long,
        help = "Dirichlet concentration for community mixing weights",
        long_help = "Concentration α of the symmetric Dirichlet prior.\n\
                     The prior sits on the community mixing weights.\n\
                     It enables variational truncation.\n\
                     Communities holding few edges are then pruned on their own.\n\
                     Pass 0 to disable it and use a uniform prior.\n\
                     \n\
                     If omitted, α is auto-scaled per level from profile sparsity.\n\
                     α = mean_size_factor / K, so sparser data gets a weaker prior."
    )]
    pub alpha: Option<f32>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Min total count to include a gene in the projection basis",
        long_help = "Genes under this total count are zeroed in the basis.\n\
                     That removes them from every profile dimension.\n\
                     Gene-pair mode ignores this flag. Pass 0 to include all genes."
    )]
    pub min_gene_count: f32,

    #[arg(
        long,
        help = "External gene-gene network file (two-column TSV: gene1, gene2)",
        long_help = "External gene-gene network, a two-column TSV of gene1, gene2. When given,\n\
                     edge profiles come from gene-pair deltas.\n\
                     They do not come from gene modules. Each edge e=(i,j) gets the profile:\n\
                     y_e[p] = sum of positive co-expression deltas for pair p."
    )]
    pub gene_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching for gene names in external network",
        hide = true
    )]
    pub gene_network_allow_prefix: bool,

    #[arg(
        long,
        default_value = "_",
        help = "Delimiter for splitting compound gene names",
        hide = true
    )]
    pub gene_network_delimiter: Option<char>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Shared-neighbor count to add an SNN edge (0 disables)",
        long_help = "Augment the gene network with shared-neighbor edges.\n\
                     A synthetic edge joins any unconnected pair (u, v).\n\
                     The pair must share at least N neighbours already.\n\
                     This densifies incomplete networks. It applies only with --gene-network;\n\
                     0 disables it.",
        hide = true
    )]
    pub snn_min_shared: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Minimum gene degree to keep before Leiden module resolution",
        long_help = "k-core trim applied before Leiden runs on the gene graph.\n\
                     Genes below this subgraph degree are dropped, iteratively.\n\
                     A gene trimmed in any round contributes to no module,\n\
                     and to no module-pair basis entry. This applies only with --gene-network.",
        hide = true
    )]
    pub gene_trim_min_degree: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution for gene-module clustering",
        long_help = "Modularity γ for Leiden on the gene graph.\n\
                     That graph is SNN-augmented and k-core-trimmed. Higher γ yields more,\n\
                     smaller modules. Lower γ yields fewer, larger ones.\n\
                     This applies only with --gene-network.",
        hide = true
    )]
    pub gene_modules_resolution: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Modularity-gain resolution for the coarsening merge veto",
        long_help = "Resolution γ for the degree-corrected merge veto.\n\
                     A proposed merge (i, j) is rejected when:\n\
                     sim(i,j) < γ · deg(i) · deg(j) / (2W).\n\
                     That is the Louvain/Leiden modularity-gain criterion,\n\
                     adapted to cosine-weighted edges.\n\
                     γ = 1.0 is the standard modularity resolution. Pass 0 to disable the veto.",
        hide = true
    )]
    pub modularity_gamma: f32,

    #[arg(
        long,
        default_value_t = 0.9,
        help = "Cosine similarity cutoff for the dictionary-merge consensus cut",
        long_help = "Merges at or above this cosine collapse into one community.\n\
                     Merges below the cutoff stay separate.\n\
                     \n\
                     Cosine runs on per-gene-centred log-rates of the gene-community posterior,\n\
                     restricted to detected genes (see --merge-min-nnz).\n\
                     Centring is per GENE only, so this is not a Pearson\n\
                     correlation between communities, which would also centre\n\
                     each community.\n\
                     \n\
                     The default of 0.90 is moderately conservative.\n\
                     It collapses obviously redundant programs.\n\
                     It keeps closely-related cell-state distinctions.\n\
                     Try 0.95 for a finer partition. Try 0.85 for an aggressive collapse.\n\
                     \n\
                     If the cut collapses far more than expected,\n\
                     check the gene filter before raising this.\n\
                     Undetected genes inflate every pairwise cosine.\n\
                     \n\
                     The cut lands in <out>.dict_merges.cut.parquet.\n\
                     Its columns are (community, consensus).\n\
                     Empty communities get consensus = −1.",
        hide = true
    )]
    pub merge_cut: f64,

    #[arg(
        long,
        value_name = "N",
        help = "Minimum cells a gene must appear in to score the dictionary merge (unset = auto)",
        long_help = "Genes detected in fewer than N cells are dropped before the\n\
                     merge cosine is computed.\n\
                     They are NOT dropped from any other output.\n\
                     \n\
                     Unset (the default) picks N by the same 2-means split\n\
                     data-beans uses for cell QC.\n\
                     Pass 0 to score every gene, which is the pre-fix behaviour.\n\
                     \n\
                     WHY THIS EXISTS.\n\
                     An undetected gene still gets a Poisson-Gamma posterior.\n\
                     Its log-rate is then set by each community's exposure\n\
                     rather than by data, so it swings across communities\n\
                     harder than a well-measured gene's does.\n\
                     Cosine is dominated by the largest-magnitude rows,\n\
                     so left in, those genes decide the merge\n\
                     and collapse the tree at any cut.\n\
                     \n\
                     Raising N past the community count is refused:\n\
                     below that the dictionary is rank-deficient\n\
                     and every pair looks identical."
    )]
    pub merge_min_nnz: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable per-level cascade outputs (still runs the V-cycle)",
        long_help = "By default the V-cycle emits per-level outputs:\n\
                     \n  \
                     <out>.L{l}.link_community.parquet\n  \
                     <out>.L{l}.propensity.parquet\n  \
                     <out>.L{l}.gene_community.parquet\n\
                     They let you inspect the clustering at every resolution.\n\
                     \n\
                     Pass this flag to skip those writes.\n\
                     Only the final fine-resolution outputs are then emitted,\n\
                     matching the pre-V-cycle behaviour. The cascade still runs internally."
    )]
    pub no_level_outputs: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable the frozen K×K incidence (RBM-style vertex prior)",
        long_help = "After the V-cycle, pinto derives a vertex propensity.\n\
                     It reads the cascade-final edge labels.\n\
                     It then freezes the K×K incidence matrix:\n\
                     log B[k, k'] = ψ(a + S[k, k']) − log(b + W[k']) That is the variational E_q[log B] under Gamma(a, b).\n\
                     \n\
                     The final EM-Gibbs and greedy score then gains the term Σ_{k'} (θ_L[k'] + θ_R[k']) · log B[k, k'].\n\
                     That pulls the labelling toward block structure.\n\
                     The factorised Poisson rate alone cannot see it.\n\
                     Our Xenium leukemia smoke test gained ~50% MI. It cost ~9% more wall time.\n\
                     \n\
                     Pass --no-incidence to disable it.",
        hide = true
    )]
    pub no_incidence: bool,

    #[arg(
        long,
        default_value_t = 1.0,
        value_name = "A",
        help = "Gamma prior shape a for the incidence term (no effect with --no-incidence)",
        hide = true
    )]
    pub incidence_a: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        value_name = "B",
        help = "Gamma prior rate b for the incidence term (no effect with --no-incidence)",
        hide = true
    )]
    pub incidence_b: f64,
}
