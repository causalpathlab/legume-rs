use clap::Args;

#[derive(Args, Debug)]
pub struct AnnotateArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest produced by `senna topic|masked-topic|joint-topic|svd|joint-svd`"
    )]
    pub from: Box<str>,

    #[arg(
        short = 'c',
        long = "clusters",
        help = "Cluster parquet from `senna cluster` (cells × 1 cluster column,",
        long_help = "Cluster parquet from `senna cluster` (cells × 1 cluster column,\n\
                     `NaN` for unassigned). When omitted, the path is read from\n\
                     `manifest.cluster.clusters` if populated; otherwise annotate\n\
                     runs Leiden internally on the latent matrix from the manifest."
    )]
    pub clusters: Option<Box<str>>,

    #[arg(
        long = "knn",
        default_value_t = 15,
        help = "Nearest neighbors for the internal Leiden cosine-KNN graph",
        long_help = "Number of nearest neighbors for the cosine-KNN graph used by the\n\
                     internal Leiden clustering. Ignored when --clusters / manifest\n\
                     cluster path is provided."
    )]
    pub knn: usize,

    #[arg(
        long = "resolution",
        default_value_t = 1.0,
        help = "Modularity resolution (CPM) for the internal Leiden clustering",
        long_help = "Modularity resolution (CPM) for the internal Leiden clustering.\n\
                     Higher → more clusters. Ignored when clusters are supplied."
    )]
    pub resolution: f64,

    #[arg(
        long = "num-clusters",
        help = "Optional target cluster count for Leiden auto-resolution",
        long_help = "Optional target cluster count for Leiden auto-resolution. When set,\n\
                     resolution is binary-searched to approximate this. Ignored when\n\
                     clusters are supplied."
    )]
    pub num_clusters: Option<usize>,

    #[arg(
        long = "min-cluster-size",
        default_value_t = 2,
        help = "Minimum cluster size; smaller clusters become unassigned"
    )]
    pub min_cluster_size: usize,

    #[arg(
        long = "cluster-seed",
        help = "Seed for internal Leiden clustering (deterministic)"
    )]
    pub cluster_seed: Option<u64>,

    #[arg(
        short = 'm',
        long = "markers",
        required = true,
        help = "Marker-gene TSV: `gene<TAB>celltype` per line",
        long_help = "Marker-gene TSV: `gene<TAB>celltype` per line. Flexible delimiter\n\
                     (tab / comma / space). Symbol / alias matching via `flexible_gene_match`."
    )]
    pub markers: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix for annotation artifacts",
        long_help = "Output prefix for annotation artifacts. When omitted, derived\n\
                     from `--from` by stripping `.senna.json` (or `.json`) — e.g.\n\
                     `--from temp.senna.json` → `--out temp`."
    )]
    pub out: Option<Box<str>>,

    #[arg(short = 'v', long, help = "Verbose logging")]
    pub verbose: bool,

    #[arg(
        long = "block-size",
        default_value_t = 1024,
        help = "Cells per CSC read block when streaming raw counts for per-cluster aggregation",
        long_help = "Cells per CSC read block when streaming raw counts for per-cluster\n\
                     aggregation and NB-Fisher trend fitting. Larger blocks → fewer reads\n\
                     but more memory."
    )]
    pub block_size: usize,

    #[arg(
        long = "num-draws",
        default_value_t = 1000,
        help = "Random gene-set draws per cell type (Efron–Tibshirani moments)",
        long_help = "Number of random gene-set draws per celltype for the\n\
                     Efron–Tibshirani row-randomization moments (used to restandardize\n\
                     both observed and permuted ES)."
    )]
    pub num_draws: usize,

    #[arg(
        long = "num-perm",
        default_value_t = 500,
        help = "Number of PB-level sample permutations for the correlation-preserving null",
        long_help = "Number of PB-level sample permutations for the correlation-preserving\n\
                     null (shuffle pb_membership, recompute β̃ = pb_gene · shuffled,\n\
                     ES per permutation, pool across clusters). Set 0 to fall back to\n\
                     row-randomization-based p-values (useful for small nClusters)."
    )]
    pub num_perm: usize,

    #[arg(
        long = "fdr-alpha",
        default_value_t = 0.10,
        help = "FDR α for the Q-matrix threshold"
    )]
    pub fdr_alpha: f32,

    #[arg(
        long = "q-temperature",
        default_value_t = 1.0,
        help = "Softmax temperature used when row-normalizing Q over significant entries",
        long_help = "Softmax temperature used when row-normalizing Q over significant\n\
                     entries. Lower → sharper; higher → more uniform."
    )]
    pub q_temperature: f32,

    #[arg(
        long = "min-confidence",
        default_value_t = 0.0,
        help = "Minimum cell-level confidence to emit a concrete label",
        long_help = "Minimum cell-level confidence to emit a concrete label; below this\n\
                     cells are labeled `unassigned` in the argmax TSV."
    )]
    pub min_confidence: f32,

    #[arg(
        long = "seed",
        default_value_t = 42,
        help = "RNG seed (deterministic; affects row randomization)"
    )]
    pub seed: u64,

    #[arg(
        long = "no-clean",
        help = "Keep existing {out}.* annotation outputs (default: erase the explicit annotation set first — never the embedding/manifest — for a fresh re-run)"
    )]
    pub no_clean: bool,

    #[arg(
        long = "preload-data",
        default_value_t = false,
        help = "Preload columns into memory after opening the zarr/h5 backend",
        long_help = "Preload columns into memory after opening the zarr/h5 backend.\n\
                     On slow disks this trades memory for I/O latency on later block reads."
    )]
    pub preload_data: bool,

    #[arg(
        long = "no-empirical-specificity",
        default_value_t = false,
        help = "Disable data-aware specificity re-weighting of marker genes",
        long_help = "Disable data-aware specificity re-weighting of marker genes. By\n\
                     default, each marker is multiplied by an empirical specificity\n\
                     score derived from the cluster expression matrix (`max simplex\n\
                     value across clusters`, rescaled to [0, 1]) — this suppresses\n\
                     markers that fire broadly (e.g. GZMB shared between NK and CD8\n\
                     effector). Set this flag to fall back to IDF-only weighting."
    )]
    pub no_empirical_specificity: bool,

    // ----- optional inline ontology annotation (TreeBH) -----
    #[arg(
        long = "obo",
        help = "Cell Ontology OBO file; with --label-cl, also runs annotate-ontology inline",
        long_help = "Cell Ontology OBO file (e.g. cl-basic.obo). When BOTH --obo and\n\
                     --label-cl are given, the restandardized-ES z-matrix is fed to the\n\
                     TreeBH ontology annotator in the same run (no separate\n\
                     `annotate-ontology` invocation needed), writing\n\
                     {out}.ontology_assignment.tsv + {out}.ontology_node_mass.parquet."
    )]
    pub obo: Option<Box<str>>,

    #[arg(
        long = "label-cl",
        help = "Curated `label<TAB>CL:id` TSV; enables inline ontology annotation (with --obo)"
    )]
    pub label_cl: Option<Box<str>>,

    #[arg(
        long = "ontology-fdr-q",
        default_value_t = 0.1,
        help = "Per-level selective-FDR target q for the inline TreeBH ontology walk"
    )]
    pub ontology_fdr_q: f64,

    #[arg(
        long = "ontology-by",
        default_value_t = false,
        help = "Use the Benjamini–Yekutieli correction within families for the inline ontology walk"
    )]
    pub ontology_by: bool,
}

/// `senna annotate-ontology` — hierarchical multi-resolution cell-type calling
/// (TreeBH) on the Cell Ontology, post-processing an `annotate-by-enrichment`
/// run's cluster × celltype matrix.
#[derive(Args, Debug)]
pub struct AnnotateOntologyArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest already annotated by `senna annotate-by-enrichment`",
        long_help = "Run manifest already annotated by `senna annotate-by-enrichment`\n\
                     (reads `annotate.cluster_celltype_q` and its sibling\n\
                     `*_es_std` / `*_p` matrices)."
    )]
    pub from: Box<str>,

    #[arg(
        long = "label-cl",
        required = true,
        help = "Curated `label<TAB>CL:id` TSV mapping each marker celltype to a Cell Ontology term"
    )]
    pub label_cl: Box<str>,

    #[arg(
        long = "obo",
        required = true,
        help = "Cell Ontology OBO file (e.g. cl-basic.obo)",
        long_help = "Cell Ontology OBO file. Fetch the basic release with:\n\
                     curl -sSL https://github.com/obophenotype/cell-ontology/\
                     releases/latest/download/cl-basic.obo -o cl-basic.obo"
    )]
    pub obo: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix (defaults to `--from` with `.senna.json`/`.json` stripped)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long = "fdr-q",
        default_value_t = 0.1,
        help = "Per-level selective-FDR target q for the TreeBH descent",
        long_help = "Per-level selective-FDR target q (Bogomolov–Peterson–Benjamini–Sabatti\n\
                     TreeBH). Benjamini–Hochberg is applied within each family at a\n\
                     working target shrunk by the rejection proportions along the\n\
                     ancestor path; lower q → descends less eagerly (more abstention)."
    )]
    pub fdr_q: f64,

    #[arg(
        long = "by",
        default_value_t = false,
        help = "Use the Benjamini–Yekutieli correction within families (arbitrary dependence; more conservative)"
    )]
    pub by: bool,

    #[arg(
        long = "use-perm-p",
        default_value_t = false,
        help = "Use the saturated permutation p-values instead of Φ(−z) from the restandardized ES",
        long_help = "Use `cluster_celltype_p` directly instead of converting the\n\
                     restandardized-ES z-scores (`cluster_celltype_es_std`) to\n\
                     p-values. The permutation p is resolution-limited (≈1/B), so the\n\
                     z→p default is usually more discriminative."
    )]
    pub use_perm_p: bool,

    #[arg(short = 'v', long, help = "Verbose logging")]
    pub verbose: bool,
}
