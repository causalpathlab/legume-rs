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
        default_value = "",
        help = "Marker-gene TSV: `gene<TAB>celltype` per line (one of --markers/--gaf/--gmt)",
        long_help = "Marker-gene TSV: `gene<TAB>celltype` per line. Flexible delimiter\n\
                     (tab / comma / space). Symbol / alias matching via `flexible_gene_match`.\n\
                     Exactly one gene-set source is required: --markers (curated cell-type\n\
                     markers), --gaf (GO annotations), or --gmt (MSigDB gene-sets)."
    )]
    pub markers: Box<str>,

    #[arg(
        long = "gaf",
        help = "GO annotation file (.gaf/.gaf.gz). Ontology mode: score each term as a \
                cross-cluster-contrasted module score on the cluster profile → per-cluster \
                signature ({out}.ontology_signature.tsv). --obo supplies term names."
    )]
    pub gaf: Option<Box<str>>,

    #[arg(
        long = "gmt",
        help = "MSigDB GMT gene-sets (`term<TAB>desc<TAB>genes…`). Ontology mode (as --gaf)"
    )]
    pub gmt: Option<Box<str>>,

    #[arg(
        long = "no-iea",
        default_value_t = false,
        help = "GAF only: drop IEA (electronic) annotations — the low-confidence bulk"
    )]
    pub no_iea: bool,

    #[arg(
        long = "min-gene-set",
        default_value_t = 15,
        help = "Ontology mode: minimum matched members for a term to be scored"
    )]
    pub min_gene_set: usize,

    #[arg(
        long = "max-gene-set",
        default_value_t = 500,
        help = "Ontology mode: maximum matched members (size window; excludes \
                near-universal terms)"
    )]
    pub max_gene_set: usize,

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

    //////////////////////////////////////////////////
    // optional inline ontology annotation (TreeBH) //
    //////////////////////////////////////////////////
    #[arg(
        long = "obo",
        help = "Cell Ontology .obo (e.g. cl-basic.obo). Given WITH --label-cl, runs TreeBH \
                ontology calling inline → {out}.ontology_assignment.tsv + .ontology_node_mass.parquet"
    )]
    pub obo: Option<Box<str>>,

    #[arg(
        long = "label-cl",
        help = "Curated `label<TAB>CL:id` map, one row per marker celltype. Required together with --obo"
    )]
    pub label_cl: Option<Box<str>>,

    #[arg(
        long = "ontology-fdr-q",
        default_value_t = 0.1,
        help = "Ontology TreeBH per-level FDR target (lower → descends less, abstains more)"
    )]
    pub ontology_fdr_q: f64,

    #[arg(
        long = "ontology-by",
        default_value_t = false,
        help = "Ontology: Benjamini–Yekutieli within families (any dependence; more conservative)"
    )]
    pub ontology_by: bool,
}

/// `senna annotate-by-projection` — firm marker-set annotation by projection
/// onto a co-embedded feature space (bge / fne / resolve-embedding-space).
/// Embedding-grounded (no raw-count re-read), complementary to
/// `annotate-by-enrichment`. Drives the shared firm term-ORA core.
#[derive(Args, Debug)]
pub struct AnnotateProjectionArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest from a co-embedding run (`senna bge|fne|resolve-embedding-space`)",
        long_help = "Run manifest with a co-embedded gene space — `senna bge`, `fne`, or\n\
                     `resolve-embedding-space` (reads `outputs.feature_embedding` +\n\
                     `outputs.cell_embedding`, falling back to `outputs.latent` for the\n\
                     cell side on plain bge/fne). topic/svd runs have no genes-on-the-\n\
                     cell-manifold embedding — use `annotate-by-enrichment` for those."
    )]
    pub from: Box<str>,

    #[arg(
        short = 'm',
        long = "markers",
        required = true,
        help = "Marker-gene TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix (default: `--from` with `.senna.json`/`.json` stripped)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long = "knn",
        default_value_t = 30,
        help = "k for the cosine cell kNN graph fed to Leiden clustering"
    )]
    pub knn: usize,

    #[arg(
        long = "resolution",
        default_value_t = 1.0,
        help = "Leiden resolution for cell clustering (higher → more, finer clusters)"
    )]
    pub resolution: f64,

    #[arg(
        long = "num-perm",
        default_value_t = 500,
        help = "Permutation draws calibrating the over-representation null (0 = analytic p only)"
    )]
    pub num_perm: usize,

    #[arg(
        long = "seed",
        default_value_t = 42,
        help = "RNG seed (clustering + permutation null)"
    )]
    pub seed: u64,

    #[arg(
        long = "no-idf",
        help = "Disable IDF down-weighting of markers shared across many types"
    )]
    pub no_idf: bool,

    #[arg(
        long = "no-assign-qc",
        help = "Keep every cell→term assignment (skip the distance-outlier prune)"
    )]
    pub no_assign_qc: bool,

    #[arg(
        long = "assign-mad",
        default_value_t = 2.5,
        help = "Outlier gate: prune a cell whose distance to its centroid exceeds median + k·MAD"
    )]
    pub assign_mad: f64,

    #[arg(
        long = "fdr-alpha",
        default_value_t = 0.1,
        help = "FDR α for the per-cluster term call + Q sparsity (BH on the permutation p)"
    )]
    pub fdr_alpha: f32,

    #[arg(
        long = "q-temperature",
        default_value_t = 1.0,
        help = "Softmax temperature when row-normalizing Q over significant terms"
    )]
    pub q_temperature: f32,

    #[arg(
        long = "obo",
        help = "Cell Ontology .obo (e.g. cl-basic.obo). With --label-cl, runs TreeBH ontology \
                calling on the cluster × term matrix → {out}.ontology_assignment.tsv"
    )]
    pub obo: Option<Box<str>>,

    #[arg(
        long = "label-cl",
        help = "Curated `label<TAB>CL:id` map, one row per marker celltype. Required with --obo"
    )]
    pub label_cl: Option<Box<str>>,

    #[arg(
        long = "ontology-fdr-q",
        default_value_t = 0.1,
        help = "Ontology TreeBH per-level FDR target (lower → descends less, abstains more)"
    )]
    pub ontology_fdr_q: f64,

    #[arg(
        long = "ontology-by",
        help = "Ontology: Benjamini–Yekutieli within families (any dependence; more conservative)"
    )]
    pub ontology_by: bool,

    #[arg(
        long = "no-clean",
        help = "Keep existing {out}.* projection outputs (default: erase the explicit set first)"
    )]
    pub no_clean: bool,

    #[arg(short = 'v', long, help = "Verbose logging")]
    pub verbose: bool,
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
        help = "Per-level selective-FDR target (TreeBH); lower → descends less, abstains more"
    )]
    pub fdr_q: f64,

    #[arg(
        long = "by",
        default_value_t = false,
        help = "Benjamini–Yekutieli within families (valid under any dependence; more conservative)"
    )]
    pub by: bool,

    #[arg(
        long = "use-perm-p",
        default_value_t = false,
        help = "Force the (saturated) permutation p-values instead of the default z→p",
        long_help = "Force `cluster_celltype_p`. By default the walk scores on Φ(−z) using\n\
                     the correlation-preserving permutation z (`*_perm_z`) when present,\n\
                     else the row-randomization restandardized ES (`*_es_std`). The\n\
                     permutation p is resolution-limited (≈1/B) and rarely preferable."
    )]
    pub use_perm_p: bool,

    #[arg(short = 'v', long, help = "Verbose logging")]
    pub verbose: bool,
}
