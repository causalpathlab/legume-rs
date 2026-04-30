use clap::Args;

#[derive(Args, Debug)]
pub struct AnnotateArgs {
    /// Run manifest produced by `senna topic|itopic|joint-topic|svd|joint-svd`.
    #[arg(short = 'f', long = "from", required = true)]
    pub from: Box<str>,

    /// Cluster parquet from `senna cluster` (cells × 1 cluster column,
    /// `NaN` for unassigned). When omitted, the path is read from
    /// `manifest.cluster.clusters` if populated; otherwise annotate
    /// runs Leiden internally on the latent matrix from the manifest.
    #[arg(short = 'c', long = "clusters")]
    pub clusters: Option<Box<str>>,

    /// Number of nearest neighbors for the cosine-KNN graph used by the
    /// internal Leiden clustering. Ignored when --clusters / manifest
    /// cluster path is provided.
    #[arg(long = "knn", default_value_t = 15)]
    pub knn: usize,

    /// Modularity resolution (CPM) for the internal Leiden clustering.
    /// Higher → more clusters. Ignored when clusters are supplied.
    #[arg(long = "resolution", default_value_t = 1.0)]
    pub resolution: f64,

    /// Optional target cluster count for Leiden auto-resolution. When set,
    /// resolution is binary-searched to approximate this. Ignored when
    /// clusters are supplied.
    #[arg(long = "num-clusters")]
    pub num_clusters: Option<usize>,

    /// Minimum cluster size; smaller clusters become unassigned.
    #[arg(long = "min-cluster-size", default_value_t = 2)]
    pub min_cluster_size: usize,

    /// Seed for internal Leiden clustering (deterministic).
    #[arg(long = "cluster-seed")]
    pub cluster_seed: Option<u64>,

    /// Marker-gene TSV: `gene<TAB>celltype` per line. Flexible delimiter
    /// (tab / comma / space). Symbol / alias matching via `flexible_gene_match`.
    #[arg(short = 'm', long = "markers", required = true)]
    pub markers: Box<str>,

    /// Output prefix for annotation artifacts.
    #[arg(short = 'o', long = "out", required = true)]
    pub out: Box<str>,

    /// Verbose logging.
    #[arg(short = 'v', long)]
    pub verbose: bool,

    /// Cells per CSC read block when streaming raw counts for per-cluster
    /// aggregation and NB-Fisher trend fitting. Larger blocks → fewer reads
    /// but more memory.
    #[arg(long = "block-size", default_value_t = 1024)]
    pub block_size: usize,

    /// Number of random gene-set draws per celltype for the
    /// Efron–Tibshirani row-randomization moments (used to restandardize
    /// both observed and permuted ES).
    #[arg(long = "num-draws", default_value_t = 1000)]
    pub num_draws: usize,

    /// Number of PB-level sample permutations for the correlation-preserving
    /// null (shuffle pb_membership, recompute β̃ = pb_gene · shuffled,
    /// ES per permutation, pool across clusters). Set 0 to fall back to
    /// row-randomization-based p-values (useful for small nClusters).
    #[arg(long = "num-perm", default_value_t = 500)]
    pub num_perm: usize,

    /// FDR α for the Q-matrix threshold.
    #[arg(long = "fdr-alpha", default_value_t = 0.10)]
    pub fdr_alpha: f32,

    /// Softmax temperature used when row-normalizing Q over significant
    /// entries. Lower → sharper; higher → more uniform.
    #[arg(long = "q-temperature", default_value_t = 1.0)]
    pub q_temperature: f32,

    /// Minimum cell-level confidence to emit a concrete label; below this
    /// cells are labeled `unassigned` in the argmax TSV.
    #[arg(long = "min-confidence", default_value_t = 0.0)]
    pub min_confidence: f32,

    /// RNG seed (deterministic; affects row randomization).
    #[arg(long = "seed", default_value_t = 42)]
    pub seed: u64,

    /// Preload columns into memory after opening the zarr/h5 backend.
    /// On slow disks this trades memory for I/O latency on later block reads.
    #[arg(long = "preload-data", default_value_t = false)]
    pub preload_data: bool,
}
