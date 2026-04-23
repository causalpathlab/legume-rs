use clap::Args;

#[derive(Args, Debug)]
pub struct AnnotateArgs {
    /// Run manifest produced by `senna topic|itopic|joint-topic|svd|joint-svd`.
    #[arg(short = 'f', long = "from", required = true)]
    pub from: Box<str>,

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

    /// Number of random gene-set draws per celltype for the
    /// Efron–Tibshirani row-randomization moments (used to restandardize
    /// both observed and permuted ES).
    #[arg(long = "num-draws", default_value_t = 1000)]
    pub num_draws: usize,

    /// Number of PB-level sample permutations for the correlation-preserving
    /// null (shuffle pb_membership, recompute β̃ = pb_gene · shuffled,
    /// ES per permutation, pool across topics). Set 0 to fall back to
    /// row-randomization-based p-values (useful for K ≤ 5).
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
}
