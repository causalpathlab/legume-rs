use clap::Args;

/// CLI arguments for `faba rna-mod-embed` (alias `rmodem`).
#[derive(Args, Debug, Clone)]
pub struct RnaModEmbedArgs {
    /// Counts (gene-level) sparse matrix prefix. Rows must follow
    /// `{gene_key}/count/{spliced|unspliced}`.
    #[arg(long, required = true)]
    pub genes: Box<str>,

    /// m6A (DART-seq) sparse matrix prefix. Rows
    /// `{gene_key}/m6A/{component|chr:pos}`.
    #[arg(long)]
    pub dartseq: Option<Box<str>>,

    /// A-to-I editing sparse matrix prefix. Rows
    /// `{gene_key}/A2I/{component|chr:pos}`.
    #[arg(long)]
    pub atoi: Option<Box<str>>,

    /// Alternative-polyA sparse matrix prefix. Rows
    /// `{gene_key}/pA/{component|chr:pos}`.
    #[arg(long)]
    pub apa: Option<Box<str>>,

    /// Optional batch-label files, one per modality input.
    #[arg(long, value_delimiter = ',')]
    pub batch_files: Option<Vec<Box<str>>>,

    /// Output prefix.
    #[arg(short, long, required = true)]
    pub out: Box<str>,

    ////////////////////////////////////////
    // Model dims
    ////////////////////////////////////////
    /// Embedding dimension H (size of ρ_g, Q_{k,m,:}).
    #[arg(short = 'd', long = "embedding-dim", default_value_t = 32)]
    pub embedding_dim: usize,

    /// Number of shared regulatory programs K.
    #[arg(short = 'k', long = "num-programs", default_value_t = 8)]
    pub n_programs: usize,

    ////////////////////////////////////////
    // Pseudobulk collapse
    ////////////////////////////////////////
    #[arg(long, default_value_t = 3)]
    pub num_levels: usize,

    #[arg(long, default_value_t = 10)]
    pub sort_dim: usize,

    #[arg(long, default_value_t = 10)]
    pub knn_pb: usize,

    #[arg(long, default_value_t = 100)]
    pub num_opt_iter: usize,

    #[arg(long, default_value_t = 64)]
    pub proj_dim: usize,

    /// Drop batch labels (treat all cells as one batch).
    #[arg(long)]
    pub ignore_batch: bool,

    ////////////////////////////////////////
    // Training
    ////////////////////////////////////////
    #[arg(long, default_value_t = 30)]
    pub epochs: usize,

    #[arg(long, default_value_t = 100)]
    pub batches_per_epoch: usize,

    #[arg(long, default_value_t = 1024)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    pub learning_rate: f64,

    ////////////////////////////////////////
    // Sampling strata
    ////////////////////////////////////////
    /// Fraction of positives drawn from AGG rows.
    #[arg(long, default_value_t = 0.25)]
    pub f_agg: f32,

    /// Fraction of positives drawn from count-component rows.
    #[arg(long, default_value_t = 0.25)]
    pub f_count: f32,

    /// Count-weight tempering exponent (τ ∈ [0, 1]; 1 = strict count-prop,
    /// 0 = uniform over rows with non-zero mass).
    #[arg(long, default_value_t = 1.0)]
    pub tau: f32,

    /// Modality-balance tempering exponent (τ_M ∈ [0, 1]; 1 = strict
    /// mass-prop, 0 = uniform across modalities).
    #[arg(long, default_value_t = 0.5)]
    pub tau_modality: f32,

    ////////////////////////////////////////
    // Negatives
    ////////////////////////////////////////
    #[arg(long, default_value_t = 10)]
    pub n_rand: usize,

    #[arg(long, default_value_t = 5)]
    pub n_swap_z: usize,

    #[arg(long, default_value_t = 5)]
    pub n_swap_q: usize,

    ////////////////////////////////////////
    // Misc
    ////////////////////////////////////////
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}
