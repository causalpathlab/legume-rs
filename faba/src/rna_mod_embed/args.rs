use clap::Args;

use super::common::ComputeDevice;

/// CLI arguments for `faba rna-mod-embed` (aliases `rmodem`, `embed`).
///
/// Flag conventions mirror `senna bge` where applicable (`-i / --epochs`,
/// `-b / --batch-files`, `--learning-rate` with `--lr` alias,
/// `--preload-data`, `--device` / `--device-no`,
/// `--feature-name-delim` / `--feature-name-exact`,
/// `-o / --out`). Rmodem-specific knobs (per-stratum fractions,
/// negative counts, etc.) stay in their own block below.
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

    ////////////////////////////////////////
    // Component annotations (region binning)
    //
    // The `*_components.parquet` sidecars emitted by `faba m6a` / `faba
    // atoi` carry each GMM component's `mu` and `gene_length`. They feed
    // the transcript-position region bin (`u = mu/gene_length`) used by
    // the model's γ_{m,r,:} offset. The modality is inferred from the
    // flag, matching the row's `{gene}/{modality}/{component}` name.
    // Optional: a missing sidecar collapses that modality's γ to a
    // single region.
    ////////////////////////////////////////
    /// `m6a_components.parquet` for the `--dartseq` (m6A) modality.
    #[arg(long)]
    pub dartseq_components: Option<Box<str>>,

    /// `atoi_components.parquet` for the `--atoi` (A2I) modality.
    #[arg(long)]
    pub atoi_components: Option<Box<str>>,

    /// Component annotation parquet for the `--apa` (pA) modality.
    #[arg(long)]
    pub apa_components: Option<Box<str>>,

    /// Optional batch-label files, one per modality input.
    #[arg(short = 'b', long, value_delimiter = ',')]
    pub batch_files: Option<Vec<Box<str>>>,

    /// Output prefix.
    #[arg(short, long, required = true)]
    pub out: Box<str>,

    ////////////////////////////////////////
    // Model dims
    ////////////////////////////////////////
    /// Embedding dimension H (size of β_g, δ_{k,m,:}, γ_{m,r,:}).
    #[arg(long, default_value_t = 32)]
    pub embedding_dim: usize,

    /// Number of shared regulatory programs K.
    #[arg(long = "num-programs", default_value_t = 8)]
    pub n_programs: usize,

    /// Number of transcript-position region bins R for the additive
    /// γ_{m,r,:} offset. Components are binned by normalized 5'-relative
    /// position `u = mu/gene_length`. R=1 collapses γ to one per-modality
    /// offset (no positional resolution).
    #[arg(long = "num-regions", default_value_t = 5)]
    pub n_regions: usize,

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

    /// Train only on the pseudobulk axes — skip the cell-axis loss.
    /// For feature-embedding (β, z, δ, γ) quality the pb axes carry the
    /// dense, low-variance signal; the cell axis adds mostly Poisson
    /// noise on the modifier side. `e_cell` / `b_cell` still get
    /// allocated and written, but stay at their init values (zero bias,
    /// random embedding) — refit them post-hoc if needed.
    #[arg(long)]
    pub no_cell_axis: bool,

    ////////////////////////////////////////
    // Feature-name canonicalization (senna bge convention)
    //
    // NOTE: rmodem rows are `{gene}/{modality}/{detail}` and the
    // FeatureTable parser depends on that full path. The Gene-style
    // last-token canonicalizer (senna bge's default) would collapse
    // `gene_5/m6A/pos1234` to `1234` and silently drop every modifier
    // row, so we default to **exact** matching here. The delim flag is
    // still exposed for symmetry with bge / for input files that mix
    // `ENSG..._SYMBOL` prefixes inside the `{gene}` slot.
    ////////////////////////////////////////
    /// Delimiter for fuzzy gene-name matching across input files (last
    /// token after the split is the canonical row name). Ignored unless
    /// `--feature-name-exact` is *off* — which is **not** the default
    /// for rmodem (see note above).
    #[arg(long, default_value_t = '_')]
    pub feature_name_delim: char,

    /// Use exact row-name match across files (no canonicalization). The
    /// rmodem default — required because the `{gene}/{modality}/{detail}`
    /// row format is sensitive to suffix-splitting. Pass
    /// `--feature-name-exact=false` only if your `{gene}` slot itself
    /// carries a stripping suffix.
    #[arg(long, default_value_t = true)]
    pub feature_name_exact: bool,

    ////////////////////////////////////////
    // I/O knobs (senna bge convention)
    ////////////////////////////////////////
    /// Preload all sparse column data into memory before any pass over
    /// cells. On by default — much faster than repeated disk reads on
    /// typical SSDs, and required on slow disks. Pass `--no-preload-data`
    /// to fall back to streaming reads (use only for datasets that don't
    /// fit in RAM).
    #[arg(long = "no-preload-data", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub preload_data: bool,

    ////////////////////////////////////////
    // Training
    ////////////////////////////////////////
    #[arg(short = 'i', long, default_value_t = 30, help = "Training epochs")]
    pub epochs: usize,

    /// Batches per epoch. **Omit for auto** — one weighted pass over the
    /// largest axis (`ceil(max(n_cells, max_pb_per_level) / batch_size)`).
    /// Pass a value to force a fixed step budget per epoch (old behavior;
    /// historical default was 100).
    #[arg(
        long,
        help = "Batches per epoch (default: auto = one pass over largest axis)"
    )]
    pub batches_per_epoch: Option<usize>,

    #[arg(long, default_value_t = 1024, help = "Positive edges per batch")]
    pub batch_size: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        alias = "lr",
        help = "AdamW learning rate"
    )]
    pub learning_rate: f64,

    /// L2 penalty λ_z · mean(z²) on the per-gene program loadings.
    /// Matches `senna bge`'s `--feature-embedding-l2` style: mean-normalized
    /// so λ stays scale-invariant across (G·K). Default 1e-4 (mild).
    #[arg(
        long,
        default_value_t = 1e-4,
        help = "L2 penalty on z (mean-normalized)"
    )]
    pub z_l2: f32,

    /// L2 penalty λ_Q · mean(Q²) on the per-modality program signatures.
    /// Default 1e-4.
    #[arg(
        long,
        default_value_t = 1e-4,
        help = "L2 penalty on Q (mean-normalized)"
    )]
    pub q_l2: f32,

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
    /// Random negatives: pick another (g', m, c) row within the
    /// positive's stratum and modality. Tests gene-cell identification
    /// (β-side classification).
    #[arg(long, default_value_t = 10)]
    pub n_rand: usize,

    /// Swap-gene-mode negatives: keep β_g and (modality m, region r)
    /// fixed, but substitute the K-program loading z from another gene
    /// g'. Tests the gene's program-loading identity given the modality
    /// (z-side classification). Was `--n-swap-z` previously.
    #[arg(long, default_value_t = 5, alias = "n-swap-z")]
    pub n_swap_gene_mode: usize,

    /// Swap-modality negatives: keep (gene, cell) fixed but swap the
    /// satellite's `(modality, region)` axis to a different one. Forces
    /// δ/γ to keep each satellite distinguishable from the gene's base
    /// and other modalities (prevents the deviation gate collapsing).
    #[arg(long, default_value_t = 5)]
    pub n_swap_modality: usize,

    /// Weight satellite edges by the denoised modification fraction
    /// `w = X·r·π` instead of the raw modified-count `w = X·π`. Per the
    /// design's locked decision #2 this ships **model-side-first**: the
    /// fraction path (Phase 4, requires per-component converted/
    /// unconverted counts from the mixture pipeline) is not yet wired, so
    /// the default is `false` (raw counts) and `true` currently has no
    /// effect beyond a warning. Kept here so the CLI contract is stable
    /// for when Phase 4 lands.
    #[arg(long, default_value_t = false)]
    pub use_modification_fraction: bool,

    ////////////////////////////////////////
    // Misc
    ////////////////////////////////////////
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Compute device. `cuda`/`metal` require the matching cargo feature.
    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    /// Device ordinal (for `cuda` / `metal`).
    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,
}
