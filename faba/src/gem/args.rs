use clap::Args;

use super::common::ComputeDevice;

/// CLI arguments for `faba gem` (alias `gem-embedding`).
///
/// Flag conventions mirror `senna bge` where applicable (`-i / --epochs`,
/// `-b / --batch-files`, `--learning-rate` with `--lr` alias,
/// `--preload-data`, `--device` / `--device-no`,
/// `--feature-name-delim` / `--feature-name-exact`,
/// `-o / --out`). Rmodem-specific knobs (per-stratum fractions,
/// negative counts, etc.) stay in their own block below.
#[derive(Args, Debug, Clone)]
pub struct GemArgs {
    /// Counts (gene-level) sparse matrix prefix(es), comma-separated.
    /// Rows must follow `{gene_key}/count/{spliced|unspliced}`. Multiple
    /// files are stacked under Union column alignment (cells merged by
    /// barcode); use an embedded `@batch` tag on the barcodes to keep
    /// samples as distinct batches (see `--batch-files`).
    #[arg(long, required = true, value_delimiter = ',')]
    pub genes: Vec<Box<str>>,

    /// m6A (DART-seq) sparse matrix prefix(es), comma-separated. Rows
    /// `{gene_key}/m6A/{component|chr:pos}`.
    #[arg(long, value_delimiter = ',')]
    pub dartseq: Option<Vec<Box<str>>>,

    /// A-to-I editing sparse matrix prefix(es), comma-separated. Rows
    /// `{gene_key}/A2I/{component|chr:pos}`.
    #[arg(long, value_delimiter = ',')]
    pub atoi: Option<Vec<Box<str>>>,

    /// Alternative-polyA sparse matrix prefix(es), comma-separated. Rows
    /// `{gene_key}/pA/{component|chr:pos}`.
    #[arg(long, value_delimiter = ',')]
    pub apa: Option<Vec<Box<str>>>,

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

    /// Optional batch labels. Under Union column alignment (gem's
    /// mode) exactly **one** file is expected, listing one label per
    /// unified cell — a barcode shared across modalities cannot carry two
    /// labels. As an alternative to this file, embed an `@batch` tag in
    /// the barcodes (e.g. `AAACCC@sampleA`); the loader infers and
    /// reconciles per-cell batches from those tags.
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
    // NOTE: gem rows are `{gene}/{modality}/{detail}` and the
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
    /// for gem (see note above).
    #[arg(long, default_value_t = '_')]
    pub feature_name_delim: char,

    /// Use exact row-name match across files (no canonicalization). The
    /// gem default — required because the `{gene}/{modality}/{detail}`
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

    /// Phase-2 gate: after phase 1 fixes the feature side (β/z/δ/γ + pb
    /// heads), phase 2 freezes it and re-evaluates the per-cell embedding
    /// `e_cell` against it. Phase 2 is now an **analytical** per-cell
    /// projection (Poisson MAP onto the frozen dictionary, solved in
    /// parallel — not SGD), so the numeric value is just an on/off gate:
    /// `0` skips phase 2, any non-zero (or omitted → default) runs it.
    /// Also skipped under `--no-cell-axis`. (The old name is kept for
    /// back-compat; the value no longer counts SGD epochs.)
    #[arg(long, help = "Phase-2 cell projection on/off (0 = skip; default = on)")]
    pub phase2_epochs: Option<usize>,

    /// Ridge prior strength λ on `e_cell` in the analytical phase-2
    /// projection. The Poisson MAP fits each cell's observed features and
    /// this Gaussian prior stands in for the (infeasible) all-feature
    /// partition — higher λ shrinks `e_cell` toward 0 / regularises cells
    /// with few features. The per-cell intercept `b_cell` is left
    /// unpenalised (it absorbs library size).
    #[arg(long, default_value_t = 1.0, help = "Phase-2 ridge prior on e_cell")]
    pub phase2_ridge: f32,

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

    /// L2 penalty `λ · mean(δ²)` on the program×modality deviation δ
    /// (`[K, M, H]`), mean-normalized — keeps the exp gate from blowing up.
    /// Default 1e-4. (Was `--q-l2`, kept as an alias.)
    #[arg(
        long = "delta-l2",
        alias = "q-l2",
        default_value_t = 1e-4,
        help = "L2 penalty on δ (program×modality deviation; mean-normalized)"
    )]
    pub delta_l2: f32,

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

    /// NB-Fisher housekeeping penalty (exponent on the per-gene Fisher
    /// weight `w_g = 1/(1 + π_g·s̄·φ(μ_g))`). Mirrors `senna bge/topic`:
    /// high-mean / high-dispersion genes (ribosomal, housekeeping,
    /// library-size drivers) are sampled less from the count-based anchor
    /// pools (agg + count-comp) so they stop monopolising the shared
    /// program loadings z. `0` disables (`w⁰ = 1`); `1` is full strength;
    /// `> 1` is more aggressive. The m6A / modifier pools are unaffected.
    /// Per-gene weights are written to `{out}.fisher_weights.parquet`.
    #[arg(
        long,
        default_value_t = 1.0,
        help = "NB-Fisher housekeeping penalty (exponent; 0 = off)"
    )]
    pub housekeeping_penalty: f32,

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

    ////////////////////////////////////////
    // Topic resolution (archetypal, post-training)
    //
    // Mirrors `senna bge --resolve-etm`: archetypal analysis on the
    // trained cell embedding e_cell → archetypes α (topic embeddings),
    // per-cell simplex θ (topic proportions), and a gene×topic dictionary
    // β = log_softmax(β_g·αᵀ). Writes the senna topic-model layout
    // (`{out}.{latent,dictionary,topic_embedding}.parquet`) consumed by
    // `senna {plot,clustering,annotate} --from`. No retraining.
    ////////////////////////////////////////
    /// Resolve archetype-based topics from the cell embedding after
    /// training and write the senna topic-model layout.
    #[arg(long, default_value_t = false)]
    pub resolve_topics: bool,

    /// Number of topics K for `--resolve-topics`. Omit to auto-select via
    /// an archetypal RSS-elbow sweep over `2..=--max-k`.
    #[arg(long = "num-topics")]
    pub num_topics: Option<usize>,

    /// Upper K for the `--resolve-topics` auto-sweep (when `--num-topics`
    /// is unset).
    #[arg(long = "max-k", default_value_t = 30)]
    pub max_k: usize,

    /// Archetypal-analysis alternating iterations for `--resolve-topics`.
    #[arg(long = "aa-iters", default_value_t = 50)]
    pub aa_iters: usize,

    /// Cap on cells used to fit archetypes for `--resolve-topics` (θ is
    /// still assigned for every cell). Unset → auto-caps at 50k cells when
    /// the dataset is larger (the K-sweep fits ~max-k times, so fitting on
    /// all cells dominates runtime); pass an explicit value to override.
    #[arg(long = "aa-subsample")]
    pub aa_subsample: Option<usize>,

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
