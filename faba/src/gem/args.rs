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
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "Counts (gene-level) sparse matrix prefix(es), comma-separated",
        long_help = "Counts (gene-level) sparse matrix prefix(es), comma-separated.\n\
                     Rows must follow `{gene_key}/count/{spliced|unspliced}`. Multiple\n\
                     files are stacked under Union column alignment (cells merged by\n\
                     barcode); use an embedded `@batch` tag on the barcodes to keep\n\
                     samples as distinct batches (see `--batch-files`)."
    )]
    pub genes: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "m6A (DART-seq) sparse matrix prefix(es), comma-separated",
        long_help = "m6A (DART-seq) sparse matrix prefix(es), comma-separated. Rows\n\
                     `{gene_key}/m6A/{component|chr:pos}`."
    )]
    pub dartseq: Option<Vec<Box<str>>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "A-to-I editing sparse matrix prefix(es), comma-separated",
        long_help = "A-to-I editing sparse matrix prefix(es), comma-separated. Rows\n\
                     `{gene_key}/A2I/{component|chr:pos}`."
    )]
    pub atoi: Option<Vec<Box<str>>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Alternative-polyA sparse matrix prefix(es), comma-separated",
        long_help = "Alternative-polyA sparse matrix prefix(es), comma-separated. Rows\n\
                     `{gene_key}/pA/{component|chr:pos}`."
    )]
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
    #[arg(
        long,
        help = "`m6a_components.parquet` for the `--dartseq` (m6A) modality"
    )]
    pub dartseq_components: Option<Box<str>>,

    #[arg(
        long,
        help = "`atoi_components.parquet` for the `--atoi` (A2I) modality"
    )]
    pub atoi_components: Option<Box<str>>,

    #[arg(
        long,
        help = "Component annotation parquet for the `--apa` (pA) modality"
    )]
    pub apa_components: Option<Box<str>>,

    #[arg(
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Optional batch labels",
        long_help = "Optional batch labels. Under Union column alignment (gem's\n\
                     mode) exactly **one** file is expected, listing one label per\n\
                     unified cell — a barcode shared across modalities cannot carry two\n\
                     labels. As an alternative to this file, embed an `@batch` tag in\n\
                     the barcodes (e.g. `AAACCC@sampleA`); the loader infers and\n\
                     reconciles per-cell batches from those tags."
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(short, long, required = true, help = "Output prefix")]
    pub out: Box<str>,

    ////////////////////////////////////////
    // Model dims
    ////////////////////////////////////////
    #[arg(
        long,
        default_value_t = 32,
        help = "Embedding dimension H (size of β_g, δ_{k,m,:}, γ_{m,r,:})"
    )]
    pub embedding_dim: usize,

    #[arg(
        long = "num-programs",
        default_value_t = 8,
        help = "Number of shared regulatory programs K"
    )]
    pub n_programs: usize,

    #[arg(
        long = "num-regions",
        default_value_t = 5,
        help = "Number of transcript-position region bins R for the additive γ_{m,r,:} offset",
        long_help = "Number of transcript-position region bins R for the additive\n\
                     γ_{m,r,:} offset. Components are binned by normalized 5'-relative\n\
                     position `u = mu/gene_length`. R=1 collapses γ to one per-modality\n\
                     offset (no positional resolution)."
    )]
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

    #[arg(long, help = "Drop batch labels (treat all cells as one batch)")]
    pub ignore_batch: bool,

    #[arg(
        long = "min-cell-nnz",
        default_value_t = 2,
        help = "Cell QC: minimum number of detected features (nonzeros) a cell must have across all modalities",
        long_help = "Cell QC: minimum number of detected features (nonzeros) a cell must\n\
                     have **across all modalities** (count + m6A/A2I/APA) to be embedded\n\
                     and written. Modality-agnostic on purpose — coverage is skewed\n\
                     across modalities, so this keeps any cell with signal in *any* one,\n\
                     dropping only near-empty cells. Under Union alignment a barcode seen\n\
                     in just one sparse modality (e.g. a single stray editing read, no\n\
                     counts) is a degenerate \"cell\" the count-anchored phase-2 projection\n\
                     maps to ~0; the default `2` drops exactly those while keeping every\n\
                     cell that yields a real embedding. Set `1` to keep all but fully\n\
                     empty cells; raise for stricter QC. No data is rewritten — this is a\n\
                     write-time selection, not a squeeze."
    )]
    pub min_cell_nnz: usize,

    #[arg(
        long,
        help = "Feature-only mode: skip the cell axis in phase 1 and skip the phase-2 cell projection",
        long_help = "Feature-only mode: skip the cell axis in phase 1 **and** skip the\n\
                     phase-2 cell projection. `e_cell` / `b_cell` are still allocated and\n\
                     written, but stay at their init values (zero bias, random embedding)\n\
                     — refit them post-hoc if needed. This differs from\n\
                     `--phase1-cells-per-pb 0` (the default), which also drops the cell\n\
                     axis from phase 1 but *still* projects every cell in phase 2."
    )]
    pub no_cell_axis: bool,

    #[arg(
        long = "phase1-cells-per-pb",
        default_value_t = 0,
        help = "Phase-1 cell-axis mode (k): controls what shapes the feature dictionary in phase 1",
        long_help = "Phase-1 cell-axis mode (`k`). Controls only what shapes the feature\n\
                     dictionary (β/z/δ/γ) in phase 1; phase 2 ALWAYS analytically projects\n\
                     every cell (unless `--no-cell-axis` / `--phase2-epochs 0`), so the\n\
                     per-cell embedding output is essentially unaffected by `k`.\n\
                     \n\
                       k = 0 (default) → suppress the cell axis in phase 1 (pure-pb:\n\
                         features shaped by pb aggregates only). Fastest, because the\n\
                         per-epoch step budget is then sized by the pb levels, not n_cells.\n\
                       1 ≤ k < n_cells → keep ≤k cells per pb-sample at EVERY collapse\n\
                         level (union), shrinking the phase-1 budget while keeping rare /\n\
                         shallow cells visible to the shared dictionary.\n\
                       k ≥ n_cells → every cell shapes the dictionary (legacy all-cells;\n\
                         slowest)."
    )]
    pub phase1_cells_per_pb: usize,

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
    #[arg(
        long,
        default_value_t = '_',
        help = "Delimiter for fuzzy gene-name matching across input files",
        long_help = "Delimiter for fuzzy gene-name matching across input files (last\n\
                     token after the split is the canonical row name). Ignored unless\n\
                     `--feature-name-exact` is *off* — which is **not** the default\n\
                     for gem (see note above)."
    )]
    pub feature_name_delim: char,

    #[arg(
        long,
        default_value_t = true,
        help = "Use exact row-name match across files (no canonicalization)",
        long_help = "Use exact row-name match across files (no canonicalization). The\n\
                     gem default — required because the `{gene}/{modality}/{detail}`\n\
                     row format is sensitive to suffix-splitting. Pass\n\
                     `--feature-name-exact=false` only if your `{gene}` slot itself\n\
                     carries a stripping suffix."
    )]
    pub feature_name_exact: bool,

    ////////////////////////////////////////
    // I/O knobs (senna bge convention)
    ////////////////////////////////////////
    #[arg(
        long = "no-preload-data",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Preload all sparse column data into memory before any pass over cells",
        long_help = "Preload all sparse column data into memory before any pass over\n\
                     cells. On by default — much faster than repeated disk reads on\n\
                     typical SSDs, and required on slow disks. Pass `--no-preload-data`\n\
                     to fall back to streaming reads (use only for datasets that don't\n\
                     fit in RAM)."
    )]
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
    #[arg(
        long,
        default_value_t = 0.25,
        help = "Fraction of positives drawn from AGG rows"
    )]
    pub f_agg: f32,

    #[arg(
        long,
        default_value_t = 0.25,
        help = "Fraction of positives drawn from count-component rows"
    )]
    pub f_count: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Count-weight tempering exponent (τ ∈ [0, 1]; 1 = strict count-prop, 0 = uniform)",
        long_help = "Count-weight tempering exponent (τ ∈ [0, 1]; 1 = strict count-prop,\n\
                     0 = uniform over rows with non-zero mass)."
    )]
    pub tau: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Modality-balance tempering exponent (τ_M ∈ [0, 1]; 1 = strict mass-prop, 0 = uniform)",
        long_help = "Modality-balance tempering exponent (τ_M ∈ [0, 1]; 1 = strict\n\
                     mass-prop, 0 = uniform across modalities)."
    )]
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
    #[arg(
        long,
        default_value_t = 10,
        help = "Random negatives: pick another (g', m, c) row within the positive's stratum and modality",
        long_help = "Random negatives: pick another (g', m, c) row within the\n\
                     positive's stratum and modality. Tests gene-cell identification\n\
                     (β-side classification)."
    )]
    pub n_rand: usize,

    #[arg(
        long,
        default_value_t = 5,
        alias = "n-swap-z",
        help = "Swap-gene-mode negatives: keep β_g and modality fixed, substitute z from another gene",
        long_help = "Swap-gene-mode negatives: keep β_g and (modality m, region r)\n\
                     fixed, but substitute the K-program loading z from another gene\n\
                     g'. Tests the gene's program-loading identity given the modality\n\
                     (z-side classification). Was `--n-swap-z` previously."
    )]
    pub n_swap_gene_mode: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Swap-modality negatives: keep (gene, cell) fixed but swap the satellite's (modality, region) axis",
        long_help = "Swap-modality negatives: keep (gene, cell) fixed but swap the\n\
                     satellite's `(modality, region)` axis to a different one. Forces\n\
                     δ/γ to keep each satellite distinguishable from the gene's base\n\
                     and other modalities (prevents the deviation gate collapsing)."
    )]
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
    #[arg(
        long,
        default_value_t = false,
        help = "Resolve archetype-based topics from the cell embedding after training"
    )]
    pub resolve_topics: bool,

    #[arg(
        long = "num-topics",
        help = "Number of topics K for `--resolve-topics`",
        long_help = "Number of topics K for `--resolve-topics`. Omit to auto-select via\n\
                     an archetypal RSS-elbow sweep over `2..=--max-k`."
    )]
    pub num_topics: Option<usize>,

    #[arg(
        long = "max-k",
        default_value_t = 30,
        help = "Upper K for the `--resolve-topics` auto-sweep (when `--num-topics` is unset)"
    )]
    pub max_k: usize,

    #[arg(
        long = "aa-iters",
        default_value_t = 50,
        help = "Archetypal-analysis alternating iterations for `--resolve-topics`"
    )]
    pub aa_iters: usize,

    #[arg(
        long = "aa-subsample",
        help = "Cap on cells used to fit archetypes for `--resolve-topics`",
        long_help = "Cap on cells used to fit archetypes for `--resolve-topics` (θ is\n\
                     still assigned for every cell). Unset → auto-caps at 50k cells when\n\
                     the dataset is larger (the K-sweep fits ~max-k times, so fitting on\n\
                     all cells dominates runtime); pass an explicit value to override."
    )]
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
