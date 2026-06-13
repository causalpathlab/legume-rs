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

    ////////////////////////////////////////////////////////////////////////
    // Per-cell sample identity (Union loader)
    //
    // Under Union column alignment, cells merge by raw barcode. To keep
    // distinct biological samples apart — and to MERGE a sample's modalities
    // into one joint cell — gem tags each input file's barcodes with a sample
    // id (`barcode@sample`) before the merge. The sample id is the file's
    // basename with the per-flag suffix below stripped, so e.g. stripping
    // `_genes` from `rep1_wt_genes` and `_m6a_mixture` from `rep1_wt_m6a_mixture`
    // both yield `rep1_wt` → their matched cells merge. The `@sample` tag is
    // also read back as the per-cell batch label. Empty (default) = tag with
    // the full basename (samples stay distinct, but modalities of one sample
    // do NOT merge unless their basenames already match). Skipped entirely
    // when `--batch-files` is given or barcodes already carry an `@` tag.
    ////////////////////////////////////////////////////////////////////////
    #[arg(
        long,
        default_value = "",
        help = "Strip this suffix from each --genes file basename to form its sample id"
    )]
    pub genes_sample_strip: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Strip this suffix from each --dartseq file basename to form its sample id"
    )]
    pub dartseq_sample_strip: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Strip this suffix from each --atoi file basename to form its sample id"
    )]
    pub atoi_sample_strip: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Strip this suffix from each --apa file basename to form its sample id"
    )]
    pub apa_sample_strip: Box<str>,

    #[arg(
        long = "no-auto-cell-cutoff",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Disable automatic cell calling (keep every cell with any spliced signal)",
        long_help = "Automatic cell calling (ON by default; pass `--no-auto-cell-cutoff`\n\
                     to disable). Picks the spliced-nnz cutoff by 2-means clustering of\n\
                     log(1+nnz) — the ambient↔real-cell boundary, same routine as\n\
                     `data-beans squeeze` — prints the spliced-nnz histogram with the\n\
                     applied cutoff marked, then **subsets the cells up front** so the\n\
                     collapse + training + outputs all run on the called cells only (faster,\n\
                     and ambient never shapes the model). nnz is counted over the spliced\n\
                     rows — the features that drive the collapse. There is no separate\n\
                     manual floor: when the distribution is unimodal (no real ambient↔cell\n\
                     split) or when this flag disables auto-calling, only genuinely\n\
                     zero-spliced cells (unplaceable by the spliced projection) are dropped."
    )]
    pub auto_cell_cutoff: bool,

    #[arg(
        long,
        help = "Feature-only mode: skip cell axis in phase 1 and phase-2 projection",
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
        help = "Phase-1 cell-axis mode (k): shapes the feature dictionary in phase 1",
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

    //////////////
    // Training //
    //////////////
    #[arg(short = 'i', long, default_value_t = 1000, help = "Training epochs")]
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

    ///////////////////////
    // Sampling features //
    ///////////////////////
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
        help = "Count-weight tempering exponent (τ ∈ [0,1]; 1=count-prop, 0=uniform)",
        long_help = "Count-weight tempering exponent (τ ∈ [0, 1]; 1 = strict count-prop,\n\
                     0 = uniform over rows with non-zero mass)."
    )]
    pub tau: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Modality-balance tempering exponent (τ_M ∈ [0,1]; 1=mass-prop, 0=uniform)",
        long_help = "Modality-balance tempering exponent (τ_M ∈ [0, 1]; 1 = strict\n\
                     mass-prop, 0 = uniform across modalities)."
    )]
    pub tau_modality: f32,

    ///////////////
    // Negatives //
    ///////////////
    #[arg(
        long,
        default_value_t = 10,
        help = "Random negatives: pick another (g', m, c) within the positive's stratum",
        long_help = "Random negatives: pick another (g', m, c) row within the\n\
                     positive's stratum and modality. Tests gene-cell identification\n\
                     (β-side classification)."
    )]
    pub n_rand: usize,

    #[arg(
        long,
        default_value_t = 5,
        alias = "n-swap-z",
        help = "Swap-gene-mode negatives: keep β_g fixed, substitute z from another gene",
        long_help = "Swap-gene-mode negatives: keep β_g and (modality m, region r)\n\
                     fixed, but substitute the K-program loading z from another gene\n\
                     g'. Tests the gene's program-loading identity given the modality\n\
                     (z-side classification). Was `--n-swap-z` previously."
    )]
    pub n_swap_gene_mode: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Swap-modality negatives: keep (gene, cell) fixed, swap satellite modality/region",
        long_help = "Swap-modality negatives: keep (gene, cell) fixed but swap the\n\
                     satellite's `(modality, region)` axis to a different one. Forces\n\
                     δ/γ to keep each satellite distinguishable from the gene's base\n\
                     and other modalities (prevents the deviation gate collapsing)."
    )]
    pub n_swap_modality: usize,

    /////////////////////////////////////////////////////////////////////////
    // Topic resolution (archetypal, post-training)			   //
    // 									   //
    // Mirrors `senna bge --resolve-etm`: archetypal analysis on the	   //
    // trained cell embedding e_cell → archetypes α (topic embeddings),	   //
    // per-cell simplex θ (topic proportions), and a gene×topic dictionary //
    // β = log_softmax(β_g·αᵀ). Writes the senna topic-model layout	   //
    // (`{out}.{latent,dictionary,topic_embedding}.parquet`) consumed by   //
    // `senna {plot,clustering,annotate} --from`. No retraining.	   //
    /////////////////////////////////////////////////////////////////////////
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
                     a SPA-anchor residual-elbow sweep over `2..=H+1` (H = embedding dim)."
    )]
    pub num_topics: Option<usize>,

    ///////////////
    // Refinement //
    ///////////////
    #[arg(
        long,
        default_value_t = false,
        help = "Run a second training pass after filtering dead genes and cells",
        long_help = "Run a second full training pass.  After pass-1 completes, the gene\n\
                     embeddings are k-means'd into --gene-groups clusters; clusters whose\n\
                     median per-gene count mass is below --empty-group-frac × the richest\n\
                     cluster are 'empty'.  Genes in empty clusters, and cells whose\n\
                     embedding direction lands in one, are removed; the pseudobulk, feature\n\
                     table, and model are rebuilt from the filtered data and training\n\
                     restarts from scratch."
    )]
    pub refine: bool,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Refine pass: target FDR for the empirical-Bayes null-gene call (keep genes significant above the null)",
        long_help = "Target false-discovery rate for the --refine gene QC.  A null gene\n\
                     (one the model never moved from init β_g ~ N(0,σ²I)) has ‖β_g‖²/σ² ~\n\
                     χ²_H; the null scale σ̂² and proportion π̂₀ are estimated from the data\n\
                     (ashr-style), each gene gets a χ²_H upper-tail p-value and a Storey\n\
                     q-value, and genes with q ≤ this FDR are kept (signal above the null).\n\
                     Per-gene and deterministic (no clustering).  Only used when --refine is\n\
                     active.  Must be in (0, 1)."
    )]
    pub gene_null_fdr: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Refine pass: drop a cell when its pre-L2 embedding norm is below this × the median",
        long_help = "Per-cell depth cutoff for the --refine QC.  e_cell is L2-normalised\n\
                     before storage, so the pre-L2 norm (√emb_sq_norm from the phase-2\n\
                     Poisson-MAP fit) is the only depth-aware signal that separates a\n\
                     near-empty droplet — which reconstructs to the gene-bias background —\n\
                     from a low-depth real cell.  A cell is dropped when that norm is below\n\
                     cell_min_norm_frac × the median norm of the cutoff-passing cells.  The\n\
                     per-cell norm and nnz, both cutoffs, and the pass/fail flags are written\n\
                     to {out}.cell_qc.parquet.  Only used when --refine is active.\n\
                     Must be in [0, 1)."
    )]
    pub cell_min_norm_frac: f32,

    //////////
    // Misc //
    //////////
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Compute device. `cuda`/`metal` require the matching cargo feature.
    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    /// Device ordinal (for `cuda` / `metal`).
    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,

    /// Number of CPU threads for rayon-parallel work (HNSW, collapse, phase-2
    /// cell projection). Defaults to all available logical CPUs.
    #[arg(long, default_value_t = 0, help = "CPU threads (0 = all available)")]
    pub threads: usize,
}

impl GemArgs {
    /// Whether phase 1 draws the cell axis (and so the per-cell pools are
    /// built). Off in feature-only mode and in the default pure-pb path
    /// (`--phase1-cells-per-pb 0`), where phase 2 streams from the backend.
    /// Single source of truth for the pool-vs-stream decision.
    pub fn use_phase1_cell_axis(&self) -> bool {
        !self.no_cell_axis && self.phase1_cells_per_pb != 0
    }
}
