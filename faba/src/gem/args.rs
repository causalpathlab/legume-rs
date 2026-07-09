use clap::Args;

use super::common::ComputeDevice;

/// Model dimensions.
#[derive(Args, Debug, Clone)]
pub struct ModelArgs {
    #[arg(
        long,
        default_value_t = 32,
        help = "Embedding dimension H (size of β_g and the cell embedding)"
    )]
    pub embedding_dim: usize,

    #[arg(
        long = "delta-l2",
        default_value_t = 0.0,
        help = "L2 (ridge) weight on the per-gene splice offset δ_g (0 = auto: a mild ridge is applied when unspliced rows are present)",
        long_help = "L2 (ridge) penalty on the per-gene splice offset δ_g. When 0 (default) and\n\
                     the input carries unspliced rows, gem auto-applies a mild ridge (L2=1.0) so a\n\
                     δ_g dictionary is always written for `faba annotate --track velocity`; set an\n\
                     explicit value to override, or 0 on a spliced-only input keeps δ off. When > 0,\n\
                     unspliced rows embed as β_g + δ_g with a ridge-shrunk δ_g learned in phase\n\
                     1: it absorbs the (dense) static per-gene nascent structure (the RNA-\n\
                     velocity γ) so cell identity (spliced θ) stays clean and the phase-2\n\
                     velocity increment δ (raw Poisson-MAP shift, θ fixed) becomes\n\
                     γ-calibrated. Larger = more shrinkage\n\
                     (δ_g pulled toward 0). Try 0.01–1.0; δ_g is written to\n\
                     `{out}.delta_dictionary.parquet`."
    )]
    pub delta_l2: f32,

    #[arg(
        long = "feature-null-fdr",
        default_value_t = 0.05,
        help = "Empirical-Bayes feature-null QC: drop genes whose β never moved off init at this FDR, then re-fit (default 0.05; 0 = off)",
        long_help = "Empirical-Bayes feature-null QC — ON by default at FDR 0.05, the same\n\
                     shared engine call `senna bge` uses. After phase 1, each feature row's\n\
                     ‖e_feat‖² (its materialized β_g) is tested against an estimated null: a\n\
                     gene the model never moved keeps ‖e_feat‖² ~ σ²·χ²_ν, and the scale σ̂²,\n\
                     effective dof ν̂, and null proportion π̂₀ are estimated from the data so\n\
                     each row gets a BH q-value. Rows with q > this FDR are the untrained\n\
                     low-detection background (the 'empty' genes that pile at the co-embed\n\
                     centre) and are DROPPED; the model then re-fits on the live feature axis\n\
                     (two-pass refine). β-sharing means a null gene drops both its spliced and\n\
                     unspliced tracks together. The live/null flags and norm² are written to\n\
                     `{out}.feature_qc.parquet`. This is the automatic, data-driven complement\n\
                     to the `--n-hvg` top-N cut (which runs first, up front); leaving both on\n\
                     gives the strongest feature gate. 0 disables. Must be in [0, 1)."
    )]
    pub feature_null_fdr: f32,
}

/// Pseudobulk collapse, phase-1 cell-axis mode, per-file sample identity, and
/// feature-name canonicalization — everything that shapes how cells/features are
/// grouped and matched before training.
///
/// Per-cell sample identity (Union loader): under Union column alignment, cells
/// merge by raw barcode. To keep distinct biological samples apart, gem tags
/// each input file's barcodes with a sample id (`barcode@sample`) before the
/// merge. The sample id is the file's basename with `--genes-sample-strip`
/// removed (e.g. `_genes` from `rep1_wt_genes` → `rep1_wt`). The `@sample` tag
/// is also read back as the per-cell batch label. Skipped when `--batch-files`
/// is given or barcodes already carry an `@` tag.
///
/// Feature-name canonicalization: gem rows are `{gene}/count/{spliced|unspliced}`
/// and the per-gene β-sharing factorization depends on that full path, so we
/// default to **exact** matching. The delim flag is exposed for input files that
/// carry an `ENSG..._SYMBOL` prefix inside the `{gene}` slot.
#[derive(Args, Debug, Clone)]
pub struct CollapseArgs {
    #[arg(
        long,
        default_value_t = 3,
        help = "Number of pseudobulk collapse levels (coarse→fine); each level is a training axis"
    )]
    pub num_levels: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Bits of the binary projection sketch used to hash cells into the finest pb-samples (≤ 2^sort_dim codes)"
    )]
    pub sort_dim: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "kNN neighbours for the cross-batch pseudobulk matching during collapse"
    )]
    pub knn_pb: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Optimization iterations for the pseudobulk collapse/refine"
    )]
    pub num_opt_iter: usize,

    #[arg(
        long,
        default_value_t = 64,
        help = "Random-projection dimension for the batch-corrected sketch that drives collapse"
    )]
    pub proj_dim: usize,

    #[arg(long, help = "Drop batch labels (treat all cells as one batch)")]
    pub ignore_batch: bool,

    #[arg(
        long = "phase1-cells-per-pb",
        default_value_t = 0,
        help = "Phase-1 cell-axis mode (k): shapes the feature dictionary in phase 1",
        long_help = "Phase-1 cell-axis mode (`k`). Controls only what shapes the feature\n\
                     dictionary (β) in phase 1; phase 2 ALWAYS analytically projects every\n\
                     cell, so the per-cell embedding output is essentially unaffected by `k`.\n\
                     \n\
                       k = 0 (default) → suppress the cell axis in phase 1 (pure-pb:\n\
                         features shaped by pb aggregates only). Fastest.\n\
                       1 ≤ k < n_cells → keep ≤k cells per pb-sample at EVERY collapse\n\
                         level (union), shrinking the phase-1 budget while keeping rare /\n\
                         shallow cells visible to the shared dictionary.\n\
                       k ≥ n_cells → every cell shapes the dictionary (slowest)."
    )]
    pub phase1_cells_per_pb: usize,

    #[arg(
        long = "n-hvg",
        default_value_t = 5000,
        help = "Keep only the top-N highly-variable genes (default 5000, matching senna/pinto; 0 = all genes)",
        long_help = "Gene-level HVG feature filter (like `senna bge`). Selects the top-N most\n\
                     variable GENES (NB dispersion-trend, spliced+unspliced pooled) and drops the\n\
                     rest — both the spliced and unspliced rows of a dropped gene go together so\n\
                     the β-sharing factorization stays aligned. This removes the abundant, uniform\n\
                     housekeeping/ribosomal genes that otherwise dominate the positive edges and\n\
                     collapse every cell onto one point; it shrinks the dictionary and restricts\n\
                     the pseudobulk projection/membership to the kept genes. Defaults to 5000 for\n\
                     consistency with `senna bge` / `pinto`; `0` keeps every gene (expect a\n\
                     proliferation/housekeeping-dominated collapse on rich data). Try 2000–5000."
    )]
    pub n_hvg: usize,

    #[arg(
        long,
        default_value = "",
        help = "Strip this suffix from each --genes file basename to form its sample id"
    )]
    pub genes_sample_strip: Box<str>,

    #[arg(
        long,
        default_value_t = '_',
        help = "Delimiter for fuzzy gene-name matching across input files",
        long_help = "Delimiter for fuzzy gene-name matching across input files \n\
		     (last token after the split is the canonical row name). \n\
		     Ignored unless `--feature-name-exact` is *off*."
    )]
    pub feature_name_delim: char,

    #[arg(
        long,
        default_value_t = true,
        help = "Use exact row-name match across files (no canonicalization)",
        long_help = "Use exact row-name match across files (no canonicalization). \n\
		     The gem default — required because the `{gene}/count/{spliced|unspliced}`\n\
                     row format is sensitive to suffix-splitting. Pass\n\
                     `--feature-name-exact=false` only if your `{gene}` slot itself\n\
                     carries a stripping suffix."
    )]
    pub feature_name_exact: bool,
}

/// Training: optimizer schedule for the phase-1 pseudobulk fit.
#[derive(Args, Debug, Clone)]
pub struct TrainArgs {
    #[arg(short = 'i', long, default_value_t = 1000, help = "Training epochs")]
    pub epochs: usize,

    #[arg(
        long,
        help = "Batches per epoch (default: auto = one pass over largest axis)",
        long_help = "Batches per epoch. Omit for auto — one weighted pass over the\n\
                     largest axis (`ceil(max(n_cells, max_pb_per_level) / batch_size)`)."
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

    #[arg(
        long = "max-grad-norm",
        default_value_t = 1.0,
        help = "Global-norm gradient clip for phase-1 AdamW (0 = off).\n\
		When > 0, each step's gradients are scaled down \n\
		if their global L2 norm exceeds this, bounding embedding inflation on loss spikes."
    )]
    pub max_grad_norm: f32,

    #[arg(
        long = "lineage-dag",
        default_value_t = false,
        help = "Inject developmental structure at pseudobulk scale (experimental; default off).",
        long_help = "Shape the embedding along a pseudobulk lineage. When set, gem reads the\n\
                     pb-level velocity (identity θ_pb + velocity δ_pb per pseudobulk per collapse\n\
                     level), orients a directed lineage over the pseudobulks, and runs a SECOND\n\
                     phase-1 pass with a velocity-drift term so the shared feature dictionary\n\
                     picks up that lineage geometry — then lifts a per-cell pseudotime + fate\n\
                     (`{out}.dag_pseudotime.parquet` / `{out}.dag_fate.parquet`). It uses the\n\
                     LEARNED DAG by default (see `--fixed-dag`). Off by default — the per-cell\n\
                     embedding is then byte-identical to a plain run; turning it ON changes the\n\
                     embedding (the second pass). Only meaningful with spliced+unspliced input\n\
                     (β-sharing)."
    )]
    pub lineage_dag: bool,

    #[arg(
        long = "fixed-dag",
        default_value_t = false,
        help = "Lineage-DAG: use the fixed velocity-KNN graph instead of the default learned DAG.",
        long_help = "Within `--lineage-dag`, orient the pb structure with a FIXED velocity-oriented\n\
                     KNN graph, instead of LEARNING the directed adjacency `W` (the default).\n\
                     Learning co-refines `W` with the embedding (velocity-drift SEM + DAGMA-style\n\
                     acyclicity + L1 + velocity-orientation prior) and gives a cleaner single-\n\
                     lineage structure; the fixed graph is faster but more fragmented. Ignored\n\
                     unless `--lineage-dag` is set."
    )]
    pub fixed_dag: bool,

    #[arg(
        long = "lineage-smooth",
        default_value_t = false,
        help = "Lineage-DAG: smooth the pb velocity readout δ_pb (opt-in).",
        long_help = "Smooth the pb velocity readout δ_pb over θ-space KNN neighbours before it\n\
                     orients the lineage graph, stabilizing sign(δ_pb). A wash on clean data\n\
                     (no noise to remove, and it can blur branch-point velocity), so it is off\n\
                     by default — the payoff is on noisy real spliced/unspliced ratios. Ignored\n\
                     unless `--lineage-dag` is set."
    )]
    pub lineage_smooth: bool,
}

/// Runtime knobs: data preload, RNG seed, compute device, threads.
#[derive(Args, Debug, Clone)]
pub struct RuntimeArgs {
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

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = ComputeDevice::Cpu,
        value_enum,
        help = "Compute device",
        long_help = "Compute device. `cuda` / `metal` require the matching cargo feature."
    )]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,

    #[arg(
        long,
        alias = "max-threads",
        default_value_t = 16,
        help = "CPU threads (0 = all available)",
        long_help = "Number of CPU threads for rayon-parallel work (HNSW, collapse,\n\
                     phase-2 cell projection). Defaults to all available logical\n\
                     CPUs (0 = all)."
    )]
    pub threads: usize,
}

/// CLI arguments for `faba gem` (alias `gem-embedding`).
///
/// Joint embedding of gene counts (spliced + unspliced) into one cell/gene
/// space over the shared `graph_embedding_util` engine. Each row
/// `{gene}/count/{spliced|unspliced}` embeds as `β_g` (β-sharing); cell identity
/// is the spliced projection θ and the splice contrast is a velocity δ on the
/// cell axis (`{out}.velocity.parquet`).
///
/// Flag conventions mirror `senna bge` where applicable (`-i / --epochs`,
/// `-b / --batch-files`, `--learning-rate` with `--lr` alias,
/// `--preload-data`, `--device` / `--device-no`,
/// `--feature-name-delim` / `--feature-name-exact`, `-o / --out`).
#[derive(Args, Debug, Clone)]
pub struct GemArgs {
    #[arg(
        value_name = "GENES",
        value_delimiter = ',',
        help = "Counts (gene-level) sparse matrix prefix(es), space- or comma-separated",
        long_help = "Counts (gene-level) sparse matrix prefix(es), given positionally —\n\
                     space-separated, so shell globs work: `faba gem out/*_genes.zarr.zip`.\n\
                     Commas are also accepted. Rows must follow\n\
                     `{gene_key}/count/{spliced|unspliced}`. Multiple files are stacked\n\
                     under Union column alignment (cells merged by barcode); use an\n\
                     embedded `@batch` tag on the barcodes to keep samples as distinct\n\
                     batches (see `--batch-files`).\n\n\
                     The `--genes a,b` flag form is still accepted, but pass one or the\n\
                     other, not both."
    )]
    pub genes_pos: Vec<Box<str>>,

    #[arg(
        long = "genes",
        value_delimiter = ',',
        help = "Deprecated alias for the positional GENES argument (comma-separated)"
    )]
    pub genes_flag: Vec<Box<str>>,

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

    #[command(flatten)]
    pub model: ModelArgs,

    #[command(flatten)]
    pub collapse: CollapseArgs,

    #[command(flatten)]
    pub train: TrainArgs,

    #[command(flatten)]
    pub runtime: RuntimeArgs,
}

impl GemArgs {
    /// The gene matrices to load, from whichever form the user gave.
    ///
    /// Positional is the primary spelling (`faba gem a.zarr.zip b.zarr.zip`, so shell
    /// globs work); `--genes a,b` is kept for the existing scripts. Accepting both at
    /// once would silently pick one, so it is an error — the user's intent is
    /// genuinely ambiguous there.
    pub fn genes(&self) -> anyhow::Result<&[Box<str>]> {
        match (self.genes_pos.is_empty(), self.genes_flag.is_empty()) {
            (false, true) => Ok(&self.genes_pos),
            (true, false) => Ok(&self.genes_flag),
            (true, true) => anyhow::bail!(
                "no gene matrices given — pass them positionally \
                 (`faba gem out/*_genes.zarr.zip -o out/gem`) or with `--genes a,b`"
            ),
            (false, false) => anyhow::bail!(
                "gene matrices given both positionally ({}) and via --genes ({}) — \
                 pass one or the other",
                self.genes_pos.len(),
                self.genes_flag.len()
            ),
        }
    }
}

#[cfg(test)]
mod tests;
