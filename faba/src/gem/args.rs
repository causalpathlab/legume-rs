use clap::Args;

use super::common::{ComputeDevice, ProjectionArg};

/// Model dimensions.
#[derive(Args, Debug, Clone)]
pub struct ModelArgs {
    #[arg(
        long,
        default_value_t = 128,
        help = "Embedding dimension H (size of β_g and the cell embedding; default 128)"
    )]
    pub embedding_dim: usize,

    #[arg(
        long = "delta-l2",
        default_value_t = 0.0,
        help = "L2 (ridge) weight on the per-gene splice offset δ_g (0 = auto: a mild ridge is applied when unspliced rows are present)",
        long_help = "L2 (ridge) penalty on the per-gene splice offset δ_g.\n\
                     When 0 (default) and the input carries unspliced rows,\n\
                     gem auto-applies a mild ridge (L2=1.0) so a δ_g dictionary is always written for `faba annotate --track velocity`;\n\
                     set an explicit value to override, or 0 on a spliced-only input keeps δ off.\n\
                     When > 0, unspliced rows embed as β_g + δ_g with a ridge-shrunk δ_g learned in phase 1:\n\
                     it absorbs the (dense) static per-gene nascent structure (the RNA-velocity γ) so cell identity (spliced θ) stays clean\n\
                     and the phase-2 velocity increment δ (raw Poisson-MAP shift, θ fixed) becomes γ-calibrated.\n\
                     Larger = more shrinkage (δ_g pulled toward 0).\n\
                     Try 0.01–1.0; δ_g is written to `{out}.delta_dictionary.parquet`."
    )]
    pub delta_l2: f32,

    #[arg(
        long = "feature-null-fdr",
        default_value_t = 0.05,
        help = "Empirical-Bayes feature-null QC: drop genes whose β never moved off init at this FDR, then re-fit (default 0.05; 0 = off)",
        long_help = "Empirical-Bayes feature-null QC — ON by default at FDR 0.05,\n\
                     the same shared ash (adaptive-shrinkage) engine `senna bge` uses.\n\
                     After phase 1, each gene's materialized β_g (its ‖e_feat‖ loadings) is\n\
                     tested one embedding dimension at a time against a per-axis null whose\n\
                     scale is estimated empirically from the data (no fixed χ² assumption),\n\
                     and a gene is live when it is confidently non-null on some axis\n\
                     (local false-sign rate below this FDR).\n\
                     The per-axis null is elicited by `--n-hvg`: the lowest-norm `n − n_hvg`\n\
                     genes stand in as the presumed null that seeds each axis's scale,\n\
                     which the sampler then refines.\n\
                     Null genes are DROPPED (β-sharing drops the spliced and unspliced tracks\n\
                     together) and the model re-fits on the live feature axis (two-pass refine);\n\
                     dropped genes still get a projected embedding.\n\
                     The live/null flags and norm² are written to `{out}.feature_qc.parquet`.\n\
                     This is the automatic ALTERNATIVE to the manual `--n-hvg` top-N cut, not a\n\
                     complement: the two are mutually exclusive. It runs only when `--n-hvg 0`;\n\
                     with `--n-hvg` set, HVG wins and this is skipped. 0 disables. Must be in [0, 1)."
    )]
    pub feature_null_fdr: f32,

    #[arg(
        long = "projection",
        default_value_t = ProjectionArg::Nce,
        value_enum,
        help = "Phase-2 cell projection: nce (stochastic frozen-feature block training;\n\
                θ on spliced edges, δ on unspliced; GPU-batched or CPU-parallel;\n\
                the default) or analytic (exact per-cell Poisson-MAP).",
        long_help = "How each cell's final embedding θ (and velocity increment δ)\n\
                     is recovered once the feature side is frozen.\n\
                     nce (default): trains θ against the frozen β_g\n\
                     on the cell's spliced edges, then δ against β_g+δ_g\n\
                     on the unspliced edges (scored at θ+δ, θ held fixed) —\n\
                     the stochastic analogue of the analytic dual solve.\n\
                     Blocked and GPU-batched or CPU-parallel, much faster\n\
                     at the default H=128; approximate and seed-dependent.\n\
                     analytic: the exact per-cell Poisson-MAP (IRLS) —\n\
                     reproducible, with library-size-calibrated ‖θ‖,\n\
                     but O(m·h²) per cell on the CPU."
    )]
    pub projection: ProjectionArg,
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
        long_help = "Phase-1 cell-axis mode (`k`). Controls only what shapes the feature dictionary (β) in phase 1;\n\
                     phase 2 ALWAYS analytically projects every cell,\n\
                     so the per-cell embedding output is essentially unaffected by `k`.\n\
                     \n\
                       k = 0 (default) → suppress the cell axis in phase 1 (pure-pb:\n\
                         features shaped by pb aggregates only). Fastest.\n\
                       1 ≤ k < n_cells → keep ≤k cells per pb-sample at EVERY collapse level (union),\n\
                         shrinking the phase-1 budget while keeping rare / shallow cells visible to the shared dictionary.\n\
                       k ≥ n_cells → every cell shapes the dictionary (slowest)."
    )]
    pub phase1_cells_per_pb: usize,

    #[arg(
        long = "n-hvg",
        default_value_t = 0,
        help = "HVG cut: keep the top-N highly-variable genes (default 0 = data-driven feature-null selection instead; >0 = HVG)",
        long_help = "Gene-level HVG feature filter (like `senna bge`).\n\
                     Selects the top-N most variable GENES (NB dispersion-trend, spliced+unspliced pooled)\n\
                     and drops the rest — both the spliced and unspliced rows of a dropped gene go together\n\
                     so the β-sharing factorization stays aligned.\n\
                     This removes the abundant, uniform housekeeping/ribosomal genes that otherwise dominate the positive edges\n\
                     and collapse every cell onto one point;\n\
                     it shrinks the dictionary and restricts the pseudobulk projection/membership to the kept genes.\n\
                     Defaults to `0`: the DATA-DRIVEN feature-null selection (the `--feature-null-fdr` branch:\n\
                     Pass 1 → LRT null call → refit on the live genes), which keeps low-abundance markers an HVG cut would drop.\n\
                     Set `--n-hvg > 0` to pick a fixed HVG cut instead (e.g. 5000, matching `senna bge` / `pinto`).\n\
                     HVG and the data-driven null are mutually exclusive — setting `--n-hvg > 0` picks HVG and skips the null call."
    )]
    pub n_hvg: usize,

    #[arg(
        long = "must-train-features",
        value_name = "FILE",
        help = "Genes to TRAIN on regardless of whether they make the HVG cut",
        long_help = "Force-include list: these genes enter the FIT\n\
                     even when they do not make the `--n-hvg` cut.\n\
                     UNIONed with the HVG selection, and also exempt from the `--feature-null-fdr` drop —\n\
                     the two gates a gene would otherwise have to pass.\n\
                     Both the spliced and unspliced rows of a kept gene are kept together,\n\
                     so the β-sharing factorization stays aligned.\n\
                     \n\
                     This is about TRAINED vs PROJECTED, not presence:\n\
                     a gene that misses the HVG cut still gets a β in `{out}.beta_dictionary.parquet`,\n\
                     but a post-hoc PROJECTED one (the held-out-feature regression), not an in-model estimate.\n\
                     Name it here and it is fit in-model instead.\n\
                     The `trained` column of `{out}.gene_qc.parquet` says which each gene got.\n\
                     \n\
                     Format is inferred from the extension: .txt / .tsv / .csv / .parquet, optionally gzipped.\n\
                     One gene per row;\n\
                     a gene-like header (`gene`, `feature`, `symbol`, …) picks the column, else the first column is used.\n\
                     EVERY OTHER COLUMN IS IGNORED, so a curated `gene<TAB>celltype` marker table can be passed as-is.\n\
                     \n\
                     Names are matched leniently against the `{gene}` slot of the `{gene}/count/{spliced|unspliced}` rows\n\
                     (case-insensitive, symbol ↔ `ENSG…_SYMBOL` either way); unmatched names are logged, not fatal.\n\
                     A no-op only when `--n-hvg 0` AND `--feature-null-fdr 0`, i.e. when nothing would drop a gene anyway."
    )]
    pub must_train_features: Option<Box<str>>,

    #[arg(
        long = "markers",
        value_name = "FILE",
        help = "Marker panel this embedding will be annotated with — force-trained, like \
                --must-train-features",
        long_help = "The `gene<TAB>celltype` marker panel that `faba annotate` / `faba lineage --markers`\n\
                     will later score against this embedding.\n\
                     Its genes are UNIONed into `--must-train-features`,\n\
                     i.e. trained in-model regardless of the `--n-hvg` cut and the `--feature-null-fdr` drop.\n\
                     \n\
                     This exists because the two ends of the pipeline are easy to leave inconsistent.\n\
                     The embedding writes only its TRAINED feature rows to `{out}.feature_embedding.parquet`,\n\
                     and that is the table the annotators read —\n\
                     so a marker that misses the HVG cut is not merely down-weighted,\n\
                     it is ABSENT, and it silently leaves the panel.\n\
                     A cell type that entered with 20 markers and scores on 1 still produces a confident-looking call.\n\
                     Naming the panel here removes the failure mode:\n\
                     the genes the calls will be made on are, by construction, the genes the model fit.\n\
                     \n\
                     Same format and lenient name matching as --must-train-features (the celltype column is ignored here);\n\
                     pass the SAME file you will pass to `faba annotate --markers`."
    )]
    pub markers: Option<Box<str>>,

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
        long_help = "Use exact row-name match across files (no canonicalization).\n\
                     The gem default — required because the `{gene}/count/{spliced|unspliced}` row format\n\
                     is sensitive to suffix-splitting.\n\
                     Pass `--feature-name-exact=false` only if your `{gene}` slot itself carries a stripping suffix."
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
        long_help = "Batches per epoch. Omit for auto —\n\
                     one weighted pass over the largest axis (`ceil(max(n_cells, max_pb_per_level) / batch_size)`)."
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
        long_help = "Shape the embedding along a pseudobulk lineage.\n\
                     When set, gem reads the pb-level velocity (identity θ_pb + velocity δ_pb per pseudobulk per collapse level),\n\
                     orients a directed lineage over the pseudobulks, and runs a SECOND phase-1 pass with a velocity-drift term\n\
                     so the shared feature dictionary picks up that lineage geometry —\n\
                     then lifts a per-cell pseudotime + fate (`{out}.dag_pseudotime.parquet` / `{out}.dag_fate.parquet`).\n\
                     It uses the LEARNED DAG by default (see `--fixed-dag`).\n\
                     Off by default — the per-cell embedding is then byte-identical to a plain run;\n\
                     turning it ON changes the embedding (the second pass).\n\
                     Only meaningful with spliced+unspliced input (β-sharing)."
    )]
    pub lineage_dag: bool,

    #[arg(
        long = "fixed-dag",
        default_value_t = false,
        help = "Lineage-DAG: use the fixed velocity-KNN graph instead of the default learned DAG.",
        long_help = "Within `--lineage-dag`, orient the pb structure with a FIXED velocity-oriented KNN graph,\n\
                     instead of LEARNING the directed adjacency `W` (the default).\n\
                     Learning co-refines `W` with the embedding (velocity-drift SEM + DAGMA-style acyclicity + L1 + velocity-orientation prior)\n\
                     and gives a cleaner single-lineage structure;\n\
                     the fixed graph is faster but more fragmented.\n\
                     Ignored unless `--lineage-dag` is set."
    )]
    pub fixed_dag: bool,

    #[arg(
        long = "lineage-smooth",
        default_value_t = false,
        help = "Lineage-DAG: smooth the pb velocity readout δ_pb (opt-in).",
        long_help = "Smooth the pb velocity readout δ_pb over θ-space KNN neighbours before it orients the lineage graph,\n\
                     stabilizing sign(δ_pb).\n\
                     A wash on clean data (no noise to remove, and it can blur branch-point velocity),\n\
                     so it is off by default — the payoff is on noisy real spliced/unspliced ratios.\n\
                     Ignored unless `--lineage-dag` is set."
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
        long_help = "Preload all sparse column data into memory before any pass over cells.\n\
                     On by default — much faster than repeated disk reads on typical SSDs, and required on slow disks.\n\
                     Pass `--no-preload-data` to fall back to streaming reads (use only for datasets that don't fit in RAM)."
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
        long_help = "Number of CPU threads for rayon-parallel work (HNSW, collapse, phase-2 cell projection).\n\
                     Defaults to all available logical CPUs (0 = all)."
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
                     Commas are also accepted.\n\
                     Rows must follow `{gene_key}/count/{spliced|unspliced}`.\n\
                     Multiple files are stacked under Union column alignment (cells merged by barcode);\n\
                     use an embedded `@batch` tag on the barcodes to keep samples as distinct batches (see `--batch-files`).\n\n\
                     The `--genes a,b` flag form is still accepted, but pass one or the other, not both."
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
        long_help = "Optional batch labels.\n\
                     Under Union column alignment (gem's mode) exactly **one** file is expected,\n\
                     listing one label per unified cell —\n\
                     a barcode shared across modalities cannot carry two labels.\n\
                     As an alternative to this file, embed an `@batch` tag in the barcodes (e.g. `AAACCC@sampleA`);\n\
                     the loader infers and reconciles per-cell batches from those tags."
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
