use clap::Args;

use super::common::{ComputeDevice, NceObjectiveArg};

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
                     Try 0.01–1.0; δ_g is written to `{out}.delta_feature_embedding.parquet`."
    )]
    pub delta_l2: f32,

    #[arg(
        long = "nce-objective",
        default_value_t = NceObjectiveArg::Softmax,
        value_enum,
        help = "NCE objective for phase-1 training:\n\
                softmax (InfoNCE — negatives compete in one softmax; sharper on\n\
                dense pseudobulk data; default) or logistic (per-pair SGNS)."
    )]
    pub nce_objective: NceObjectiveArg,

    // Per-gene softmax feature gate — ALWAYS ON for gem (the standard training):
    // β_g ⊙ softmax(S_g), a per-gene SuSiE variational single-effect (spike-and-slab:
    // categorical selection + Gaussian effect KL) over the H embedding dims + a null
    // 'load-nothing' slot. A gene with no cell-state signal sends its mass to null and
    // contributes ≈0 → single-pass feature selection. The velocity δ_g gets its own
    // independent gate too (→ velocity_selection). Temperature is the one knob.
    #[arg(
        long = "feature-softmax-temp",
        default_value_t = 1.0,
        help = "Softmax feature-gate temperature τ (< 1 sharpens the per-gene selection)."
    )]
    pub feature_softmax_temp: f32,
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
        help = "Optional HVG cut: keep the top-N highly-variable genes (default 0 = train ALL genes; the softmax gate selects)",
        long_help = "Optional gene-level HVG feature filter.\n\
                     Selects the top-N most variable GENES (NB dispersion-trend, spliced+unspliced pooled)\n\
                     and drops the rest — both the spliced and unspliced rows of a dropped gene go together\n\
                     so the β-sharing factorization stays aligned.\n\
                     It shrinks the dictionary and speeds the fit; the `--n-hvg` remainder is restored post-hoc\n\
                     by the held-out-feature projection (with velocity).\n\
                     Defaults to `0`: train ALL genes and let the per-gene softmax FEATURE GATE do the selecting\n\
                     (a junk gene sends its gate mass to null → β̃_g ≈ 0), no HVG cut needed. This is the recommended path;\n\
                     set `--n-hvg > 0` only for a fixed smaller dictionary (e.g. 5000, matching `senna bge` / `pinto`)."
    )]
    pub n_hvg: usize,

    #[arg(
        long = "must-train-features",
        value_name = "FILE",
        help = "Genes to TRAIN on regardless of whether they make the HVG cut",
        long_help = "Force-include list: these genes enter the FIT\n\
                     even when they do not make the `--n-hvg` cut.\n\
                     UNIONed with the HVG selection so a named gene is always trained in-model.\n\
                     Both the spliced and unspliced rows of a kept gene are kept together,\n\
                     so the β-sharing factorization stays aligned.\n\
                     \n\
                     This is about TRAINED vs PROJECTED, not presence:\n\
                     a gene that misses the HVG cut still gets a β in `{out}.beta_feature_embedding.parquet`,\n\
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
                     A no-op when `--n-hvg 0` (all genes trained), i.e. when the HVG cut wouldn't drop a gene anyway."
    )]
    pub must_train_features: Option<Box<str>>,

    #[arg(
        long = "markers",
        value_name = "FILE",
        help = "Marker panel this embedding will be annotated with — force-trained, like \
                --must-train-features (a no-op at the default --n-hvg 0)",
        long_help = "The `gene<TAB>celltype` marker panel that `faba annotate` / `faba lineage --markers`\n\
                     will later score against this embedding.\n\
                     Its genes are UNIONed into `--must-train-features`,\n\
                     i.e. trained in-model regardless of the `--n-hvg` cut.\n\
                     \n\
                     Like `--must-train-features`, this is a NO-OP at the default `--n-hvg 0`:\n\
                     every gene is trained there anyway, so the panel is on the trained axis by construction.\n\
                     It matters only when you set `--n-hvg > 0` and the HVG cut could drop a marker.\n\
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
        default_value_t = 1e-2,
        alias = "lr",
        help = "AdamW learning rate"
    )]
    pub learning_rate: f64,

    #[arg(
        long,
        default_value_t = 1e-2,
        help = "AdamW decoupled weight decay (all phase-1 params). Default 1e-2.",
        long_help = "AdamW decoupled weight decay applied uniformly to every phase-1 parameter\n\
                     (β_g, δ_g, per-axis heads, biases).\n\
                     Post-update shrinkage `θ ← θ − lr·wd·θ`; it does NOT enter the backward graph,\n\
                     so unlike an explicit E_feat L2 it is compatible with β-sharing.\n\
                     Mild by construction: the per-step pull is far below the clipped adaptive step,\n\
                     so it sets an equilibrium scale rather than decaying params away.\n\
                     0.0 = off (plain Adam)."
    )]
    pub weight_decay: f64,

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
        help = "CPU threads (default 16; 0 = all available)",
        long_help = "Number of CPU threads for rayon-parallel work (HNSW, collapse, phase-2 cell projection).\n\
                     Defaults to 16; pass `0` to use every available logical CPU."
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

    #[arg(
        short,
        long,
        required = true,
        help = "Output prefix",
        long_help = "Output file prefix.\n\
                     \n\
                     NOTE the per-cell tables (cell_embedding, velocity, ...) may contain\n\
                     FEWER ROWS than the input: cell QC drops failing cells from the OUTPUTS\n\
                     (never from the fit — every cell still informs the embedding and the\n\
                     feature dictionary). Join downstream tables by the cell/barcode column,\n\
                     never by row position. --no-qc keeps every cell; --qc-report writes the\n\
                     per-cell keep/drop table."
    )]
    pub out: Box<str>,

    #[command(flatten)]
    pub model: ModelArgs,

    #[command(flatten)]
    pub collapse: CollapseArgs,

    #[command(flatten)]
    pub train: TrainArgs,

    /// Cell QC, applied as an OUTPUT FILTER only — see the note on `--out`.
    #[command(flatten)]
    pub qc: data_beans::qc_lib::QcArgs,

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
