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
        long = "independent-delta-gate",
        default_value_t = false,
        help = "Let the velocity gate select dims the identity gate did not\n\
                (--posterior runs only). Default: nested.",
        long_help = "gem carries TWO feature-side gates and --posterior samples both: the\n\
                     identity gate on β_g, pinned by the spliced rows, and the velocity gate\n\
                     on δ_g, pinned by the unspliced rows with β_g carried as an offset that\n\
                     is refreshed every sweep. This flag decides how the two relate.\n\
                     \n\
                     By default the velocity gate is NESTED inside the identity gate:\n\
                     δ_g may be included on an embedding dim only where β_g already is.\n\
                     Velocity is a deviation from the identity loading, so a gene moving\n\
                     along a dim its identity does not load is a state the model should\n\
                     not visit.\n\
                     \n\
                     Nesting also removes a real failure mode. β is pinned by the spliced\n\
                     rows and δ by the unspliced-minus-spliced contrast, so a gene observed\n\
                     ONLY in the unspliced track pins β+δ but neither alone; two independent\n\
                     gates then split inclusion mass between (z_β=1, z_δ=0) and (z_β=0, z_δ=1)\n\
                     and read confidently wrong on both. Such genes are reported as\n\
                     unidentified and written as NaN in the δ tables either way.\n\
                     \n\
                     Pass this flag to sample the two gates independently, e.g. to check\n\
                     whether any gene really does carry velocity on a dim its identity misses."
    )]
    pub independent_delta_gate: bool,

    #[arg(
        long = "nce-objective",
        default_value_t = NceObjectiveArg::Softmax,
        value_enum,
        help = "NCE objective for phase-1 training:\n\
                softmax (InfoNCE — negatives compete in one softmax; sharper on\n\
                dense pseudobulk data; default) or logistic (per-pair SGNS).",
        long_help = "Which objective phase-1 SGD trains the feature side with.\n\
                     `softmax` (default) is sampled-softmax / InfoNCE: the positive competes\n\
                     with its negatives in one distribution, which separates cell types better\n\
                     on dense pseudobulk counts. `logistic` is the per-pair SGNS loss.\n\
                     \n\
                     `logistic` CANNOT be combined with `--posterior`. The sampler's likelihood\n\
                     is the profiled Poisson, whose normalizer is the same estimand as\n\
                     sampled-softmax — dividing it by the anchor total gives InfoNCE exactly —\n\
                     but SGNS is a sum of independent per-pair decisions with no logsumexp\n\
                     anywhere. Sampling a logistic fit with it would report a posterior for a\n\
                     different model, so the combination is a hard error rather than a silent\n\
                     mismatch."
    )]
    pub nce_objective: NceObjectiveArg,

    // Per-gene softmax feature gate — ALWAYS ON for gem (the standard training):
    // β_g ⊙ softmax(S_g), a per-dim distribution over genes (slab:
    // categorical selection + Gaussian effect KL) over the H embedding dims + a null
    // 'load-nothing' slot. A gene with no cell-state signal sends its mass to null and
    // contributes ≈0 → single-pass feature selection. The velocity δ_g gets its own
    // independent gate too (→ velocity_selection). Temperature is the one knob.
    #[arg(
        long = "feature-softmax-temp",
        default_value_t = 1.0,
        help = "Softmax feature-gate temperature τ (< 1 sharpens the per-gene selection).",
        hide = true
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
                       k ≥ n_cells → every cell shapes the dictionary (slowest).",
        hide = true
    )]
    pub phase1_cells_per_pb: usize,

    #[arg(
        long = "n-hvg",
        default_value_t = 5000,
        help = "HVG weighting: the top-N highly-variable genes carry the pseudobulk\n\
                projection (default 5000, matching `senna bge`; 0 = every gene carries it).",
        long_help = "Optional gene-level HVG weighting, with the SAME meaning as in `senna bge`.\n\
                     Selects the top-N most variable GENES (NB dispersion-trend, spliced+unspliced\n\
                     pooled, both tracks of a gene together so the β-sharing factorization stays\n\
                     aligned) and gives the rest projection weight ZERO.\n\
                     \n\
                     WEIGHTS, DOES NOT DROP. The non-selected genes sit out the basis the multilevel\n\
                     pseudobulk partition is built from, but they stay on the feature axis: still\n\
                     trained, still gated, still present in the dictionary, the δ_g velocity table\n\
                     and `--posterior`'s anchor set. So the selection shapes WHERE the pseudobulks\n\
                     land without deciding which genes the model may use.\n\
                     \n\
                     This changed: `--n-hvg N` used to hard-subset the feature axis, which made a gem\n\
                     run and a `senna bge` run at the same N different experiments. It no longer does,\n\
                     and the fit is no longer smaller or faster for setting it — that was the one\n\
                     thing the hard cut bought.\n\
                     \n\
                     Defaults to `5000`, the same default `senna bge` and `pinto` carry. It used\n\
                     to default to 0, which was the right answer while this flag DROPPED genes —\n\
                     there, 0 meant `keep everything`. Now that it only weights, 0 means `let\n\
                     depth and abundance decide where the pseudobulks land`, which is a weaker\n\
                     default, not a safer one.\n\
                     \n\
                     Pass `--n-hvg 0` to give every gene equal projection weight. That is a\n\
                     reasonable choice — the per-gene softmax FEATURE GATE still selects during\n\
                     training, so nothing is lost from the model either way — but it lets the\n\
                     partition be shaped by whatever is most abundant rather than by what varies."
    )]
    pub n_hvg: usize,

    #[arg(
        long = "must-train-features",
        value_name = "FILE",
        help = "Genes forced into the pseudobulk projection basis even if not HVGs",
        long_help = "Force-include list: these genes are UNIONed into the `--n-hvg` selection, so\n\
                     they carry projection weight 1.0 even when their variance does not earn it.\n\
                     Both the spliced and unspliced rows of a named gene come along, so the\n\
                     β-sharing factorization stays aligned.\n\
                     \n\
                     NARROWER THAN THE NAME SUGGESTS, since `--n-hvg` started weighting rather\n\
                     than dropping. Every gene is now trained in-model whatever you pass here —\n\
                     there is no trained-vs-projected split left to force, and the held-out\n\
                     feature regression no longer runs at all. What this still does is decide\n\
                     which genes shape WHERE the pseudobulks land. Useful when a curated panel\n\
                     is biologically important but not variable enough to make the cut; useless\n\
                     as a way to get a gene into the output, because it is already there.\n\
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
        help = "Marker panel this embedding will be annotated with — forced into the\n\
                projection basis, like --must-train-features (a no-op at --n-hvg 0)",
        long_help = "The `gene<TAB>celltype` marker panel that `faba annotate` / `faba lineage --markers`\n\
                     will later score against this embedding. Its genes are UNIONed into\n\
                     `--must-train-features`, so they carry projection weight regardless of the\n\
                     `--n-hvg` selection.\n\
                     \n\
                     THE FAILURE MODE THIS EXISTED FOR IS GONE. It was written when `--n-hvg`\n\
                     hard-subsetted the feature axis: a marker that missed the cut was ABSENT\n\
                     from `{out}.feature_embedding.parquet` — the table the annotators read — so\n\
                     it silently left the panel, and a cell type that entered with 20 markers and\n\
                     scored on 1 still produced a confident-looking call. `--n-hvg` now weights\n\
                     instead of dropping, so every marker is on the trained axis by construction\n\
                     and cannot silently leave.\n\
                     \n\
                     What remains is a modelling nudge, not a safety net: naming the panel biases\n\
                     the pseudobulk geometry toward separating the compartments the panel will\n\
                     later call. Read `annotate`'s agreement as a check on the grouping rather\n\
                     than an independent confirmation — which is what the run already logs.\n\
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
                     orients a fixed velocity-KNN lineage over the pseudobulks, and runs a SECOND phase-1 pass with a\n\
                     velocity-drift SEM residual so the shared feature dictionary picks up that lineage geometry —\n\
                     then lifts a per-cell pseudotime + fate (`{out}.dag_pseudotime.parquet` / `{out}.dag_fate.parquet`).\n\
                     Off by default — the per-cell embedding is then byte-identical to a plain run;\n\
                     turning it ON changes the embedding (the second pass).\n\
                     Only meaningful with spliced+unspliced input (β-sharing).\n\
                     \n\
                     CANNOT be combined with `--posterior`, which REPLACES phase-1 training\n\
                     rather than refining it — there is then no trained fit for the second pass\n\
                     to refine. That combination is a hard error, not a silent skip."
    )]
    pub lineage_dag: bool,

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

    #[arg(
        long = "dense-dag",
        default_value_t = false,
        help = "Lineage-DAG: use the dense velocity-KNN pb graph instead of the default MST tree (opt-out).",
        long_help = "Within `--lineage-dag`, build the pb structure as the dense velocity-KNN graph\n\
                     (each node → its velocity-forward θ-neighbours) instead of the DEFAULT minimum spanning\n\
                     tree oriented into a DAG.\n\
                     The MST is a sparse single-tree lineage (n−1 edges per level) that gives a\n\
                     better-conditioned embedding (measured: PC1 further from the ‖θ‖ norm axis);\n\
                     the dense graph keeps more branch edges for the fate readout.\n\
                     Ignored unless `--lineage-dag` is set."
    )]
    pub dense_dag: bool,

    #[arg(
        long = "sequential-velocity",
        default_value_t = false,
        help = "Phase 2: fit identity θ then velocity δ sequentially, not jointly (opt-out).",
        long_help = "Revert to the SEQUENTIAL phase-2 velocity fit: identity θ from the spliced edges,\n\
                     then the velocity increment δ from the unspliced edges with θ held fixed.\n\
                     The DEFAULT is the JOINT solve — θ and δ estimated together, θ pulled by both\n\
                     the spliced and unspliced tracks — which gives a better-powered θ embedding\n\
                     (measured: PC1 further from the ‖θ‖ norm axis).\n\
                     Use this to pin θ to the mature/spliced state for a cleaner δ velocity readout.\n\
                     Only meaningful on spliced+unspliced input (β-sharing)."
    )]
    pub sequential_velocity: bool,
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

    /// The shared `--posterior` / `--mcmc` flag group (see
    /// `ge::posterior::PosteriorArgs`); `senna bge` flattens the same one.
    #[command(flatten)]
    pub posterior: graph_embedding_util::posterior::PosteriorArgs,

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
