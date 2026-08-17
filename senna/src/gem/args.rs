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
        help = "L2 (ridge) weight on the per-gene splice offset δ_g. 0 = auto:\n\
                a mild ridge when unspliced rows are present.",
        long_help = "L2 (ridge) penalty on the per-gene splice offset δ_g.\n\
                     When 0 (default) and the input carries unspliced rows,\n\
                     gem auto-applies a mild ridge (L2=1.0),\n\
                     so a δ_g dictionary is always written for `senna annotate-gem --track velocity`;\n\
                     set an explicit value to override,\n\
                     or 0 on a spliced-only input keeps δ off. When > 0,\n\
                     unspliced rows embed as β_g + δ_g with a ridge-shrunk δ_g learned in phase 1:\n\
                     It absorbs the dense static per-gene nascent structure,\n\
                     the RNA-velocity γ. Cell identity, the spliced θ, therefore stays clean.\n\
                     The phase-2 velocity increment δ then becomes γ-calibrated.\n\
                     That increment is a raw Poisson-MAP shift with θ fixed.\n\
                     Larger = more shrinkage (δ_g pulled toward 0). Try 0.01–1.0;\n\
                     δ_g is written to `{out}.delta_feature_embedding.parquet`."
    )]
    pub delta_l2: f32,

    #[arg(
        long = "independent-delta-gate",
        default_value_t = false,
        help = "Let the velocity gate select dims the identity gate did not (--posterior runs only).\n\
                Default: nested.",
        long_help = "gem carries TWO feature-side gates, and --posterior samples both.\n\
                     The identity gate sits on β_g, pinned by the spliced rows.\n\
                     The velocity gate sits on δ_g, pinned by the unspliced rows,\n\
                     with β_g carried as an offset that refreshes every sweep.\n\
                     This flag decides how the two relate.\n\
                     \n\
                     By default the velocity gate is NESTED inside the identity gate:\n\
                     δ_g may be included on an embedding dim only where β_g already is.\n\
                     Velocity is a deviation from the identity loading.\n\
                     So a gene should not move along a dim its identity misses.\n\
                     The model treats that as a state not to visit.\n\
                     \n\
                     Nesting also removes a real failure mode. β is pinned by the spliced rows.\n\
                     δ is pinned by the unspliced-minus-spliced contrast.\n\
                     A gene seen ONLY in the unspliced track pins β+δ.\n\
                     It pins neither of them alone.\n\
                     \n\
                     Two independent gates then split inclusion mass between (z_β=1, z_δ=0) and (z_β=0, z_δ=1).\n\
                     Both read confidently wrong. Such genes are reported as unidentified.\n\
                     They are written as NaN in the δ tables, either way.\n\
                     \n\
                     Pass this flag to sample the two gates independently.\n\
                     That checks for genes carrying velocity on a dim their identity misses."
    )]
    pub independent_delta_gate: bool,

    #[arg(
        long = "nce-objective",
        default_value_t = NceObjectiveArg::Softmax,
        value_enum,
        help = "NCE objective for phase-1 training: softmax or logistic",
        long_help = "Which objective phase-1 SGD trains the feature side with.\n\
                     \n\
                     `softmax` is the default: sampled-softmax, or InfoNCE.\n\
                     The positive competes with its negatives in one distribution.\n\
                     That separates cell types better on dense pseudobulk counts.\n\
                     \n\
                     `logistic` is the per-pair SGNS loss.\n\
                     \n\
                     `logistic` CANNOT be combined with `--posterior`.\n\
                     The sampler's likelihood is the profiled Poisson.\n\
                     Its normalizer is the same estimand as sampled-softmax:\n\
                     dividing by the anchor total gives InfoNCE exactly.\n\
                     \n\
                     SGNS is different. It is a sum of independent per-pair decisions,\n\
                     with no logsumexp anywhere.\n\
                     Sampling a logistic fit would report the wrong posterior.\n\
                     The combination is a hard error, not a silent mismatch."
    )]
    pub nce_objective: NceObjectiveArg,

    // Per-gene spike-and-slab feature gate — ALWAYS ON for gem (the standard
    // training): β_g ⊙ σ(S_g), an INDEPENDENT inclusion probability per (gene, dim)
    // — Bernoulli selection + Gaussian effect KL, with each dim's inclusion rate π_h
    // learned. There is no null slot and no per-dim budget: a gene with no cell-state
    // signal simply has σ(S) → 0 in every dim and contributes ≈0, giving single-pass
    // feature selection. σ(S) is the same estimand `--posterior` reports as a PIP, so
    // feature_selection.parquet and beta_pip.parquet are now comparable. The velocity
    // δ_g gets its own independent gate and its own π_h (→ velocity_selection).
    // Temperature is the one knob.
    #[arg(
        long = "feature-gate-temp",
        alias = "feature-softmax-temp",
        default_value_t = 1.0,
        help = "Feature-gate temperature τ (< 1 sharpens each inclusion probability toward 0/1).",
        hide = true
    )]
    pub feature_gate_temp: f32,

    #[arg(
        long = "gate-ibp-alpha",
        help = "Truncated-IBP concentration for the gate's per-dim inclusion ladder;\n\
                unset = auto",
        long_help = "Concentration alpha of the truncated Indian Buffet Process whose\n\
                     ladder tilts the feature gate: dim h carries a fixed logit\n\
                     offset h * ln(alpha/(alpha+1)), so later dims must earn their\n\
                     inclusion against a steeper prior. Chosen, never fitted.\n\
                     \n\
                     Unset (the default) derives alpha from the embedding dimension\n\
                     so the ladder spans 4 logits end to end, leaving the last dim\n\
                     at the sigmoid's most responsive point rather than frozen.\n\
                     \n\
                     SMALLER alpha means a steeper ladder and more sparsity. This\n\
                     replaced a KL toward a Beta(1,9) inclusion prior, which had no\n\
                     natural weight under gem's noise-contrastive phase-1 objective."
    )]
    pub gate_ibp_alpha: Option<f64>,
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
        help = "Number of pseudobulk collapse levels (coarse→fine);\n\
                each level is a training axis"
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
        long_help = "Phase-1 cell-axis mode (`k`).\n\
                     Controls only what shapes the feature dictionary (β) in phase 1;\n\
                     phase 2 ALWAYS analytically projects every cell,\n\
                     so the per-cell embedding output is essentially unaffected by `k`.\n\
                     \n\
                     k = 0 (default) → suppress the cell axis in phase 1 (pure-pb: features shaped by pb aggregates only).\n\
                     Fastest.\n\
                     1 ≤ k < n_cells → keep ≤k cells per pb-sample at EVERY collapse level (union),\n\
                     shrinking the phase-1 budget while keeping rare / shallow cells visible to the shared dictionary.\n\
                     k ≥ n_cells → every cell shapes the dictionary (slowest).",
        hide = true
    )]
    pub phase1_cells_per_pb: usize,

    #[arg(
        long = "n-hvg",
        default_value_t = 5000,
        help = "HVG weighting: the top-N variable genes carry the pseudobulk projection.\n\
                Default 5000, matching `senna bge`; 0 = all genes.",
        long_help = "Optional gene-level HVG weighting.\n\
                     It has the SAME meaning here as in `senna bge`.\n\
                     \n\
                     Selects the top-N most variable GENES by NB dispersion trend.\n\
                     Spliced and unspliced are pooled, both tracks of a gene together,\n\
                     so the β-sharing factorization stays aligned.\n\
                     The rest get projection weight ZERO.\n\
                     \n\
                     WEIGHTS, DOES NOT DROP.\n\
                     Non-selected genes sit out the basis that the multilevel pseudobulk partition is built from.\n\
                     They stay on the feature axis regardless: still trained, still gated,\n\
                     still present in the dictionary, in the δ_g velocity table,\n\
                     and in `--posterior`'s anchor set.\n\
                     So the selection shapes WHERE the pseudobulks land.\n\
                     It does not decide which genes the model may use.\n\
                     \n\
                     This changed. `--n-hvg N` used to hard-subset the feature axis,\n\
                     which made a gem run and a `senna bge` run at the same N into different experiments.\n\
                     It no longer does.\n\
                     The fit is also no longer smaller or faster for setting it —\n\
                     that was the one thing the hard cut bought.\n\
                     \n\
                     Defaults to `5000`, the default `senna bge` and `pinto` carry.\n\
                     It used to default to 0, the right answer while this flag DROPPED genes:\n\
                     there, 0 meant `keep everything`. Now that it only weights,\n\
                     0 means `let depth and abundance decide where the pseudobulks land`.\n\
                     That is a weaker default, not a safer one.\n\
                     \n\
                     Pass `--n-hvg 0` to give every gene equal projection weight.\n\
                     That is a reasonable choice.\n\
                     The per-gene softmax FEATURE GATE still selects during training,\n\
                     so nothing is lost from the model either way.\n\
                     But it lets the partition be shaped by whatever is most abundant,\n\
                     rather than by what varies."
    )]
    pub n_hvg: usize,

    #[arg(
        long = "must-train-features",
        value_name = "FILE",
        help = "Genes forced into the pseudobulk projection basis even if not HVGs",
        long_help = "Force-include list. These genes are UNIONed into the `--n-hvg` selection,\n\
                     so they carry projection weight 1.0 even when their variance does not earn it.\n\
                     Both the spliced and unspliced rows of a named gene come along,\n\
                     so the β-sharing factorization stays aligned.\n\
                     \n\
                     NARROWER THAN THE NAME SUGGESTS,\n\
                     since `--n-hvg` started weighting rather than dropping.\n\
                     Every gene is now trained in-model whatever you pass here.\n\
                     There is no trained-vs-projected split left to force,\n\
                     and the held-out feature regression no longer runs at all.\n\
                     What this still does is decide which genes shape WHERE the pseudobulks land.\n\
                     Useful when a curated panel is biologically important but not variable enough to make the cut.\n\
                     Useless as a way to get a gene into the output,\n\
                     because it is already there.\n\
                     \n\
                     Format is inferred from the extension, optionally gzipped:\n\
                     .txt / .tsv / .csv / .parquet. One gene per row.\n\
                     A gene-like header (`gene`, `feature`, `symbol`, …) picks the column;\n\
                     otherwise the first column is used. EVERY OTHER COLUMN IS IGNORED,\n\
                     so a curated `gene<TAB>celltype` marker table can be passed as-is.\n\
                     \n\
                     Names are matched leniently against the `{gene}` slot of the `{gene}/count/{spliced|unspliced}` rows:\n\
                     case-insensitive, symbol ↔ `ENSG…_SYMBOL` either way;\n\
                     unmatched names are logged, not fatal.\n\
                     A no-op when `--n-hvg 0` (all genes trained),\n\
                     i.e. when the HVG cut wouldn't drop a gene anyway."
    )]
    pub must_train_features: Option<Box<str>>,

    #[arg(
        long = "markers",
        value_name = "FILE",
        help = "Marker panel this embedding will be annotated with —\n\
                forced into the projection basis,\n\
                like --must-train-features (a no-op at --n-hvg 0)",
        long_help = "The `gene<TAB>celltype` marker panel,\n\
                     which `senna annotate-gem` or `senna lineage --markers` will later score against this embedding.\n\
                     Its genes are UNIONed into `--must-train-features`,\n\
                     so they carry projection weight regardless of the `--n-hvg` selection.\n\
                     \n\
                     THE FAILURE MODE THIS EXISTED FOR IS GONE.\n\
                     It was written when `--n-hvg` hard-subsetted the feature axis:\n\
                     a marker that missed the cut was ABSENT from `{out}.feature_embedding.parquet` —\n\
                     the table the annotators read — so it silently left the panel,\n\
                     and a cell type that entered with 20 markers and scored on 1 still produced a confident-looking call.\n\
                     `--n-hvg` now weights instead of dropping,\n\
                     so every marker is on the trained axis by construction and cannot silently leave.\n\
                     \n\
                     What remains is a modelling nudge, not a safety net:\n\
                     naming the panel biases the pseudobulk geometry,\n\
                     toward separating the compartments the panel will later call.\n\
                     Read `annotate`'s agreement as a check on the grouping rather than an independent confirmation —\n\
                     which is what the run already logs.\n\
                     \n\
                     Same format and lenient name matching as --must-train-features (the celltype column is ignored here);\n\
                     pass the SAME file you will pass to `senna annotate-gem --markers`."
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
        long_help = "Delimiter for fuzzy gene-name matching across input files.\n\
                     The last token after the split is the canonical row name.\n\
                     Ignored unless `--feature-name-exact` is *off*."
    )]
    pub feature_name_delim: char,

    #[arg(
        long,
        default_value_t = true,
        help = "Use exact row-name match across files (no canonicalization)",
        long_help = "Use exact row-name match across files (no canonicalization).\n\
                     The gem default —\n\
                     required because the `{gene}/count/{spliced|unspliced}` row format is sensitive to suffix-splitting.\n\
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
        long_help = "AdamW decoupled weight decay,\n\
                     applied uniformly to every phase-1 parameter: β_g, δ_g, per-axis heads,\n\
                     biases. Post-update shrinkage `θ ← θ − lr·wd·θ`;\n\
                     it does NOT enter the backward graph,\n\
                     so unlike an explicit E_feat L2 it is compatible with β-sharing.\n\
                     Mild by construction:\n\
                     the per-step pull is far below the clipped adaptive step,\n\
                     so it sets an equilibrium scale rather than decaying params away.\n\
                     0.0 = off (plain Adam)."
    )]
    pub weight_decay: f64,

    #[arg(
        long = "max-grad-norm",
        default_value_t = 1.0,
        help = "Global-norm gradient clip for phase-1 AdamW (0 = off). When > 0,\n\
                each step's gradients are scaled down if their global L2 norm exceeds this,\n\
                bounding embedding inflation on loss spikes."
    )]
    pub max_grad_norm: f32,

    #[arg(
        long = "lineage-dag",
        default_value_t = false,
        help = "Inject developmental structure at pseudobulk scale (experimental; default off).",
        long_help = "Shape the embedding along a pseudobulk lineage. When set,\n\
                     gem reads the pb-level velocity (identity θ_pb + velocity δ_pb per pseudobulk per collapse level),\n\
                     orients a fixed velocity-KNN lineage over the pseudobulks,\n\
                     and runs a SECOND phase-1 pass with a velocity-drift SEM residual,\n\
                     so the shared feature dictionary picks up that lineage geometry —\n\
                     then lifts a per-cell pseudotime + fate (`{out}.dag_pseudotime.parquet` / `{out}.dag_fate.parquet`).\n\
                     Off by default —\n\
                     the per-cell embedding is then byte-identical to a plain run;\n\
                     turning it ON changes the embedding (the second pass).\n\
                     Only meaningful with spliced+unspliced input (β-sharing).\n\
                     \n\
                     CANNOT be combined with `--posterior`,\n\
                     which REPLACES phase-1 training rather than refining it —\n\
                     there is then no trained fit for the second pass to refine.\n\
                     That combination is a hard error, not a silent skip."
    )]
    pub lineage_dag: bool,

    #[arg(
        long = "lineage-smooth",
        default_value_t = false,
        help = "Lineage-DAG: smooth the pb velocity readout δ_pb (opt-in).",
        long_help = "Smooth the pb velocity readout δ_pb over θ-space KNN neighbours before it orients the lineage graph,\n\
                     stabilizing sign(δ_pb).\n\
                     A wash on clean data (no noise to remove, and it can blur branch-point velocity),\n\
                     so it is off by default —\n\
                     the payoff is on noisy real spliced/unspliced ratios.\n\
                     Ignored unless `--lineage-dag` is set."
    )]
    pub lineage_smooth: bool,

    #[arg(
        long = "dense-dag",
        default_value_t = false,
        help = "Lineage-DAG:\n\
                use the dense velocity-KNN pb graph instead of the default MST tree (opt-out).",
        long_help = "Within `--lineage-dag`,\n\
                     build the pb structure as the dense velocity-KNN graph,\n\
                     each node → its velocity-forward θ-neighbours,\n\
                     instead of the DEFAULT minimum spanning tree oriented into a DAG.\n\
                     The MST is a sparse single-tree lineage, n−1 edges per level,\n\
                     that gives a better-conditioned embedding: measured,\n\
                     PC1 lands further from the ‖θ‖ norm axis;\n\
                     the dense graph keeps more branch edges for the fate readout.\n\
                     Ignored unless `--lineage-dag` is set."
    )]
    pub dense_dag: bool,

    #[arg(
        long = "sequential-velocity",
        default_value_t = false,
        help = "Phase 2: fit identity θ then velocity δ sequentially,\n\
                not jointly (opt-out).",
        long_help = "Revert to the SEQUENTIAL phase-2 velocity fit:\n\
                     identity θ from the spliced edges,\n\
                     then the velocity increment δ from the unspliced edges with θ held fixed.\n\
                     The DEFAULT is the JOINT solve — θ and δ estimated together,\n\
                     θ pulled by both the spliced and unspliced tracks —\n\
                     which gives a better-powered θ embedding (measured: PC1 further from the ‖θ‖ norm axis).\n\
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
                     On by default — much faster than repeated disk reads on typical SSDs,\n\
                     and required on slow disks.\n\
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

/// CLI arguments for `senna gem` (alias `gem-embedding`).
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
                     space-separated, so shell globs work: `senna gem out/*_genes.zarr.zip`.\n\
                     Commas are also accepted.\n\
                     Rows must follow `{gene_key}/count/{spliced|unspliced}`.\n\
                     Multiple files are stacked under Union column alignment (cells merged by barcode);\n\
                     use an embedded `@batch` tag on the barcodes to keep samples as distinct batches (see `--batch-files`).\n\
                     \n\
                     The `--genes a,b` flag form is still accepted, but pass one or the other,\n\
                     not both."
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
                     As an alternative to this file,\n\
                     embed an `@batch` tag in the barcodes (e.g. `AAACCC@sampleA`);\n\
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
                     NOTE the per-cell tables (cell_embedding, velocity, ...) may contain FEWER ROWS than the input:\n\
                     cell QC drops failing cells from the OUTPUTS, never from the fit —\n\
                     every cell still informs the embedding and the feature dictionary.\n\
                     Join downstream tables by the cell/barcode column, never by row position.\n\
                     --no-qc keeps every cell; --qc-report writes the per-cell keep/drop table."
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
    /// Positional is the primary spelling (`senna gem a.zarr.zip b.zarr.zip`, so shell
    /// globs work); `--genes a,b` is kept for the existing scripts. Accepting both at
    /// once would silently pick one, so it is an error — the user's intent is
    /// genuinely ambiguous there.
    pub fn genes(&self) -> anyhow::Result<&[Box<str>]> {
        match (self.genes_pos.is_empty(), self.genes_flag.is_empty()) {
            (false, true) => Ok(&self.genes_pos),
            (true, false) => Ok(&self.genes_flag),
            (true, true) => anyhow::bail!(
                "no gene matrices given — pass them positionally \
                 (`senna gem out/*_genes.zarr.zip -o out/gem`) or with `--genes a,b`"
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
