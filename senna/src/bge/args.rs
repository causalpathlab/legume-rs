//! `senna bge`'s command-line surface: the [`BgeArgs`] clap struct and its
//! `--update` rebase.
//!
//! Split out of the module driver because it is almost entirely help text —
//! several hundred lines of it — and reading `fit_bge` meant scrolling past all
//! of it first. Nothing here computes; the translation to
//! `graph_embedding_util::FitConfig` stays with the driver that performs it.

use data_beans::alg::hvg::HvgCliArgs;
use graph_embedding_util as ge;
use senna::embed_common::*;

#[derive(Args, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub struct BgeArgs {
    #[arg(
        value_delimiter = ',',
        help = "Sparse count matrices (zarr/h5), comma-separated",
        long_help = "Count matrices to embed. Pass them all, in any order.\n\
                     \n\
                     Multiome is detected, not declared. Files whose feature axes\n\
                     mostly agree are one modality and share a row block.\n\
                     Files of DIFFERENT modalities whose barcodes mostly agree\n\
                     are the same cells measured twice, and become one sample group.\n\
                     \x20\n\
                     RNA + ADT for four donors is therefore just:\n\
                     \x20 senna bge scRNA_*.zarr.zip scADT_*.zarr.zip\n\
                     \x20\n\
                     A layout is only claimed when matched cells exist.\n\
                     Two feature axes with no shared cells are as likely two assays\n\
                     on different donors, so that case loads as it always has:\n\
                     one feature axis, cells stacked.\n\
                     \x20\n\
                     The resolved layout is printed before training. Override it\n\
                     with --multiome when the modalities SHARE feature names\n\
                     (spliced versus unspliced, say), which no rule can read off the axes."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    /// The parent's carried pseudobulks, when `senna update` chose to reuse
    /// them instead of re-reading its cells. Derived per invocation, so it is
    /// neither a CLI flag nor part of the recorded configuration.
    #[arg(skip)]
    #[serde(skip)]
    pub(crate) pb_reference: Option<senna::pb_reference::ReferenceInput>,

    /// The parent run this one continues, set by `senna update`. Under `senna
    /// update` the parent's module partition (its membership, argmax per gene,
    /// unmatched genes initialised through the parent's modules) seeds phase 1;
    /// module vectors and per-gene residuals are re-learned. The ETM is
    /// re-derived by archetypal analysis each run, and the emitted reference's
    /// generation counter is chained through here.
    #[arg(skip)]
    #[serde(skip)]
    pub(crate) init_from: Option<Box<str>>,

    #[command(flatten)]
    pub(crate) hvg: HvgCliArgs,

    #[arg(
        long,
        default_value_t = graph_embedding_util::EmbeddingDim::Fixed(128),
        value_name = "H|auto",
        help = "Embedding dimension H (auto = the width of a given feature embedding)",
        alias = "dim-embedding"
    )]
    pub(crate) embedding_dim: graph_embedding_util::EmbeddingDim,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) feature_embedding: crate::feature_embedding_args::FeatureEmbeddingArgs,

    #[command(flatten)]
    pub(crate) collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    pub(crate) qc: QcArgs,

    #[arg(
        long = "phase1-cells-per-pb",
        default_value_t = 16,
        help = "Cells injected per pseudobulk into phase-1 training (k); 0 = pure-pb.",
        long_help = "Phase-1 cell-axis mode (k): how many raw cells per pseudobulk-sample,\n\
                     at each collapse level, train alongside the pseudobulks.\n\
                     Phase 2 ALWAYS analytically projects every cell,\n\
                     so this shapes the feature dictionary, not the per-cell output.\n\
                     \n\
                     A pseudobulk averages many cells, and averaging smooths away the\n\
                     sharp present-versus-absent contrast that separates closely related\n\
                     states. A few raw cells restore it: measured, a moderate k raised\n\
                     purity on exactly the within-lineage classes pure-pb lost, and\n\
                     several-fold more of the embedding's dimensions carry variance.\n\
                     \n\
                     It has an interior optimum. Injecting EVERY cell is worse than\n\
                     injecting none, and costs the most; the default is the moderate\n\
                     setting that measured best, at a few times pure-pb's runtime.\n\
                     \n\
                     k = 0 → suppress the cell axis (pure-pb, fastest).\n\
                     1 ≤ k < n_cells → keep ≤ k cells per pb-sample at each level (union).\n\
                     k ≥ n_cells → every cell (slowest, and measured worse)."
    )]
    pub(crate) phase1_cells_per_pb: usize,

    #[arg(
        long = "modules-per-unit",
        default_value_t = 8,
        value_name = "K",
        help = "Modules scored at the gene level per unit per step (phase 1)",
        long_help = "Phase 1 scores every module exactly, every step.\n\
                     At the gene level it scores K modules per unit per step,\n\
                     drawn in proportion to the unit's share of counts in them.\n\
                     Higher K covers more of a unit's genes per epoch at linear cost."
    )]
    pub(crate) modules_per_unit: usize,

    #[arg(
        long = "module-only-min-rows",
        default_value_t = 100_000,
        value_name = "N",
        help = "Under --multiome, a modality with at least N features drops its residual (0 = off)",
        long_help = "Under --multiome, a modality with at least N features is module-only.\n\
                     Its modules are its own, partitioned once from the finest pseudobulks'\n\
                     counts and fixed for the run, and a feature's row is its module's row,\n\
                     with a bias equal to its share of the module's counts.\n\
                     Phase 1 then skips the within-module softmax for that modality,\n\
                     and phase 2 reads each such module as one row, exactly,\n\
                     which is what keeps a very wide axis such as ATAC peaks affordable.\n\
                     Features in a module share one embedding, so per-feature structure\n\
                     inside a module is given up. 0 keeps every modality's residual."
    )]
    pub(crate) module_only_min_rows: usize,

    #[arg(
        long = "skip-etm",
        default_value_t = false,
        help = "Skip ETM resolution; emit raw bge embeddings (Z and ρ) only.",
        long_help = "Skip the default ETM resolution.\n\
                     Only the raw bge embeddings are then emitted: cell_embedding = Z,\n\
                     dictionary = ρ, and no latent.\n\
                     \n\
                     By default bge resolves ETM topics from the cell embedding,\n\
                     by anchor analysis. It then ALSO writes the topic-model tables:\n\
                     latent = log θ, dictionary = β, topic_embedding = α.\n\
                     \n\
                     Either way, Z lands in {out}.cell_embedding.parquet."
    )]
    pub(crate) skip_etm: bool,

    #[arg(
        long = "num-topics",
        help = "ETM topics K (omit to take one topic per cell cluster)."
    )]
    pub(crate) num_topics: Option<usize>,

    #[arg(short = 'i', long, default_value_t = 1000, help = "Training epochs")]
    pub(crate) epochs: usize,

    /// Batches per epoch. **Omit for auto** — one weighted pass per
    /// epoch over the largest axis (`ceil(max_axis_units / batch_size)`).
    /// Pass a value to force a fixed step budget per epoch (historical
    /// default: 100).
    #[arg(
        long,
        help = "Batches per epoch (default: auto = one pass over largest axis)",
        hide = true
    )]
    pub(crate) batches_per_epoch: Option<usize>,

    #[arg(
        long,
        help = "Units (pseudobulks + phase-1 cells) per phase-1 step (unset: 256)",
        long_help = "Units per phase-1 step: the pseudobulks at every collapse level,\n\
                     plus each pseudobulk's phase-1 cell subsample.\n\
                     Unset, the default is 256."
    )]
    pub(crate) batch_size: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Learning rate: the row-wise Adagrad step of phase 1.",
        alias = "lr"
    )]
    pub(crate) learning_rate: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Weight decay: a per-row shrink 1 − lr·wd on every touched row.",
        long_help = "Weight decay: a per-row shrink 1 − lr·wd is applied to every row a step\n\
                     touches, right before that row's Adagrad update.\n\
                     Per-step post-update shrinkage; doesn't enter the backward graph.\n\
                     Default 0.0 (off)."
    )]
    pub(crate) weight_decay: f64,

    #[arg(
        long,
        help = "Cells per block for column I/O / streaming (omit for auto).",
        long_help = "Cells per parallel block for streaming column-block I/O.\n\
                     Omit for auto-scaling, which clamps to 100 for large feature counts.\n\
                     That is slow on rotational disks.\n\
                     Pass 1024+ when you have RAM, especially without --preload-data.",
        hide = true
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse column data into memory. Faster when data fits in RAM;\n\
                required on slow disks.",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        value_name = "FILE[,FILE...]",
        help = "Declare the multiome layout by hand; one sample (group) per flag.",
        long_help = "Declare the multiome layout instead of letting it be detected\n\
                     from the inputs. Reach for this when the modalities share\n\
                     feature names, which the detector cannot see.\n\
                     \x20\n\
                     Pass one sample (group) per flag, comma-separated,\n\
                     as in `--multiome rna.zarr,atac.zarr`. Cells are the shared axis.\n\
                     Each modality keeps its own features.\n\
                     Repeat the flag for each additional sample or group:\n\
                     \n\
                     --multiome rna1.zarr,atac1.zarr \\\n\
                     --multiome rna2.zarr,atac2.zarr\n\
                     \n\
                     Cell (barcode) identity. Within a group, equal barcodes are the same cell.\n\
                     Modalities Union-merge. A cell present in only some files is fine.\n\
                     Patchy multiome therefore works. ACROSS groups, barcodes must be disjoint.\n\
                     A shared barcode would merge cells from different samples.\n\
                     This is validated, and a collision is an error.\n\
                     \n\
                     Feature (modality) identity. Features are namespaced `{name}/{modality}`.\n\
                     The SAME modality across samples therefore merges, sharing one gene panel.\n\
                     DIFFERENT modalities stay on separate rows.\n\
                     That holds even when names collide.\n\
                     Spliced versus unspliced `TSPAN6` is the usual case.\n\
                     \n\
                     The modality tag defaults to file position: m0, m1, and so on.\n\
                     Override it with a `label=` prefix:\n\
                     --multiome spliced=spliced.zarr,unspliced=unspliced.zarr\n\
                     File ORDER within a group defines modality order,\n\
                     so the positional default lines up across groups.\n\
                     \n\
                     Batch identity.\n\
                     Each group becomes its own batch when --batch-files is omitted.\n\
                     That is modality-presence auto-batching. Pass a single --batch-files,\n\
                     one label per unified cell, to set batches explicitly.\n\
                     This flag replaces the positional data files.\n\
                     \n\
                     Note: comma-separate files within one group, with no spaces.\n\
                     Use a separate --multiome flag for each additional group.\n\
                     \x20\n\
                     Declared groups are NOT barcode-namespaced, so a barcode shared\n\
                     across groups is an error. Detected groups are namespaced\n\
                     `{barcode}@{group}`, which is why several samples need no\n\
                     pre-processing there."
    )]
    pub(crate) multiome: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = 1,
        value_name = "N",
        help = "Seed for training (default 1).",
        long_help = "Seed for the fit's sampling RNG and parameter initialization.\n\
                     \n\
                     Changing it gives an INDEPENDENT fit.\n\
                     Initialization and minibatch order both differ.\n\
                     That is what an A/B across seeds needs.\n\
                     \n\
                     It does NOT make a run bit-reproducible.\n\
                     Two runs at the same seed still differ slightly."
    )]
    pub(crate) seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub(crate) device_no: usize,

    /// The `--feature-modules` flag group (see `ge::FeatureModuleArgs`); on by default
    /// here, resolved in `build_config`.
    #[command(flatten)]
    pub(crate) modules: ge::FeatureModuleArgs,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix. It produces {out}.cell_embedding.parquet, which is Z,\n\
                     {out}.feature_embedding.parquet, which is the raw gene table,\n\
                     {out}.feature_coembedding.parquet, the genes on the cell manifold,\n\
                     {out}.feature_bias.parquet, {out}.cell_bias.parquet, and {out}.senna.json.\n\
                     Unless --skip-etm, it adds three more:\n\
                     {out}.latent.parquet, {out}.dictionary.parquet and {out}.topic_embedding.parquet."
    )]
    pub(crate) out: Box<str>,
}

impl crate::update::Updatable for BgeArgs {
    fn rebase(&mut self, r: crate::update::Rebase) {
        self.data_files = r.data_files;
        self.batch_files = r.batch_files;
        self.out = r.out;
        self.pb_reference = r.reference;
        // `update` re-fits on the union with the recorded configuration; the
        // carried reference (when used) keeps that O(new), and `init_from` is
        // where `build_config` finds the parent's modules to warm-start from.
        self.init_from = Some(r.init_from);
        if let Some(e) = r.epochs {
            self.epochs = e;
        }
    }
}
