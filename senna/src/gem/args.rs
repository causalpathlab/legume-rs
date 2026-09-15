//! `senna gem`'s command-line surface: the [`GemArgs`] clap struct.
//!
//! Mirrors `senna/src/bge/args.rs` field for field: the same shared groups
//! (`HvgCliArgs`, `refine_weighting::CollapseArgs`, `QcArgs`,
//! `ge::GeneModuleArgs`), the same top-level knobs, and the same help text
//! for every flag they share, so `senna bge` and `senna gem` read as one
//! flag surface. gem adds its own modality inputs (`GENES...`, `--modality`,
//! `--genes-sample-strip`) and its own ridge on the per-track feature
//! offsets (`--offset-l2`). Unlike `BgeArgs`, `GemArgs` does not implement
//! `Updatable`; `senna update` does not (yet) continue a gem run.

use crate::embed_common::*;
use data_beans_alg::hvg::HvgCliArgs;
use graph_embedding_util as ge;

#[derive(Args, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "crate::embed_common::clap_defaults")]
pub(crate) struct GemArgs {
    #[arg(
        value_name = "GENES",
        value_delimiter = ',',
        required = true,
        help = "Gene-count matrices (zarr/h5), comma- or space-separated",
        long_help = "Gene-count matrices to embed. Pass them all, in any order;\n\
                     space-separated so shell globs work (`senna gem out/*_genes.zarr.zip`),\n\
                     commas also accepted.\n\
                     \x20\n\
                     Rows must follow `{gene}/count/{spliced|unspliced}`. The gene key is the first field,\n\
                     and the row itself is the join key across files.\n\
                     \x20\n\
                     Cells are matched across files by barcode within a sample;\n\
                     see --genes-sample-strip and --modality for co-measured tracks."
    )]
    pub(crate) genes: Vec<Box<str>>,

    #[arg(
        long = "modality",
        value_name = "FILE[,FILE...]",
        value_delimiter = ',',
        action = clap::ArgAction::Append,
        help = "Modality count matrices, comma separated; the flag may repeat.",
        long_help = "Modality count matrices, comma separated; the flag may repeat.\n\
                     Each file holds one modality (m6a, atoi or apa)\n\
                     with its two channels as rows; the modality is read from the rows.\n\
                     Cells are matched to the gene files by barcode within a sample;\n\
                     see --genes-sample-strip."
    )]
    pub(crate) modality_files: Vec<Box<str>>,

    #[arg(
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value = "",
        help = "Suffix stripped from every input basename to form its sample id.",
        long_help = "Suffix stripped from every input basename to form its sample id.\n\
                     Empty (the default) strips _genes from gene files and\n\
                     _{modality} from modality files.\n\
                     Files of one sample must land on the same id."
    )]
    pub(crate) genes_sample_strip: Box<str>,

    #[command(flatten)]
    pub(crate) hvg: HvgCliArgs,

    #[arg(
        long,
        default_value_t = 128,
        help = "Embedding dimension H",
        alias = "dim-embedding"
    )]
    pub(crate) embedding_dim: usize,

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
        help = "ETM topics K (omit to auto-select via SPA-anchor residual-elbow sweep)."
    )]
    pub(crate) num_topics: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable BBKNN + DC-Poisson refinement of the multi-level pseudobulk partition.\n\
                Default: enabled."
    )]
    pub(crate) no_refine: bool,

    #[arg(short = 'i', long, default_value_t = 1000, help = "Training epochs")]
    pub(crate) epochs: usize,

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
        help = "Learning rate: the row-wise Adagrad step of phase 1 on the plain path\n\
                (AdamW on the splice path).",
        alias = "lr"
    )]
    pub(crate) learning_rate: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Weight decay: a per-row shrink 1 − lr·wd on every touched row (plain path);\n\
                AdamW decoupled decay on the splice path.",
        long_help = "Weight decay: on the plain path (bge) a per-row shrink 1 − lr·wd is applied\n\
                     to every row a step touches, right before that row's Adagrad update.\n\
                     \n\
                     On the splice path (gem) this is AdamW's decoupled weight decay instead,\n\
                     applied uniformly to every parameter: E_feat, b_feat, and the per-axis heads.\n\
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
        long = "offset-l2",
        default_value_t = 1.0,
        help = "Ridge on the per-track offsets of the feature loading; 0 = off",
        long_help = "Ridge penalty on the per-track offsets.\n\
                     Every row of a gene shares the gene's loading;\n\
                     each track other than count/spliced adds an offset to it,\n\
                     and this ridge shrinks that offset toward zero.\n\
                     Larger values pull the tracks of a gene together,\n\
                     so a modality's contrast in {out}.feature_contrast.parquet\n\
                     keeps only what its counts insist on.\n\
                     Default 1.0, the ridge the previous gem applied to its splice offset.\n\
                     0 disables it."
    )]
    pub(crate) offset_l2: f32,

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

    #[command(flatten)]
    pub(crate) modules: ge::GeneModuleArgs,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix. It produces {out}.cell_embedding.parquet, which is Z,\n\
                     {out}.dictionary.parquet, {out}.feature_embedding.parquet,\n\
                     {out}.feature_bias.parquet, {out}.cell_bias.parquet, and {out}.senna.json.\n\
                     Unless --skip-etm, it adds two more:\n\
                     {out}.latent.parquet and {out}.topic_embedding.parquet.\n\
                     With --modality tracks it adds {out}.feature_contrast.parquet."
    )]
    pub(crate) out: Box<str>,
}

#[cfg(test)]
#[path = "args/tests.rs"]
mod tests;
