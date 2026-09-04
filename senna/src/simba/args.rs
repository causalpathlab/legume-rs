//! `senna simba`'s command-line surface. Defaults are SIMBA's own (the
//! PyTorch-BigGraph settings its `pbg_train` uses), so a bare invocation is
//! the published recipe.

use crate::embed_common::*;
use data_beans_alg::hvg::HvgCliArgs;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "crate::embed_common::clap_defaults")]
pub struct SimbaArgs {
    #[arg(
        value_delimiter = ',',
        help = "Sparse count matrices (zarr/h5), comma-separated",
        long_help = "One or more count matrices sharing a feature axis.\n\
                     Cells are unified by barcode; each file is a batch unless -b says otherwise.\n\
                     Batches do not enter the model: SIMBA has no batch term.\n\
                     They only name the cells (barcode@batch), as bge does."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    /// Shared HVG flags. Here the selection HARD-SUBSETS the embedded genes
    /// (SIMBA's `use_highly_variable=True`), unlike bge where it only weights.
    #[command(flatten)]
    pub(crate) hvg: HvgCliArgs,

    #[command(flatten)]
    pub(crate) qc: QcArgs,

    #[arg(
        long,
        default_value_t = 50,
        alias = "dim-embedding",
        help = "Embedding dimension D (SIMBA: 50)"
    )]
    pub(crate) embedding_dim: usize,

    #[arg(
        long,
        short = 'i',
        default_value_t = 10,
        help = "Training epochs (PBG: 10)"
    )]
    pub(crate) epochs: usize,

    #[arg(
        long,
        alias = "lr",
        default_value_t = 0.1,
        help = "Row-wise Adagrad learning rate (PBG: 0.1)"
    )]
    pub(crate) learning_rate: f64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Edges per batch (PBG: 1000)",
        long_help = "Edges per batch. Every batch holds ONE expression level,\n\
                     drawn with probability proportional to that level's remaining edges.\n\
                     One optimizer step per batch."
    )]
    pub(crate) batch_size: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Batch negatives, i.e. the chunk size (PBG: 50)",
        long_help = "A batch is cut into chunks of this many positives.\n\
                     Within a chunk every other positive's cell and gene is a negative.\n\
                     A positive never competes with itself."
    )]
    pub(crate) num_batch_negs: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Uniform negatives per chunk (PBG: 50)",
        long_help = "Random cells and random genes drawn per chunk and shared by its positives.\n\
                     Both sides are corrupted: cells against random genes, genes against random cells."
    )]
    pub(crate) num_uniform_negs: usize,

    #[arg(
        long,
        help = "Weight decay; omit for SIMBA's automatic value",
        long_help = "L2 weight decay on both node tables.\n\
                     Omit it for SIMBA's `auto_wd`, which scales a reference value by the edge count.\n\
                     Pass 0 to disable."
    )]
    pub(crate) weight_decay: Option<f64>,

    #[arg(
        long,
        default_value_t = 50,
        help = "Draw the weight decay with probability 1/N per batch (PBG: 50)"
    )]
    pub(crate) wd_interval: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Fraction of edges held out for the eval loss (PBG: 0.05)",
        long_help = "Edges never trained on, scored with the same loss after every epoch.\n\
                     They are drawn once; PBG re-draws them each epoch.\n\
                     Pass 0 to train on every edge."
    )]
    pub(crate) eval_fraction: f64,

    #[arg(long, default_value_t = 5, help = "Expression levels (SIMBA: 5)")]
    pub(crate) n_bins: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Softmax temperature of the gene co-embedding (SIMBA: 0.5)",
        long_help = "Each gene is placed at the softmax-weighted mean of the cells,\n\
                     weighted by exp(score / T) over the raw dot scores.\n\
                     Lower T pins a gene to its best cells; higher T spreads it."
    )]
    pub(crate) coembed_temp: f64,

    #[arg(long, default_value_t = 1, help = "Random seed")]
    pub(crate) seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub(crate) device_no: usize,

    #[arg(
        long,
        hide = true,
        help = "Column block size for the QC and HVG passes"
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        hide = true,
        help = "Preload the count matrices into memory"
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix. It produces {out}.cell_embedding.parquet (Z),\n\
                     {out}.feature_loading.parquet (the raw gene table),\n\
                     {out}.feature_embedding.parquet (the co-embedded genes),\n\
                     {out}.feature_scores.parquet, {out}.simba_bins.parquet\n\
                     and {out}.senna.json."
    )]
    pub(crate) out: Box<str>,
}

impl crate::update::Updatable for SimbaArgs {
    fn rebase(&mut self, r: crate::update::Rebase) {
        self.data_files = r.data_files;
        self.batch_files = r.batch_files;
        self.out = r.out;
        // No checkpoint and no pseudobulks: `init_from` and `reference` have
        // nothing to act on, so `update` re-fits on the union.
        if let Some(e) = r.epochs {
            self.epochs = e;
        }
    }
}
