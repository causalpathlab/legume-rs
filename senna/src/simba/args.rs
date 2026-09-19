//! `senna simba`'s command-line surface. Defaults are SIMBA's own (the
//! PyTorch-BigGraph settings its `pbg_train` uses), so a bare invocation is
//! the published recipe.

use senna::embed_common::*;
use data_beans_alg::hvg::HvgCliArgs;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
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

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) feature_embedding: crate::feature_embedding_args::FeatureEmbeddingArgs,

    #[arg(
        long,
        default_value_t = graph_embedding_util::EmbeddingDim::Fixed(50),
        value_name = "D|auto",
        alias = "dim-embedding",
        help = "Embedding dimension D (SIMBA: 50; auto = the width of a given feature embedding)"
    )]
    pub(crate) embedding_dim: graph_embedding_util::EmbeddingDim,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) train: crate::pbg_train_args::PbgTrainArgs,

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
                     {out}.feature_embedding.parquet (the raw gene table),\n\
                     {out}.feature_coembedding.parquet (the co-embedded genes),\n\
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
        // `reference` has nothing to act on: simba trains on cells, never on
        // pseudobulks. `init_from` is a different story — simba writes a
        // gene x H node table and the gene axis is shared across rounds, so a
        // gene-side warm start is available and simply is not wired up yet.
        // Until it is, `update` re-fits on the union.
        if let Some(e) = r.epochs {
            self.train.epochs = e;
        }
    }
}
