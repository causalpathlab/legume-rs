//! PyTorch-BigGraph (Lerer et al.) over a typed graph: nodes of several
//! types share one embedding table, every relation names its lhs and rhs
//! types and a loss weight, and training is PBG's softmax loss with
//! in-batch and uniform negatives on both corrupted sides, row-wise
//! Adagrad and stochastic weight decay. Uniform negatives are drawn inside
//! the relation's own node types, so a gene is never corrupted by a cell
//! type and a term never by a gene.
//!
//! This is the general engine; [`crate::simba`] is its two-type,
//! expression-binned special case and shares the constants below.

pub(crate) mod batch;
pub mod graph;
pub(crate) mod model;
pub mod train;

pub use candle_util::masking::MASK_NEG;
pub use candle_util::optim::{RowAdagrad, ADAGRAD_EPS};
pub use graph::{auto_wd, NodeTypeTable, Relation, RelationTable, TypedEdgeList};
pub use train::{train, FneOutput, RelationStats};

pub use crate::preset_mode::{PresetMode, PresetRows};
use candle_util::candle_core::Device;

/// PBG `init_scale`: each coordinate starts at `N(0, 1e-3)`.
pub const INIT_STDEV: f64 = 1e-3;

/// Per-epoch record: losses are per edge, weight decay excluded (PBG's
/// `Stats.loss`).
#[derive(Clone, Debug)]
pub struct EpochStats {
    pub epoch: usize,
    pub train_loss: f64,
    pub eval_loss: Option<f64>,
    /// Batches that drew the weight-decay term this epoch.
    pub wd_hits: usize,
}

/// Every knob of the recipe; `Default` is PBG's configuration with the
/// workspace's embedding dimension.
#[derive(Clone, Debug)]
pub struct FneConfig {
    /// PBG `dimension`.
    pub dim: usize,
    /// PBG `num_epochs`.
    pub epochs: usize,
    /// PBG `lr` (RowAdagrad).
    pub lr: f64,
    /// PBG `batch_size` (edges per single-relation batch).
    pub batch_size: usize,
    /// PBG `num_batch_negs` (chunk size; the chunk's other positives are negatives).
    pub num_batch_negs: usize,
    /// PBG `num_uniform_negs` (per chunk, shared by its positives).
    pub num_uniform_negs: usize,
    /// PBG `wd`; `None` = SIMBA's `auto_wd` from the edge count.
    pub wd: Option<f64>,
    /// PBG `wd_interval`: the decay is drawn with probability `1/wd_interval`
    /// per batch and scaled by `wd_interval`.
    pub wd_interval: usize,
    /// PBG `eval_fraction`, applied per relation: edges held out (once) and
    /// scored every epoch.
    pub eval_fraction: f64,
    /// Floor on the held-out edges of each relation when the fraction is
    /// positive, so a small relation still reports an eval loss; never more
    /// than leaves one training edge.
    pub eval_min_per_relation: usize,
    /// Passes over each relation's training edges per epoch, by relation
    /// index (missing entries count as 1). PBG draws batches in proportion
    /// to the edges left, so a relation a hundred times smaller than another
    /// gets a hundred times fewer updates; repeating it restores its share
    /// without touching the loss weight.
    pub relation_repeats: Vec<usize>,
    /// Rows given from outside, started from or pinned (see [`PresetRows`]).
    pub preset: Option<PresetRows>,
    pub seed: u64,
    pub device: Device,
}

impl Default for FneConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            epochs: 10,
            lr: 0.1,
            batch_size: 1000,
            num_batch_negs: 50,
            num_uniform_negs: 50,
            wd: None,
            wd_interval: 50,
            eval_fraction: 0.05,
            eval_min_per_relation: 1,
            relation_repeats: Vec::new(),
            preset: None,
            seed: 1,
            device: Device::Cpu,
        }
    }
}
