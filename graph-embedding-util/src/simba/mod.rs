//! A faithful, standalone re-implementation of SIMBA (Chen et al.), the
//! single-cell embedding method built on PyTorch-BigGraph: cells and genes are
//! nodes of a bipartite graph whose edges are the nonzero entries of the
//! log-normalized expression matrix, binned into a handful of expression
//! levels that each form their own (weighted) relation. Both node tables are
//! trained as free embeddings with PBG's softmax loss, batch + uniform
//! negatives on both sides, row-wise Adagrad and stochastic weight decay.
//!
//! Self-contained on purpose: this module never touches the composite trainer
//! (`fit`, `JointEmbedModel`, pseudobulks, gene modules). It borrows only
//! [`crate::loss::softmax_nce`], the progress bar and the stop flag, so it can
//! serve as an independent baseline for `senna bge`.
//!
//! Recipe (verified against `pinellolab/simba` and `pinellolab/simba_pbg`):
//! library-size normalize to [`SCALE_FACTOR`] and `log1p`; discretize the
//! nonzero values of ALL genes with a [`HIST_BINS`]-bin histogram and a
//! weighted 1-D k-means; one edge per nonzero HVG entry with its bin as the
//! relation, relation weights `linspace(1, 5)` over the present levels; PBG
//! defaults `dim 50, 10 epochs, lr 0.1, batch 1000, 50 batch + 50 uniform
//! negatives, wd auto, wd_interval 50, eval_fraction 0.05`; then a fixed-T
//! softmax co-embedding of genes onto cells and SIMBA's marker metrics.
//!
//! Stated deviations from SIMBA/PBG: the 5% evaluation edges are held out once
//! (PBG re-draws them every epoch); the 1-D k-means is solved exactly by
//! dynamic programming rather than sklearn's seeded local search; and
//! `si.pp.filter_genes(min_n_cells=3)` is not ported, so library sizes and the
//! histogram see every gene as loaded (the HVG selection is the gene filter).
//! The caller co-embeds genes onto the cells it keeps after QC, where SIMBA
//! uses every cell of the graph.

pub(crate) mod batch;
pub(crate) mod discretize;
pub(crate) mod graph;
pub(crate) mod metrics;
pub(crate) mod row_adagrad;
pub(crate) mod train;

pub use discretize::Discretization;
pub use graph::{auto_wd, EdgeList, RelationTable};
pub use metrics::{compare_entities, EntityMetrics};
pub use row_adagrad::RowAdagrad;
pub use train::{train, EpochStats, TrainOutput};

use candle_util::candle_core::{Device, Tensor};
use data_beans::sparse_io_vector::SparseIoVec;

/// PBG `init_scale`: each coordinate of both tables starts at `N(0, 1e-3)`.
pub const INIT_STDEV: f64 = 1e-3;
/// `si.pp.normalize(method='lib_size')` scale factor.
pub const SCALE_FACTOR: f64 = 1e4;
/// `si.tl.discretize(max_bins=100)`: bins of the initial histogram.
pub const HIST_BINS: usize = 100;
/// PBG's "ignore this negative" score.
pub const MASK_NEG: f64 = -1e9;
/// PBG `RowAdagrad` denominator floor.
pub const ADAGRAD_EPS: f64 = 1e-10;
/// `si.tl.compare_entities(n_top_cells=50)`.
pub const N_TOP_CELLS: usize = 50;
/// `si.tl.compare_entities(T=1)`.
pub const METRICS_T: f64 = 1.0;

/// Every knob of the recipe; `Default` is SIMBA's own configuration.
#[derive(Clone, Debug)]
pub struct SimbaConfig {
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
    /// PBG `eval_fraction`: edges held out (once) and scored every epoch.
    pub eval_fraction: f64,
    /// `si.tl.discretize(n_bins)`.
    pub n_bins: usize,
    /// `si.tl.embed(T)` for the caller's co-embedding.
    pub coembed_t: f64,
    pub seed: u64,
    pub device: Device,
}

impl Default for SimbaConfig {
    fn default() -> Self {
        Self {
            dim: 50,
            epochs: 10,
            lr: 0.1,
            batch_size: 1000,
            num_batch_negs: 50,
            num_uniform_negs: 50,
            wd: None,
            wd_interval: 50,
            eval_fraction: 0.05,
            n_bins: 5,
            coembed_t: 0.5,
            seed: 1,
            device: Device::Cpu,
        }
    }
}

pub struct SimbaOutput {
    /// `[N, D]` on the CPU.
    pub e_cell: Tensor,
    /// `[G, D]` on the CPU; row `g` is `hvg_rows[g]`.
    pub e_gene: Tensor,
    pub epochs: Vec<EpochStats>,
    pub discretization: Discretization,
    pub relations: RelationTable,
    /// Edges per relation, aligned with `relations.levels`.
    pub level_counts: Vec<usize>,
    pub wd: f64,
    pub n_edges: usize,
    pub n_train_edges: usize,
    pub n_eval_edges: usize,
}

/// Build the graph over `hvg_rows` (backend row indices) and train it.
/// Co-embedding and marker metrics are the caller's (see
/// [`crate::postprocess::feature_coembedding_fixed_t`] and
/// [`compare_entities`]).
pub fn run_simba(
    data: &SparseIoVec,
    hvg_rows: &[usize],
    cfg: &SimbaConfig,
) -> anyhow::Result<SimbaOutput> {
    let (edges, discretization) = graph::build_edge_list(data, hvg_rows, cfg.n_bins)?;
    let n_edges = edges.len();
    let mut counts = [0usize; 256];
    for &l in &edges.level {
        counts[l as usize] += 1;
    }
    let t = train::train(edges, cfg)?;
    let level_counts = t
        .relations
        .levels
        .iter()
        .map(|&l| counts[l as usize])
        .collect();
    Ok(SimbaOutput {
        e_cell: t.e_cell,
        e_gene: t.e_gene,
        epochs: t.epochs,
        discretization,
        relations: t.relations,
        level_counts,
        wd: t.wd,
        n_edges,
        n_train_edges: t.n_train_edges,
        n_eval_edges: t.n_eval_edges,
    })
}
