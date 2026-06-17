//! Count-NCE loss: NEG-style binary logistic over count-weighted
//! positive (cell, feature) edges vs within-batch marginal^α negatives.
//!
//! Negatives are drawn from features observed *in the positive cell's
//! batch*, so the model can't earn signal by separating cells along
//! technical-batch confounders — features that distinguish batches are
//! also exactly the candidate negatives for cells in those batches.
//!
//! ## Submodules
//!
//! - [`feat`] — cell-feature (bipartite) samplers and NCE losses used by
//!   `senna gbe` and the chain trainer.
//! - [`cell`] — cell-cell sampling primitives over a caller-provided
//!   graph (e.g. spatial KNN). Cross-batch and pb-mismatched edges are
//!   filtered, with per-chain-position sibling pools precomputed.
//! - [`chain`] — multi-level cell-cell NCE over the chain hierarchy.
//!   Consumed by `pinto cage`; includes the batched-over-genes variant.
//!
//! Public items are re-exported here so callers continue using
//! `graph_embedding_util::loss::Foo` paths unchanged.

use candle_util::candle_core::{Result, Tensor};

pub mod cell;
pub mod chain;
pub mod feat;

#[cfg(test)]
mod tests;

pub use cell::{
    build_per_batch_cell_samplers, CellCellSamplerStats, PbChainFilter, PerBatchCellSampler,
};
pub use chain::{
    cell_cell_nce_loss_per_level_batched_gated, cell_cell_nce_loss_per_level_gated,
    sample_cell_chain_batch, sample_cell_chain_batch_with_pos, CellChainBatch, CellChainBatchArgs,
    CellChainBatchStats,
};
pub use feat::{
    build_per_batch_samplers, build_per_batch_stratified_cell_samplers, build_stratified_sampler,
    nce_loss, nce_loss_chain, nce_loss_identity, sample_chain_batch, sample_edge_batch,
    sample_per_batch_stratified_edge_batch, sample_stratified_edge_batch, CellFeatureSampler,
    ChainAxis, ChainBatch, ChainBatchArgs, ChainFeatureSide, ChainSampler, EdgeBatch,
    EdgeBatchArgs, PbFeatureSampler, PerBatchSampler, PerBatchStratifiedCellSampler,
    PerBatchStratifiedEdgeBatchArgs, StratifiedEdgeBatchArgs, StratifiedSampler,
};

/// Numerically stable `log σ(x) = x - log_sum_exp([0, x])`. Shared by
/// `feat` (bipartite NCE) and `chain` (cell-cell chain NCE).
pub(super) fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    let stacked = Tensor::stack(&[x.zeros_like()?, x.clone()], 0)?;
    let softplus = stacked.log_sum_exp(0)?;
    x - softplus
}
