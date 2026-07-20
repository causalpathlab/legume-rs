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
    EdgeBatchArgs, FeatPairing, PbFeatureSampler, PerBatchSampler, PerBatchStratifiedCellSampler,
    PerBatchStratifiedEdgeBatchArgs, StratifiedEdgeBatchArgs, StratifiedSampler,
};

/// The one canonical numerically-stable `log σ(x)` lives in `candle_util`;
/// re-exported here so `feat` (bipartite NCE), `chain` (cell-cell chain NCE),
/// and [`logistic_nce`] all share a single implementation.
pub(super) use candle_util::loss::log_sigmoid;

/// Per-positive logistic (SGNS) NCE loss, shared by the geu embedders gbe
/// (`feat`, bipartite NCE) and cage (`chain`, cell-cell chain NCE):
///
/// ```text
///   ℓ_i = -( log σ(pos_i) + Σ_blocks Σ_k log σ(-neg_{i,k}) )
/// ```
///
/// `pos` is `[B]`; each `negs` block is `[B, K]`. The `&[Tensor]` form admits
/// several concatenated negative slates with differing K; the current callers
/// (`feat`/`chain`) each pass a single block. Returns the per-positive loss
/// `[B]`; callers mean/weight as needed. The [`softmax_nce`] alternative (selected
/// per axis via [`NceObjective`]) is what `faba gem` trains its feature side with.
pub fn logistic_nce(pos: &Tensor, negs: &[Tensor]) -> Result<Tensor> {
    let mut term = log_sigmoid(pos)?;
    for neg in negs {
        term = (term + log_sigmoid(&neg.neg()?)?.sum(1)?)?;
    }
    term.neg()
}

/// Which NCE objective a feature-side loss uses. `Softmax` (default) is sampled-
/// softmax / InfoNCE: it normalizes the positive against its negatives in one
/// distribution, which separates cell types better on dense count data than
/// independent per-pair decisions — the default for `faba gem` and `senna bge`
/// (`--nce-objective`). `Logistic` is the SGNS per-pair loss, still selectable
/// (and the historical bge behaviour, kept byte-identical when chosen).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NceObjective {
    /// Per-pair logistic (SGNS) — [`logistic_nce`].
    Logistic,
    /// Sampled-softmax / InfoNCE — [`softmax_nce`].
    #[default]
    Softmax,
}

/// Per-positive **sampled-softmax (InfoNCE)** NCE loss: the positive competes
/// against its `negs` in a single softmax.
///
/// ```text
///   ℓ_i = −pos_i + logsumexp([pos_i, neg_{i,1..K}])
/// ```
///
/// Same signature/shape contract as [`logistic_nce`] (`pos` `[B]`, each `negs`
/// block `[B, K]`, returns `[B]`), so callers weight/mean identically. Unlike the
/// logistic loss — which decides each (pos, neg) pair independently and saturates
/// when many pairs are "somewhat positive" — the softmax normalization makes the
/// negatives compete with the positive, which sharpens separation on dense data.
pub fn softmax_nce(pos: &Tensor, negs: &[Tensor]) -> Result<Tensor> {
    // logits = [pos | neg blocks] along dim 1 → `[B, 1 + ΣK]`; the positive is
    // column 0, so the cross-entropy target is class 0 and `ℓ = logsumexp − pos`.
    let mut cols = Vec::with_capacity(1 + negs.len());
    cols.push(pos.unsqueeze(1)?);
    cols.extend(negs.iter().cloned());
    let logits = Tensor::cat(&cols, 1)?; // [B, 1 + ΣK]
                                         // Numerically-stable logsumexp over the candidates (subtract the row max).
    let m = logits.max_keepdim(1)?; // [B, 1]
    let lse = ((logits.broadcast_sub(&m)?).exp()?.sum_keepdim(1)?.log()? + m)?; // [B, 1]
    lse.squeeze(1)?.sub(pos)
}
