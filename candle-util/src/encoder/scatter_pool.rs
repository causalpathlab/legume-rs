//! Attention pooling over a cell's slots, re-associated so nothing `[N, K, H]`
//! is ever formed.
//!
//! The masked encoder pools a cell's context by gating one embedding row per
//! slot, scoring it against a learned query and summing the softmax-weighted
//! rows. Written slot by slot that needs the `[N, K, H]` block of gathered
//! rows, whose backward is the largest term in a training step and whose size
//! is what caps `K`.
//!
//! The same three sums, re-associated:
//!
//! ```text
//! rq_d    = ρ q                               [D]      one matvec per step
//! s_nk    = a_nk · rq[idx] / √H                [N, K]   gather from a vector
//! attn_nk = softmax_k(s_nk + mask)             [N, K]   unchanged
//! W_nd    = scatter_add(attn_nk · a_nk, idx)   [N, D]   repeated ids ADD
//! pool_nh = W_nd ρ                             [N, H]   one gemm
//! ```
//!
//! Exactly the same value: `⟨a·ρ_g, q⟩ = a·⟨ρ_g, q⟩` and
//! `Σ_k w_k ρ_{g(k)} = Σ_g (Σ_{k: g(k)=g} w_k) ρ_g`. Only the summation order
//! changes, so both paths carry the same gradient to every parameter — no
//! detach, no stop-gradient.

use crate::fast_index::{gather_rows, scatter_add_cols};
use crate::feature_embedding::FeatureEmbedding;
use candle_core::{Result, Tensor};

/// `scores + (1 − visible)·(−1e9)` — the ONE additive visibility mask.
///
/// Written as a single affine `v·1e9 − 1e9`, which is `(1 − v)·(−1e9)`
/// rearranged and exact for `v ∈ {0, 1}` (`1e9` is representable in `f32`, so
/// `1e9 − 1e9` is exactly `0` and a visible score is untouched). Both pools —
/// the slot scorer here and [`super::dense_pool::attention_scores_dense`] —
/// call this, so a hidden position leaves the softmax with the same weight
/// whichever path scored it.
pub fn masked_scores(scores: &Tensor, visible: &Tensor) -> Result<Tensor> {
    scores + visible.affine(1e9, -1e9)?
}

/// `s_nk = gate · rq[idx] · scale`, with `-1e9` added where `visible == 0`.
///
/// The gather is from the `[D]` vector `rq`, not from the `[D, H]` table, so
/// the scores cost `N·K` rather than `N·K·H`. `scale` is the caller's `1/√H`.
pub fn attention_scores_from_vector(
    gate_nk: &Tensor,
    idx_nk: &Tensor,
    rq_d: &Tensor,
    visible_nk: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    let (n, k) = gate_nk.dims2()?;
    let rq_nk = gather_rows(rq_d, &idx_nk.flatten_all()?.contiguous()?)?.reshape((n, k))?;
    let scores = (gate_nk * rq_nk)?.affine(scale, 0.0)?;
    // The same additive mask the block path used, so a masked slot leaves the
    // softmax with exactly the weight it had there.
    masked_scores(&scores, visible_nk)
}

/// `pool_nh = scatter_add(attn · gate, idx) ρ`, through the feature side.
///
/// A cell's weight on a gene is the sum over the slots holding it: ids repeat
/// (a context can name the same gene twice), and the scatter must ADD, not
/// overwrite, or the identity fails exactly where it is hardest to see.
pub fn pool_by_scatter(
    attn_nk: &Tensor,
    gate_nk: &Tensor,
    idx_nk: &Tensor,
    features: &FeatureEmbedding,
) -> Result<Tensor> {
    Ok(pool_parts(attn_nk, gate_nk, idx_nk, features)?.2)
}

/// `(w_nk, W_nd, pool_nh)` — [`pool_by_scatter`] with its intermediates, so a
/// test can pin both what they hold and how big they are.
fn pool_parts(
    attn_nk: &Tensor,
    gate_nk: &Tensor,
    idx_nk: &Tensor,
    features: &FeatureEmbedding,
) -> Result<(Tensor, Tensor, Tensor)> {
    let w_nk = (attn_nk * gate_nk)?; // [N, K]
    let w_nd = scatter_add_cols(idx_nk, &w_nk, features.n_features())?; // [N, D]
    let pool_nh = features.map_rows_linear(|rows| w_nd.matmul(rows))?; // [N, H]
    Ok((w_nk, w_nd, pool_nh))
}

/// `rq_d = ρ q` for the learned query `q [1, H]`: one matvec over the feature
/// side per step, replacing the per-slot inner products.
pub fn query_over_features(features: &FeatureEmbedding, attn_query_1h: &Tensor) -> Result<Tensor> {
    let h = attn_query_1h.elem_count();
    let q_h1 = attn_query_1h.reshape((h, 1))?;
    features.project_dims(&q_h1)?.squeeze(1)
}

#[cfg(test)]
#[path = "scatter_pool_tests.rs"]
mod scatter_pool_tests;
