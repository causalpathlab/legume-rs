//! Attention pooling over EVERY feature, zeros included — the window-free
//! masked encoder's pool.
//!
//! [`super::scatter_pool`] re-associates the same three sums over a cell's `K`
//! context slots. Once nothing scales with `K · H`, the slots stop earning
//! their keep: feeding all `D` features costs one `[N, D]` product either way,
//! so the context window — and the shortlist that had to choose it — goes.
//!
//! ```text
//! rq_d    = ρ q                            [D]      one matvec per step
//! s_nd    = a_nd · rq_d / √H                [N, D]   no gather at all
//! attn_nd = softmax_d(s_nd + mask)          [N, D]   over the whole gene axis
//! pool_nh = (attn_nd · a_nd) ρ              [N, H]   one gemm
//! ```
//!
//! This is NOT the same model as the indexed path. A zero-count gene carries a
//! below-mean gate rather than no gate at all, and it takes part in the
//! softmax, so the encoder reads what a cell does not express as well as what
//! it does — the set the decoder was already scoring. With the visible mask
//! hiding everything outside a cell's support the two pools agree exactly,
//! which is what ties this module to [`super::scatter_pool`].

use crate::feature_embedding::FeatureEmbedding;
use candle_core::{Result, Tensor};

/// `s_nd = gate · rq · scale`, with `-1e9` added where `visible == 0`; no mask
/// when `visible_nd` is `None` (every gene visible).
///
/// `rq_d` is `[D]` (see [`super::scatter_pool::query_over_features`]), broadcast
/// across the rows; `scale` is the caller's `1/√H`, applied to the `[D]` vector
/// rather than to the `[N, D]` product.
pub fn attention_scores_dense(
    gate_nd: &Tensor,
    rq_d: &Tensor,
    visible_nd: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let rq_1d = rq_d.reshape((1, rq_d.elem_count()))?.affine(scale, 0.0)?;
    let scores = gate_nd.broadcast_mul(&rq_1d)?;
    match visible_nd {
        // The same additive mask the indexed path uses, so a hidden gene leaves
        // the softmax with exactly the weight it would have had as a masked slot.
        Some(v) => super::scatter_pool::masked_scores(&scores, v),
        None => Ok(scores),
    }
}

/// `pool_nh = (attn · gate) ρ`, through the feature side.
///
/// [`super::scatter_pool::pool_by_scatter`]'s scatter has nothing left to do
/// here: a gene appears once per row, so the `[N, D]` weight matrix IS the
/// product `attn · gate`.
pub fn pool_dense(
    attn_nd: &Tensor,
    gate_nd: &Tensor,
    features: &FeatureEmbedding,
) -> Result<Tensor> {
    let w_nd = (attn_nd * gate_nd)?; // [N, D]
    features.map_rows_linear(|rows| w_nd.matmul(rows))
}

#[cfg(test)]
#[path = "dense_pool_tests.rs"]
mod dense_pool_tests;
