//! Query decoder: a gene reads the visible context.
//!
//! The masked heads predict a gene from the cell summary alone, `μ_g = ℓ·(θβ)_g`,
//! so two cells with the same `θ` get the same prediction for every gene and
//! nothing gene-specific ever looks at the context. This module is the other
//! direction: a target gene becomes a **query** `ρ_g + e_mask` and attends over
//! the visible slots of its own row, the way a masked position in a masked
//! autoencoder reads the encoded patches. The read is a scalar residual `r_g`
//! that the caller adds to the cell-level log rate, so the mixture explains
//! what it can and attention carries only what the mixture cannot: the
//! co-expression a topic latent has no room for.
//!
//! Because a query is built from the gene identity alone, any gene can be a
//! query — a masked context gene or one outside the context with a zero count
//! — at the same cost.
//!
//! ## Projection through ρ
//!
//! A context token is `a·ρ_g` with a scalar Anscombe gate `a`, so the key and
//! value projections commute with the gate: `W (a·ρ_g) = a·(W ρ)_g`. The
//! decoder therefore projects the whole `[D, H]` table once per step and
//! gathers rows at the context indices, which costs `D·H·r` instead of
//! `N·K·H·r`, and takes the loader's `(indices, gate)` directly instead of the
//! encoder's tokens. Keys, values and queries all live in `rank` dimensions,
//! so the gene-by-gene score `(ρW_qᵀ)(W_kρᵀ)` has rank at most `rank` and the
//! `D × D` co-expression it learns is never materialized. The per-row
//! attention over the visible slots is returned so a caller can inspect which
//! genes a target read from: with no positions in a bag of genes, that is the
//! learned co-expression.

use candle_core::{Result, Tensor};
use candle_nn::{linear, linear_no_bias, ops, Linear, Module, VarBuilder};

/// One row-batch of context and targets, all `[N, ·]`, borrowed from the
/// minibatch.
pub struct QueryInput<'a> {
    /// `[N, K]` u32 context gene ids (pads carry index 0).
    pub indices: &'a Tensor,
    /// `[N, K]` the scalar gate each slot's `ρ` is multiplied by — the
    /// encoder's Anscombe value, 0 on pads.
    pub gate: &'a Tensor,
    /// `[N, K]` 1 where a slot may be read, 0 for masked and padded slots.
    pub visible: &'a Tensor,
    /// `[N, Q]` u32 target genes per row.
    pub query_ids: &'a Tensor,
}

/// What one forward returns.
pub struct QueryRead {
    /// `[N, Q]` residual log-rate per query; exactly 0 on a row with no
    /// visible slot.
    pub residual: Tensor,
    /// `[N, Q, K]` attention from each query over the row's context slots;
    /// zero on masked and padded slots.
    pub attention: Tensor,
}

pub struct QueryDecoder {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    /// `[1, H]` the mask vector added to a query gene's `ρ_g`.
    e_mask: Tensor,
    out: Linear,
    /// Width of the query, key and value projections.
    rank: usize,
}

impl QueryDecoder {
    /// Vars under `vs`: `q`, `k`, `v` (`[rank, H]`, no bias), `mask`
    /// (`[1, H]`), `out` (`[1, rank]` + bias).
    pub fn new(embedding_dim: usize, rank: usize, vs: VarBuilder) -> Result<Self> {
        let h = embedding_dim;
        if rank == 0 || rank > h {
            candle_core::bail!("query decoder: rank must be in 1..={h}, got {rank}");
        }
        let e_mask = vs.get_with_hints((1, h), "mask", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;
        Ok(Self {
            w_q: linear_no_bias(h, rank, vs.pp("q"))?,
            w_k: linear_no_bias(h, rank, vs.pp("k"))?,
            w_v: linear_no_bias(h, rank, vs.pp("v"))?,
            e_mask,
            out: linear(rank, 1, vs.pp("out"))?,
            rank,
        })
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// `rho` is the shared `[D, H]` gene embedding.
    pub fn forward(
        &self,
        features: &crate::feature_embedding::FeatureEmbedding,
        x: &QueryInput<'_>,
    ) -> Result<QueryRead> {
        let (n, k) = x.indices.dims2()?;
        let q = x.query_ids.dim(1)?;
        let r = self.rank;
        let h = features.embedding_dim();

        // Gather the slots this minibatch touches, THEN project. The other
        // order costs a pass over every feature — and under a composed feature
        // side it would compose the whole table to read a few thousand rows.
        let flat_idx = x.indices.flatten_all()?;
        let gate_nk1 = x.gate.unsqueeze(2)?; // [N, K, 1]
        let context = features.gather(&flat_idx)?; // [N*K, H]
        let keys = self
            .w_k
            .forward(&context)?
            .reshape((n, k, r))?
            .broadcast_mul(&gate_nk1)?; // [N, K, r]
        let values = self
            .w_v
            .forward(&context)?
            .reshape((n, k, r))?
            .broadcast_mul(&gate_nk1)?; // [N, K, r]
        let queries = features
            .gather(&x.query_ids.flatten_all()?)?
            .reshape((n, q, h))?
            .broadcast_add(&self.e_mask)?; // [N, Q, H]
        let qh = self.w_q.forward(&queries)?; // [N, Q, r]

        let scores = qh
            .matmul(&keys.transpose(1, 2)?.contiguous()?)?
            .affine(1.0 / (r as f64).sqrt(), 0.0)?; // [N, Q, K]
                                                    // (1 − visible)·(−1e9) keeps masked and padded slots out of the softmax.
        let neg_inf = x
            .visible
            .affine(-1.0, 1.0)?
            .affine(-1e9, 0.0)?
            .unsqueeze(1)?; // [N, 1, K]
        let attn = ops::softmax(&scores.broadcast_add(&neg_inf)?, 2)?; // [N, Q, K]
                                                                       // A row with nothing visible softmaxes uniformly over its pads: zero its
                                                                       // attention so the read, and everything downstream, is exactly 0.
        let has_visible = x.visible.sum_keepdim(1)?.gt(0.0)?.to_dtype(attn.dtype())?; // [N, 1]
        let attention = attn.broadcast_mul(&has_visible.unsqueeze(2)?)?;
        let read = attention.matmul(&values)?; // [N, Q, r]
                                               // The readout has a bias; a row that read nothing must still be exactly 0.
        let residual = self
            .out
            .forward(&read)?
            .squeeze(2)?
            .broadcast_mul(&has_visible)?; // [N, Q]
        Ok(QueryRead {
            residual,
            attention,
        })
    }
}

#[cfg(test)]
#[path = "query_decoder_tests.rs"]
mod query_decoder_tests;
