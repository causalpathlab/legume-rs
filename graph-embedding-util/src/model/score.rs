//! The bilinear score and the row gathers that feed it.
//!
//! The hot path from parameter tables to a `[B]` / `[B, K]` score: the factored
//! row composition, the cell-axis mean pool, and the four score kernels
//! (feature-cell and cell-cell, positives and negatives).

use candle_util::batched_dot::batched_matvec;
use candle_util::candle_core::{Device, Result, Tensor};

use super::vars::pool_axis;
use super::{FeatFactor, JointEmbedModel};

impl JointEmbedModel {
    /// Compose the factored feature rows: `β + mask·δ`. `genes` gathers the
    /// per-gene tables (`row_to_gene[idx]` in training, the full `row_to_gene` at
    /// materialize); `mask_rows` is the `[N,1]` unspliced selector (already
    /// gathered), `None` when there is no velocity.
    pub(crate) fn factored_feat_rows(
        &self,
        f: &FeatFactor,
        genes: &Tensor,
        mask_rows: Option<&Tensor>,
    ) -> Result<Tensor> {
        let beta = f.beta.index_select(genes, 0)?;
        match (&f.splice_delta, mask_rows) {
            // + mask ⊙ δ on the unspliced rows
            (Some((delta, _)), Some(m)) => {
                beta.add(&delta.index_select(genes, 0)?.broadcast_mul(m)?)
            }
            _ => Ok(beta),
        }
    }

    /// Mean-pool the cell embedding table over the fine children of a
    /// list of coarse-block indices. Output `[n_blocks, H]` plus a
    /// matching `[n_blocks]` bias vector.
    pub fn pool_cells(
        &self,
        coarse_blocks: &[u32],
        coarse_to_fine: &[Vec<usize>],
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        pool_axis(
            &self.e_cell,
            &self.b_cell,
            coarse_blocks,
            coarse_to_fine,
            dev,
        )
    }

    /// Bilinear score with bias terms.
    ///
    /// `e_f`: `[B, H]` pooled feature embeddings (one row per positive's
    /// feature block).
    /// `e_c`: `[B, H]` pooled cell embeddings (one row per positive's
    /// cell block).
    /// `b_f`, `b_c`: `[B]` bias scalars per row.
    /// Returns `[B]` scores.
    pub fn score_diag(e_f: &Tensor, e_c: &Tensor, b_f: &Tensor, b_c: &Tensor) -> Result<Tensor> {
        let dot = (e_f * e_c)?.sum(1)?;
        (dot + b_f)? + b_c
    }

    /// Bilinear score for negatives: score positive cells against
    /// alternative feature blocks. `e_f_neg`: `[B, K, H]`,
    /// `e_c`: `[B, H]`, `b_f_neg`: `[B, K]`, `b_c`: `[B]`. Returns `[B, K]`.
    pub fn score_negatives(
        e_f_neg: &Tensor,
        e_c: &Tensor,
        b_f_neg: &Tensor,
        b_c: &Tensor,
    ) -> Result<Tensor> {
        let b = e_f_neg.dim(0)?;
        let k = e_f_neg.dim(1)?;
        // Gemm, not broadcast-multiply-then-sum — see `candle_util::batched_dot`
        // for why. Measured in `pinto cage`, which had re-derived this same
        // expression: forward 18.1s → 7.4s, backward 39.4s → 16.3s over 5
        // epochs, taking the run from 122s to 86s.
        let dot = batched_matvec(e_f_neg, e_c)?; // [B, K]
        let b_c_b = b_c.unsqueeze(1)?.broadcast_as((b, k))?;
        (dot + b_f_neg)? + b_c_b
    }
}
