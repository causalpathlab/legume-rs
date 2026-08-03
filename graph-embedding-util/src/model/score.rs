//! The bilinear score and the gated row gathers that feed it.
//!
//! The hot path from parameter tables to a `[B]` / `[B, K]` score: the
//! reparameterized effect draw, the per-row gate application, the cell-axis mean
//! pool, and the four score kernels (feature-cell and cell-cell, positives and
//! negatives). Split from the gate because these consume `α`, never define it.

use candle_util::candle_core::{Device, Result, Tensor};

use super::gate::{GateKind, GATE_LOGSTD_CLAMP};
use super::vars::pool_axis;
use super::{FeatFactor, JointEmbedModel};

impl JointEmbedModel {
    /// Reparameterize the Gaussian effect for the variational gate: `μ + σ·ε`, with
    /// `σ = exp(logstd_rows)` and a fresh `ε ~ N(0,1)` each call. `mu_rows` /
    /// `logstd_rows` are `[N, H]`. Used in the TRAINING gather so the posterior
    /// variance feeds the likelihood; output/materialize use the mean (`σ=0`).
    pub fn sample_effect(&self, mu_rows: &Tensor, logstd_rows: &Tensor) -> Result<Tensor> {
        let eps = Tensor::randn(0f32, 1f32, mu_rows.shape(), mu_rows.device())?;
        let sigma = logstd_rows
            .clamp(-GATE_LOGSTD_CLAMP, GATE_LOGSTD_CLAMP)?
            .exp()?;
        mu_rows.add(&sigma.mul(&eps)?)
    }

    /// One gated single-effect: reparam-sample the Gaussian effect (`μ + σ·ε`) when
    /// `sample` and a log-std is present (else use the mean `μ`), then apply the softmax
    /// gate when selection logits are present. `mu` / `logstd` are `[N, H]`, `s` is
    /// `[N, H+1]`. The shared per-row primitive for both the `β` and `δ` sides (and the
    /// free `e_feat`). With no logits/logstd it is the plain ungated gather.
    pub(crate) fn gated_rows(
        &self,
        mu: &Tensor,
        logstd: Option<&Tensor>,
        w: Option<&Tensor>,
        sample: bool,
    ) -> Result<Tensor> {
        let eff = match (sample, logstd) {
            (true, Some(ls)) => self.sample_effect(mu, ls)?,
            _ => mu.clone(),
        };
        match w {
            Some(w) => eff.mul(w),
            None => Ok(eff),
        }
    }

    /// Compose the effective factored feature rows: `β̃ + mask·δ̃`, where each side is
    /// its own gated effect ([`Self::gated_rows`]) — the IDENTITY gate on `β_g`
    /// and the INDEPENDENT velocity gate on `δ_g`. `genes` gathers the per-gene tables
    /// (`row_to_gene[idx]` in training, the full `row_to_gene` at materialize);
    /// `mask_rows` is the `[N,1]` unspliced selector (already gathered), `None` when
    /// there is no velocity. `sample` reparam-samples (training) vs uses means (output).
    pub(crate) fn factored_feat_rows(
        &self,
        f: &FeatFactor,
        genes: &Tensor,
        mask_rows: Option<&Tensor>,
        sample: bool,
    ) -> Result<Tensor> {
        let gather = |t: &Option<Tensor>| -> Result<Option<Tensor>> {
            t.as_ref().map(|x| x.index_select(genes, 0)).transpose()
        };
        let (beta_ls, beta_s) = (
            gather(&f.beta_logstd)?,
            self.gathered_gate_weights(GateKind::Identity, f.s_beta.as_ref(), genes)?,
        );
        let beta = self.gated_rows(
            &f.beta.index_select(genes, 0)?,
            beta_ls.as_ref(),
            beta_s.as_ref(),
            sample,
        )?;
        match (&f.splice_delta, mask_rows) {
            (Some((delta, _)), Some(m)) => {
                let (delta_ls, delta_s) = (
                    gather(&f.delta_logstd)?,
                    self.gathered_gate_weights(GateKind::Velocity, f.s_delta.as_ref(), genes)?,
                );
                let delta = self.gated_rows(
                    &delta.index_select(genes, 0)?,
                    delta_ls.as_ref(),
                    delta_s.as_ref(),
                    sample,
                )?;
                beta.add(&delta.broadcast_mul(m)?) // + mask ⊙ δ̃ on unspliced rows
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
        // Batched mat-vec, not broadcast-multiply-then-sum. Both compute
        // `Σ_h e_f_neg[i,k,h] · e_c[i,h]`, but the broadcast form materializes a
        // `[B, K, H]` product that backward turns into several more of the same,
        // and the expanded `e_c` has its stride-0 dim BETWEEN two non-zero
        // strides — the one layout candle's `offsets_b()` rejects, so it fell
        // through to a scalar `StridedIndex` loop with no SIMD and no threading.
        //
        // Measured in `pinto cage`, which had re-derived this same expression:
        // forward 18.1s → 7.4s and backward 39.4s → 16.3s over 5 epochs, taking
        // the whole run from 122s to 86s. `senna bge` and `faba gem` reach this
        // through `nce_loss`, so they inherit it.
        let dot = e_f_neg.matmul(&e_c.unsqueeze(2)?)?.squeeze(2)?; // [B, K]
        let b_c_b = b_c.unsqueeze(1)?.broadcast_as((b, k))?;
        (dot + b_f_neg)? + b_c_b
    }
}
