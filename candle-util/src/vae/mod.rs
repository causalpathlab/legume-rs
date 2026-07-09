//! VAE-style training drivers for topic / link-community models.
//!
//! - [`topic`]: dense `EncoderModuleT` + `DecoderModuleT` trainer.
//! - [`masked_topic`]: `IndexedEmbeddingEncoder` + `EmbeddedTopicDecoder`
//!   trainer driven by [`crate::data::indexed::IndexedInMemoryData`].
//!
//! Shared utilities (`TrainScores`, `smooth_topics`, `PhaseTimers`,
//! grad-clipping helpers) live here at the module root.

pub mod masked_topic;
pub mod topic;

use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer};
use log::info;
use std::time::Duration;

/// Per-epoch llik / kl trace.
pub struct TrainScores {
    pub llik: Vec<f32>,
    pub kl: Vec<f32>,
}

/// Apply topic smoothing in log-space: `exp → mix with uniform → log`.
pub fn smooth_topics(log_z_nk: Tensor, alpha: f64) -> candle_core::Result<Tensor> {
    if alpha > 0.0 {
        let kk = log_z_nk.dim(1)? as f64;
        ((log_z_nk.exp()? * (1.0 - alpha))? + alpha / kk)?.log()
    } else {
        Ok(log_z_nk)
    }
}

/// Deterministic stick-breaking map `logits [N,K] → log θ [N,K]` on the simplex.
///
/// Interprets the first `K−1` columns as stick logits `η_k`; the stick
/// fractions are `v_k = σ(η_k)` and `θ_k = v_k ∏_{j<k}(1−v_j)`, with the last
/// topic taking the closing mass `θ_{K−1} = ∏_{j<K}(1−v_j)`. Computed entirely
/// in log-space (`log σ` via [`crate::loss::log_sigmoid`] + an inclusive cumsum
/// of `log(1−v)`), so it is numerically stable for `|η|` up to the encoder
/// clamp (±8) and any `K`. Rows sum to 1 exactly by telescoping — no explicit
/// normalization.
///
/// A drop-in for `log_softmax` on the masked head: no sampling, no KL, still a
/// point estimate. Unlike softmax it **breaks topic exchangeability** — early
/// sticks carry more mass a priori, giving an intrinsic ordering and a
/// self-pruning tail (later topics shrink toward 0 unless the data needs them).
pub fn stick_breaking_log_simplex(logits_nk: &Tensor) -> candle_core::Result<Tensor> {
    let k = logits_nk.dim(1)?;
    if k == 1 {
        // Degenerate simplex: θ ≡ 1, so log θ ≡ 0.
        return logits_nk.zeros_like();
    }
    // θ_k = v_k·∏_{j<k}(1−v_j), v_k = σ(η_k). Using the identity
    // log σ(η) − log σ(−η) = η, the log-simplex collapses to
    // log θ_k = η_k + Σ_{j≤k} log(1−v_j) — so only a single `log σ` (for
    // `log(1−v)`) and its inclusive cumsum are needed; `log v` and the
    // exclusive-cumsum subtraction both drop out.
    let eta = logits_nk.narrow(1, 0, k - 1)?; // [N, K−1] stick logits (strided view)
    let log_1mv = crate::loss::log_sigmoid(&eta.neg()?)?; // log(1−v_k) = log σ(−η_k)
    let incl = log_1mv.cumsum(1)?; // inclusive: Σ_{j≤k} log(1−v_j)
    let head = (&eta + &incl)?; // log θ_0..θ_{K−2} = η_k + Σ_{j≤k} log(1−v_j)
    let tail = incl.narrow(1, k - 2, 1)?; // log θ_{K−1} = full remaining mass [N, 1]
    Tensor::cat(&[&head, &tail], 1) // [N, K], rows sum to 1
}

/// Wall-clock breakdown of the indexed-topic training hot loop.
///
/// Candle's CPU backend is eager, so an `Instant` straddling each phase
/// attributes time accurately (no async kernels to sync). On CUDA the
/// kernels are async, so the per-phase split is only indicative.
#[derive(Default)]
pub struct PhaseTimers {
    pub precompute: Duration,
    pub encoder_fwd: Duration,
    pub decoder_fwd: Duration,
    pub backward: Duration,
    pub optimize: Duration,
}

impl PhaseTimers {
    pub fn log_summary(&self) {
        let total =
            self.precompute + self.encoder_fwd + self.decoder_fwd + self.backward + self.optimize;
        let total_s = total.as_secs_f64().max(1e-9);
        let pct = |d: Duration| 100.0 * d.as_secs_f64() / total_s;
        info!(
            "phase timing — precompute {:.1}s ({:.0}%), encoder_fwd {:.1}s ({:.0}%), \
             decoder_fwd {:.1}s ({:.0}%), backward {:.1}s ({:.0}%), opt_step {:.1}s ({:.0}%)",
            self.precompute.as_secs_f64(),
            pct(self.precompute),
            self.encoder_fwd.as_secs_f64(),
            pct(self.encoder_fwd),
            self.decoder_fwd.as_secs_f64(),
            pct(self.decoder_fwd),
            self.backward.as_secs_f64(),
            pct(self.backward),
            self.optimize.as_secs_f64(),
            pct(self.optimize),
        );
    }
}

/// Rescale every grad in-place so the global L2 norm is at most `max_norm`.
///
/// The `+1e-6` in the inverse-norm protects against zero-norm degeneracy;
/// the `clamp(0, 1)` makes this a no-op when the actual norm is already
/// below `max_norm`. Caller guarantees `max_norm > 0`.
fn apply_global_l2_clip(
    grads: &mut candle_core::backprop::GradStore,
    max_norm: f64,
) -> anyhow::Result<()> {
    let ids: Vec<_> = grads.get_ids().copied().collect();
    let mut sumsq: Option<Tensor> = None;
    for id in &ids {
        if let Some(g) = grads.get_id(*id) {
            let s = g.sqr()?.sum_all()?;
            sumsq = Some(match sumsq {
                None => s,
                Some(prev) => (prev + s)?,
            });
        }
    }
    let Some(sumsq) = sumsq else {
        return Ok(());
    };
    let inv_norm = sumsq.sqrt()?.affine(1.0, 1e-6)?.powf(-1.0)?;
    let scale = inv_norm.affine(max_norm, 0.0)?.clamp(0.0_f64, 1.0_f64)?;
    for id in &ids {
        if let Some(g) = grads.get_id(*id) {
            let scaled = g.broadcast_mul(&scale)?;
            grads.insert_id(*id, scaled);
        }
    }
    Ok(())
}

/// Backward + global-L2-norm gradient clipping + optimizer step.
/// `max_norm <= 0` skips clipping (falls back to plain `backward_step`).
///
/// Use this when you have a `loss` tensor in hand and want a one-call
/// step. The indexed trainer instead uses [`clip_and_step_dense`] which
/// takes pre-computed gradients so it can time `backward()` separately.
pub fn clip_grads_and_step<O: Optimizer>(
    opt: &mut O,
    loss: &Tensor,
    max_norm: f64,
) -> anyhow::Result<()> {
    if max_norm <= 0.0 {
        opt.backward_step(loss)?;
        return Ok(());
    }
    let mut grads = loss.backward()?;
    apply_global_l2_clip(&mut grads, max_norm)?;
    opt.step(&grads)?;
    Ok(())
}

/// Global-L2-norm clip + dense `AdamW` step from a precomputed `GradStore`.
///
/// Mirrors [`clip_grads_and_step`] but takes the already-computed grads so
/// the caller can time `backward()` separately (used by the indexed
/// trainer's [`PhaseTimers`]).
pub fn clip_and_step_dense(
    adam: &mut AdamW,
    mut grads: candle_core::backprop::GradStore,
    max_norm: f64,
) -> anyhow::Result<()> {
    if max_norm > 0.0 {
        apply_global_l2_clip(&mut grads, max_norm)?;
    }
    adam.step(&grads)?;
    Ok(())
}

/// Per-level loss-extension closure.
///
/// Trainers call this once per minibatch after computing the ELBO loss
/// and before backward. Senna uses it to inject the anchor-prior cross-
/// entropy penalty; pinto and other callers pass `None`.
///
/// Signature: `(loss, level) -> extended_loss`.
pub type LevelLossHook<'a> = dyn Fn(Tensor, usize) -> anyhow::Result<Tensor> + 'a;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Hand-computed reference for the stick-breaking map. With logits
    /// `η = [0, ln2, *]` (the 3rd column is ignored — only the first K−1 are
    /// sticks): `v₀ = σ(0) = 1/2`, `v₁ = σ(ln2) = 2/3`, so
    /// `θ = [v₀, v₁(1−v₀), (1−v₀)(1−v₁)] = [1/2, 1/3, 1/6]`.
    #[test]
    fn stick_breaking_matches_reference_and_normalizes() {
        let dev = Device::Cpu;
        let ln2 = std::f32::consts::LN_2;
        // Row 0: the reference case. Row 1: arbitrary logits → check it still
        // normalizes. The last column (999.0 / −999.0) must not affect output.
        let logits =
            Tensor::from_vec(vec![0.0f32, ln2, 999.0, -1.5, 2.0, -999.0], (2, 3), &dev).unwrap();

        let log_theta = stick_breaking_log_simplex(&logits).unwrap();
        assert_eq!(log_theta.dims(), &[2, 3]);
        let theta = log_theta.exp().unwrap().to_vec2::<f32>().unwrap();

        // Row 0 matches the closed form.
        let expect0 = [0.5f32, 1.0 / 3.0, 1.0 / 6.0];
        for (got, want) in theta[0].iter().zip(expect0.iter()) {
            assert!((got - want).abs() < 1e-5, "θ₀ {got} vs {want}");
        }
        // Both rows lie on the simplex (sum to 1) despite the huge 3rd logit.
        for row in &theta {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row sum {sum} ≠ 1");
            assert!(row.iter().all(|&p| p > 0.0), "non-positive θ entry");
        }
    }

    /// `K = 1` is a degenerate simplex: θ ≡ 1, log θ ≡ 0.
    #[test]
    fn stick_breaking_k1_is_degenerate() {
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![3.7f32, -2.0], (2, 1), &dev).unwrap();
        let log_theta = stick_breaking_log_simplex(&logits).unwrap();
        for v in log_theta.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert_eq!(v, 0.0, "K=1 log θ must be 0");
        }
    }
}
