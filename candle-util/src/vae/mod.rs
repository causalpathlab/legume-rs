//! VAE-style training drivers for topic / link-community models.
//!
//! - [`topic`]: dense `EncoderModuleT` + `DecoderModuleT` trainer.
//! - [`indexed_topic`]: `IndexedEmbeddingEncoder` + `EmbeddedTopicDecoder`
//!   trainer driven by [`crate::data::indexed::IndexedInMemoryData`].
//!
//! Shared utilities (`TrainScores`, `smooth_topics`, `PhaseTimers`,
//! grad-clipping helpers) live here at the module root.

pub mod indexed_topic;
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
