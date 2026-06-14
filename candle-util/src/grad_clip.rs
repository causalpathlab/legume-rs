//! Global-norm gradient clipping for candle optimizers.
//!
//! candle's `Optimizer::backward_step` fuses `loss.backward()` + `step`, so
//! gradients flow into the optimizer unbounded. [`clipped_backward_step`]
//! restores the one-call ergonomics with a global-norm clip in between:
//! compute `‖g‖ = sqrt(Σ_i ‖g_i‖²)` over every parameter gradient and, when it
//! exceeds `max_norm`, scale them all by `max_norm / ‖g‖` — bounding the update
//! magnitude without changing its direction. Keeps embedding norms from
//! inflating on loss spikes.

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor};
use candle_nn::optim::Optimizer;

/// `loss.backward()` → global-norm clip → `opt.step()`, the clipped equivalent
/// of [`Optimizer::backward_step`]. `max_norm <= 0` disables clipping (plain
/// backward-step). Returns the pre-clip global gradient norm.
pub fn clipped_backward_step<O: Optimizer>(
    opt: &mut O,
    loss: &Tensor,
    max_norm: f64,
) -> Result<f64> {
    let mut grads = loss.backward()?;
    let norm = clip_grad_global_norm(&mut grads, max_norm)?;
    opt.step(&grads)?;
    Ok(norm)
}

/// Clip every gradient in `grads` to a global L2 norm of `max_norm`. No-op when
/// `max_norm <= 0` (clipping disabled) or the global norm is already within
/// bound. Returns the pre-clip global norm (for logging / diagnostics).
///
/// The per-parameter sums of squares are accumulated **on-device** and read
/// back as a single scalar (one host sync, not one per parameter).
pub fn clip_grad_global_norm(grads: &mut GradStore, max_norm: f64) -> Result<f64> {
    if max_norm <= 0.0 {
        return Ok(0.0);
    }
    // Release the immutable `get_ids` borrow before the mutable rescale pass.
    let ids: Vec<_> = grads.get_ids().copied().collect();
    let mut sumsq: Option<Tensor> = None;
    for id in &ids {
        if let Some(g) = grads.get_id(*id) {
            let s = g.sqr()?.sum_all()?;
            sumsq = Some(match sumsq {
                None => s,
                Some(t) => (t + s)?,
            });
        }
    }
    let norm = match sumsq {
        Some(t) => (t.to_scalar::<f32>()? as f64).sqrt(),
        None => 0.0,
    };
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for id in &ids {
            if let Some(g) = grads.get_id(*id) {
                let scaled = g.affine(scale, 0.0)?;
                grads.insert_id(*id, scaled);
            }
        }
    }
    Ok(norm)
}
