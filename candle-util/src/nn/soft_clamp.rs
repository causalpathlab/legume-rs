//! Gradient-preserving symmetric bound for latent logits.

use candle_core::{Result, Tensor};

/// Symmetric bound on a masked encoder's per-topic latent logits.
///
/// Shared by every masked head so the bound cannot drift between them.
pub const MASKED_LOGIT_CLAMP: f64 = 8.0;

/// Bound `x` to `(−c, c)` **without killing the gradient**: `c·tanh(x/c)`.
///
/// # Why not `clamp`
///
/// A hard `clamp` has EXACTLY zero gradient outside its range, so once the loss
/// pushes a logit past the bound that unit stops learning and can never come
/// back. Measured, not hypothetical: on a six-sample fit **99.6 % of cells had
/// a latent logit pinned at `+8`**, and the encoder froze while the likelihood
/// trace kept improving — so nothing in the trace showed it. The same failure is
/// recorded for the Gaussian head of `crate::vae::masked_topic` ("drove itself
/// into the ±8 clamp, where the gradient is exactly zero").
///
/// `c·tanh(x/c)` maps to the open interval `(−c, c)` with a gradient that decays
/// smoothly, so a unit driven to the edge is still recoverable.
///
/// # The bound this does not remove
///
/// Not unconditional: `tanh` reaches exactly `±1` in `f32` around `|x/c| ≈ 9`,
/// so beyond `|x| ≈ 9c` the gradient underflows to zero again. That is an order
/// of magnitude past where the hard clamp bit (it bit AT `c`), and these
/// pre-activations come from a `Linear` on batch-normed features, so they sit in
/// the low tens. It is a real limit, not an eliminated one.
///
/// # Applies to log-variance heads too
///
/// A `z_lnvar` clamp exists to stop `exp(lnvar)` overflowing, and `tanh`
/// preserves that bound exactly while keeping the head trainable at the edge.
pub fn soft_clamp(x: &Tensor, c: f64) -> Result<Tensor> {
    x.affine(1.0 / c, 0.0)?.tanh()?.affine(c, 0.0)
}

#[cfg(test)]
#[path = "soft_clamp_tests.rs"]
mod tests;
