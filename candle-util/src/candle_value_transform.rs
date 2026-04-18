//! Unified, variance-stabilized, batch-adjusted, outlier-robust value
//! transform shared by all encoders.
//!
//! # Model
//!
//! The upstream PB adjustment (`data-beans-alg::collapse_data`) fits a
//! **multiplicative** model (matching the generative comment in
//! `optimize`: `E[y] = E[μ_resid] · E[μ_adj]`):
//!
//! ```text
//! μ_mixed  ≈ μ_adjusted · μ_residual
//! ```
//!
//! `μ_residual` is the per-gene/per-PB fold-factor (~1 for batch-clean
//! cells, >1 for batch-inflated genes, <1 for batch-depleted genes). The
//! encoder-side batch correction is therefore **division** — `y / μ_resid`
//! recovers the batch-clean count. Subtraction would be nearly a no-op
//! because `μ_resid` sits on a ratio scale (~1) while `y` sits on a count
//! scale (10s–1000s).
//!
//! After division we Anscombe-transform (`2√(x + 3/8)`) so the downstream
//! Gaussian embedding sees ~Poisson-standardized input with unit variance.
//!
//! # Formula
//!
//! ```text
//! clean_ig = y_ig / max(x0_ig, ε_div)         # multiplicative batch correction
//! a_ig     = 2√(clean_ig + 3/8)               # Anscombe variance stabilize
//! r_ig     = a_ig − mean_g(a_i,·)             # per-cell library-size center
//! s_g     = k · std_n(r_·,g) + ε              # per-gene clip scale
//! r̃_ig    = s_g · tanh(r_ig / s_g)            # soft winsorize
//! ```
//!
//! When `x0` is `None`: skip the division, everything else the same.
//!
//! # Why this handles spikes
//!
//! A raw 10000-count spike becomes `2√10000 ≈ 200` after Anscombe — already
//! tamed. The per-gene tanh then bounds it to roughly `k·std_g`, where
//! `std_g ≈ 1` under Poisson (Anscombe's whole point is that
//! Anscombe(Poisson) has unit variance). Spikes end up contributing ~k ≈ 4
//! to the encoder input, not 10000.
//!
//! # Why the `ε_div` floor
//!
//! `μ_resid` can get small for rare genes with sparse counterfactual
//! matches. Without a floor, `y / tiny` blows up. `ε_div = 0.1` caps the
//! amplification at ~10×, which is far beyond any plausible biological
//! batch effect and easily handled by the downstream tanh clip.

use candle_core::{Result, Tensor};

const TANH_K: f64 = 4.0;
const EPS: f64 = 1e-6;
const EPS_DIV: f64 = 0.1;

/// Apply the unified value transform. Output shape matches `y_nf`.
///
/// `y_nf` is non-negative count-space (raw PB counts or indexed subset).
/// `x0_nf`, when `Some`, is the multiplicative batch residual `μ_residual`
/// (ratio-space, centered at ~1) with the same shape as `y_nf`. Division
/// is floored at `EPS_DIV` to prevent blowup on rare genes.
pub fn anscombe_residual(y_nf: &Tensor, x0_nf: Option<&Tensor>) -> Result<Tensor> {
    debug_assert_eq!(y_nf.dims().len(), 2);
    if let Some(x0) = x0_nf {
        debug_assert_eq!(x0.dims(), y_nf.dims());
    }

    let clean = match x0_nf {
        Some(x0) => {
            let x0_safe = x0.clamp(EPS_DIV, f64::INFINITY)?;
            y_nf.broadcast_div(&x0_safe)?
        }
        None => y_nf.clone(),
    };
    let a = anscombe(&clean)?;

    let a_mean_n1 = a.mean_keepdim(1)?;
    let r = a.broadcast_sub(&a_mean_n1)?;

    let mean_1f = r.mean_keepdim(0)?;
    let diff = r.broadcast_sub(&mean_1f)?;
    let var_1f = diff.sqr()?.mean_keepdim(0)?;
    let std_1f = (var_1f + EPS)?.sqrt()?;
    let scale_1f = (std_1f * TANH_K)?;

    let r_scaled = r.broadcast_div(&scale_1f)?;
    let r_tanh = r_scaled.tanh()?;
    r_tanh.broadcast_mul(&scale_1f)
}

fn anscombe(t: &Tensor) -> Result<Tensor> {
    let shifted = (t + 0.375)?;
    let root = shifted.sqrt()?;
    root * 2.0
}
