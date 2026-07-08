//! Shared primitives for the two Bayesian binomial-logit assoc tests
//! ([`super::contrast_bayes`] and [`super::trend_bayes`]).
//!
//! Both fit a binomial GLM `logit(p_i) = ηᵢ(β)` with vague intercept + Gaussian shrinkage
//! prior on the effect coefficients, via [`mcmc_util::engine::EssSampler`], and summarize the
//! posterior of a **linear contrast** of `β` (start→end for the trend, branch indicator for
//! the contrast) as a mean + 90% credible interval + local false sign rate. The scaffolding
//! is orthogonal to the specific design matrix, so it lives here to prevent drift between
//! the two files.

/// Vague prior standard deviation for the intercept, on the logit scale (weakly informative;
/// the per-bin baselines / spline intercept sit here). The effect coefficients use a
/// caller-supplied `τ`.
pub(super) const INTERCEPT_SD: f32 = 10.0;

/// Numerically-stable `log(1 + exp(e))` with the standard ±20 clamp — the softplus in the
/// binomial-logit log-likelihood `Σ_i (k_i·ηᵢ − n_i·log(1 + e^ηᵢ))`.
#[inline]
pub(super) fn softplus(e: f32) -> f32 {
    if e > 20.0 {
        e
    } else if e < -20.0 {
        0.0
    } else {
        (1.0 + e.exp()).ln()
    }
}

/// Posterior summary of the linear contrast: mean, sd, 5% / 95% quantiles, and
/// `lfsr = min(P(effect > 0), P(effect < 0))` — the local false sign rate. Consumes
/// `effects` for the O(n) `select_nth_unstable`-based quantiles (no full sort).
/// Returns `None` when the chain is empty (a degenerate fit).
pub(super) fn summarize_posterior(mut effects: Vec<f32>) -> Option<Posterior> {
    let ns = effects.len();
    if ns == 0 {
        return None;
    }
    let mean = effects.iter().sum::<f32>() / ns as f32;
    let var = effects.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / ns as f32;
    let pos = effects.iter().filter(|&&e| e > 0.0).count() as f32 / ns as f32;
    let neg = effects.iter().filter(|&&e| e < 0.0).count() as f32 / ns as f32;

    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap();
    let idx = |pr: f32| ((pr * (ns as f32 - 1.0)).round() as usize).min(ns - 1);
    let (lo_i, hi_i) = (idx(0.05), idx(0.95));
    effects.select_nth_unstable_by(hi_i, cmp);
    let effect_hi = effects[hi_i];
    let effect_lo = if lo_i < hi_i {
        *effects[..hi_i].select_nth_unstable_by(lo_i, cmp).1
    } else {
        effect_hi
    };
    Some(Posterior {
        effect: mean,
        effect_sd: var.sqrt(),
        effect_lo,
        effect_hi,
        lfsr: pos.min(neg),
    })
}

/// Distinct per-(site, branch) seed so parallel ESS chains stay reproducible regardless of
/// thread order. `1009` is prime, keeping the site strides mutually offset from the branch
/// index.
#[inline]
pub(super) fn derive_seed(base: u64, si: usize, l: usize) -> u64 {
    base.wrapping_add((si as u64).wrapping_mul(1009))
        .wrapping_add(l as u64)
}

/// Posterior summary shared by both Bayesian tests. Consumers own their own `*Result` types
/// (which add test-specific metadata like `n_cells`, `total_cov`, `n_obs`).
pub(super) struct Posterior {
    pub effect: f32,
    pub effect_sd: f32,
    pub effect_lo: f32,
    pub effect_hi: f32,
    pub lfsr: f32,
}
