//! Hierarchical hyperpriors — conjugate Gibbs draws for the prior scales the
//! gate/sweep otherwise hold fixed. This is full Bayes on what the *variational*
//! gate hard-codes (`GATE_EFFECT_PRIOR_VAR = 1.0`, the sparsity, the ridge): learn
//! them and pool across genes. NOTE the trained gate no longer has a null column at
//! all (see `crate::model::SoftmaxGateSpec`); `π₀` here is this module's OWN
//! spike-and-slab null mass, not a readout of the gate's.
//!
//! Each is a cheap closed-form draw from sufficient statistics, run as a **serial
//! reduce** between the parallel per-gene/per-cell sweeps (map-reduce cadence).
//!
//! ## Numerical stability (why not IG(ε,ε))
//!
//! A variance estimated from few effects, in f32, collapses toward 0 under the
//! usual "non-informative" `IG(ε,ε)` (Gelman 2006). Instead each variance gets a
//! **half-Cauchy prior on the SD**, sampled through its Inverse-Gamma scale-mixture
//! auxiliary (Wand et al. 2011) so it stays a conjugate two-step Gibbs — mass at 0
//! for shrinkage, a heavy tail, and well-behaved as σ → 0. Draws are clamped to
//! the model's `GATE_LOGSTD_CLAMP`-scale band and accumulated in f64. The global
//! sparsity `π₀` is a Beta-Binomial draw, clamped off the `{0,1}` boundary exactly
//! as `compute_pip` guards with `.max(1e-15)`.
//!
//! Tiered escalation (turn on only once the previous mixes): Tier 0 = all fixed;
//! Tier 1 = the two global scalars σ₀² + π₀ ([`HalfCauchyVar`] + [`sample_pi0`]);
//! Tier 2 = per-dim variances under a shared prior, a `Vec<HalfCauchyVar>` composed
//! by [`super::dim_block`]. Only Tier 2 is on the CLI path — the whole-gene Tier 1
//! sampler was retired, and these primitives are what it left behind.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Distribution, Gamma};

/// SD-band clamp mirroring `model::GATE_LOGSTD_CLAMP` (`σ = exp(logσ)` stays finite).
const LOG_SD_CLAMP: f64 = 8.0;
/// Boundary guard on `π₀`, matching `compute_pip`'s `.max(1e-15)` spirit.
const PI0_EPS: f64 = 1e-4;

/// Clamp a variance into `[e^{-2·CLAMP}, e^{2·CLAMP}]` so `σ` and `log σ²` are finite.
fn clamp_var(v: f64) -> f64 {
    v.clamp((-2.0 * LOG_SD_CLAMP).exp(), (2.0 * LOG_SD_CLAMP).exp())
}

/// `X ~ Gamma(shape, rate)` then `1/X ~ Inverse-Gamma(shape, rate)`. `rand_distr`'s
/// `Gamma` is `(shape, scale = 1/rate)`.
fn inv_gamma(shape: f64, rate: f64, rng: &mut impl Rng) -> f64 {
    let g = Gamma::new(shape, 1.0 / rate.max(1e-30)).expect("valid IG params");
    let x: f64 = g.sample(rng);
    1.0 / x.max(1e-30)
}

/// A variance with a **half-Cauchy(scale) prior on its SD**, carried as the
/// IG-scale-mixture: `σ² | a ~ IG(½, 1/a)`, `a ~ IG(½, 1/scale²)` ⟹ `σ ~
/// HalfCauchy(scale)`. One [`Self::sample`] does the full two-step Gibbs given the
/// current effects' `Σx²`.
#[derive(Clone, Copy, Debug)]
pub struct HalfCauchyVar {
    /// Half-Cauchy scale `A` (the weakly-informative prior width).
    pub scale: f64,
    /// Auxiliary `a` (the scale-mixture latent), carried between sweeps.
    aux: f64,
}

impl HalfCauchyVar {
    /// New variance hyperparameter with half-Cauchy scale `A`.
    #[must_use]
    pub fn new(scale: f64) -> Self {
        Self {
            scale,
            aux: scale * scale,
        }
    }

    /// Draw `σ²` given `sum_sq = Σ_i x_i²` over `n` zero-mean Gaussians
    /// `x_i ~ N(0, σ²)`, and refresh the auxiliary. Clamped + f64.
    ///
    /// `σ² | ·  ~ IG((n+1)/2, Σx²/2 + 1/a)`,  `a | σ² ~ IG(1, 1/A² + 1/σ²)`.
    pub fn sample(&mut self, sum_sq: f64, n: usize, rng: &mut impl Rng) -> f64 {
        let shape = (n as f64 + 1.0) / 2.0;
        let rate = 0.5 * sum_sq + 1.0 / self.aux;
        let sigma2 = clamp_var(inv_gamma(shape, rate, rng));
        self.aux = inv_gamma(1.0, 1.0 / (self.scale * self.scale) + 1.0 / sigma2, rng);
        sigma2
    }
}

/// Beta-Binomial global sparsity `π₀` (null mass): given `n_null` of `n_total`
/// genes at this module's null state, `π₀ | · ~ Beta(a + n_null, b + n_active)`,
/// clamped off the boundary. `(a, b)` are the Beta hyperprior counts; `(9, 1)`
/// centers on 0.9, a weak prior that genes are mostly OFF. (Historically that
/// matched the trainer's null column, which no longer exists.)
#[must_use]
pub fn sample_pi0(n_null: usize, n_total: usize, a: f64, b: f64, rng: &mut impl Rng) -> f64 {
    let n_active = (n_total.saturating_sub(n_null)) as f64;
    let beta = Beta::new(a + n_null as f64, b + n_active).expect("valid Beta params");
    let p: f64 = beta.sample(rng);
    p.clamp(PI0_EPS, 1.0 - PI0_EPS)
}

/// Numerically stable logistic, branching on the sign so neither `exp` overflows.
/// The spike-and-slab inclusion draw in every sweep goes through this, so it lives
/// here with the other primitives rather than being re-rolled per sampler.
#[must_use]
pub(super) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Per-(gene, sweep) independent RNG stream — reproducible, no stored state. This
/// keying is what makes [`super::dim_block`] reproducible under any rayon schedule,
/// so it must stay a function of `(seed, unit, sweep)` and nothing else.
pub(super) fn gene_rng(seed: u64, unit: usize, sweep: usize) -> SmallRng {
    let s = seed
        ^ (unit as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (sweep as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    SmallRng::seed_from_u64(s)
}

#[cfg(test)]
#[path = "hyper_tests.rs"]
mod hyper_tests;
