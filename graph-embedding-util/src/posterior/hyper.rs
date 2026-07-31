//! Hierarchical hyperpriors — conjugate Gibbs draws for the prior scales the
//! gate/sweep otherwise hold fixed. This is full Bayes on what the *variational*
//! gate hard-codes (`GATE_EFFECT_PRIOR_VAR = 1.0`, the sparsity, the ridge): learn
//! them and pool across genes. NOTE the trained gate no longer has a null column at
//! all (see `crate::model::FeatureGateSpec`); `π₀` here is this module's OWN
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
use rand::{Rng, RngExt, SeedableRng};
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

/////////////////////////////////////////
// Truncated IBP: stick-breaking sticks //
/////////////////////////////////////////

/// Max shrinkage steps in the univariate slice sampler below. Reaching it means the
/// bracket never found an acceptable point, which on a log-concave target means the
/// current value is already at a near-degenerate mode; keeping it is the safe fallback.
const SLICE_MAX_SHRINK: usize = 64;

/// Truncated Indian-Buffet-Process inclusion rates via stick-breaking
/// (Teh, Görür & Ghahramani 2007), held as the sticks themselves.
///
/// ```text
///   v_j ~ Beta(α, 1),      π_h = ∏_{j ≤ h} v_j
/// ```
///
/// NOT the same construction as `candle_util::vae::stick_breaking_log_simplex`, which
/// is the DP/GEM form `θ_k = v_k ∏_{j < k}(1 − v_j)` and sums to 1. This one is the IBP
/// form: the `π_h` are independent inclusion RATES and do not sum to anything.
///
/// Each `v_j ∈ (0,1)`, so `π` is **monotonically decreasing** in the dim index. That
/// ordering is the whole point, and it is what an independent `Beta(a,b)` per dim
/// cannot express: with ~34k genes on every dim, an O(1) Beta prior is swamped, so
/// each unused dim independently re-estimates its own inclusion rate back to the same
/// value — measured flat at 0.787–0.930 across 16 dims on BM1 while the likelihood
/// supported only ~3.4 of them. Here the tail is squeezed toward zero by CONSTRUCTION,
/// which no amount of data can outvote.
///
/// Truncating at `H` keeps the dimension fixed, so every downstream interface — the
/// `[rows, H]` mask, the `h0..hH` outputs, the phase-2 projection — is untouched. The
/// only thing the model sees is [`Self::pi0`], a length-`H` vector, exactly as before.
///
/// A second consequence worth naming: ordering the dims removes the dim-PERMUTATION
/// gauge, the last symmetry left once the gate breaks rotation. Dims become comparable
/// across runs the way PCA components are.
#[derive(Clone, Debug)]
pub struct StickBreaking {
    /// Concentration `α`. Larger ⇒ sticks nearer 1 ⇒ slower decay ⇒ more dims used.
    /// Under the IBP it is the expected number of dims a gene loads.
    pub alpha: f64,
    /// The sticks `v_j`, one per dim.
    v: Vec<f64>,
}

impl StickBreaking {
    /// Sticks at the prior mean `α/(α+1)`, so an untrained state states the prior.
    #[must_use]
    pub fn new(alpha: f64, h: usize) -> Self {
        Self {
            alpha,
            v: vec![alpha / (alpha + 1.0); h],
        }
    }

    /// Per-dim EXCLUSION rates `π₀ₕ = 1 − ∏_{j ≤ h} v_j`, the vector the sampler
    /// consumes. Monotonically INCREASING, since the inclusion rates decrease.
    #[must_use]
    pub fn pi0(&self) -> Vec<f64> {
        let mut cum = 1.0f64;
        self.v
            .iter()
            .map(|&vj| {
                cum *= vj;
                (1.0 - cum).clamp(PI0_EPS, 1.0 - PI0_EPS)
            })
            .collect()
    }

    /// One Gibbs sweep over the sticks given per-dim inclusion counts: `m[h]` of
    /// `n[h]` coordinates included on dim `h`.
    ///
    /// The conditional for a single `v_j` with the others fixed is, up to a constant,
    ///
    /// ```text
    ///   (α − 1 + Σ_{h ≥ j} m_h)·ln v_j  +  Σ_{h ≥ j} (n_h − m_h)·ln(1 − v_j·c_h)
    ///   where c_h = ∏_{i ≤ h, i ≠ j} v_i
    /// ```
    ///
    /// The first term is Beta-like; the `ln(1 − v·c)` term is what breaks conjugacy —
    /// hence slice sampling rather than a closed form. Both terms are concave in `v_j`,
    /// so the target is log-concave on `(0,1)` and simple interval shrinkage (no
    /// stepping-out needed on a bounded support) mixes well.
    pub fn sample(&mut self, m: &[usize], n: &[usize], rng: &mut impl Rng) {
        debug_assert_eq!(m.len(), self.v.len());
        debug_assert_eq!(n.len(), self.v.len());
        let h = self.v.len();
        for j in 0..h {
            // `c_h` for every h ≥ j: the product of the OTHER sticks up to h.
            let mut c = Vec::with_capacity(h - j);
            let mut prod = self.v[..j].iter().product::<f64>();
            for hh in j..h {
                if hh > j {
                    prod *= self.v[hh];
                }
                c.push(prod);
            }
            let pow: f64 = self.alpha - 1.0 + m[j..].iter().sum::<usize>() as f64;
            let ln_target = |v: f64| -> f64 {
                if !(v > 0.0 && v < 1.0) {
                    return f64::NEG_INFINITY;
                }
                let mut lp = pow * v.ln();
                for (k, hh) in (j..h).enumerate() {
                    let off = (n[hh] - m[hh].min(n[hh])) as f64;
                    if off > 0.0 {
                        lp += off * (1.0 - v * c[k]).max(f64::MIN_POSITIVE).ln();
                    }
                }
                lp
            };
            self.v[j] = slice_unit(self.v[j], ln_target, rng);
        }
    }
}

/// Univariate slice sampler for a log-concave density on `(0, 1)`.
///
/// Bounded support, so the bracket starts as the whole interval and only shrinks —
/// the stepping-out phase of Neal (2003) is unnecessary and would only cost evaluations.
fn slice_unit(x0: f64, ln_f: impl Fn(f64) -> f64, rng: &mut impl Rng) -> f64 {
    let y = ln_f(x0) + rng.random::<f64>().max(f64::MIN_POSITIVE).ln();
    if !y.is_finite() {
        return x0;
    }
    // Bracket the OPEN interval directly rather than clamping the accepted point
    // afterwards: a clamp can return a value the slice never accepted, which is a
    // silent bias rather than a guard.
    let (mut lo, mut hi) = (PI0_EPS, 1.0 - PI0_EPS);
    for _ in 0..SLICE_MAX_SHRINK {
        let x = lo + rng.random::<f64>() * (hi - lo);
        if ln_f(x) > y {
            return x;
        }
        if x < x0 {
            lo = x;
        } else {
            hi = x;
        }
    }
    x0
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
