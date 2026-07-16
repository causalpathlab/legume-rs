//! Adaptive shrinkage (ash) for the normal-means model, used to calibrate the
//! feature-null call. Faithful to `ashr::ash(..., mixcompdist = "normal")`:
//!
//!   Stephens, M. (2017). "False discovery rates: a new deal."
//!   *Biostatistics* 18(2):275–294. doi:10.1093/biostatistics/kxw041.
//!
//! Observations `betahat_i ~ N(beta_i, s²)` with a unimodal, zero-centred prior
//! `beta_i ~ π₀ δ₀ + Σ_k π_k N(0, σ_k²)` over a variance grid. The point mass
//! `σ₀² = 0` gives the null component; a large `π₀` = mostly null. Each
//! observation then gets a local false-discovery rate
//! `lfdr_i = P(beta_i = 0 | betahat_i) = π₀·N(betahat_i;0,s²) / f(betahat_i)`
//! and a false-sign rate (the robust call quantity), where `f` is the fitted
//! marginal.
//!
//! **Empirical null + collapsed Gibbs.** Canonical ash is handed the standard
//! errors `s_i` by the upstream measurement model. An embedding has none — the
//! "noise" is just how far a *never-moved* feature drifts from its init — so we
//! **estimate a common null variance `s²` jointly with the fit**. That makes the
//! `(π, s²)` objective non-convex (the convex-π guarantee only holds for fixed
//! `s`), so instead of a greedy EM that can stick in a local optimum we run a
//! **collapsed Gibbs sampler**: `π` is integrated out (Dirichlet–multinomial with
//! a null-biased concentration), the assignments `z_i` are resampled from the
//! Pólya-urn conditional `(n_{-i,j} + α_j)·N(betahat_i;0,σ_j²+s²)`, and `s²` is
//! drawn from a β-augmented inverse-gamma (residuals `betahat_i − β_i ~ N(0,s²)`
//! give exact conjugacy). The lfsr/lfdr are **Rao-Blackwellised** — averaged over
//! sweeps under each draw's plug-in `π̂`, so they integrate over the `s²`
//! posterior. The caller supplies only an **init** for `s`. One sampler runs per
//! embedding dimension (the callers parallelise across dimensions).
//!
//! Standalone (not the SGVB `MixtureGaussianPrior`) so the null call has no
//! variational dependency. Deterministic; the only inputs are the data.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma, StandardNormal};
use statrs::distribution::{ContinuousCDF, Normal};

/// `1/√(2π)`, the zero-mean Gaussian density's normalizing constant.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Zero-mean Gaussian density `N(x; 0, var)` — inlined so the `n·m` `lmat`
/// build doesn't construct a `statrs` distribution per cell.
#[inline]
fn gaussian0(x: f64, var: f64) -> f64 {
    let var = var.max(1e-24);
    INV_SQRT_2PI / var.sqrt() * (-0.5 * x * x / var).exp()
}

/// Fitted mixture + per-observation posterior summaries.
pub struct AshResult {
    /// Per-observation local FDR: posterior probability the effect is exactly 0
    /// (`= w₀`). Under-identified when the point mass and the smallest grid
    /// components overlap — prefer [`lfsr`](AshResult::lfsr) for the call.
    pub lfdr: Vec<f64>,
    /// Per-observation local false-sign rate: `P(β=0) + min(P(β>0), P(β<0))`.
    /// Well-identified even when `π₀` isn't (a tiny effect has an ambiguous
    /// sign ⇒ high lfsr), so this is the quantity to threshold for the call.
    pub lfsr: Vec<f64>,
    /// Null-component weight π₀ (point mass at 0).
    pub pi0: f64,
    /// Mixture weights aligned with [`grid`](AshResult::grid) (`pi[0]` = π₀).
    pub pi: Vec<f64>,
    /// Variance grid `σ_k²`; `grid[0] = 0` is the null point mass.
    pub grid: Vec<f64>,
    /// Posterior-mean common null variance `s²` (the empirical-null scale,
    /// averaged over the Gibbs sweeps).
    pub null_var: f64,
}

/// Tuning for [`ash_normal`]'s collapsed-Gibbs fit.
#[derive(Clone, Copy)]
pub struct AshOpts {
    /// Number of non-null variance-grid components (the null point mass is added
    /// on top). ~20 spans several orders of magnitude finely enough for lfsr.
    pub n_grid: usize,
    /// Burn-in sweeps discarded before collecting posterior summaries.
    pub burnin: usize,
    /// Sampling sweeps averaged for the Rao-Blackwellised lfsr/lfdr.
    pub n_sweeps: usize,
    /// RNG seed (the sampler is stochastic; a fixed seed ⇒ reproducible).
    pub seed: u64,
}

impl Default for AshOpts {
    fn default() -> Self {
        Self {
            n_grid: 20,
            burnin: 50,
            n_sweeps: 150,
            seed: 0xA5A5_0017_u64,
        }
    }
}

/// Fit the empirical-null ash normal-means model by **collapsed Gibbs** and
/// return per-observation lfdr/lfsr plus the posterior-mean null variance.
/// `se_init > 0` only **seeds** `s`; the sampler refines the common null
/// variance `s²`, so the calibration needs no external scale rule. Single
/// threaded (callers parallelise across embedding dimensions).
#[must_use]
pub fn ash_normal(betahat: &[f64], se_init: f64, opts: &AshOpts) -> AshResult {
    let n = betahat.len();
    let k = opts.n_grid.max(1);
    let m = k + 1; // K slab components + the null point mass (grid[0] = 0)
    if n == 0 {
        return AshResult {
            lfdr: vec![],
            lfsr: vec![],
            pi0: 1.0,
            pi: vec![1.0],
            grid: vec![0.0],
            null_var: (se_init * se_init).max(1e-24),
        };
    }

    // Fixed slab variance grid σ_j² (grid[0] = 0 is the null point mass, β = 0),
    // geometric from a floor below the seeded noise up to the largest loading.
    let bmax = betahat.iter().map(|b| b.abs()).fold(0.0, f64::max);
    let sigma_min = (se_init / 10.0).max(1e-6);
    let sigma_max = (bmax * 2.0).max(sigma_min * 8.0);
    let ratio = (sigma_max / sigma_min).powf(1.0 / (k as f64 - 1.0).max(1.0));
    let mut grid = vec![0.0f64; m];
    for j in 0..k {
        let sigma = sigma_min * ratio.powi(j as i32);
        grid[j + 1] = sigma * sigma;
    }

    // Dirichlet prior on π, null-biased: 10 extra pseudo-counts on the point mass
    // (ashr's default `prior = "nullbiased"`), 1 on each slab.
    let mut alpha = vec![1.0f64; m];
    alpha[0] = 11.0;
    let alpha_sum: f64 = alpha.iter().sum();
    // Weakly-informative inverse-gamma prior on s².
    let (ig_a0, ig_b0) = (1e-3_f64, 1e-3_f64);

    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut s2 = (se_init * se_init).max(1e-24);
    let std_normal = Normal::new(0.0, 1.0).expect("normal");

    // Initial assignments from the seeded prior×likelihood, tracking counts.
    let mut z = vec![0usize; n];
    let mut counts = vec![0.0f64; m];
    let mut probs = vec![0.0f64; m];
    for i in 0..n {
        let bi = betahat[i];
        for j in 0..m {
            probs[j] = alpha[j] * gaussian0(bi, grid[j] + s2);
        }
        let zi = sample_categorical(&probs, &mut rng);
        z[i] = zi;
        counts[zi] += 1.0;
    }

    let mut lfsr_acc = vec![0.0f64; n];
    let mut lfdr_acc = vec![0.0f64; n];
    let mut pi_acc = vec![0.0f64; m];
    let mut s2_acc = 0.0f64;
    let mut collected = 0usize;

    for sweep in 0..(opts.burnin + opts.n_sweeps) {
        // (1) Collapsed z-update (π integrated out): the Pólya-urn conditional
        //     p(z_i=j | z_{-i}) ∝ (n_{-i,j} + α_j)·N(betahat_i; 0, σ_j² + s²).
        for i in 0..n {
            let bi = betahat[i];
            counts[z[i]] -= 1.0;
            for j in 0..m {
                probs[j] = (counts[j] + alpha[j]) * gaussian0(bi, grid[j] + s2);
            }
            let zi = sample_categorical(&probs, &mut rng);
            z[i] = zi;
            counts[zi] += 1.0;
        }

        // (2) s² draw. Augment the effect β_i | z_i (0 if null, else its Gaussian
        //     posterior); the residuals betahat_i − β_i ~ N(0, s²) give exact
        //     conjugacy ⇒ s² ~ InvGamma(a₀ + n/2, b₀ + ½·Σ residual²).
        let mut rss = 0.0f64;
        for i in 0..n {
            let bi = betahat[i];
            let j = z[i];
            let r = if j == 0 {
                bi
            } else {
                let g = grid[j];
                let mu = bi * g / (g + s2);
                let tau = (g * s2 / (g + s2)).sqrt();
                let noise: f64 = StandardNormal.sample(&mut rng);
                bi - (mu + tau * noise)
            };
            rss += r * r;
        }
        let a_post = ig_a0 + 0.5 * n as f64;
        let b_post = ig_b0 + 0.5 * rss;
        let gdraw: f64 = Gamma::new(a_post, 1.0 / b_post)
            .expect("gamma")
            .sample(&mut rng);
        s2 = (1.0 / gdraw).max(1e-24);

        // (3) After burn-in, accumulate the Rao-Blackwellised posterior under this
        //     draw's plug-in π̂ = (counts + α)/(n + Σα). `lfdr = w₀` (null point
        //     mass); for slab j the effect posterior is N(μ_j, τ_j²) with
        //     μ_j = betahat·σ_j²/(σ_j²+s²), τ_j² = σ_j²s²/(σ_j²+s²), so
        //     P(β>0|j) = Φ(μ_j/τ_j); lfsr = w₀ + min(P(β>0), P(β<0)).
        if sweep >= opts.burnin {
            let denom_pi = n as f64 + alpha_sum;
            for j in 0..m {
                probs[j] = (counts[j] + alpha[j]) / denom_pi; // reuse as π̂
            }
            for i in 0..n {
                let bi = betahat[i];
                let mut denom = 0.0f64;
                for j in 0..m {
                    denom += probs[j] * gaussian0(bi, grid[j] + s2);
                }
                let denom = denom.max(1e-300);
                let w0 = (probs[0] * gaussian0(bi, s2) / denom).clamp(0.0, 1.0);
                let mut pos = 0.0;
                for j in 1..m {
                    let wj = probs[j] * gaussian0(bi, grid[j] + s2) / denom;
                    let g = grid[j];
                    let mu = bi * g / (g + s2);
                    let tau = (g * s2 / (g + s2)).sqrt().max(1e-12);
                    pos += wj * std_normal.cdf(mu / tau);
                }
                let neg = (1.0 - w0 - pos).max(0.0);
                lfdr_acc[i] += w0;
                lfsr_acc[i] += (w0 + pos.min(neg)).clamp(0.0, 1.0);
            }
            for j in 0..m {
                pi_acc[j] += probs[j];
            }
            s2_acc += s2;
            collected += 1;
        }
    }

    let inv = 1.0 / collected.max(1) as f64;
    let lfsr: Vec<f64> = lfsr_acc.iter().map(|v| v * inv).collect();
    let lfdr: Vec<f64> = lfdr_acc.iter().map(|v| v * inv).collect();
    let pi: Vec<f64> = pi_acc.iter().map(|v| v * inv).collect();
    AshResult {
        pi0: pi[0],
        lfdr,
        lfsr,
        pi,
        grid,
        null_var: s2_acc * inv,
    }
}

/// Draw an index from unnormalised categorical `weights` (inverse-CDF).
fn sample_categorical(weights: &[f64], rng: &mut StdRng) -> usize {
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 || sum.is_nan() {
        return 0;
    }
    let mut u = rng.random::<f64>() * sum;
    for (j, &w) in weights.iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            return j;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
#[path = "ash_tests.rs"]
mod tests;
