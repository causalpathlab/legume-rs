//! Two-groups **location mixture** (Efron-style local FDR) for embedding
//! feature-null calls, fit by a **fully collapsed** Gibbs sampler.
//!
//!   Efron, B. (2004). "Large-scale simultaneous hypothesis testing: the choice
//!   of a null hypothesis." *JASA* 99(465):96–104 — the two-groups model + lfdr.
//!   Neal, R. M. (2000). "Markov chain sampling methods for Dirichlet process
//!   mixture models." *JCGS* 9(2):249–265 — the collapsed (Algorithm 3) sampler.
//!   cf. Stephens, M. (2017). "False discovery rates: a new deal."
//!   *Biostatistics* 18(2):275–294 — the zero-centred scale-mixture special case.
//!
//! Observations `x_i` (the signed loadings on one embedding axis) follow
//! `x_i ~ π₀ N(0, σ₀²) + Σ_k π_k N(μ_k, σ_k²)`: one **null component pinned at
//! mean 0** (features the model never moved off init) plus **K free-location
//! non-null components** whose means `μ_k` AND variances `σ_k²` are free.
//! Allowing a non-zero mean lets a *shifted* signal bulk (e.g. an HVG-subset gem
//! run where every gene has moved off init) read as non-null instead of being
//! forced into the null — the failure mode of a zero-centred scale mixture. As
//! null and non-null now differ in **location**, `π₀` is identifiable and the
//! call quantity is the **lfdr** = `P(z_i = null)`, not ash's lfsr.
//!
//! **Fully collapsed Gibbs.** Both `π` (Dirichlet–multinomial) and each
//! component's `(μ, σ²)` (conjugate NIG / Inverse-Gamma) are integrated out, so
//! the assignment conditional is the Pólya-urn weight times a **Student-t
//! posterior predictive** over the *other* points in the component:
//! `P(z_i=j | z_{-i}) ∝ (n_{-i,j}+α)·T_j(x_i)`. Per-component sufficient stats
//! `(n_j, Σx, Σx²)` are maintained incrementally; `x_i` is removed from its
//! component before the conditional is formed and re-added after the draw. lfdr
//! is the Rao-Blackwellised null responsibility, averaged over sweeps. One
//! sampler runs per embedding dimension (callers parallelise across dimensions).
//! Deterministic given the seed.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use statrs::function::gamma::ln_gamma;

/// log Student-t density `t_ν(x; loc, scale²)`. `lg_diff = lnΓ((ν+1)/2) −
/// lnΓ(ν/2)` is passed in (cached by ν, which only depends on the component
/// count) so the two `lnΓ` calls are not repeated per observation.
#[inline]
fn student_t_logpdf(x: f64, nu: f64, loc: f64, scale2: f64, lg_diff: f64) -> f64 {
    let scale2 = scale2.max(1e-24);
    let z2 = (x - loc) * (x - loc) / scale2;
    lg_diff
        - 0.5 * (nu * std::f64::consts::PI * scale2).ln()
        - 0.5 * (nu + 1.0) * (1.0 + z2 / nu).ln()
}

/// Fitted per-observation posterior summaries.
pub struct AshResult {
    /// Per-observation local false-discovery rate `lfdr_i = P(z_i = null)` — the
    /// call quantity. A null (didn't-move) feature → high lfdr; a shifted/moved
    /// feature → low lfdr.
    pub lfdr: Vec<f64>,
    /// Posterior-mean null-component weight `π₀`.
    pub pi0: f64,
    /// Posterior-mean null variance `σ₀²` (the empirical "didn't-move" scale).
    pub null_var: f64,
}

/// Tuning for [`ash_normal`]'s fully-collapsed Gibbs fit.
#[derive(Clone, Copy)]
pub struct AshOpts {
    /// Free (non-null) location components; the mean-0 null atom is added on top.
    /// Generous is cheap under the collapsed sampler — empty components take no mass.
    pub n_components: usize,
    /// Burn-in sweeps discarded before collecting lfdr.
    pub burnin: usize,
    /// Sampling sweeps averaged for the Rao-Blackwellised lfdr.
    pub n_sweeps: usize,
    /// RNG seed (a fixed seed ⇒ reproducible).
    pub seed: u64,
}

impl Default for AshOpts {
    fn default() -> Self {
        Self {
            n_components: 24,
            burnin: 200,
            n_sweeps: 800,
            seed: 0xA5A5_0017_u64,
        }
    }
}

/// Fit the two-groups location mixture by fully-collapsed Gibbs and return the
/// per-observation lfdr = `P(z = null)`. `se_init > 0` seeds the prior scale.
/// Single threaded (callers parallelise across embedding dimensions).
#[must_use]
pub fn ash_normal(betahat: &[f64], se_init: f64, opts: &AshOpts) -> AshResult {
    let n = betahat.len();
    let k = opts.n_components.max(1);
    let m = k + 1; // + the mean-0 null atom (index 0)
    let s2_init = (se_init * se_init).max(1e-24);
    if n == 0 {
        return AshResult {
            lfdr: vec![],
            pi0: 1.0,
            null_var: s2_init,
        };
    }

    // Priors. Inverse-Gamma(a₀, b₀) on every variance, seeded to the "didn't-move"
    // scale (prior mean σ² = b₀/(a₀−1) = s2_init); a weak N(0, σ²/κ₀) on non-null
    // means; symmetric Dirichlet(α) weights (no null bias — the null earns mass).
    let a0 = 2.0_f64;
    let b0 = s2_init; // ⇒ prior mean σ² = s2_init
    let kappa0 = 0.01_f64;
    let alpha = 1.0_f64;
    let alpha_sum = alpha * m as f64;
    // Empty non-null component: broad prior predictive t_{2a₀}(0, (b₀/a₀)(κ₀+1)/κ₀).
    let empty_nu = 2.0 * a0;
    let empty_scale2 = (b0 / a0) * (kappa0 + 1.0) / kappa0;
    let empty_lg = ln_gamma(0.5 * (empty_nu + 1.0)) - ln_gamma(0.5 * empty_nu);

    // lnΓ((ν+1)/2) − lnΓ(ν/2) cached by component count c (ν = 2a₀ + c).
    let lg_diff: Vec<f64> = (0..=n)
        .map(|c| {
            let nu = 2.0 * a0 + c as f64;
            ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu)
        })
        .collect();

    let mut rng = StdRng::seed_from_u64(opts.seed);

    // Nearest-centre init: null centre 0, non-null centres quantile-spread.
    let mut sorted = betahat.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = |p: f64| sorted[((p * n as f64) as usize).min(n - 1)];
    let mut centre = vec![0.0f64; m];
    for (j, c) in centre.iter_mut().enumerate().skip(1) {
        let p = if k == 1 {
            0.5
        } else {
            0.05 + 0.90 * (j - 1) as f64 / (k as f64 - 1.0)
        };
        *c = quantile(p);
    }
    let mut z = vec![0usize; n];
    let mut cnt = vec![0.0f64; m];
    let mut s1 = vec![0.0f64; m]; // Σx per component
    let mut s2 = vec![0.0f64; m]; // Σx² per component
    for i in 0..n {
        let xi = betahat[i];
        let mut best = 0usize;
        let mut bd = (xi - centre[0]).abs();
        for (j, &c) in centre.iter().enumerate().skip(1) {
            let d = (xi - c).abs();
            if d < bd {
                bd = d;
                best = j;
            }
        }
        z[i] = best;
        cnt[best] += 1.0;
        s1[best] += xi;
        s2[best] += xi * xi;
    }

    let mut lfdr_acc = vec![0.0f64; n];
    let mut logw = vec![0.0f64; m];
    let mut pi0_acc = 0.0f64;
    let mut null_var_acc = 0.0f64;
    let mut collected = 0usize;

    for sweep in 0..(opts.burnin + opts.n_sweeps) {
        let collect = sweep >= opts.burnin;
        for i in 0..n {
            let xi = betahat[i];
            // Remove i from its component so every component's stats exclude it.
            let j0 = z[i];
            cnt[j0] -= 1.0;
            s1[j0] -= xi;
            s2[j0] -= xi * xi;

            // Pólya-urn weight × Student-t posterior predictive, in log space.
            for j in 0..m {
                let cj = cnt[j];
                let (nu, loc, scale2, lgd) = if j == 0 {
                    // Null: mean fixed 0; predictive t_{2a_n}(0, b_n/a_n).
                    let a_n = a0 + 0.5 * cj;
                    let b_n = b0 + 0.5 * s2[0];
                    (2.0 * a0 + cj, 0.0, b_n / a_n, lg_diff[cj as usize])
                } else if cj < 0.5 {
                    // Empty non-null: broad prior predictive.
                    (empty_nu, 0.0, empty_scale2, empty_lg)
                } else {
                    // Non-null: NIG posterior predictive (prior mean 0).
                    let xbar = s1[j] / cj;
                    let ss = (s2[j] - cj * xbar * xbar).max(0.0);
                    let kn = kappa0 + cj;
                    let mn = cj * xbar / kn;
                    let a_n = a0 + 0.5 * cj;
                    let b_n = b0 + 0.5 * ss + 0.5 * (kappa0 * cj / kn) * xbar * xbar;
                    (
                        2.0 * a0 + cj,
                        mn,
                        (b_n / a_n) * (kn + 1.0) / kn,
                        lg_diff[cj as usize],
                    )
                };
                logw[j] = (cj + alpha).ln() + student_t_logpdf(xi, nu, loc, scale2, lgd);
            }

            // Softmax (shared by the lfdr accumulation and the assignment draw).
            let mx = logw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0f64;
            for w in &mut logw {
                *w = (*w - mx).exp();
                sum += *w;
            }
            let inv = 1.0 / sum.max(1e-300);
            if collect {
                lfdr_acc[i] += logw[0] * inv; // P(z_i = null | z_{-i})
            }
            // Draw z_i.
            let mut u = rng.random::<f64>() * sum;
            let mut zi = m - 1;
            for (j, &w) in logw.iter().enumerate() {
                u -= w;
                if u <= 0.0 {
                    zi = j;
                    break;
                }
            }
            z[i] = zi;
            cnt[zi] += 1.0;
            s1[zi] += xi;
            s2[zi] += xi * xi;
        }

        if collect {
            pi0_acc += (cnt[0] + alpha) / (n as f64 + alpha_sum);
            let a_n = a0 + 0.5 * cnt[0];
            let b_n = b0 + 0.5 * s2[0];
            null_var_acc += b_n / (a_n - 1.0).max(1e-6); // posterior mean σ₀²
            collected += 1;
        }
    }

    let inv = 1.0 / collected.max(1) as f64;
    AshResult {
        lfdr: lfdr_acc.iter().map(|v| v * inv).collect(),
        pi0: pi0_acc * inv,
        null_var: null_var_acc * inv,
    }
}

#[cfg(test)]
#[path = "ash_tests.rs"]
mod tests;
