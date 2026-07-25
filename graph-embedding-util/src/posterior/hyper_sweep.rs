//! Interleaved Tier-1 joint Gibbs: sample the per-gene effects **and** the global
//! slab variance `σ₀²` in one chain, the "sample hypers per sweep" cadence.
//!
//! Each outer sweep:
//!   1. **θ-update** — for every gene in parallel, `K` elliptical-slice transitions
//!      of `θ_g` with the prior `N(0, σ²·I)` **at the current σ²** (the likelihood
//!      `cur_lnpdf` carries across sweeps unchanged — it does not depend on σ²);
//!   2. **reduce** — `Σ_g ‖e_g‖²` over the embedding coordinates (a *live* draw, not
//!      a posterior mean);
//!   3. **hyper** — one conjugate [`HalfCauchyVar`] draw of `σ²` from that sum.
//!
//! **Centered, on purpose.** Non-centering is the funnel fix for a *data-poor*
//! variance that can wander to 0. `σ₀²` here is pooled over `n_genes × H` effects,
//! so its Inverse-Gamma conditional is razor-tight and never approaches the funnel's
//! throat — the centered conjugate draw is both correct and stable at this tier
//! (Tier 2's per-dim variances are the ones that need non-centering + pooling).
//! The σ²-chain diagnostics ([`Self::sigma_diag`]) are the arbiter: a stuck or
//! low-ESS σ²-chain would show it.
//!
//! The bias coordinate keeps its own fixed wide prior `σ_b` (it absorbs library
//! size and is not part of `σ₀²`).
//!
//! **SIGINT** is polled at the outer (per-sweep) boundary via
//! [`crate::stop::stop_flag`], the same flag `fit`/`train_composite` use. MCMC
//! degrades gracefully: on Ctrl+C the loop breaks and finalizes from whatever
//! draws were already collected (a shorter but valid chain), rather than hanging
//! or discarding everything. The per-gene θ-update within a sweep is a bounded
//! rayon `par_iter` and runs to the sweep boundary before the flag is checked.

use super::diagnostics::{scalar_diagnostics, ChainDiag};
use super::hyper::{gene_rng, HalfCauchyVar};
use super::lnpdf::{poisson_lnpdf, FrozenSide, NodeTerm};
use mcmc_util::engine::elliptical_slice_step;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Configuration for [`hyper_sweep`].
pub struct HyperSweepConfig {
    /// Outer sweeps (each = a θ-update + one σ² draw).
    pub n_sweeps: usize,
    /// Warm-up sweeps discarded before collecting.
    pub burnin: usize,
    /// ESS transitions per gene per sweep (`K`). `1` is the riskiest for mixing;
    /// a handful decorrelates θ from the last σ² without running to convergence.
    pub transitions_per_sweep: usize,
    pub seed: u64,
    /// Fixed wide prior SD on the bias coordinate.
    pub sigma_b: f32,
    /// Half-Cauchy scale `A` on the effect SD.
    pub half_cauchy_scale: f64,
}

impl HyperSweepConfig {
    #[must_use]
    pub fn new(n_sweeps: usize, burnin: usize, seed: u64) -> Self {
        Self {
            n_sweeps,
            burnin,
            transitions_per_sweep: 10,
            seed,
            sigma_b: 10.0,
            half_cauchy_scale: 1.0,
        }
    }
}

/// Result of an interleaved Tier-1 sweep.
pub struct HyperSweepResult {
    /// Per-gene posterior-mean `θ = [e ; b]` (length `h + 1`).
    pub theta_mean: Vec<Vec<f32>>,
    /// Retained `σ₀²` draws, one per post-burn-in sweep.
    pub effect_var_chain: Vec<f64>,
    /// Posterior mean of `σ₀²`.
    pub effect_var_mean: f64,
    /// Mixing diagnostics of the `σ₀²` chain (the Tier-1 stability read).
    pub sigma_diag: ChainDiag,
}

/// Run the interleaved joint Gibbs over `(θ per gene, σ₀²)`. `nodes[g]` is gene
/// `g`'s likelihood terms against the frozen `side`; `inits[g]` is its warm-start
/// `θ` (length `h + 1`).
#[must_use]
pub fn hyper_sweep(
    nodes: &[NodeTerm],
    inits: &[Vec<f32>],
    side: &FrozenSide,
    cfg: &HyperSweepConfig,
) -> HyperSweepResult {
    let h = side.h;
    let n_genes = nodes.len();
    assert_eq!(n_genes, inits.len(), "nodes / inits length mismatch");
    let k = cfg.transitions_per_sweep.max(1);

    // Live state: θ per gene + its cached likelihood. cur_ll is the likelihood
    // only (the prior lives in the ellipse), so it stays valid as σ² moves.
    let mut theta: Vec<DVector<f32>> = inits.iter().map(|v| DVector::from_vec(v.clone())).collect();
    // One-time, independent per gene (a full-partition poisson_lnpdf each) → parallel.
    let mut cur_ll: Vec<f32> = theta
        .par_iter()
        .zip(nodes.par_iter())
        .map(|(t, node)| poisson_lnpdf(t, node, side))
        .collect();

    let mut hv = HalfCauchyVar::new(cfg.half_cauchy_scale);
    let mut sigma2 = cfg.half_cauchy_scale * cfg.half_cauchy_scale; // start at A²
    let mut hyper_rng = StdRng::seed_from_u64(cfg.seed ^ 0xABCD_1234);

    let mut var_chain: Vec<f64> = Vec::new();
    let mut theta_acc = vec![vec![0.0f64; h + 1]; n_genes];
    let mut n_kept = 0usize;

    // SIGINT: the same graceful-stop flag fit/train_composite poll. Checked at
    // the sweep boundary so a Ctrl+C finalizes the collected draws instead of
    // hanging (see the module doc).
    let stop = crate::stop::stop_flag();

    for sweep in 0..cfg.n_sweeps {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let sigma_e = (sigma2.max(1e-12)).sqrt() as f32;
        let sigma_b = cfg.sigma_b;

        // (1) θ-update: K ESS transitions per gene, prior N(0, σ²I), parallel.
        let updated: Vec<(DVector<f32>, f32)> = (0..n_genes)
            .into_par_iter()
            .map(|g| {
                let mut rng = gene_rng(cfg.seed, g, sweep);
                let node = &nodes[g];
                let lnpdf = |x: &DVector<f32>| poisson_lnpdf(x, node, side);
                let mut th = theta[g].clone();
                let mut ll = cur_ll[g];
                for _ in 0..k {
                    let nu = DVector::from_fn(h + 1, |c, _| {
                        let z: f64 = StandardNormal.sample(&mut rng);
                        z as f32 * if c < h { sigma_e } else { sigma_b }
                    });
                    let (nt, nl) = elliptical_slice_step(&th, &nu, &lnpdf, ll, &mut rng);
                    th = nt;
                    ll = nl;
                }
                (th, ll)
            })
            .collect();
        for (g, (th, ll)) in updated.into_iter().enumerate() {
            theta[g] = th;
            cur_ll[g] = ll;
        }

        // (2) reduce: Σ‖e_g‖² over the embedding coords only (a live draw).
        let sum_sq: f64 = theta
            .iter()
            .map(|t| {
                t.as_slice()[..h]
                    .iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum::<f64>()
            })
            .sum();

        // (3) hyper: one conjugate σ² draw pooled over n_genes·H effects.
        sigma2 = hv.sample(sum_sq, n_genes * h, &mut hyper_rng);

        if sweep >= cfg.burnin {
            var_chain.push(sigma2);
            // Per-gene running sums are independent (no cross-gene reduction), so
            // the accumulation parallelizes bit-identically.
            theta_acc
                .par_iter_mut()
                .zip(theta.par_iter())
                .for_each(|(acc, t)| {
                    for c in 0..h + 1 {
                        acc[c] += f64::from(t[c]);
                    }
                });
            n_kept += 1;
        }
    }

    let inv = 1.0 / n_kept.max(1) as f64;
    let theta_mean: Vec<Vec<f32>> = theta_acc
        .iter()
        .map(|acc| acc.iter().map(|&s| (s * inv) as f32).collect())
        .collect();
    let effect_var_mean = var_chain.iter().sum::<f64>() * inv;
    let sigma_diag = scalar_diagnostics(&var_chain);

    HyperSweepResult {
        theta_mean,
        effect_var_chain: var_chain,
        effect_var_mean,
        sigma_diag,
    }
}

#[cfg(test)]
#[path = "hyper_sweep_tests.rs"]
mod hyper_sweep_tests;
