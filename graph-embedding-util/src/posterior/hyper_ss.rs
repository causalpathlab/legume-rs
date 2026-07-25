//! Complete Tier-1: an interleaved **spike-and-slab** joint Gibbs that samples,
//! per sweep, both global hyperparameters the gate hard-codes — the slab variance
//! `σ₀²` **and** the sparsity `π₀` (the gate's null mass).
//!
//! Model, per gene `g` (bias `b_g` fixed at the MAP, as in the gate):
//!   * slab effect `e_g ∈ ℝ^H`, prior `N(0, σ₀²·I)`;
//!   * inclusion `z_g ∈ {0,1}`, prior `Bernoulli(1 − π₀)` — `z_g = 0` is "load
//!     nothing" (the null column), so `θ_g = z_g · e_g`;
//!   * `σ₀² ~ HalfCauchy(A)` on the SD; `π₀ ~ Beta(a, b)`.
//!
//! Each sweep (all per-gene work is one rayon `par_iter`):
//!   1. **effect** — `K` elliptical-slice transitions of `e_g` under `N(0, σ₀²·I)`
//!      at the current `σ₀²`;
//!   2. **inclusion** — draw `z_g ~ Bernoulli(σ(logit))`, `logit = ln((1−π₀)/π₀) +
//!      ℓ(e_g) − ℓ(0)` (the on-vs-null log-likelihood ratio + prior odds);
//!   3. **reduce** — `Σ_{z_g=1}‖e_g‖²` over the *included* effects, and the null
//!      count `#{z_g=0}`;
//!   4. **hypers** — one [`HalfCauchyVar`] draw of `σ₀²` from the included Σ‖e‖²,
//!      and one [`sample_pi0`] Beta-Binomial draw of `π₀` from the null count.
//!
//! `σ₀²` uses only the *included* genes (spike-and-slab slab variance); `π₀` pools
//! the null indicators over all genes. Both are pooled over thousands of genes, so
//! (per the `hyper_sweep` note) the centered conjugate draws are stable. SIGINT is
//! polled at the sweep boundary.

use super::diagnostics::{scalar_diagnostics, ChainDiag};
use super::hyper::{gene_rng, sample_pi0, HalfCauchyVar};
use super::lnpdf::{poisson_ll, FrozenSide, NodeTerm};
use mcmc_util::engine::elliptical_slice_step;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Configuration for [`hyper_ss`].
pub struct HyperSsConfig {
    pub n_sweeps: usize,
    pub burnin: usize,
    /// ESS transitions per gene per sweep.
    pub transitions_per_sweep: usize,
    pub seed: u64,
    /// Half-Cauchy prior scale `A` on the effect SD (the slab width).
    pub half_cauchy_scale: f64,
    /// Beta hyperprior `(a, b)` on `π₀` (e.g. `(9, 1)` centers on `GATE_NULL_PRIOR`).
    pub beta_a: f64,
    pub beta_b: f64,
}

impl HyperSsConfig {
    #[must_use]
    pub fn new(n_sweeps: usize, burnin: usize, seed: u64) -> Self {
        Self {
            n_sweeps,
            burnin,
            transitions_per_sweep: 10,
            seed,
            half_cauchy_scale: 1.0,
            beta_a: 9.0,
            beta_b: 1.0,
        }
    }
}

/// Result of a spike-and-slab Tier-1 run.
pub struct HyperSsResult {
    /// Retained `σ₀²` draws (slab variance over included genes).
    pub sigma2_chain: Vec<f64>,
    /// Retained `π₀` draws (global null mass).
    pub pi0_chain: Vec<f64>,
    pub sigma2_mean: f64,
    pub pi0_mean: f64,
    /// Per-gene posterior inclusion probability `P(z_g = 1)` (mean over draws).
    pub inclusion_prob: Vec<f32>,
    /// Mixing diagnostics of the `σ₀²` and `π₀` chains.
    pub sigma_diag: ChainDiag,
    pub pi0_diag: ChainDiag,
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Run the interleaved spike-and-slab Tier-1. `nodes[g]` is gene `g`'s likelihood
/// terms against the frozen `side`; `biases[g]` its fixed MAP bias `b_g`.
#[must_use]
pub fn hyper_ss(
    nodes: &[NodeTerm],
    biases: &[f32],
    side: &FrozenSide,
    cfg: &HyperSsConfig,
) -> HyperSsResult {
    let h = side.h;
    let n_genes = nodes.len();
    assert_eq!(n_genes, biases.len(), "nodes / biases length mismatch");
    let k = cfg.transitions_per_sweep.max(1);
    let zeros_h = DVector::<f32>::zeros(h);

    // Live per-gene state: slab effect e_g + cached on-likelihood ℓ(e_g). The two
    // setup passes each evaluate `poisson_ll` over the whole partition per gene, so
    // run them in parallel (one-time, independent per gene, bit-identical).
    let mut e: Vec<DVector<f32>> = (0..n_genes).map(|_| DVector::zeros(h)).collect();
    let mut cur_ll: Vec<f32> = (0..n_genes)
        .into_par_iter()
        .map(|g| poisson_ll(e[g].as_slice(), f64::from(biases[g]), &nodes[g], side))
        .collect();
    // ℓ(0) per gene — the null score (bias only), constant across sweeps.
    let ll_off: Vec<f32> = (0..n_genes)
        .into_par_iter()
        .map(|g| poisson_ll(zeros_h.as_slice(), f64::from(biases[g]), &nodes[g], side))
        .collect();
    // BIC/Occam inclusion penalty `½·H·ln(m)` per gene — loop-invariant (the slab
    // dimension `H` and the gene's observation count are fixed), so precompute it
    // once rather than in the per-gene, per-sweep hot map.
    let bic_pen: Vec<f64> = nodes
        .iter()
        .map(|node| 0.5 * h as f64 * (node.partition.len().max(2) as f64).ln())
        .collect();

    let mut hv = HalfCauchyVar::new(cfg.half_cauchy_scale);
    let mut sigma2 = cfg.half_cauchy_scale * cfg.half_cauchy_scale;
    let mut pi0 = cfg.beta_a / (cfg.beta_a + cfg.beta_b);
    let mut hyper_rng = StdRng::seed_from_u64(cfg.seed ^ 0x5151_2323);

    let mut sigma2_chain: Vec<f64> = Vec::new();
    let mut pi0_chain: Vec<f64> = Vec::new();
    let mut incl_acc = vec![0.0f64; n_genes];
    let mut n_kept = 0usize;
    let stop = crate::stop::stop_flag();

    for sweep in 0..cfg.n_sweeps {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let sigma_e = (sigma2.max(1e-12)).sqrt() as f32;
        let log_prior_odds = ((1.0 - pi0).max(1e-12) / pi0.max(1e-12)).ln();

        // (1)+(2): per gene — sample the slab effect, then the inclusion.
        let out: Vec<(DVector<f32>, f32, bool)> = (0..n_genes)
            .into_par_iter()
            .map(|g| {
                let mut rng = gene_rng(cfg.seed, g, sweep);
                let node = &nodes[g];
                let b_g = f64::from(biases[g]);
                let lnpdf = |x: &DVector<f32>| poisson_ll(x.as_slice(), b_g, node, side);
                let mut eg = e[g].clone();
                let mut ll = cur_ll[g];
                for _ in 0..k {
                    let nu = DVector::from_fn(h, |_, _| {
                        let z: f64 = StandardNormal.sample(&mut rng);
                        z as f32 * sigma_e
                    });
                    let (ne, nl) = elliptical_slice_step(&eg, &nu, &lnpdf, ll, &mut rng);
                    eg = ne;
                    ll = nl;
                }
                // z_g ~ Bernoulli(σ(logit)). The likelihood gain ℓ(e_g) − ℓ(0) is
                // penalized by the slab's extra `H` parameters via the BIC/Occam term
                // `bic_pen[g]` (model selection: null bias-only vs bias + H-dim slab).
                // Without it a null gene's noisy slab fits ≈ as well as 0, so its
                // inclusion reverts to the prior and π₀ collapses downward.
                let logit = log_prior_odds + f64::from(ll) - f64::from(ll_off[g]) - bic_pen[g];
                let z = rng.random::<f64>() < sigmoid(logit);
                (eg, ll, z)
            })
            .collect();

        // (3): reduce over the included effects + the null count.
        let mut sum_sq = 0.0f64;
        let mut n_incl = 0usize;
        for (g, (eg, ll, z)) in out.into_iter().enumerate() {
            if z {
                sum_sq += eg
                    .as_slice()
                    .iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum::<f64>();
                n_incl += 1;
            }
            e[g] = eg;
            cur_ll[g] = ll;
            if sweep >= cfg.burnin && z {
                incl_acc[g] += 1.0;
            }
        }
        let n_null = n_genes - n_incl;

        // (4): draw both hypers.
        sigma2 = hv.sample(sum_sq, (n_incl * h).max(1), &mut hyper_rng);
        pi0 = sample_pi0(n_null, n_genes, cfg.beta_a, cfg.beta_b, &mut hyper_rng);

        if sweep >= cfg.burnin {
            sigma2_chain.push(sigma2);
            pi0_chain.push(pi0);
            n_kept += 1;
        }
    }

    let inv = 1.0 / n_kept.max(1) as f64;
    let inclusion_prob: Vec<f32> = incl_acc.iter().map(|&a| (a * inv) as f32).collect();
    let sigma2_mean = sigma2_chain.iter().sum::<f64>() * inv;
    let pi0_mean = pi0_chain.iter().sum::<f64>() * inv;
    let sigma_diag = scalar_diagnostics(&sigma2_chain);
    let pi0_diag = scalar_diagnostics(&pi0_chain);

    HyperSsResult {
        sigma2_chain,
        pi0_chain,
        sigma2_mean,
        pi0_mean,
        inclusion_prob,
        sigma_diag,
        pi0_diag,
    }
}

#[cfg(test)]
#[path = "hyper_ss_tests.rs"]
mod hyper_ss_tests;
