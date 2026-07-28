//! **Tier 2** — per-dimension block Gibbs: independent per-dim inclusion with
//! per-dim hyperparameters `(σ₀h², π₀h)`, the `hyper.rs` doc's "per-dim variances
//! under a shared prior".
//!
//! # Why this exists
//!
//! [`super::gate`] and [`super::hyper_ss`] both ask a *whole-gene* question. The
//! gate is a SuSiE single-effect — a softmax over dims, so exactly one dim wins
//! and the per-dim PIPs **compete**. `hyper_ss` is coarser still: one Bernoulli
//! per gene ("does this gene load anything"), with two global scalars `(σ₀², π₀)`
//! shared across all `H` dims.
//!
//! Selection is really a **per-dim** question — "which genes load dim `h`" — and
//! that is what this samples:
//!
//! ```text
//!   θ_gh = z_gh · β_gh,   β_gh ~ N(0, σ₀h²),   z_gh ~ Bernoulli(1 − π₀h)
//! ```
//!
//! Two consequences, and they are the point:
//!
//! 1. **PIPs no longer compete.** A gene loading many dims can read `PIP ≈ 1` on
//!    all of them, instead of having one softmax's worth of mass to divide. That
//!    is the whole argument: a set-valued truth cannot be reported by a model with
//!    one unit of mass to spend. Measured on planted data, a three-dim truth gets
//!    3.00 of 3 dims named here against 0.00 of 3 from the gate — not a tuning
//!    gap, since one unit split three ways cannot reach 0.5 anywhere.
//! 2. **Per-dim hypers instead of two global scalars.** One shared `π₀` forced to
//!    describe all `H` dims at once collapses to "include everything" (measured
//!    `π₀ = 0.0044` on BMMC). Each `(σ₀h², π₀h)` here pools over *all genes* on
//!    its own dim, so it is well identified — thousands of observations per
//!    parameter — and a coherent gene module can reinforce itself on its own dim
//!    without competing against every other dim's sparsity.
//!
//! # This is a MODELLING CHANGE, not a bug fix
//!
//! Independent per-dim inclusion is **not** the softmax single-effect the trained
//! variational gate uses, so this deliberately stops being "the exact posterior of
//! the deployed gate" — which [`super::gate`] remains. The two samplers are
//! therefore kept side by side rather than one replacing the other: pick `gate`
//! when fidelity to the deployed model is what matters, and this when the question
//! is genuinely "which dims does this gene use".
//!
//! NOTE the trained gate has since moved to normalizing over GENES within a dim
//! (`crate::model::SoftmaxGateSpec`), so neither sampler here is its exact
//! posterior — but this one, being per-dim, is the closer match.
//!
//! # Blocking, and the mixing trade
//!
//! Given the frozen cell side, genes are conditionally independent, so the outer
//! `rayon par_iter` is over **genes** and the dim scan runs *inside* each gene's
//! closure. That ordering is deliberate:
//!
//!   * it is a proper systematic-scan Gibbs — dim `h` conditions on the already
//!     updated dims `< h` of the same gene;
//!   * it amortizes the per-gene scratch buffers over the whole scan instead of
//!     reallocating per `(gene, dim)`;
//!   * the hypers update once per sweep, after every gene — standard blocked Gibbs.
//!
//! The honest cost: a coordinate-wise scan mixes **worse** than [`super::gate`]'s
//! joint-over-dims block when the dims are correlated, and they are — the dims are
//! coupled through the cell Gram `Σ = Eᶜᵀ Eᶜ`, which is not diagonal. We take that
//! trade because the *selection* estimand is what was wrong, not the mixing. The
//! standard fix if it bites is to scan in a whitened basis (`θ = Σ^{-1/2} z`) and
//! transform back for reporting; the per-dim readout is what must stay in the
//! original basis, not the scan.
//!
//! # Identifiability caveat — read the PIPs as a set
//!
//! With the cell side frozen the rotational gauge is gone *within a fit*, so
//! per-dim quantities are well defined there. They are **not** comparable across
//! fits: two seeds give cross-seed ARI 0.0365 on `argmax|E_feat|` against a
//! same-seed floor of 1.0000, because the basis rotates between runs even though
//! the geometry does not. Never compare a dim index from one fit to another.
//!
//! The residual within-fit risk is **collinearity**: if two dims of `e_cell` are
//! near-parallel, "which of them does gene `g` load" is not identifiable, and
//! independent per-dim marginals split the mass between them and can look
//! confidently wrong on *both*. This is not hypothetical — measured max VIF on
//! real 12k BMMC fits at `H=16` is **29–37**, far past the conventional 5. Check
//! [`super::frozen_diag`], which reports it per run and warns at 5. When it is
//! high, read a gene's row as a profile rather than a winner, and prefer
//! [`super::gate`]'s credible set for a set-valued answer.

use super::diagnostics::{scalar_diagnostics, ChainDiag};
use super::hyper::{bic_penalty, gene_rng, sample_pi0, sigmoid, HalfCauchyVar};
use super::lnpdf::{multinomial_ll, FrozenSide, NodeTerm};
use mcmc_util::engine::elliptical_slice_step;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Configuration for [`dim_block`].
pub struct DimBlockConfig {
    pub n_sweeps: usize,
    pub burnin: usize,
    /// ESS transitions per `(gene, dim)` per sweep. The scan already costs `H`
    /// likelihood passes per gene per sweep, so `1` here is the analogue of
    /// [`super::hyper_ss`]'s `transitions_per_sweep = 10`, not of `1`.
    pub transitions_per_dim: usize,
    pub seed: u64,
    /// Half-Cauchy prior scale `A` on each dim's slab SD.
    pub half_cauchy_scale: f64,
    /// Beta hyperprior `(a, b)` on every `π₀h`. Shared across dims — it is the
    /// prior, not the parameter, that is shared, which is what makes this a
    /// partial-pooled per-dim model rather than `H` unrelated fits.
    pub beta_a: f64,
    pub beta_b: f64,
}

impl DimBlockConfig {
    #[must_use]
    pub fn new(n_sweeps: usize, burnin: usize, seed: u64) -> Self {
        Self {
            n_sweeps,
            burnin,
            transitions_per_dim: 1,
            seed,
            half_cauchy_scale: 1.0,
            beta_a: 9.0,
            beta_b: 1.0,
        }
    }
}

/// Result of a per-dim block run. The `[n_genes × H]` tables are **row-major**,
/// gene-major — `pip[g * h + d]` — matching the `{out}.{tag}_pip.parquet` layout.
pub struct DimBlockResult {
    /// Per-`(gene, dim)` posterior inclusion probability `P(z_gh = 1)`. Unlike the
    /// gate's softmax PIP these do **not** sum to 1 across a gene's row.
    pub pip: Vec<f32>,
    /// Per-`(gene, dim)` posterior mean loading `E[β_gh]`.
    pub mean_beta: Vec<f32>,
    /// Posterior-mean slab variance per dim `[H]`.
    pub sigma2: Vec<f64>,
    /// Posterior-mean null mass per dim `[H]`.
    pub pi0: Vec<f64>,
    /// Mixing diagnostics per dim `[H]`.
    pub sigma_diag: Vec<ChainDiag>,
    pub pi0_diag: Vec<ChainDiag>,
    pub h: usize,
    /// Sweeps actually retained (less than `n_sweeps − burnin` only if SIGINT hit).
    pub n_kept: usize,
}

impl DimBlockResult {
    /// Gene `g`'s per-dim PIP row `[H]`.
    #[must_use]
    pub fn pip_row(&self, g: usize) -> &[f32] {
        &self.pip[g * self.h..(g + 1) * self.h]
    }

    /// Gene `g`'s per-dim posterior-mean loading row `[H]`.
    #[must_use]
    pub fn beta_row(&self, g: usize) -> &[f32] {
        &self.mean_beta[g * self.h..(g + 1) * self.h]
    }
}

/// Per-gene sweep state handed back from the parallel map.
struct GeneDraw {
    beta: DVector<f32>,
    z: Vec<bool>,
}

/// Run the per-dim block Gibbs. `nodes[g]` is gene `g`'s likelihood terms against
/// the frozen `side`; the per-gene intercept is profiled out by
/// [`multinomial_ll`], so none is supplied.
///
/// SIGINT is polled at the sweep boundary: the accumulators are already valid
/// posterior means over the sweeps completed so far, so an interrupted run
/// returns a coarser answer rather than a wrong or truncated one. Check
/// [`DimBlockResult::n_kept`] to see how many sweeps it actually got.
#[must_use]
pub fn dim_block(nodes: &[NodeTerm], side: &FrozenSide, cfg: &DimBlockConfig) -> DimBlockResult {
    let h = side.h;
    let n_genes = nodes.len();
    let k = cfg.transitions_per_dim.max(1);

    // Live per-gene slab effects. Start at zero, matching `hyper_ss`.
    // Live per-gene slab effects `β_g`, and one shared all-zero vector for the
    // "dim off" likelihood (hoisted out of the parallel map, as `hyper_ss` does).
    let mut beta: Vec<DVector<f32>> = (0..n_genes).map(|_| DVector::zeros(h)).collect();
    let zeros_h = DVector::<f32>::zeros(h);

    // Occam penalty for turning ONE coordinate on — `hyper_ss` passes `h` here
    // instead, because it prices the whole `H`-dim slab in a single decision.
    let bic_pen: Vec<f64> = nodes
        .iter()
        .map(|node| bic_penalty(1.0, node.partition.len()))
        .collect();

    let mut hv: Vec<HalfCauchyVar> = (0..h)
        .map(|_| HalfCauchyVar::new(cfg.half_cauchy_scale))
        .collect();
    let mut sigma2 = vec![cfg.half_cauchy_scale * cfg.half_cauchy_scale; h];
    let mut pi0 = vec![cfg.beta_a / (cfg.beta_a + cfg.beta_b); h];
    let mut hyper_rng = StdRng::seed_from_u64(cfg.seed ^ 0x7A11_0C0D);

    let mut sigma2_chain: Vec<Vec<f64>> = vec![Vec::new(); h];
    let mut pi0_chain: Vec<Vec<f64>> = vec![Vec::new(); h];
    let mut pip_acc = vec![0.0f64; n_genes * h];
    let mut beta_acc = vec![0.0f64; n_genes * h];
    let mut n_kept = 0usize;
    let stop = crate::stop::stop_flag();

    for sweep in 0..cfg.n_sweeps {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let sd: Vec<f32> = sigma2
            .iter()
            .map(|v| (v.max(1e-12)).sqrt() as f32)
            .collect();
        let log_prior_odds: Vec<f64> = pi0
            .iter()
            .map(|p| ((1.0 - p).max(1e-12) / p.max(1e-12)).ln())
            .collect();

        let draws: Vec<GeneDraw> = (0..n_genes)
            .into_par_iter()
            .map(|g| {
                let mut rng = gene_rng(cfg.seed, g, sweep);
                let node = &nodes[g];
                let mut th = beta[g].clone();
                let mut z = vec![false; h];

                // `off` is scratch, allocated once per gene per sweep and rewritten
                // per dim — the reason genes are the parallel axis. (`cur` is NOT
                // reused: `elliptical_slice_step` returns a fresh vector, so the
                // binding is rebound rather than written through.)
                let mut off = vec![0.0f32; h];
                let mut cur = DVector::<f32>::zeros(h);
                let mut nu = DVector::<f32>::zeros(h);

                // The likelihood of the gene's FULL current loading. It carries
                // across the whole dim scan: at every `d`, `off + cur` reconstructs
                // exactly `base + th`, so the value the previous ESS returned is
                // still the value this dim starts from. Recomputing it per dim —
                // which an earlier version did — costs a full pass over `pos` +
                // `partition` for every one of the `n_genes × H` pairs per sweep.
                let mut ll = multinomial_ll(th.as_slice(), node, side);

                for d in 0..h {
                    // The frozen direction for this coordinate: the gene's own
                    // caller-supplied offset (gem's velocity gate holds β_g here)
                    // PLUS every other dim of its current loading. That is what
                    // makes this the conditional θ_gd | θ_g,−d.
                    for (kk, o) in off.iter_mut().enumerate() {
                        let base = node.offset.map_or(0.0, |b| b[kk]);
                        *o = base + if kk == d { 0.0 } else { th[kk] };
                    }
                    let node_d = NodeTerm {
                        offset: Some(&off),
                        ..*node
                    };
                    let lnpdf = |x: &DVector<f32>| multinomial_ll(x.as_slice(), &node_d, side);

                    // `cur` and `nu` are zero off coordinate `d`, so the ESS
                    // ellipse `cur·cos + nu·sin` moves ONLY that coordinate —
                    // a 1-D slice through the joint, with the rest carried in
                    // `off`. Everything else about the step is the shared,
                    // already-tested elliptical slice sampler.
                    cur.fill(0.0);
                    cur[d] = th[d];
                    nu.fill(0.0); // only index `d` is ever written below
                    for _ in 0..k {
                        let g_std: f64 = StandardNormal.sample(&mut rng);
                        nu[d] = g_std as f32 * sd[d];
                        let (nc, nl) = elliptical_slice_step(&cur, &nu, &lnpdf, ll, &mut rng);
                        cur = nc;
                        ll = nl;
                    }
                    th[d] = cur[d];

                    // z_gd ~ Bernoulli(σ(logit)); the on-vs-off log-likelihood
                    // ratio for THIS coordinate alone, since `off` already holds
                    // every other dim. This evaluation buys the READOUT, not the
                    // chain — `θ_gd` is never actually spiked to zero — and it
                    // cannot be hoisted, since it moves with `θ_g,−d`.
                    let ll_off = lnpdf(&zeros_h);
                    let logit = log_prior_odds[d] + f64::from(ll) - f64::from(ll_off) - bic_pen[g];
                    z[d] = rng.random::<f64>() < sigmoid(logit);
                }
                GeneDraw { beta: th, z }
            })
            .collect();

        // Reduce per DIM: each column's included effects give that dim's slab
        // variance, and its null count gives that dim's sparsity.
        let mut sum_sq = vec![0.0f64; h];
        let mut n_incl = vec![0usize; h];
        let keep = sweep >= cfg.burnin;
        for (g, draw) in draws.into_iter().enumerate() {
            for d in 0..h {
                let v = f64::from(draw.beta[d]);
                if draw.z[d] {
                    sum_sq[d] += v * v;
                    n_incl[d] += 1;
                    if keep {
                        pip_acc[g * h + d] += 1.0;
                    }
                }
                if keep {
                    beta_acc[g * h + d] += v;
                }
            }
            beta[g] = draw.beta;
        }

        for d in 0..h {
            sigma2[d] = hv[d].sample(sum_sq[d], n_incl[d].max(1), &mut hyper_rng);
            pi0[d] = sample_pi0(
                n_genes - n_incl[d],
                n_genes,
                cfg.beta_a,
                cfg.beta_b,
                &mut hyper_rng,
            );
        }

        if keep {
            for d in 0..h {
                sigma2_chain[d].push(sigma2[d]);
                pi0_chain[d].push(pi0[d]);
            }
            n_kept += 1;
        }
    }

    let inv = 1.0 / n_kept.max(1) as f64;
    DimBlockResult {
        pip: pip_acc.iter().map(|&a| (a * inv) as f32).collect(),
        mean_beta: beta_acc.iter().map(|&a| (a * inv) as f32).collect(),
        sigma2: sigma2_chain
            .iter()
            .map(|c| c.iter().sum::<f64>() * inv)
            .collect(),
        pi0: pi0_chain
            .iter()
            .map(|c| c.iter().sum::<f64>() * inv)
            .collect(),
        sigma_diag: sigma2_chain.iter().map(|c| scalar_diagnostics(c)).collect(),
        pi0_diag: pi0_chain.iter().map(|c| scalar_diagnostics(c)).collect(),
        h,
        n_kept,
    }
}

#[cfg(test)]
#[path = "dim_block_tests.rs"]
mod dim_block_tests;
