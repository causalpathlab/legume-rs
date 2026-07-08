//! Bayesian within-branch trajectory association via elliptical slice sampling.
//!
//! Same spline design as the frequentist [`trend`](super::trend), but the coefficients
//! get a Gaussian smoothing prior — vague on the intercept, `N(0, τ²)` on the
//! standardized spline terms — which is the Bayesian form of a penalized spline (the
//! posterior mode is exactly a ridge-penalized GAM). ESS samples the posterior with no
//! tuning (mcmc-util). The association summary is the posterior of the net log-odds
//! change along the branch: its mean and 90% credible interval, plus an
//! `lfsr = min(P(effect > 0), P(effect < 0))` — the local false sign rate, small when
//! the branch's editing rate moves in a consistent direction.
//!
//! The prior regularizes `β`, so the near-separation blow-up that destabilizes the
//! frequentist point estimate simply does not occur here. Each (site, branch) runs one
//! sequential ESS chain; parallelism is over sites (no nested rayon).

use nalgebra::DVector;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use mcmc_util::engine::EssSampler;

use super::bayes_common::{derive_seed, softplus, summarize_posterior, Posterior, INTERCEPT_SD};
use super::gam::{build_spline_design, SplineDesign};
use super::io::{branch_buckets, Lineage, Site};

pub struct BayesTrendConfig {
    pub n_knots: usize,
    pub min_total_coverage: u64,
    pub min_cells: usize,
    /// Prior sd `τ` for the standardized spline coefficients (smaller ⇒ smoother).
    pub prior_sd: f32,
    pub n_samples: usize,
    pub warmup: usize,
    pub seed: u64,
}

pub struct BayesTrendResult {
    pub site: usize,
    pub branch: usize,
    pub n_cells: usize,
    pub total_cov: u64,
    /// Posterior mean net change in log-odds from branch start → end.
    pub effect: f32,
    pub effect_sd: f32,
    /// 5% and 95% posterior quantiles of the effect (90% credible interval).
    pub effect_lo: f32,
    pub effect_hi: f32,
    /// Local false sign rate `min(P(effect > 0), P(effect < 0))`.
    pub lfsr: f32,
}

/// Fit the Bayesian within-branch association for every (site, branch) that passes QC.
/// Parallel over sites; each branch is one sequential ESS chain.
pub fn run_trends_bayes(
    sites: &[Site],
    lin: &Lineage,
    cfg: &BayesTrendConfig,
) -> Vec<BayesTrendResult> {
    sites
        .par_iter()
        .enumerate()
        .flat_map_iter(|(si, s)| site_trends_bayes(si, s, lin, cfg).into_iter())
        .collect()
}

fn site_trends_bayes(
    si: usize,
    s: &Site,
    lin: &Lineage,
    cfg: &BayesTrendConfig,
) -> Vec<BayesTrendResult> {
    let mut out = Vec::new();
    for (l, bd) in branch_buckets(s, lin).into_iter().enumerate() {
        if bd.k.len() < cfg.min_cells || bd.cov < cfg.min_total_coverage {
            continue;
        }
        let seed = derive_seed(cfg.seed, si, l);
        if let Some((post, n_obs)) = fit_branch(&bd.k, &bd.n, &bd.x, cfg, seed) {
            out.push(BayesTrendResult {
                site: si,
                branch: l,
                n_cells: n_obs,
                total_cov: bd.cov,
                effect: post.effect,
                effect_sd: post.effect_sd,
                effect_lo: post.effect_lo,
                effect_hi: post.effect_hi,
                lfsr: post.lfsr,
            });
        }
    }
    out
}

/// ESS posterior for the net log-odds change of one branch, plus the number of design rows.
/// `None` if the branch cannot support a spline (degenerate design).
fn fit_branch(
    k: &[u32],
    n: &[u32],
    x: &[f32],
    cfg: &BayesTrendConfig,
    seed: u64,
) -> Option<(Posterior, usize)> {
    // The spline design is already f32 — consume it directly (no per-branch copy).
    let SplineDesign {
        x: xf,
        k: kf,
        n: nf,
        contrast,
    } = build_spline_design(k, n, x, cfg.n_knots)?;
    let m = xf.nrows();
    let p = xf.ncols();

    // Binomial log-likelihood (ESS wants the likelihood only; the Gaussian prior is
    // supplied through prior_draw).
    let lnpdf = |beta: &DVector<f32>| -> f32 {
        let eta = &xf * beta;
        let mut ll = 0.0f32;
        for i in 0..m {
            ll += kf[i] * eta[i] - nf[i] * softplus(eta[i]);
        }
        ll
    };

    // Prior sds: vague intercept, τ on each standardized spline coefficient.
    let mut sd = vec![cfg.prior_sd; p];
    sd[0] = INTERCEPT_SD;
    let prior_draw = |rng: &mut SmallRng| -> DVector<f32> {
        DVector::from_fn(p, |j, _| {
            let z: f64 = StandardNormal.sample(rng);
            sd[j] * z as f32
        })
    };

    let init = DVector::from_element(p, 0.0f32);
    let sampler = EssSampler {
        n_samples: cfg.n_samples,
        warmup: cfg.warmup,
        thin: 1,
        seed,
    };
    let chain = sampler.run(&lnpdf, &prior_draw, &init);
    // Posterior of the start→end net log-odds change.
    let effects: Vec<f32> = chain.samples.iter().map(|b| contrast.dot(b)).collect();
    summarize_posterior(effects).map(|post| (post, m))
}

#[cfg(test)]
mod tests;
