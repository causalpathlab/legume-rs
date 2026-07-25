//! One posterior sweep of a side, holding the other fixed.
//!
//! Each anchor node's `θ_a = [e_a ; b_a]` is conditionally independent given the
//! frozen other side (Poisson likelihood — see [`super::lnpdf`]), so the sweep is
//! `rayon par_iter` over nodes, one elliptical-slice chain each. This is the same
//! embarrassingly-parallel-over-units idiom as `crate::cell_projection::project_cells`;
//! the per-node chain reuses `mcmc_util`'s `EssSampler` verbatim.
//!
//! The prior is the model's ridge: `e_a ~ N(0, σ_e²·I_H)` and a wide `b_a ~ N(0,
//! σ_b²)` (the bias absorbs library size and stays near-unpenalized). ESS draws
//! its ellipse from this prior and the likelihood is passed as `lnpdf`, so the
//! stationary distribution is the exact posterior `∝ likelihood · N(0, Σ)`.
//!
//! Returns per-node posterior **mean** plus the retained **draws**, so a caller
//! can form any gauge-invariant summary (reconstructed scores, norms, credible
//! intervals) — see the recovery test. When one side is frozen at a fixed basis
//! there is no rotational gauge freedom, so the draws are directly comparable.

use super::lnpdf::{poisson_lnpdf, FrozenSide, NodeTerm};
use mcmc_util::engine::EssSampler;
use nalgebra::DVector;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Chain + prior configuration for a sweep.
pub struct SweepConfig {
    pub n_samples: usize,
    pub warmup: usize,
    pub thin: usize,
    /// Base seed; node `i` runs at `seed + i` (reproducible, independent streams).
    pub seed: u64,
    /// Prior SD on the embedding coordinates (`σ_e`, the ridge `√(1/2λ)`).
    pub sigma_e: f32,
    /// Prior SD on the bias coordinate (`σ_b`, wide — the bias absorbs depth).
    pub sigma_b: f32,
    /// Retain the per-draw samples (for coverage / credible intervals). When
    /// `false`, only the mean is kept and `samples` is empty.
    pub keep_samples: bool,
}

impl SweepConfig {
    /// A sensible default: `σ_e` from the ridge, a wide bias prior, samples kept.
    #[must_use]
    pub fn new(n_samples: usize, warmup: usize, sigma_e: f32, seed: u64) -> Self {
        Self {
            n_samples,
            warmup,
            thin: 1,
            seed,
            sigma_e,
            sigma_b: 10.0,
            keep_samples: true,
        }
    }
}

/// One node's posterior after a sweep.
pub struct NodePosterior {
    /// Posterior mean of `θ_a = [e_a ; b_a]` (length `h + 1`).
    pub mean: Vec<f32>,
    /// Retained draws (each length `h + 1`); empty when `keep_samples` is off.
    pub samples: Vec<Vec<f32>>,
}

impl NodePosterior {
    /// Posterior-mean embedding `e_a` (drops the bias coordinate).
    #[must_use]
    pub fn e_mean(&self) -> &[f32] {
        &self.mean[..self.mean.len() - 1]
    }

    /// Posterior-mean bias `b_a`.
    #[must_use]
    pub fn b_mean(&self) -> f32 {
        self.mean[self.mean.len() - 1]
    }
}

/// Sample every anchor node's posterior against the frozen side, in parallel.
///
/// `nodes[i]` is anchor `i`'s likelihood terms; `inits[i]` is its warm-start
/// `θ` (length `h + 1`, e.g. the SGD MAP). Returns one [`NodePosterior`] per node,
/// aligned with `nodes`.
#[must_use]
pub fn sweep_side(
    nodes: &[NodeTerm],
    inits: &[Vec<f32>],
    side: &FrozenSide,
    cfg: &SweepConfig,
) -> Vec<NodePosterior> {
    let h = side.h;
    assert_eq!(nodes.len(), inits.len(), "nodes / inits length mismatch");
    nodes
        .par_iter()
        .zip(inits.par_iter())
        .enumerate()
        .map(|(i, (node, init))| {
            debug_assert_eq!(init.len(), h + 1, "init θ must be length h+1");
            let lnpdf = |theta: &DVector<f32>| poisson_lnpdf(theta, node, side);
            let (sigma_e, sigma_b) = (cfg.sigma_e, cfg.sigma_b);
            let prior_draw = move |rng: &mut SmallRng| -> DVector<f32> {
                DVector::from_fn(h + 1, |k, _| {
                    let z: f64 = StandardNormal.sample(rng);
                    z as f32 * if k < h { sigma_e } else { sigma_b }
                })
            };
            let init_v = DVector::from_vec(init.clone());
            let sampler = EssSampler {
                n_samples: cfg.n_samples,
                warmup: cfg.warmup,
                thin: cfg.thin,
                seed: cfg.seed.wrapping_add(i as u64),
            };
            let chain = sampler.run(&lnpdf, &prior_draw, &init_v);
            let mean = chain.posterior_mean();
            let samples = if cfg.keep_samples {
                chain.samples.iter().map(|s| s.as_slice().to_vec()).collect()
            } else {
                Vec::new()
            };
            NodePosterior { mean, samples }
        })
        .collect()
}
