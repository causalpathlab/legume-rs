//! Per-gene SuSiE **gate** posterior against the frozen cell side — the exact
//! version of the variational softmax feature gate.
//!
//! The live model selects features with a per-gene softmax gate
//! (`crate::model::SoftmaxGateSpec`): a categorical selection over the `H`
//! embedding dims times a Gaussian effect, i.e. a **SuSiE single-effect**. Its
//! *variational* readout is `JointEmbedModel::feature_selection()`. `mcmc-util`
//! already ships the **exact** sampler for that model —
//! `sparse_regression::{mcmc_sparse, SoftmaxNormalPrior}` — so per gene we run it
//! against the gene's own Poisson likelihood (its edges vs the frozen cell side,
//! bias fixed at the MAP) and read the *posterior* selection: per-dim **PIP**, a
//! **credible set** over the loaded dims, and the **effect posterior** — replacing
//! the point-estimate gate with a calibrated one.
//!
//! `rayon par_iter` over genes; each gene is one `mcmc_sparse` call. The bias
//! `b_g` is held at the MAP (the gate selects which *dim* a gene loads, not the
//! library-size intercept), so `θ_g` is the `H`-dim gated loading only.
//!
//! First cut uses `SoftmaxNormalPrior` as-is: softmax over the `H` real dims, no
//! explicit "load-nothing" null column. A junk gene still reads as null through a
//! near-zero effect + a wide credible set; the null-biased absorber
//! (`GATE_NULL_PRIOR`, mirroring `enable_softmax_gate`) is a later refinement.

use super::lnpdf::{poisson_ll, FrozenSide, NodeTerm};
use mcmc_util::engine::McmcConfig;
use mcmc_util::sparse_regression::{compute_posterior_std_beta, mcmc_sparse, SoftmaxNormalPrior};
use nalgebra::DVector;
use rayon::prelude::*;

/// Chain + prior configuration for the per-gene gate posterior.
pub struct GateConfig {
    pub n_samples: usize,
    pub warmup: usize,
    pub thin: usize,
    /// Base seed; gene `g` runs at `seed + g`.
    pub seed: u64,
    /// Prior variance on the selection logits (higher ⇒ sharper selection).
    pub logit_var: f32,
    /// Slab variance `σ₀²` on the effect (the model's `GATE_EFFECT_PRIOR_VAR`).
    pub effect_var: f32,
    /// Number of SuSiE components `L` (single-effect gate ⇒ `1`).
    pub num_components: usize,
    /// Credible-set coverage target (e.g. `0.95`).
    pub coverage: f32,
}

impl GateConfig {
    /// Defaults matching the model's gate: single-effect, `σ₀² = 1.0`
    /// (`GATE_EFFECT_PRIOR_VAR`), 95% credible sets.
    #[must_use]
    pub fn new(n_samples: usize, warmup: usize, seed: u64) -> Self {
        Self {
            n_samples,
            warmup,
            thin: 1,
            seed,
            logit_var: 1.0,
            effect_var: crate::model::GATE_EFFECT_PRIOR_VAR as f32,
            num_components: 1,
            coverage: 0.95,
        }
    }
}

/// One gene's exact gate posterior.
pub struct GenePosterior {
    /// Per-dim posterior inclusion probability `[H]` — which embedding dim the
    /// gene loads. The exact analogue of the variational `feature_selection()` row.
    pub pip: Vec<f32>,
    /// Posterior-mean loading `E[Σ_l α_l·β_l]` `[H]`.
    pub mean_beta: Vec<f32>,
    /// Posterior SD of the loading `[H]`.
    pub std_beta: Vec<f32>,
    /// Dims (descending PIP) that capture `coverage` of the selection mass.
    pub credible_set: Vec<usize>,
    /// Achieved coverage of `credible_set` (`≥ coverage`).
    pub cs_coverage: f32,
}

/// Sample every gene's gate posterior against the frozen cell side, in parallel.
///
/// `nodes[g]` is gene `g`'s observed edges + partition (cells); `biases[g]` is its
/// fixed MAP bias `b_g`; `side` is the frozen cell embedding. Returns one
/// [`GenePosterior`] per gene, aligned with `nodes`.
#[must_use]
pub fn gate_posterior(
    nodes: &[NodeTerm],
    biases: &[f32],
    side: &FrozenSide,
    cfg: &GateConfig,
) -> Vec<GenePosterior> {
    let h = side.h;
    assert_eq!(nodes.len(), biases.len(), "nodes / biases length mismatch");
    let prior = SoftmaxNormalPrior::new(cfg.logit_var, cfg.effect_var);
    nodes
        .par_iter()
        .zip(biases.par_iter())
        .enumerate()
        .map(|(g, (node, &b_g))| {
            let b_g = f64::from(b_g);
            let lnpdf = |theta: &DVector<f32>| poisson_ll(theta.as_slice(), b_g, node, side);
            let config = McmcConfig {
                n_samples: cfg.n_samples,
                warmup: cfg.warmup,
                thin: cfg.thin,
                seed: cfg.seed.wrapping_add(g as u64),
            };
            let res = mcmc_sparse(&lnpdf, h, &prior, &config, cfg.num_components);
            let std_beta = compute_posterior_std_beta(&res.samples, h);
            let (credible_set, cs_coverage) = res
                .credible_sets(cfg.coverage)
                .first()
                .map_or((Vec::new(), 0.0), |c| (c.indices.clone(), c.coverage));
            GenePosterior {
                pip: res.pip,
                mean_beta: res.posterior_mean_beta,
                std_beta,
                credible_set,
                cs_coverage,
            }
        })
        .collect()
}
