//! Posterior summaries shared by both reference modes.
//!
//! The sampler allocates counts against `R` reference *components* and reports
//! against `C` *cell types*. In the low-rank mode the two coincide (one anchor
//! per type); in the archetype mode `R ≫ C` and the mapping is the soft
//! annotation readout. Everything here is already at `C`.

use super::reference::{ComponentAxis, FractionUnits};
use crate::embed_common::Mat;
use mcmc_util::engine::{ess, split_rhat};

/// Posterior-predictive fit of one bulk sample at the posterior-mean reference.
pub struct ResidualStat {
    pub total: f64,
    pub deviance: f64,
    pub pearson: f64,
}

pub struct DeconvResult {
    pub fractions_mean: Mat,
    pub fractions_sd: Mat,
    pub fractions_lo: Mat,
    pub fractions_hi: Mat,
    pub abundance_mean: Mat,
    /// One `D×S` block per cell type: posterior-mean allocated counts
    /// `E[Z_{s,c,g}]`, the within-type expression tensor.
    pub expression: Vec<Mat>,
    /// Posterior-mean component coordinates in the embedding: anchors `t_c`
    /// (low-rank) or archetype positions `z_m` (archetype mode).
    pub anchors_post: Mat,
    /// Row names for `anchors_post` — cell types or archetype ids.
    pub anchor_names: Vec<Box<str>>,
    /// What those rows are. Carried rather than inferred: an archetype run whose
    /// partition happens to collapse to `R == C` would otherwise be mislabelled.
    pub anchor_axis: ComponentAxis,
    /// What the reported fractions are shares of.
    pub units: FractionUnits,
    pub residual: Vec<ResidualStat>,
    pub celltype_names: Vec<Box<str>>,
    /// Per-(sample, celltype) split-R̂ and effective sample size, in the draw
    /// order of `fractions_mean`. Empty when too few draws were collected.
    pub convergence: Vec<Convergence>,
    /// Chains pooled into the posterior. Above one, R̂ compares chains rather
    /// than halves of one chain, and disagreement means the references differ —
    /// not that a chain failed to settle.
    pub n_chains: usize,
}

/// Convergence diagnostics for one scalar chain.
#[derive(Clone, Copy)]
pub struct Convergence {
    pub sample: usize,
    pub celltype: usize,
    /// Split-R̂: the chain is halved and treated as two chains.
    pub rhat: f32,
    /// Effective sample size from the initial-positive-sequence autocorrelation sum.
    pub ess: f32,
}

/// Poisson deviance and Pearson correlation of sample `si` against the
/// posterior-mean rate.
///
/// `lam_sum` is the running sum of the sampler's own per-gene rate over
/// `n_draws` retained draws, so the comparison is against the rate the model
/// actually used rather than one reconstructed from marginal summaries.
pub fn residual_stat(bulk: &Mat, lam_sum: &[f64], n_draws: f64, si: usize) -> ResidualStat {
    let d = bulk.nrows();
    let mut total = 0f64;
    let mut deviance = 0f64;
    let (mut sy, mut sl, mut syy, mut sll, mut syl) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for g in 0..d {
        let y = f64::from(bulk[(g, si)]).max(0.0);
        let mut lam = lam_sum[g] / n_draws;
        lam = lam.max(1e-12);
        total += y;
        deviance += if y > 0.0 {
            2.0 * (y * (y / lam).ln() - (y - lam))
        } else {
            2.0 * lam
        };
        sy += y;
        sl += lam;
        syy += y * y;
        sll += lam * lam;
        syl += y * lam;
    }
    let n = d as f64;
    let cov = syl - sy * sl / n;
    let vy = (syy - sy * sy / n).max(0.0);
    let vl = (sll - sl * sl / n).max(0.0);
    let pearson = if vy > 0.0 && vl > 0.0 {
        cov / (vy.sqrt() * vl.sqrt())
    } else {
        0.0
    };
    ResidualStat {
        total,
        deviance,
        pearson,
    }
}

////////////////////////////
// Convergence diagnostics //
////////////////////////////

/// Minimum retained draws before split-R̂ / ESS are meaningful.
pub const MIN_DRAWS_FOR_DIAGNOSTICS: usize = 8;

/// Split-R̂ and effective sample size for one scalar chain.
///
/// Split-R̂ segments the chain and compares within- to between-segment variance,
/// so a drifting chain is caught even with a single run. It cannot detect
/// multimodality — for that, run independent seeds and compare.
#[must_use]
pub fn split_rhat_ess(draws: &[f32]) -> Option<(f32, f32)> {
    if draws.len() < MIN_DRAWS_FOR_DIAGNOSTICS {
        return None;
    }
    Some((split_rhat(draws), ess(draws)))
}
