//! Posterior summaries shared by both reference modes.
//!
//! The sampler allocates counts against `R` reference *components* and reports
//! against `C` *cell types*. In the low-rank mode the two coincide (one anchor
//! per type); in the archetype mode `R ≫ C` and the mapping is the soft
//! annotation readout. Everything here is already at `C`.

use crate::embed_common::Mat;

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
    pub residual: Vec<ResidualStat>,
    pub celltype_names: Vec<Box<str>>,
    /// Per-(sample, celltype) split-R̂ and effective sample size, in the draw
    /// order of `fractions_mean`. Empty when too few draws were collected.
    pub convergence: Vec<Convergence>,
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
/// posterior-mean rate `Σ_c abundance_{s,c}·mu_{g,c}`.
pub fn residual_stat(bulk: &Mat, mu: &Mat, abundance: &Mat, si: usize) -> ResidualStat {
    let d = bulk.nrows();
    let c = mu.ncols();
    let mut total = 0f64;
    let mut deviance = 0f64;
    let (mut sy, mut sl, mut syy, mut sll, mut syl) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for g in 0..d {
        let y = f64::from(bulk[(g, si)]).max(0.0);
        let mut lam = 0f64;
        for ct in 0..c {
            lam += f64::from(abundance[(si, ct)]) * f64::from(mu[(g, ct)]);
        }
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
const MIN_DRAWS_FOR_DIAGNOSTICS: usize = 8;

/// Split-R̂ and effective sample size for one scalar chain.
///
/// Split-R̂ halves the chain and compares within- to between-half variance, so a
/// drifting chain is caught even with a single run. It cannot detect
/// multimodality — for that, run independent seeds and compare.
#[must_use]
pub fn split_rhat_ess(draws: &[f32]) -> Option<(f32, f32)> {
    let n = draws.len();
    if n < MIN_DRAWS_FOR_DIAGNOSTICS {
        return None;
    }
    let half = n / 2;
    let a = &draws[..half];
    let b = &draws[n - half..];
    let mean = |x: &[f32]| x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64;
    let var = |x: &[f32], m: f64| {
        x.iter()
            .map(|&v| (f64::from(v) - m).powi(2))
            .sum::<f64>()
            .max(0.0)
            / (x.len() as f64 - 1.0).max(1.0)
    };
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (var(a, ma), var(b, mb));
    let m = half as f64;
    let w = 0.5 * (va + vb);
    // Between-half variance of two chains of length `half`.
    let grand = 0.5 * (ma + mb);
    let bvar = m * ((ma - grand).powi(2) + (mb - grand).powi(2));
    let rhat = if w > 0.0 {
        let var_hat = (m - 1.0) / m * w + bvar / m;
        (var_hat / w).max(0.0).sqrt() as f32
    } else {
        1.0
    };

    // ESS from the initial-positive-sequence estimator on the pooled chain.
    let gm = mean(draws);
    let gv = var(draws, gm);
    if gv <= 0.0 {
        return Some((rhat, n as f32));
    }
    let mut rho_sum = 0f64;
    let max_lag = (n / 2).min(256);
    let mut lag = 1;
    while lag < max_lag {
        let acf = |k: usize| {
            let mut acc = 0f64;
            for t in 0..(n - k) {
                acc += (f64::from(draws[t]) - gm) * (f64::from(draws[t + k]) - gm);
            }
            acc / ((n - k) as f64 * gv)
        };
        // Sum consecutive pairs; stop when a pair turns negative (Geyer).
        let pair = acf(lag) + acf(lag + 1);
        if pair <= 0.0 {
            break;
        }
        rho_sum += pair;
        lag += 2;
    }
    let ess = (n as f64 / (1.0 + 2.0 * rho_sum)).clamp(1.0, n as f64) as f32;
    Some((rhat, ess))
}
