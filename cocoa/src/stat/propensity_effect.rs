//! Stage 2: the exposure effect on individuals.
//!
//! Stage 1 (the pseudobulk baseline) gives each individual a count `y(d,i)`
//! and an expected count at its own cell states `m(d,i)`, so
//! `omega(d,i) = y / m` is the individual's rate multiplier. With exposure
//! invariance,
//!
//! ```text
//!   omega(d,i) = tau(d, x(i)) * delta(d,i),   delta independent of X given V
//! ```
//!
//! and, for exposure levels `x = 0..K` with level 0 as the reference,
//! `psi_d(x) = log tau(d,x) - log tau(d,0)`.
//!
//! Estimation follows Dukes & Vansteelandt (2018, Am J Epidemiol, "A note on
//! G-estimation of causal risk ratios", eq. 6): a Gamma GLM with a log link,
//!
//! ```text
//!   E[omega | V, X] = exp( b0 + b_e' e(V) + psi(X) )
//! ```
//!
//! where `e(V)` are the (unclipped) propensities `P(X = x | V)`. With the
//! Gamma working variance the score for `psi` is the g-estimating equation
//! `sum_i (1{X_i = x} - e_x(V_i)) (omega_i exp(-psi(X_i)) - q(V_i)) / q(V_i) = 0`
//! with `q(V) = exp(b0 + b_e' e(V))` (Robins, Mark & Newey 1992). It is
//! doubly robust: `psi` is consistent if the propensity is right, whatever the
//! outcome looks like, or if the outcome depends on V log-linearly through
//! `e(V)`, even when `e` is misspecified;
//! individuals without overlap (`e` near 0 or 1) drop out on their own, so no
//! clipping is needed. Without a propensity the model holds only the level
//! indicators and `psi` is the log ratio of means. With a log link and Gamma
//! variance the working weights are all one, so each scoring step is ordinary
//! least squares on the working response `eta + omega / mu - 1`, with one
//! projection shared by every gene.

use crate::common::*;
use rayon::prelude::*;

#[cfg(test)]
mod tests;

const MAX_ITER: usize = 100;
const BETA_TOL: f64 = 1e-8;
const PSI_MAX: f32 = 10.0;
/// Largest change of the linear predictor for any individual in one scoring
/// step; larger steps are scaled down as a whole.
const MAX_STEP: f64 = 5.0;
/// Genes per block: the unit of parallel work in the observed-data run.
const GENE_BLOCK: usize = 256;

/// Exposure effect for one topic, over `K` exposure levels
/// with level 0 as the reference.
pub struct ExposureEffect {
    /// psi(x): log tau(x) - log tau(0), gene x level (column 0 is zero)
    pub psi: Mat,
    /// tau(x): average exposure effect, gene x level
    pub tau: Mat,
    /// delta: exposure-free individual effect, H / tau(0), gene x individual
    /// (zero for individuals left out)
    pub delta: Mat,
    /// individuals with a level and cells in this topic
    pub used: Vec<bool>,
    /// most fixed-point iterations any gene block needed
    pub iterations: usize,
}

/// Log effects of one topic: `psi` (gene x level, column 0 zero), which
/// individuals entered, and the most scoring steps any gene block needed.
struct LogEffects {
    psi: Mat,
    used: Vec<bool>,
    iterations: usize,
}

/// Solve the working Gamma GLM for every gene of one topic and assemble
/// `tau` and `delta` around `psi`.
///
/// * `y_di`, `m_di` - individual counts and stage-1 offsets
/// * `x` - exposure level per individual, `0..n_levels`; others are left out
/// * `n_levels` - number of exposure levels `K`
/// * `e` - P(X = x | V_i), individual x level (unclipped); `None` without
///   confounders
/// * `parallel` - run gene blocks in parallel; leave it off when the caller
///   is itself parallel (permutation draws)
pub fn estimate_exposure_effect(
    y_di: &Mat,
    m_di: &Mat,
    x: &[usize],
    n_levels: usize,
    e: Option<&Mat>,
    parallel: bool,
) -> ExposureEffect {
    let LogEffects {
        psi,
        used: use_i,
        iterations,
    } = fit_log_effects(y_di, m_di, x, n_levels, e, parallel);
    let (n_genes, n_indv) = y_di.shape();
    let used: Vec<usize> = (0..n_indv).filter(|&i| use_i[i]).collect();

    // exposure-free rates H = omega exp(-psi(X)), tau(0) their mean
    let n_use = used.len().max(1) as f32;
    let mut tau0 = DVec::zeros(n_genes);
    let mut delta = Mat::zeros(n_genes, n_indv);
    for &i in &used {
        let mut dc = delta.column_mut(i);
        for d in 0..n_genes {
            let m = m_di[(d, i)];
            let l = if m > 0.0 { y_di[(d, i)] / m } else { 0.0 };
            dc[d] = l * (-psi[(d, x[i])]).exp();
        }
        tau0 += &dc;
    }
    tau0 /= n_use;
    for &i in &used {
        let mut dc = delta.column_mut(i);
        for d in 0..n_genes {
            dc[d] = if tau0[d] > 0.0 { dc[d] / tau0[d] } else { 0.0 };
        }
    }
    let tau = Mat::from_fn(n_genes, n_levels, |d, l| tau0[d] * psi[(d, l)].exp());
    ExposureEffect {
        psi,
        tau,
        delta,
        used: use_i,
        iterations,
    }
}

/// `psi` alone (gene x level, column 0 zero), for permutation draws, which
/// need neither `tau` nor `delta`. Arguments as [`estimate_exposure_effect`].
pub fn estimate_log_effect(
    y_di: &Mat,
    m_di: &Mat,
    x: &[usize],
    n_levels: usize,
    e: Option<&Mat>,
    parallel: bool,
) -> Mat {
    fit_log_effects(y_di, m_di, x, n_levels, e, parallel).psi
}

fn fit_log_effects(
    y_di: &Mat,
    m_di: &Mat,
    x: &[usize],
    n_levels: usize,
    e: Option<&Mat>,
    parallel: bool,
) -> LogEffects {
    let (n_genes, n_indv) = y_di.shape();
    // individuals with a level and cells in this topic
    let m_col = m_di.row_sum_tr();
    let use_i: Vec<bool> = (0..n_indv)
        .map(|i| x[i] < n_levels && m_col[i] > 0.0)
        .collect();
    let used: Vec<usize> = (0..n_indv).filter(|&i| use_i[i]).collect();

    // design over the used individuals: [1, e_1.., 1{X = 1}..]
    let mut cols: Vec<DVec> = vec![DVec::from_element(used.len(), 1.0)];
    if let Some(e) = e {
        for l in 1..n_levels {
            cols.push(DVec::from_fn(used.len(), |r, _| e[(used[r], l)]));
        }
    }
    let first_level_col = cols.len();
    for l in 1..n_levels {
        cols.push(DVec::from_fn(used.len(), |r, _| {
            (x[used[r]] == l) as u8 as f32
        }));
    }
    let design = Mat::from_columns(&cols);
    let proj = least_squares_projection(&design);

    let blocks: Vec<(usize, usize)> = (0..n_genes)
        .step_by(GENE_BLOCK)
        .map(|r0| (r0, GENE_BLOCK.min(n_genes - r0)))
        .collect();
    let solve = |&(r0, nr): &(usize, usize)| {
        let omega = Mat::from_fn(nr, used.len(), |d, r| {
            let (yv, mv) = (y_di[(r0 + d, used[r])], m_di[(r0 + d, used[r])]);
            if mv > 0.0 {
                yv / mv
            } else {
                0.0
            }
        });
        fit_gamma_block(&omega, &design, &proj)
    };
    let outs: Vec<(Mat, usize)> = if parallel {
        blocks.par_iter().map(solve).collect()
    } else {
        blocks.iter().map(solve).collect()
    };

    let mut psi = Mat::zeros(n_genes, n_levels);
    let mut iterations = 0;
    for (&(r0, nr), (beta, it)) in blocks.iter().zip(outs) {
        for l in 1..n_levels {
            let col = beta.column(first_level_col + l - 1);
            for d in 0..nr {
                psi[(r0 + d, l)] = col[d].clamp(-PSI_MAX, PSI_MAX);
            }
        }
        iterations = iterations.max(it);
    }
    LogEffects {
        psi,
        used: use_i,
        iterations,
    }
}

type Mat64 = nalgebra::DMatrix<f64>;

/// Relative singular-value cutoff for the design's pseudo-inverse.
const RCOND: f64 = 1e-6;

/// Least-squares projection `X^+` (p x n) by SVD, in f64, dropping
/// directions with singular values below `RCOND` of the largest. With weak
/// confounding the fitted propensity is nearly linear in V, so its column is
/// almost a combination of the intercept and V columns; those directions get
/// the minimum-norm solution instead of exploding, while the exposure
/// columns, which are not collinear with them, keep their estimate.
fn least_squares_projection(x: &Mat) -> Mat64 {
    let svd = x.map(f64::from).svd(true, true);
    let eps = RCOND * svd.singular_values.max();
    svd.pseudo_inverse(eps).expect("svd with u and v_t")
}

/// Fisher scoring for a Gamma GLM with a log link on one block of genes
/// (rows of `omega`), all sharing `design`: each step regresses the working
/// response `eta + omega / mu - 1` on the design, and a gene keeps the step
/// only if its quasi-log-likelihood `sum(-omega / mu - log mu)` does not
/// fall, halving it otherwise (plain scoring can oscillate when the working
/// model is wrong). The linear predictor is carried from step to step, and a
/// halving trial evaluates only the genes still pending. Runs in f64. Genes
/// without any positive rate keep a zero effect. Returns the coefficients
/// (genes x p) and the number of steps.
fn fit_gamma_block(omega: &Mat, design: &Mat, proj: &Mat64) -> (Mat, usize) {
    let omega = omega.map(f64::from);
    let (n_genes, n) = omega.shape();
    let mut beta = Mat64::zeros(n_genes, design.ncols());
    // start at the overall mean rate
    for d in 0..n_genes {
        let mean = omega.row(d).sum() / n.max(1) as f64;
        beta[(d, 0)] = mean.max(1e-12).ln();
    }
    let active: Vec<bool> = (0..n_genes).map(|d| omega.row(d).sum() > 0.0).collect();
    let design_t = design.transpose().map(f64::from);
    let proj_t = proj.transpose();

    // quasi-log-likelihood of gene d at linear predictor eta_d + t * deta_d
    let quasi_ll = |d: usize, eta: &Mat64, deta: &Mat64, t: f64| -> f64 {
        (0..n)
            .map(|i| {
                let ev = (eta[(d, i)] + t * deta[(d, i)]).clamp(-60.0, 60.0);
                -omega[(d, i)] * (-ev).exp() - ev
            })
            .sum()
    };

    let mut eta = &beta * &design_t;
    let zero = Mat64::zeros(n_genes, n);
    let mut ll: Vec<f64> = (0..n_genes)
        .map(|d| quasi_ll(d, &eta, &zero, 0.0))
        .collect();
    let mut z = Mat64::zeros(n_genes, n);
    let mut iterations = 0;
    for it in 0..MAX_ITER {
        iterations = it + 1;
        for (zv, (&ev, &wv)) in z.iter_mut().zip(eta.iter().zip(omega.iter())) {
            let ev = ev.clamp(-60.0, 60.0);
            *zv = ev + wv * (-ev).exp() - 1.0;
        }
        let mut step = &z * &proj_t - &beta;
        let mut deta = &step * &design_t;
        // Cap each gene's step by the change it makes to the linear
        // predictor, keeping its direction. Capping coefficients instead
        // fails when columns are nearly collinear (a propensity almost linear
        // in V): the step then needs large, offsetting coefficients whose net
        // effect on eta is small.
        for (d, &is_active) in active.iter().enumerate() {
            let bad = !is_active || step.row(d).iter().any(|v| !v.is_finite());
            let big = deta.row(d).amax();
            let scale = if bad {
                0.0
            } else if big > MAX_STEP {
                MAX_STEP / big
            } else {
                1.0
            };
            if scale != 1.0 {
                step.row_mut(d).scale_mut(scale);
                deta.row_mut(d).scale_mut(scale);
            }
        }

        // step halving; each gene keeps its first step that does not lower
        // its quasi-log-likelihood
        let mut max_change = 0f64;
        for d in (0..n_genes).filter(|&d| active[d]) {
            let mut t = 1.0f64;
            for _ in 0..30 {
                let cand = quasi_ll(d, &eta, &deta, t);
                if cand >= ll[d] - 1e-12 * ll[d].abs().max(1.0) {
                    ll[d] = cand;
                    let db = step.row(d) * t;
                    let mut b = beta.row_mut(d);
                    b += db;
                    let de = deta.row(d) * t;
                    let mut e = eta.row_mut(d);
                    e += de;
                    max_change = max_change.max(t * deta.row(d).amax());
                    break;
                }
                t *= 0.5;
            }
        }
        // converged when no accepted step moves any linear predictor
        if max_change < BETA_TOL {
            break;
        }
    }
    (beta.map(|v| v as f32), iterations)
}
