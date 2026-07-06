//! Binomial / quasi-binomial spline GAM for within-branch trajectory association.
//!
//! Given per-observation edited counts `k_i` out of coverage `n_i` and a continuous
//! predictor `x_i` (pseudotime along one branch), fit
//!
//! ```text
//!     logit(p_i) = f(x_i),      k_i ~ Binomial(n_i, p_i)
//! ```
//!
//! where `f` is a restricted cubic (natural) spline, and test the smooth against an
//! intercept-only null — the tradeSeq `associationTest` question: *does the rate
//! change along this branch?* Coverage `n_i` is the binomial denominator, so detection
//! depth is conditioned out (the same logic the between-branch contrast uses).
//!
//! Editing proportions are overdispersed relative to a plain Binomial (which is why
//! the single-site caller uses a beta-binomial). The default test is therefore
//! **quasi-binomial**: a dispersion `φ` is estimated from the Pearson statistic and an
//! F-test is used; set `overdispersion = false` for the plain Binomial deviance LRT
//! (χ²). `φ` is floored at 1 so the quasi-binomial can only widen the null, never
//! sharpen it below the Binomial.
//!
//! The spline is unpenalized (no GCV/REML smoothing), so a branch whose editing is
//! near-separable by pseudotime can be fit almost perfectly and read as significant;
//! the quasi-binomial dispersion and the caller's coverage/cell QC temper this, and
//! the reported `effect` is clamped to the fitted-probability scale.
//!
//! The shared [`SplineDesign`] is `f32` — the default Bayesian ESS path consumes it
//! natively (no per-branch copy). This frequentist fit is precision-sensitive near the
//! significance threshold, so [`association_test`] upconverts to `f64` for the IRLS
//! solve and the p-value tail (a transient per-branch cost, not the memory-heavy path).

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

/// Knobs for [`association_test`].
#[derive(Clone, Copy, Debug)]
pub struct GamArgs {
    /// Requested spline knots (≥ 3 gives at least one nonlinear term). Reduced
    /// automatically when the branch has few cells or few distinct pseudotimes.
    pub n_knots: usize,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Relative deviance tolerance for IRLS convergence.
    pub tol: f64,
    /// Quasi-binomial dispersion + F-test (true) vs plain Binomial deviance LRT (false).
    pub overdispersion: bool,
}

impl Default for GamArgs {
    fn default() -> Self {
        Self {
            n_knots: 5,
            max_iter: 50,
            tol: 1e-8,
            overdispersion: true,
        }
    }
}

/// Result of one within-branch association test.
#[derive(Clone, Debug)]
pub struct GamFit {
    pub n_obs: usize,
    /// Estimated dispersion `φ` (1.0 when `overdispersion == false`).
    pub dispersion: f32,
    /// Test statistic: deviance-difference χ² (Binomial) or F ratio (quasi-binomial).
    pub stat: f32,
    pub p_value: f32,
    /// Net change in fitted log-odds from the branch start to its end (signed).
    pub effect: f32,
}

/// Standardized within-branch spline design, shared by the frequentist and Bayesian
/// association tests. Column 0 is the intercept; the remaining columns are the
/// mean-centered, unit-scaled spline terms. `contrast · β` gives the net change in
/// fitted log-odds from the branch start to its end — invariant to the column
/// standardization, so both tests report the same effect scale.
pub struct SplineDesign {
    /// `m × p` design (intercept + spline columns), covered observations only.
    pub x: DMatrix<f32>,
    /// Edited counts for the covered observations.
    pub k: Vec<f32>,
    /// Coverage (edited + unedited) for the covered observations.
    pub n: Vec<f32>,
    /// `p`-vector `row(argmax pseudotime) − row(argmin pseudotime)`.
    pub contrast: DVector<f32>,
}

/// Build the standardized spline design for one branch's covered cells.
///
/// Drops uncovered observations, standardizes pseudotime to `[0, 1]`, places
/// `n_knots` knots at its quantiles (auto-reduced for small/degenerate branches),
/// builds a restricted-cubic-spline (or straight-line) design, and centers + scales
/// the spline columns. Returns `None` when the branch cannot support a spline: fewer
/// than 4 covered observations, no pseudotime spread, or too few observations for the
/// spline parameters.
pub fn build_spline_design(
    k: &[u32],
    n: &[u32],
    x: &[f32],
    n_knots: usize,
) -> Option<SplineDesign> {
    let n_obs = k.len();
    if n_obs != n.len() || n_obs != x.len() || n_obs < 4 {
        return None;
    }

    // Drop uncovered observations; standardize x to [0, 1] for conditioning.
    let mut xs = Vec::with_capacity(n_obs);
    let mut kk = Vec::with_capacity(n_obs);
    let mut nn = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        if n[i] > 0 {
            xs.push(x[i]);
            kk.push(k[i] as f32);
            nn.push(n[i] as f32);
        }
    }
    let m = xs.len();
    if m < 4 {
        return None;
    }
    let (lo, hi) = xs
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let span = hi - lo;
    if span <= 0.0 {
        return None; // no pseudotime variation → nothing to associate
    }
    for v in xs.iter_mut() {
        *v = (*v - lo) / span;
    }

    // Effective knots: cap by requested, distinct-x, and observations (need m > p + 1).
    let mut distinct = xs.clone();
    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    let n_distinct = distinct.len();
    let k_eff = n_knots.min(n_distinct).min((m - 2).max(2)).max(2);

    // Design matrix (intercept in column 0). ≥3 knots → restricted cubic spline;
    // exactly 2 → a straight logit line.
    let knots = quantile_knots(&distinct, k_eff);
    let mut xd = if knots.len() >= 3 {
        restricted_cubic_design(&xs, &knots)
    } else {
        linear_design(&xs)
    };
    let p = xd.ncols();
    if m <= p + 1 {
        return None;
    }

    // Center + scale the spline columns (intercept left alone) so a shared prior /
    // ridge is scale-appropriate; the column space — hence the GLM fit — is unchanged.
    // Moments accumulate in f64 to keep the f32 design well-conditioned.
    for j in 1..p {
        let col = xd.column(j);
        let mean = col.iter().map(|&v| v as f64).sum::<f64>() / m as f64;
        let var = col.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / m as f64;
        let sd = if var.sqrt() < 1e-9 { 1.0 } else { var.sqrt() };
        for i in 0..m {
            xd[(i, j)] = ((xd[(i, j)] as f64 - mean) / sd) as f32;
        }
    }

    // Start→end contrast on the standardized rows (intercept term cancels to 0).
    let imin = (0..m)
        .min_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap())
        .unwrap();
    let imax = (0..m)
        .max_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap())
        .unwrap();
    let contrast = DVector::from_fn(p, |j, _| xd[(imax, j)] - xd[(imin, j)]);

    Some(SplineDesign {
        x: xd,
        k: kk,
        n: nn,
        contrast,
    })
}

/// Fit `logit(k_i / n_i) = spline(x_i)` and test the smooth against a constant rate.
///
/// Returns `None` when the branch cannot support a test: fewer than 4 covered
/// observations, no coverage, no pseudotime spread, too few residual degrees of
/// freedom, or an ill-conditioned IRLS solve.
pub fn association_test(k: &[u32], n: &[u32], x: &[f32], args: &GamArgs) -> Option<GamFit> {
    let d = build_spline_design(k, n, x, args.n_knots)?;
    let (m, p_full) = (d.x.nrows(), d.x.ncols());

    // The shared design is f32 (the default Bayesian path is native f32); the
    // frequentist LRT p-value is precision-sensitive near the threshold, so this fit
    // upconverts to f64 (a transient per-branch copy, not the memory-heavy path).
    let x64 = d.x.map(|v| v as f64);
    let k64: Vec<f64> = d.k.iter().map(|&v| v as f64).collect();
    let n64: Vec<f64> = d.n.iter().map(|&v| v as f64).collect();
    let full = irls_binomial(&x64, &k64, &n64, args)?;
    let (sum_k, sum_n): (f64, f64) = (k64.iter().sum(), n64.iter().sum());
    let dev_null = binomial_deviance_const(&k64, &n64, sum_k / sum_n);

    let dstat = (dev_null - full.deviance).max(0.0);
    let df1 = (p_full - 1) as f64;
    if df1 < 1.0 {
        return None;
    }

    // Net change in fitted log-odds along the branch (clamped: β can diverge under
    // near-separation, which would otherwise blow the reported effect up).
    let effect = d
        .contrast
        .map(|v| v as f64)
        .dot(&full.beta)
        .clamp(-60.0, 60.0) as f32;

    let (stat, p_value, dispersion) = if args.overdispersion {
        let df2 = (m - p_full) as f64;
        let phi = (full.pearson / df2).max(1.0);
        let f = (dstat / df1) / phi;
        let p = FisherSnedecor::new(df1, df2)
            .ok()
            .map_or(1.0, |dist| dist.sf(f).clamp(0.0, 1.0));
        (f as f32, p as f32, phi as f32)
    } else {
        let p = ChiSquared::new(df1)
            .ok()
            .map_or(1.0, |dist| dist.sf(dstat).clamp(0.0, 1.0));
        (dstat as f32, p as f32, 1.0)
    };

    Some(GamFit {
        n_obs: m,
        dispersion,
        stat,
        p_value,
        effect,
    })
}

////////////////////////////////////////////////////////////////////////
// Spline basis
////////////////////////////////////////////////////////////////////////

/// Knots at evenly spaced quantiles (type-7) of the distinct, sorted values.
fn quantile_knots(sorted_distinct: &[f32], k: usize) -> Vec<f32> {
    let n = sorted_distinct.len();
    if n == 0 {
        return Vec::new();
    }
    if k <= 1 || n == 1 {
        return vec![sorted_distinct[0]];
    }
    let quantile = |p: f32| -> f32 {
        let h = (n as f32 - 1.0) * p;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        sorted_distinct[lo] + (h - lo as f32) * (sorted_distinct[hi] - sorted_distinct[lo])
    };
    let mut knots: Vec<f32> = (0..k)
        .map(|i| quantile(i as f32 / (k as f32 - 1.0)))
        .collect();
    knots.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    knots
}

/// One restricted-cubic-spline design row `[1, x, term_0, …]` for scalar `x`.
fn design_row(knots: &[f32], x: f32) -> DVector<f32> {
    if knots.len() < 3 {
        return DVector::from_vec(vec![1.0, x]);
    }
    let k = knots.len();
    let (t1, tk, tkm1) = (knots[0], knots[k - 1], knots[k - 2]);
    let denom = tk - tkm1;
    let scale = (tk - t1).powi(2);
    let cube = |u: f32| {
        let v = u.max(0.0);
        v * v * v
    };
    let mut row = Vec::with_capacity(k);
    row.push(1.0);
    row.push(x);
    for tj in knots.iter().take(k - 2) {
        let term = (cube(x - tj) - cube(x - tkm1) * (tk - tj) / denom
            + cube(x - tk) * (tkm1 - tj) / denom)
            / scale;
        row.push(term);
    }
    DVector::from_vec(row)
}

/// Restricted-cubic-spline design matrix, `m × k` (intercept in column 0).
fn restricted_cubic_design(x: &[f32], knots: &[f32]) -> DMatrix<f32> {
    let (m, k) = (x.len(), knots.len());
    let mut d = DMatrix::<f32>::zeros(m, k);
    for (i, &xi) in x.iter().enumerate() {
        let row = design_row(knots, xi);
        for j in 0..k {
            d[(i, j)] = row[j];
        }
    }
    d
}

/// Straight-line design `[1, x]`, `m × 2`.
fn linear_design(x: &[f32]) -> DMatrix<f32> {
    let m = x.len();
    let mut d = DMatrix::<f32>::zeros(m, 2);
    for (i, &xi) in x.iter().enumerate() {
        d[(i, 0)] = 1.0;
        d[(i, 1)] = xi;
    }
    d
}

////////////////////////////////////////////////////////////////////////
// IRLS
////////////////////////////////////////////////////////////////////////

struct IrlsFit {
    beta: DVector<f64>,
    deviance: f64,
    pearson: f64,
}

/// Fisher-scoring (IRLS) fit of a Binomial GLM with logit link and per-observation
/// trials `n_i`. The normal equations use BLAS matrix products — `Xw = diag(√w)·X`
/// (per-row `scal`), then `XᵀWX = XwᵀXw` and `XᵀWz = Xᵀ(wz)` (`gemm`) — not a scalar
/// `O(m·p²)` loop; a tiny ridge keeps `XᵀWX` invertible under near-separation.
fn irls_binomial(x: &DMatrix<f64>, k: &[f64], n: &[f64], args: &GamArgs) -> Option<IrlsFit> {
    let (m, p) = (x.nrows(), x.ncols());
    let y: Vec<f64> = (0..m).map(|i| k[i] / n[i]).collect();

    // Init on the response scale, nudged off 0/1.
    let mut eta = DVector::<f64>::zeros(m);
    let mut mu = vec![0.0f64; m];
    for i in 0..m {
        let m0 = (k[i] + 0.5) / (n[i] + 1.0);
        mu[i] = m0;
        eta[i] = (m0 / (1.0 - m0)).ln();
    }

    let ridge = 1e-8;
    let mut dev_prev = f64::INFINITY;
    let mut beta = DVector::<f64>::zeros(p);
    // Buffers reused across IRLS iterations (no per-iteration reallocation).
    let mut w = DVector::<f64>::zeros(m);
    let mut wz = DVector::<f64>::zeros(m);
    let mut xw = x.clone();
    for _ in 0..args.max_iter {
        // IRLS working weights wᵢ and weighted response wᵢzᵢ.
        for i in 0..m {
            let v = (mu[i] * (1.0 - mu[i])).max(1e-9);
            w[i] = n[i] * v;
            wz[i] = w[i] * (eta[i] + (y[i] - mu[i]) / v);
        }
        // Xw = diag(√w)·X via per-row scaling (BLAS scal); XᵀWX = XwᵀXw (BLAS gemm),
        // the symmetric Gram form, with a ridge on the diagonal.
        xw.copy_from(x);
        for (i, mut row) in xw.row_iter_mut().enumerate() {
            row.scale_mut(w[i].sqrt());
        }
        let mut xtwx = xw.tr_mul(&xw);
        for a in 0..p {
            xtwx[(a, a)] += ridge;
        }
        let xtwz = x.tr_mul(&wz);
        beta = xtwx.cholesky().map(|c| c.solve(&xtwz))?;

        // Update η, μ (clamped); stop on relative deviance change.
        eta = x * &beta;
        for i in 0..m {
            let e = eta[i].clamp(-30.0, 30.0);
            eta[i] = e;
            mu[i] = 1.0 / (1.0 + (-e).exp());
        }
        let dev = binomial_deviance(k, n, &mu);
        if (dev_prev - dev).abs() < args.tol * (dev.abs() + 0.1) {
            break;
        }
        dev_prev = dev;
    }
    Some(finish(beta, k, n, &mu))
}

fn finish(beta: DVector<f64>, k: &[f64], n: &[f64], mu: &[f64]) -> IrlsFit {
    IrlsFit {
        deviance: binomial_deviance(k, n, mu),
        pearson: pearson_chi2(k, n, mu),
        beta,
    }
}

////////////////////////////////////////////////////////////////////////
// Goodness-of-fit
////////////////////////////////////////////////////////////////////////

/// `a · ln(a / b)` with the `0 · ln 0 = 0` convention (f64).
fn xlogy_ratio(a: f64, b: f64) -> f64 {
    if a <= 0.0 {
        0.0
    } else {
        a * (a / b.max(1e-300)).ln()
    }
}

/// Binomial deviance `2 Σ nᵢ [ yᵢ ln(yᵢ/μᵢ) + (1−yᵢ) ln((1−yᵢ)/(1−μᵢ)) ]`.
fn binomial_deviance(k: &[f64], n: &[f64], mu: &[f64]) -> f64 {
    let mut dev = 0.0;
    for i in 0..k.len() {
        let (ni, y, m) = (n[i], k[i] / n[i], mu[i]);
        dev += 2.0 * ni * (xlogy_ratio(y, m) + xlogy_ratio(1.0 - y, 1.0 - m));
    }
    dev
}

/// Deviance of the intercept-only model (all `μᵢ = μ̄`).
fn binomial_deviance_const(k: &[f64], n: &[f64], mu_bar: f64) -> f64 {
    let mut dev = 0.0;
    for i in 0..k.len() {
        let (ni, y) = (n[i], k[i] / n[i]);
        dev += 2.0 * ni * (xlogy_ratio(y, mu_bar) + xlogy_ratio(1.0 - y, 1.0 - mu_bar));
    }
    dev
}

/// Pearson statistic `Σ nᵢ (yᵢ − μᵢ)² / (μᵢ(1−μᵢ))`.
fn pearson_chi2(k: &[f64], n: &[f64], mu: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..k.len() {
        let (ni, y, m) = (n[i], k[i] / n[i], mu[i]);
        let v = (m * (1.0 - m)).max(1e-9);
        s += ni * (y - m).powi(2) / v;
    }
    s
}

#[cfg(test)]
mod tests;
