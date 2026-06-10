//! Mean-variance trend fit for Negative-Binomial count data.
//!
//! Fits a smooth trend `σ²(μ) = μ + φ(μ) · μ²` across features (genes) and
//! exposes two derived quantities from the same fit:
//!
//! - [`DispersionTrend::excess`] — HVG score: how far a feature's empirical
//!   `φ̂_g` sits above the global trend at its mean. Replaces Scanpy/Seurat
//!   binned residual variance with a continuous, self-reference-free score.
//! - [`DispersionTrend::fisher_weight`] — DC-Poisson gene weight:
//!   `w_g = 1 / (1 + π_g · s̄ · φ(μ_g))`. Bounded in `(0, 1]`, monotone
//!   decreasing in baseline abundance, recovers `w_g = 1` when `φ → 0`.
//!
//! ## Trend fit
//!
//! MVP: weighted log-linear regression `log φ = a + b · log μ` over features
//! with positive per-gene MoM dispersion `φ̂_g = (σ²_g − μ_g) / μ²_g`.
//! Weights are `μ_g` (down-weights noisy low-mean features whose `φ̂`
//! estimates are dominated by variance in the variance). For features with
//! `φ̂_g ≤ 0` (underdispersed — Poisson or sub-Poisson), we still evaluate
//! the fitted trend at their mean; they just don't contribute to the fit.
//!
//! Accuracy is sufficient for HVG scoring and DC-Poisson weighting. A
//! smoother (LOESS / Gaussian-kernel) fit is a follow-up if the log-linear
//! shape proves inadequate.

use matrix_util::sparse_stat::SparseRunningStatistics;
use matrix_util::traits::RunningStatOps;

/// Minimum absolute mean to include a feature in the trend fit. Below this
/// threshold, `μ²` is too small to yield a usable MoM `φ̂` and we would be
/// regressing mostly on noise.
const MIN_MEAN_FOR_FIT: f32 = 1e-4;

/// Floor for `φ(μ)` evaluation. Prevents negative dispersion from the
/// fitted line when extrapolating to regions not represented in the fit.
const PHI_FLOOR: f32 = 0.0;

/// Ceiling for `φ(μ)` evaluation. Keeps `fisher_weight` from collapsing to
/// zero (and numerical underflow downstream) when `μ` is very large and the
/// fitted `b` slope is positive.
const PHI_CEIL: f32 = 100.0;

/// Smooth trend of Negative-Binomial dispersion against feature mean.
///
/// Evaluates as `φ(μ) = exp(a + b · log μ)` clamped to `[PHI_FLOOR, PHI_CEIL]`.
#[derive(Clone, Debug)]
pub struct DispersionTrend {
    a: f32,
    b: f32,
    /// Number of features that contributed to the fit (i.e. had
    /// `μ_g > MIN_MEAN_FOR_FIT` and `φ̂_g > 0`). Exposed for diagnostics.
    pub num_fit: usize,
}

impl DispersionTrend {
    /// Fit the trend from per-feature mean and variance vectors (parallel,
    /// equal-length).
    pub fn fit(means: &[f32], vars: &[f32]) -> Self {
        assert_eq!(means.len(), vars.len(), "means and vars length mismatch");

        // Collect fit points: (log_mu, log_phi_hat, weight = mu).
        let mut x: Vec<f64> = Vec::with_capacity(means.len());
        let mut y: Vec<f64> = Vec::with_capacity(means.len());
        let mut w: Vec<f64> = Vec::with_capacity(means.len());
        for (&mu, &var) in means.iter().zip(vars.iter()) {
            if !mu.is_finite() || !var.is_finite() || mu < MIN_MEAN_FOR_FIT {
                continue;
            }
            let phi_hat = ((var - mu) / (mu * mu)) as f64;
            if phi_hat <= 0.0 {
                continue;
            }
            x.push((mu as f64).ln());
            y.push(phi_hat.ln());
            w.push(mu as f64);
        }
        let num_fit = x.len();

        // Degenerate fallbacks: not enough points, or zero variance in x.
        // Fall back to the Poisson limit (φ → 0), implemented as a = -∞
        // so `exp(a + b·log μ) = 0` for any finite `μ`.
        if num_fit < 2 {
            return Self {
                a: f32::NEG_INFINITY,
                b: 0.0,
                num_fit,
            };
        }
        let w_sum: f64 = w.iter().sum();
        let x_mean: f64 = x.iter().zip(&w).map(|(xi, wi)| xi * wi).sum::<f64>() / w_sum;
        let y_mean: f64 = y.iter().zip(&w).map(|(yi, wi)| yi * wi).sum::<f64>() / w_sum;
        let mut sxx = 0.0f64;
        let mut sxy = 0.0f64;
        for ((xi, yi), wi) in x.iter().zip(&y).zip(&w) {
            let dx = xi - x_mean;
            sxx += wi * dx * dx;
            sxy += wi * dx * (yi - y_mean);
        }
        if sxx <= 0.0 {
            // All log-means identical — trend collapses to a constant.
            return Self {
                a: y_mean as f32,
                b: 0.0,
                num_fit,
            };
        }
        let b = sxy / sxx;
        let a = y_mean - b * x_mean;
        Self {
            a: a as f32,
            b: b as f32,
            num_fit,
        }
    }

    /// Convenience: fit directly from a feature-row `SparseRunningStatistics`.
    pub fn from_sparse_stats(stats: &SparseRunningStatistics<f32>) -> Self {
        let means = stats.mean();
        let vars = stats.variance();
        Self::fit(&means, &vars)
    }

    /// Smooth φ at the given mean. Clamped to `[PHI_FLOOR, PHI_CEIL]`.
    pub fn phi_at(&self, mu: f32) -> f32 {
        if !mu.is_finite() || mu <= 0.0 {
            return PHI_FLOOR;
        }
        let log_mu = mu.ln();
        let log_phi = self.a + self.b * log_mu;
        log_phi.exp().clamp(PHI_FLOOR, PHI_CEIL)
    }

    /// HVG score: `(σ² − μ)/μ² − φ(μ)`. Positive = more biological variance
    /// than the NB trend predicts; rank features by descending value.
    pub fn excess(&self, mu: f32, var: f32) -> f32 {
        if !mu.is_finite() || mu <= 0.0 || !var.is_finite() {
            return f32::NEG_INFINITY;
        }
        let phi_hat = (var - mu) / (mu * mu);
        phi_hat - self.phi_at(mu)
    }

    /// DC-Poisson gene weight: `1 / (1 + π · s̄ · φ(μ))`.
    ///
    /// `pi` is the feature's empirical marginal probability, `avg_s` the
    /// mean entity size factor. Bounded in `(0, 1]`.
    pub fn fisher_weight(&self, pi: f32, avg_s: f32, mu: f32) -> f32 {
        let phi = self.phi_at(mu);
        1.0 / (1.0 + pi * avg_s * phi)
    }

    /// Evaluate [`fisher_weight`] over feature-parallel vectors in one pass.
    pub fn fisher_weights(&self, pi: &[f32], means: &[f32], avg_s: f32) -> Vec<f32> {
        assert_eq!(pi.len(), means.len());
        pi.iter()
            .zip(means.iter())
            .map(|(&p, &m)| self.fisher_weight(p, avg_s, m))
            .collect()
    }
}

#[cfg(test)]
#[path = "nb_dispersion_tests.rs"]
mod tests;
