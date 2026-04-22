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
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::RngExt;
    use rand::SeedableRng;

    /// Synthesize per-gene (mean, variance) observations under NB with a
    /// known dispersion `phi_true`. Variance is set exactly to
    /// `mu + phi_true * mu^2` with a small multiplicative jitter to mimic
    /// empirical variance noise.
    fn synth_nb_moments(
        n_genes: usize,
        phi_true: f32,
        mu_range: (f32, f32),
        jitter: f32,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut means = Vec::with_capacity(n_genes);
        let mut vars = Vec::with_capacity(n_genes);
        let log_lo = mu_range.0.ln();
        let log_hi = mu_range.1.ln();
        for _ in 0..n_genes {
            let log_mu = rng.random_range(log_lo..log_hi);
            let mu = log_mu.exp();
            let var = (mu + phi_true * mu * mu) * (1.0 + rng.random_range(-jitter..jitter));
            means.push(mu);
            vars.push(var.max(1e-6));
        }
        (means, vars)
    }

    #[test]
    fn test_trend_recovers_constant_phi() {
        // When `phi(mu)` is actually constant across genes, the fitted slope
        // `b` should be close to zero and `exp(a)` should be close to the
        // true `phi`.
        let (means, vars) = synth_nb_moments(500, 0.1, (0.1, 50.0), 0.02, 1);
        let trend = DispersionTrend::fit(&means, &vars);
        assert!(trend.num_fit > 100, "expected most genes to contribute");
        assert!(
            trend.b.abs() < 0.2,
            "slope should be near zero, got {}",
            trend.b
        );
        let phi_center = trend.phi_at(5.0);
        assert!(
            (phi_center - 0.1).abs() < 0.05,
            "phi at center should be ≈ 0.1, got {}",
            phi_center
        );
    }

    #[test]
    fn test_fisher_weight_bounds_and_poisson_limit() {
        let (means, vars) = synth_nb_moments(200, 0.2, (1.0, 100.0), 0.01, 2);
        let trend = DispersionTrend::fit(&means, &vars);

        // Bounded in (0, 1].
        for &pi in &[1e-5f32, 1e-3, 1e-1, 0.5] {
            for &mu in &[0.5f32, 5.0, 50.0] {
                let w = trend.fisher_weight(pi, 1000.0, mu);
                assert!(w > 0.0 && w <= 1.0, "weight out of (0, 1]: {}", w);
            }
        }

        // Monotone decreasing in pi.
        let mu = 10.0f32;
        let w_low = trend.fisher_weight(1e-4, 1000.0, mu);
        let w_high = trend.fisher_weight(1e-1, 1000.0, mu);
        assert!(
            w_low > w_high,
            "weight should decrease with pi: low={}, high={}",
            w_low,
            w_high
        );

        // Poisson limit: phi = 0 → w = 1.
        let poisson_trend = DispersionTrend {
            a: f32::NEG_INFINITY, // exp(-inf) = 0
            b: 0.0,
            num_fit: 0,
        };
        let w = poisson_trend.fisher_weight(0.01, 1000.0, 10.0);
        assert!(
            (w - 1.0).abs() < 1e-6,
            "Poisson limit w should be 1, got {}",
            w
        );
    }

    #[test]
    fn test_excess_ranks_outliers_on_top() {
        // Build a dataset with a trend phi_true = 0.1 and inject a handful
        // of outlier genes with phi = 1.0 at the same means. The excess
        // score should rank those outliers above the typical genes.
        let (mut means, mut vars) = synth_nb_moments(300, 0.1, (1.0, 20.0), 0.02, 3);
        let mut outlier_indices = Vec::new();
        for _ in 0..5 {
            let mu = 5.0f32;
            let var = mu + 1.0 * mu * mu; // phi = 1.0
            outlier_indices.push(means.len());
            means.push(mu);
            vars.push(var);
        }
        let trend = DispersionTrend::fit(&means, &vars);
        let mut scored: Vec<(usize, f32)> = means
            .iter()
            .zip(vars.iter())
            .enumerate()
            .map(|(i, (&m, &v))| (i, trend.excess(m, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<usize> = scored.iter().take(5).map(|(i, _)| *i).collect();
        for &outlier in &outlier_indices {
            assert!(
                top5.contains(&outlier),
                "outlier {} not in top-5 HVG set {:?}",
                outlier,
                top5
            );
        }
    }

    #[test]
    fn test_degenerate_input_does_not_panic() {
        // Empty input.
        let trend = DispersionTrend::fit(&[], &[]);
        assert_eq!(trend.num_fit, 0);
        assert_eq!(trend.phi_at(1.0), PHI_FLOOR);

        // All underdispersed.
        let means = vec![1.0f32, 2.0, 3.0];
        let vars = vec![0.5f32, 1.0, 1.5]; // all var < mean
        let trend = DispersionTrend::fit(&means, &vars);
        assert_eq!(trend.num_fit, 0);
        assert_eq!(trend.phi_at(2.0), PHI_FLOOR);

        // All zero means — skipped by MIN_MEAN_FOR_FIT.
        let trend = DispersionTrend::fit(&[0.0f32; 10], &[0.0f32; 10]);
        assert_eq!(trend.num_fit, 0);

        // Single point — degenerate regression.
        let trend = DispersionTrend::fit(&[5.0], &[30.0]);
        assert!(trend.num_fit <= 1);
    }
}
