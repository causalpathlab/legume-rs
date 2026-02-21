//! Weighted Gaussian likelihood with per-observation variance.

use candle_core::{Result, Tensor};

use crate::sgvb::BlackBoxLikelihood;

/// Fixed-variance Gaussian likelihood with per-observation variance tensor.
///
/// Like `FixedGaussianLikelihood` but accepts a variance tensor `(n, k)` instead of
/// a scalar, enabling uncertainty-aware likelihoods where each observation has its
/// own known variance (e.g., from Gamma-Poisson pseudobulk posterior).
///
/// # Model
/// ```text
/// log p(y_ik | η_ik) = -0.5 * [(y_ik - η_ik)² / σ²_ik + log(2π σ²_ik)]
/// ```
pub struct WeightedGaussianLikelihood {
    /// Observed data, shape (n, k)
    y: Tensor,
    /// 0.5 / variance, shape (1, n, k) — pre-expanded for broadcasting against (S, n, k)
    inv_2var: Tensor,
    /// 0.5 * log(2π · variance), shape (1, n, k)
    half_log_2pi_var: Tensor,
}

impl WeightedGaussianLikelihood {
    /// Create a new weighted Gaussian likelihood.
    ///
    /// # Arguments
    /// * `y` - Observed data, shape `(n, k)`
    /// * `variance` - Per-observation variance, shape `(n, k)` (must be > 0)
    pub fn new(y: Tensor, variance: &Tensor) -> Result<Self> {
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();
        let inv_2var = (variance * 2.0)?.recip()?.unsqueeze(0)?; // (1, n, k)
        let half_log_2pi_var = ((variance.log()? + ln_2pi)? * 0.5)?.unsqueeze(0)?; // (1, n, k)
        Ok(Self {
            y,
            inv_2var,
            half_log_2pi_var,
        })
    }
}

impl BlackBoxLikelihood for WeightedGaussianLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        let eta = etas[0]; // (S, n, k)
        let diff_sq = eta.broadcast_sub(&self.y)?.sqr()?; // (S, n, k)
                                                          // -diff² * inv_2var - half_log_2pi_var
        let scaled = diff_sq.broadcast_mul(&self.inv_2var)?; // (S, n, k)
        let log_prob = scaled.neg()?.broadcast_sub(&self.half_log_2pi_var)?; // (S, n, k)
        log_prob.sum(2)?.sum(1) // (S,)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::FixedGaussianLikelihood;
    use candle_core::Device;

    #[test]
    fn test_weighted_matches_fixed_uniform_variance() -> Result<()> {
        let device = Device::Cpu;

        // y = [0, 1, 2], eta = [0, 1, 2] (perfect fit)
        let y = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], (3, 1), &device)?;
        let eta = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], (1, 3, 1), &device)?;

        // Uniform variance = 1.0
        let variance = Tensor::ones((3, 1), candle_core::DType::F32, &device)?;

        let weighted = WeightedGaussianLikelihood::new(y.clone(), &variance)?;
        let fixed = FixedGaussianLikelihood::new(y, 1.0);

        let ll_weighted = weighted.log_likelihood(&[&eta])?;
        let ll_fixed = fixed.log_likelihood(&[&eta])?;

        let w_val: f32 = ll_weighted.get(0)?.to_scalar()?;
        let f_val: f32 = ll_fixed.get(0)?.to_scalar()?;

        assert!(
            (w_val - f_val).abs() < 1e-4,
            "Weighted and fixed should match with uniform variance: weighted={}, fixed={}",
            w_val,
            f_val,
        );

        Ok(())
    }

    #[test]
    fn test_weighted_high_variance_downweights() -> Result<()> {
        let device = Device::Cpu;

        // Two observations, one with low variance (informative), one with high variance
        let y = Tensor::from_vec(vec![1.0f32, 1.0], (2, 1), &device)?;
        // eta far from y
        let eta = Tensor::from_vec(vec![5.0f32, 5.0], (1, 2, 1), &device)?;

        // Low variance for both
        let var_low = Tensor::from_vec(vec![0.1f32, 0.1], (2, 1), &device)?;
        // High variance for second obs
        let var_mixed = Tensor::from_vec(vec![0.1f32, 1e6], (2, 1), &device)?;

        let ll_low = WeightedGaussianLikelihood::new(y.clone(), &var_low)?;
        let ll_mixed = WeightedGaussianLikelihood::new(y, &var_mixed)?;

        let val_low: f32 = ll_low.log_likelihood(&[&eta])?.get(0)?.to_scalar()?;
        let val_mixed: f32 = ll_mixed.log_likelihood(&[&eta])?.get(0)?.to_scalar()?;

        // High variance should give LESS negative log-likelihood (closer to 0)
        assert!(
            val_mixed > val_low,
            "High variance obs should be less penalized: mixed={}, low={}",
            val_mixed,
            val_low,
        );

        Ok(())
    }
}
