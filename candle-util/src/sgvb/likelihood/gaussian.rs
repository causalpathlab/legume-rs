//! Gaussian likelihood for continuous data.

use candle_core::{Result, Tensor};

use crate::sgvb::BlackBoxLikelihood;

/// Gaussian likelihood: y ~ N(η₁, exp(η₂))
///
/// # Model
/// ```text
/// log p(y | η₁, η₂) = -0.5 * [log(2π) + η₂ + (y - η₁)² / exp(η₂)]
/// ```
///
/// Requires two etas:
/// - etas[0]: mean (μ)
/// - etas[1]: log-variance (log σ²)
pub struct GaussianLikelihood {
    y: Tensor,
}

impl GaussianLikelihood {
    pub fn new(y: Tensor) -> Self {
        Self { y }
    }
}

impl BlackBoxLikelihood for GaussianLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        assert!(
            etas.len() >= 2,
            "GaussianLikelihood requires 2 etas (mean, log_var)"
        );
        let mu = etas[0]; // mean: (S, n, k)
        let log_var_raw = etas[1]; // log-variance: (S, n, k)

        // Clamp log_var to avoid numerical issues with exp()
        // Range [-10, 10] gives variance in [4.5e-5, 22026]
        let log_var = log_var_raw.clamp(-10.0, 10.0)?;

        // log N(y; μ, exp(log_var)) = -0.5 * [log(2π) + log_var + (y-μ)²/exp(log_var)]
        let ln_2pi: f64 = (2.0 * std::f64::consts::PI).ln();

        let diff = mu.broadcast_sub(&self.y)?;
        let diff_sq = diff.sqr()?;
        let var = log_var.exp()?;
        let scaled_diff_sq = (diff_sq / &var)?;

        let log_prob = ((scaled_diff_sq + &log_var)? + ln_2pi)? * (-0.5);

        // Sum over (n, k) dimensions
        log_prob?.sum(2)?.sum(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_gaussian_likelihood() -> Result<()> {
        let device = Device::Cpu;

        // y = [0, 1, 2], mu = [0, 1, 2] (perfect fit), log_var = [0, 0, 0] (var = 1)
        let y = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], (3, 1), &device)?;
        let mu = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], (1, 3, 1), &device)?;
        let log_var = Tensor::zeros((1, 3, 1), candle_core::DType::F32, &device)?;

        let likelihood = GaussianLikelihood::new(y);
        let log_lik = likelihood.log_likelihood(&[&mu, &log_var])?;

        let val: f32 = log_lik.get(0)?.to_scalar()?;
        assert!(val.is_finite());
        // With perfect fit and var=1, log_lik ≈ -0.5 * 3 * log(2π) ≈ -2.76
        println!("Gaussian log_lik (perfect fit): {}", val);

        Ok(())
    }
}
