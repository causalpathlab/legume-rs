use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::traits::Prior;

/// Maximum value for ln(τ) to prevent numerical overflow.
/// ln(100) ≈ 4.6, so τ is capped at ~100.
const MAX_LN_TAU: f64 = 4.6;

/// Learnable Gaussian prior p(θ) = N(0, τ²I)
///
/// The prior scale τ is a learnable parameter stored as ln(τ).
/// τ is capped at exp(MAX_LN_TAU) ≈ 100 to prevent numerical issues.
pub struct GaussianPrior {
    /// Log scale parameter ln(τ)
    ln_tau: Tensor,
}

impl GaussianPrior {
    /// Create a new Gaussian prior with learnable scale.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `init_tau` - Initial value for τ (will be stored as ln(τ))
    ///
    /// # Returns
    /// Initialized GaussianPrior
    pub fn new(vb: VarBuilder, init_tau: f32) -> Result<Self> {
        let ln_tau_init = init_tau.ln();
        let ln_tau = vb.get_with_hints((), "ln_tau", candle_nn::Init::Const(ln_tau_init as f64))?;
        Ok(Self { ln_tau })
    }

    /// Get the prior scale τ = exp(clamp(ln_tau)).
    pub fn tau(&self) -> Result<f32> {
        let clamped = self.ln_tau.clamp(-MAX_LN_TAU, MAX_LN_TAU)?;
        // Move to CPU for dtype conversion (Metal doesn't support F64)
        let val: f32 = clamped.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.exp()?.to_scalar()?;
        Ok(val)
    }

    /// Get the device of the parameters.
    pub fn device(&self) -> &Device {
        self.ln_tau.device()
    }

    /// Get the dtype of the parameters.
    pub fn dtype(&self) -> DType {
        self.ln_tau.dtype()
    }
}

impl Prior for GaussianPrior {
    /// Compute log p(θ) = sum over all elements of log N(θ; 0, τ²)
    ///
    /// log N(θ; 0, τ²) = -0.5 * [θ²/τ² + 2*ln(τ) + ln(2π)]
    ///
    /// # Arguments
    /// * `theta` - Parameter samples, shape (S, p, k)
    ///
    /// # Returns
    /// Log prior probability, shape (S,)
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        let dtype = theta.dtype();
        let device = theta.device();
        // Create scalar constants directly in the target dtype to avoid Metal F64 conversion issues
        let ln_2pi = Tensor::new((2.0 * std::f64::consts::PI).ln() as f32, device)?
            .to_dtype(dtype)?;

        // Clamp ln_tau to prevent overflow
        let ln_tau_clamped = self.ln_tau.clamp(-MAX_LN_TAU, MAX_LN_TAU)?.to_dtype(dtype)?;

        // τ = exp(ln_tau_clamped)
        let tau = ln_tau_clamped.exp()?;
        let tau_sq = tau.sqr()?;

        // θ²/τ²: shape (S, p, k)
        let theta_sq_normalized = theta.sqr()?.broadcast_div(&tau_sq)?;

        // 2*ln(τ) + ln(2π) is a scalar, broadcast to all elements
        let const_term = (ln_tau_clamped * 2.0)?.broadcast_add(&ln_2pi)?;

        // log p = -0.5 * [θ²/τ² + 2*ln(τ) + ln(2π)]
        let log_prob_element = (theta_sq_normalized.broadcast_add(&const_term)? * (-0.5))?;

        // Sum over dimensions 1 and 2 (p and k)
        log_prob_element.sum(2)?.sum(1)
    }
}

/// Fixed (non-learnable) Gaussian prior p(θ) = N(0, τ²I)
pub struct FixedGaussianPrior {
    /// Fixed scale parameter τ
    tau: f32,
}

impl FixedGaussianPrior {
    /// Create a new fixed Gaussian prior.
    ///
    /// # Arguments
    /// * `tau` - Prior scale τ
    pub fn new(tau: f32) -> Self {
        Self { tau }
    }

    /// Get the prior scale τ.
    pub fn tau(&self) -> f32 {
        self.tau
    }
}

impl Prior for FixedGaussianPrior {
    /// Compute log p(θ) = sum over all elements of log N(θ; 0, τ²)
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        let ln_2pi: f64 = (2.0 * std::f64::consts::PI).ln();
        let ln_tau: f64 = (self.tau as f64).ln();
        let tau_sq: f64 = (self.tau as f64).powi(2);

        // θ²/τ²: shape (S, p, k)
        let theta_sq_normalized = (theta.sqr()? / tau_sq)?;

        // 2*ln(τ) + ln(2π)
        let const_term = 2.0 * ln_tau + ln_2pi;

        // log p = -0.5 * [θ²/τ² + 2*ln(τ) + ln(2π)]
        let log_prob_element = ((theta_sq_normalized + const_term)? * (-0.5))?;

        // Sum over dimensions 1 and 2 (p and k)
        log_prob_element.sum(2)?.sum(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_tau_value() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let init_tau = 2.0f32;
        let prior = GaussianPrior::new(vb, init_tau)?;

        let tau = prior.tau()?;
        assert!((tau - init_tau).abs() < 1e-5, "Expected {}, got {}", init_tau, tau);

        Ok(())
    }

    #[test]
    fn test_log_prob_shape() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let s = 10;
        let p = 5;
        let k = 3;

        let prior = GaussianPrior::new(vb, 1.0)?;
        let theta = Tensor::randn(0f32, 1f32, (s, p, k), &Device::Cpu)?;
        let log_prob = prior.log_prob(&theta)?;

        assert_eq!(log_prob.dims(), &[s]);

        Ok(())
    }

    #[test]
    fn test_fixed_prior_log_prob() -> Result<()> {
        let prior = FixedGaussianPrior::new(1.0);

        let s = 5;
        let p = 2;
        let k = 2;

        // Theta at zero
        let theta = Tensor::zeros((s, p, k), DType::F64, &Device::Cpu)?;
        let log_prob = prior.log_prob(&theta)?;

        // Expected: -0.5 * ln(2π) * (p * k) per sample
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() * (p * k) as f64;

        for i in 0..s {
            let actual: f64 = log_prob.get(i)?.to_scalar()?;
            assert!((actual - expected).abs() < 1e-5, "Expected {}, got {}", expected, actual);
        }

        Ok(())
    }
}
