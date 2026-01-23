use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::traits::VariationalDistribution;

/// Gaussian variational distribution q(θ) = N(μ, σ²I)
///
/// Uses mean-field approximation with diagonal covariance.
/// Parameters are stored as mean μ and log standard deviation ln(σ).
pub struct GaussianVariational {
    /// Variational mean μ: shape (p, k)
    mean: Tensor,
    /// Log standard deviation ln(σ): shape (p, k)
    ln_std: Tensor,
}

impl GaussianVariational {
    /// Create a new Gaussian variational distribution.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `p` - Number of input features
    /// * `k` - Number of output dimensions
    ///
    /// # Returns
    /// Initialized GaussianVariational with small random mean and ln_std = 0 (std = 1)
    pub fn new(vb: VarBuilder, p: usize, k: usize) -> Result<Self> {
        let mean = vb.get_with_hints((p, k), "mean", candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 })?;
        let ln_std = vb.get_with_hints((p, k), "ln_std", candle_nn::Init::Const(0.0))?;
        Ok(Self { mean, ln_std })
    }

    /// Get the variational mean μ.
    pub fn mean(&self) -> &Tensor {
        &self.mean
    }

    /// Get the variational standard deviation σ = exp(ln_std).
    pub fn std(&self) -> Result<Tensor> {
        self.ln_std.exp()
    }

    /// Get the device of the parameters.
    pub fn device(&self) -> &Device {
        self.mean.device()
    }

    /// Get the dtype of the parameters.
    pub fn dtype(&self) -> DType {
        self.mean.dtype()
    }
}

impl VariationalDistribution for GaussianVariational {
    /// Sample using reparameterization: θ = μ + σ * ε where ε ~ N(0, I)
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples S
    ///
    /// # Returns
    /// (theta, epsilon) where:
    /// - theta: shape (S, p, k)
    /// - epsilon: shape (S, p, k)
    fn sample(&self, num_samples: usize) -> Result<(Tensor, Tensor)> {
        let (p, k) = self.mean.dims2()?;
        let device = self.mean.device();
        let dtype = self.mean.dtype();

        // ε ~ N(0, I): shape (S, p, k)
        let epsilon = Tensor::randn(0f32, 1f32, (num_samples, p, k), device)?.to_dtype(dtype)?;

        // σ = exp(ln_std): shape (p, k)
        let std = self.ln_std.exp()?;

        // θ = μ + σ * ε: broadcast (p, k) + (p, k) * (S, p, k) -> (S, p, k)
        let theta = self.mean.unsqueeze(0)?.broadcast_add(&epsilon.broadcast_mul(&std)?)?;

        Ok((theta, epsilon))
    }

    /// Compute log q(θ|μ,σ) = sum over (p,k) of log N(θ; μ, σ²)
    ///
    /// log N(θ; μ, σ²) = -0.5 * [(θ-μ)²/σ² + 2*ln(σ) + ln(2π)]
    ///
    /// # Arguments
    /// * `theta` - Parameter samples, shape (S, p, k)
    ///
    /// # Returns
    /// Log probability, shape (S,)
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        let dtype = theta.dtype();
        let device = theta.device();
        let ln_2pi = Tensor::new((2.0 * std::f64::consts::PI).ln(), device)?.to_dtype(dtype)?;

        // σ = exp(ln_std): shape (p, k)
        let std = self.ln_std.exp()?;

        // (θ - μ)²/σ²: shape (S, p, k)
        let diff = theta.broadcast_sub(&self.mean)?;
        let normalized_sq = diff.powf(2.0)?.broadcast_div(&std.powf(2.0)?)?;

        // log q = -0.5 * [(θ-μ)²/σ² + 2*ln(σ) + ln(2π)]
        // Sum over (p, k) dimensions
        // 2*ln(σ): broadcast (p, k) to (S, p, k)
        let two_ln_std = (&self.ln_std * 2.0)?;
        let log_prob_element = (normalized_sq.broadcast_add(&two_ln_std)?.broadcast_add(&ln_2pi)? * (-0.5))?;

        // Sum over dimensions 1 and 2 (p and k)
        log_prob_element.sum(2)?.sum(1)
    }

    fn mean(&self) -> &Tensor {
        &self.mean
    }

    fn var(&self) -> Result<Tensor> {
        // σ² = exp(2 * ln_std)
        (&self.ln_std * 2.0)?.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_sample_shape() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let p = 5;
        let k = 3;
        let s = 10;

        let gauss = GaussianVariational::new(vb, p, k)?;
        let (theta, epsilon) = gauss.sample(s)?;

        assert_eq!(theta.dims(), &[s, p, k]);
        assert_eq!(epsilon.dims(), &[s, p, k]);

        Ok(())
    }

    #[test]
    fn test_log_prob_shape() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let p = 5;
        let k = 3;
        let s = 10;

        let gauss = GaussianVariational::new(vb, p, k)?;
        let (theta, _) = gauss.sample(s)?;
        let log_prob = gauss.log_prob(&theta)?;

        assert_eq!(log_prob.dims(), &[s]);

        Ok(())
    }

    #[test]
    fn test_log_prob_at_mean() -> Result<()> {
        // For N(μ, σ²) evaluated at θ = μ, log p(μ) = -0.5 * ln(2π) - ln(σ) per element
        // With σ = 1 (ln_std = 0), this is just -0.5 * ln(2π) per element
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);

        let p = 2;
        let k = 2;

        let gauss = GaussianVariational::new(vb, p, k)?;

        // Create samples at the actual mean
        let theta = gauss.mean().unsqueeze(0)?;
        let log_prob = gauss.log_prob(&theta)?;

        // Expected: -0.5 * ln(2π) * (p * k) = -0.5 * ln(2π) * 4
        // Because at θ = μ, the (θ-μ)²/σ² term is 0
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() * (p * k) as f64;
        let actual: f64 = log_prob.get(0)?.to_scalar()?;

        assert!((actual - expected).abs() < 1e-5, "Expected {}, got {}", expected, actual);

        Ok(())
    }
}
