use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::traits::VariationalDistribution;

/// Gaussian variational distribution q(θ) = N(μ, σ²I)
///
/// Uses mean-field approximation with diagonal covariance.
/// Parameters are stored as mean μ and log standard deviation ln(σ).
pub struct GaussianVar {
    /// Variational mean μ: shape (p, k)
    mean: Tensor,
    /// Log standard deviation ln(σ): shape (p, k)
    ln_std: Tensor,
}

impl GaussianVar {
    /// Create a new Gaussian variational distribution.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `p` - Number of input features
    /// * `k` - Number of output dimensions
    ///
    /// # Returns
    /// Initialized GaussianVar with small random mean and ln_std = 0 (std = 1)
    pub fn new(vb: VarBuilder, p: usize, k: usize) -> Result<Self> {
        let mean = vb.get_with_hints(
            (p, k),
            "mean",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;
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

impl VariationalDistribution for GaussianVar {
    fn mean(&self) -> Result<Tensor> {
        Ok(self.mean.clone())
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
    fn test_mean_var_shapes() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let p = 5;
        let k = 3;

        let gauss = GaussianVar::new(vb, p, k)?;
        let mean = VariationalDistribution::mean(&gauss)?;
        let var = VariationalDistribution::var(&gauss)?;

        assert_eq!(mean.dims(), &[p, k]);
        assert_eq!(var.dims(), &[p, k]);

        Ok(())
    }
}
