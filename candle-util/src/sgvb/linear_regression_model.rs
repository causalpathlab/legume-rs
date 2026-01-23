use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::gaussian::GaussianVariational;
use super::sgvb::SGVBConfig;
use super::traits::{Prior, SgvbModel, SgvbSample, VariationalDistribution};

/// Linear regression SGVB model: η = X * θ where θ ~ q(θ) = N(μ, σ²I)
///
/// This is a convenience wrapper combining:
/// - Gaussian variational distribution for the regression coefficients
/// - Prior distribution for coefficients
/// - Linear predictor computation
/// - SGVB loss computation
pub struct LinearRegressionSGVB<P> {
    /// Variational distribution for coefficients θ
    pub variational: GaussianVariational,
    /// Prior distribution p(θ)
    pub prior: P,
    /// Design matrix X: (n, p)
    pub x_design: Tensor,
    /// SGVB configuration
    pub config: SGVBConfig,
}

impl<P: Prior> LinearRegressionSGVB<P> {
    /// Create a new linear regression SGVB model.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `x_design` - Design matrix X, shape (n, p)
    /// * `k` - Number of output dimensions
    /// * `prior` - Prior distribution
    /// * `config` - SGVB configuration
    ///
    /// # Returns
    /// Initialized LinearRegressionSGVB model
    pub fn new(vb: VarBuilder, x_design: Tensor, k: usize, prior: P, config: SGVBConfig) -> Result<Self> {
        let p = x_design.dim(1)?;
        let variational = GaussianVariational::new(vb, p, k)?;

        Ok(Self {
            variational,
            prior,
            x_design,
            config,
        })
    }

    /// Compute the posterior mean prediction η = X @ μ_θ
    ///
    /// # Returns
    /// Mean prediction, shape (n, k)
    pub fn eta_mean(&self) -> Result<Tensor> {
        self.x_design.matmul(self.variational.mean())
    }

    /// Get the variational mean of coefficients μ_θ
    ///
    /// # Returns
    /// Coefficient mean, shape (p, k)
    pub fn coef_mean(&self) -> &Tensor {
        self.variational.mean()
    }

    /// Get the variational standard deviation of coefficients σ_θ
    ///
    /// # Returns
    /// Coefficient std, shape (p, k)
    pub fn coef_std(&self) -> Result<Tensor> {
        self.variational.std()
    }
}

impl<P: Prior> SgvbModel for LinearRegressionSGVB<P> {
    fn sample(&self, num_samples: usize) -> Result<SgvbSample> {
        // 1. Compute η by propagating uncertainty through linear transformation
        // η_mean = X @ μ: shape (n, k)
        let theta_mean = self.variational.mean();
        let theta_var = self.variational.var()?;

        let eta_mean = self.x_design.matmul(theta_mean)?;

        // η_var = X² @ σ²: shape (n, k)
        let x_sq = self.x_design.powf(2.0)?;
        let eta_var = x_sq.matmul(&theta_var)?;
        let eta_std = eta_var.sqrt()?;

        // ε ~ N(0, 1): shape (S, n, k)
        let (n, k) = eta_mean.dims2()?;
        let device = eta_mean.device();
        let dtype = eta_mean.dtype();
        let eps = Tensor::randn(0f32, 1f32, (num_samples, n, k), device)?.to_dtype(dtype)?;

        // η = η_mean + η_std * ε: shape (S, n, k)
        let eta = eta_mean.unsqueeze(0)?.broadcast_add(&eps.broadcast_mul(&eta_std)?)?;

        // 2. Sample θ for log_prior and log_q (generalizable to non-Gaussian)
        let (theta, _) = self.variational.sample(num_samples)?;

        let log_prior = self.prior.log_prob(&theta)?;
        let log_q = self.variational.log_prob(&theta)?;
        let log_q_grad = self.variational.log_prob(&theta.detach())?;

        Ok(SgvbSample {
            eta,
            log_prior,
            log_q,
            log_q_grad,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::traits::BlackBoxLikelihood;
    use crate::sgvb::{compute_elbo, sgvb_loss, GaussianPrior};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    /// Simple Gaussian likelihood for testing
    struct TestGaussianLikelihood {
        y: Tensor,
        sigma: f64,
    }

    impl TestGaussianLikelihood {
        fn new(y: Tensor, sigma: f64) -> Self {
            Self { y, sigma }
        }
    }

    impl BlackBoxLikelihood for TestGaussianLikelihood {
        fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
            let eta = etas[0];
            let sigma_sq = self.sigma.powi(2);
            let ln_2pi = (2.0 * std::f64::consts::PI).ln();
            let ln_sigma = self.sigma.ln();
            let const_term = 2.0 * ln_sigma + ln_2pi;

            let diff_sq = eta.broadcast_sub(&self.y)?.powf(2.0)?;
            let log_prob = (((diff_sq / sigma_sq)? + const_term)? * (-0.5))?;

            log_prob.sum(2)?.sum(1)
        }
    }

    #[test]
    fn test_linear_sgvb_construction() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 50;
        let p = 10;
        let k = 3;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = LinearRegressionSGVB::new(vb.pp("model"), x, k, prior, config)?;

        // Check shapes
        assert_eq!(model.coef_mean().dims(), &[p, k]);
        assert_eq!(model.coef_std()?.dims(), &[p, k]);
        assert_eq!(model.eta_mean()?.dims(), &[n, k]);

        Ok(())
    }

    #[test]
    fn test_linear_sgvb_loss() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 50;
        let p = 10;
        let k = 3;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let y = Tensor::randn(0f32, 1f32, (n, k), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let likelihood = TestGaussianLikelihood::new(y, 1.0);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = LinearRegressionSGVB::new(vb.pp("model"), x, k, prior, config.clone())?;

        let loss = sgvb_loss(&model, &likelihood, &config)?;
        assert!(loss.dims().is_empty());

        let elbo = compute_elbo(&model, &likelihood, 50)?;
        assert!(elbo.dims().is_empty());

        Ok(())
    }
}
