use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::variational_gaussian::GaussianVar;
use super::sgvb::SGVBConfig;
use super::traits::{Prior, SgvbModel, SgvbSample, VariationalDistribution};

/// Generic linear model SGVB: η = X * θ where θ ~ q(θ)
///
/// This is a generic wrapper that works with any variational distribution:
/// - V: VariationalDistribution for the regression coefficients
/// - P: Prior distribution for coefficients
pub struct LinearModelSGVB<V, P> {
    /// Variational distribution for coefficients θ
    pub variational: V,
    /// Prior distribution p(θ)
    pub prior: P,
    /// Design matrix X: (n, p)
    pub x_design: Tensor,
    /// SGVB configuration
    pub config: SGVBConfig,
}

impl<V: VariationalDistribution, P: Prior> LinearModelSGVB<V, P> {
    /// Create a new linear model with a given variational distribution.
    ///
    /// # Arguments
    /// * `variational` - Variational distribution for θ
    /// * `x_design` - Design matrix X, shape (n, p)
    /// * `prior` - Prior distribution
    /// * `config` - SGVB configuration
    pub fn from_variational(variational: V, x_design: Tensor, prior: P, config: SGVBConfig) -> Self {
        Self {
            variational,
            prior,
            x_design,
            config,
        }
    }

    /// Compute the posterior mean prediction η = X @ μ_θ
    pub fn eta_mean(&self) -> Result<Tensor> {
        let theta_mean = self.variational.mean()?;
        self.x_design.matmul(&theta_mean)
    }

    /// Get the variational mean of coefficients μ_θ
    pub fn coef_mean(&self) -> Result<Tensor> {
        self.variational.mean()
    }

    /// Get the variational variance of coefficients
    pub fn coef_var(&self) -> Result<Tensor> {
        self.variational.var()
    }
}

impl<V: VariationalDistribution, P: Prior> SgvbModel for LinearModelSGVB<V, P> {
    fn sample(&self, num_samples: usize) -> Result<SgvbSample> {
        // 1. Sample θ from variational distribution
        let (theta, _) = self.variational.sample(num_samples)?;

        // 2. Compute η = X @ θ
        let eta = self.x_design.unsqueeze(0)?.broadcast_matmul(&theta)?;

        // 3. Compute log_prior and log_q
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

/// Linear regression SGVB model with Gaussian variational distribution.
///
/// Convenience type alias for LinearModelSGVB with GaussianVar.
pub type LinearRegressionSGVB<P> = LinearModelSGVB<GaussianVar, P>;

impl<P: Prior> LinearRegressionSGVB<P> {
    /// Create a new linear regression SGVB model with Gaussian variational distribution.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `x_design` - Design matrix X, shape (n, p)
    /// * `k` - Number of output dimensions
    /// * `prior` - Prior distribution
    /// * `config` - SGVB configuration
    pub fn new(
        vb: VarBuilder,
        x_design: Tensor,
        k: usize,
        prior: P,
        config: SGVBConfig,
    ) -> Result<Self> {
        let p = x_design.dim(1)?;
        let variational = GaussianVar::new(vb, p, k)?;
        Ok(Self::from_variational(variational, x_design, prior, config))
    }

    /// Get the variational standard deviation of coefficients σ_θ
    pub fn coef_std(&self) -> Result<Tensor> {
        self.variational.std()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::traits::BlackBoxLikelihood;
    use crate::sgvb::{compute_elbo, direct_elbo_loss, sgvb_loss, GaussianPrior};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Optimizer, VarBuilder, VarMap};

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

            let diff_sq = eta.broadcast_sub(&self.y)?.sqr()?;
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
        assert_eq!(model.coef_mean()?.dims(), &[p, k]);
        assert_eq!(model.coef_std()?.dims(), &[p, k]);
        assert_eq!(model.eta_mean()?.dims(), &[n, k]);

        Ok(())
    }

    #[test]
    fn test_linear_sgvb_loss() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 150;
        let p = 30;
        let k = 1;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        // Generate y from first column of X: y = X[:, 0] * 2.0 + noise
        let true_coef = 2.0f64;
        let x_first = x.narrow(1, 0, 1)?; // (n, 1)
        let noise = Tensor::randn(0f32, 0.5f32, (n, k), &device)?;
        let y = (x_first * true_coef)?.add(&noise)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let likelihood = TestGaussianLikelihood::new(y, 0.5);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::new(50); // 50 samples, normalized

        let model = LinearRegressionSGVB::new(vb.pp("model"), x, k, prior, config.clone())?;

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.01)?;

        for i in 0..200 {
            let loss = sgvb_loss(&model, &likelihood, &config)?;
            optimizer.backward_step(&loss)?;

            let elbo = compute_elbo(&model, &likelihood, 100)?;

            if i % 20 == 0 {
                let loss_val: f32 = loss.to_scalar()?;
                let elbo_val: f32 = elbo.to_scalar()?;
                println!("iter {}: loss = {:.4}, elbo = {:.4}", i, loss_val, elbo_val);
            }

            assert!(loss.dims().is_empty());
            assert!(elbo.dims().is_empty());
        }

        // Check learned coefficients
        let coef_mean = model.coef_mean()?;
        let coef_first: f32 = coef_mean.get(0)?.get(0)?.to_scalar()?;

        // Compute mean of other coefficients
        let mut other_sum = 0.0f32;
        for i in 1..p {
            let val: f32 = coef_mean.get(i)?.get(0)?.to_scalar()?;
            other_sum += val.abs();
        }
        let other_mean = other_sum / (p - 1) as f32;

        println!("\nCoefficients:");
        println!("  First (true={:.1}): {:.4}", true_coef, coef_first);
        println!("  Others mean abs: {:.4}", other_mean);

        // First coefficient should be close to true_coef and much larger than others
        assert!(coef_first > 1.0, "First coef should be > 1.0, got {}", coef_first);
        assert!(coef_first.abs() > other_mean * 2.0, "First coef should dominate others");

        Ok(())
    }

    #[test]
    fn test_susie_linear_construction() -> Result<()> {
        use crate::sgvb::SusieVar;

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 50;
        let p = 20;
        let k = 2;
        let l = 3; // 3 components

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new(vb.pp("susie"), l, p, k)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = LinearModelSGVB::from_variational(susie, x, prior, config);

        // Check shapes
        assert_eq!(model.coef_mean()?.dims(), &[p, k]);
        assert_eq!(model.coef_var()?.dims(), &[p, k]);
        assert_eq!(model.eta_mean()?.dims(), &[n, k]);

        // Check variational-specific methods
        assert_eq!(model.variational.alpha()?.dims(), &[l, p, k]);
        assert_eq!(model.variational.pip()?.dims(), &[p, k]);

        Ok(())
    }

    #[test]
    fn test_susie_linear_sparse_recovery() -> Result<()> {
        use crate::sgvb::SusieVar;

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 150;
        let p = 50;
        let k = 1;
        let l = 2; // 2 components for 2 true effects

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        // Generate y from features 0 and 5: y = X[:,0] * 2.0 + X[:,5] * 1.5 + noise
        let x_0 = x.narrow(1, 0, 1)?;
        let x_5 = x.narrow(1, 5, 1)?;
        let noise = Tensor::randn(0f32, 0.5f32, (n, k), &device)?;
        let y = ((x_0 * 2.0)? + (x_5 * 1.5)? + noise)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let likelihood = TestGaussianLikelihood::new(y, 0.5);
        let susie = SusieVar::new(vb.pp("susie"), l, p, k)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::new(50);

        let model = LinearModelSGVB::from_variational(susie, x, prior, config.clone());

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.05)?;

        for i in 0..500 {
            // Direct ELBO with reparameterization gradients
            let loss = direct_elbo_loss(&model, &likelihood, config.num_samples)?;
            optimizer.backward_step(&loss)?;

            if i % 5 == 0 {
                let loss_val: f32 = loss.to_scalar()?;
                let pip = model.variational.pip()?;
                let pip_0: f32 = pip.get(0)?.get(0)?.to_scalar()?;
                let pip_5: f32 = pip.get(5)?.get(0)?.to_scalar()?;
                println!(
                    "iter {}: loss = {:.4}, PIP[0] = {:.4}, PIP[5] = {:.4}",
                    i, loss_val, pip_0, pip_5
                );
            }
        }

        // Check PIPs - features 0 and 5 should have high PIPs
        let pip = model.variational.pip()?;
        let pip_0: f32 = pip.get(0)?.get(0)?.to_scalar()?;
        let pip_5: f32 = pip.get(5)?.get(0)?.to_scalar()?;

        // Mean PIP of other features
        let mut other_sum = 0.0f32;
        for j in 0..p {
            if j != 0 && j != 5 {
                let val: f32 = pip.get(j)?.get(0)?.to_scalar()?;
                other_sum += val;
            }
        }
        let other_mean = other_sum / (p - 2) as f32;

        println!("\nPosterior Inclusion Probabilities:");
        println!("  PIP[0] (true): {:.4}", pip_0);
        println!("  PIP[5] (true): {:.4}", pip_5);
        println!("  Others mean:   {:.4}", other_mean);

        // True features should have higher PIPs than others
        assert!(
            pip_0 > other_mean * 3.0,
            "PIP[0] should be > 3x other mean, got {} vs {}",
            pip_0, other_mean
        );
        assert!(
            pip_5 > other_mean * 3.0,
            "PIP[5] should be > 3x other mean, got {} vs {}",
            pip_5, other_mean
        );

        Ok(())
    }

    #[test]
    fn test_susie_linear_sparse_recovery_reinforce() -> Result<()> {
        use crate::sgvb::SusieVar;

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 150;
        let p = 50;
        let k = 1;
        let l = 2; // 2 components for 2 true effects

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        // Generate y with stronger signal for REINFORCE (higher variance estimator)
        // y = X[:,0] * 3.0 + X[:,5] * 2.5 + noise
        let x_0 = x.narrow(1, 0, 1)?;
        let x_5 = x.narrow(1, 5, 1)?;
        let noise = Tensor::randn(0f32, 0.3f32, (n, k), &device)?;
        let y = ((x_0 * 3.0)? + (x_5 * 2.5)? + noise)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let likelihood = TestGaussianLikelihood::new(y, 0.3);
        let susie = SusieVar::new(vb.pp("susie"), l, p, k)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        // More samples for variance reduction in REINFORCE
        let config = SGVBConfig::new(100);

        let model = LinearModelSGVB::from_variational(susie, x, prior, config.clone());

        // Lower learning rate for REINFORCE stability
        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.02)?;

        for i in 0..800 {
            // REINFORCE with Gaussian moment-matching approximation for Susie
            let loss = sgvb_loss(&model, &likelihood, &config)?;
            optimizer.backward_step(&loss)?;

            if i % 50 == 0 {
                let loss_val: f32 = loss.to_scalar()?;
                let pip = model.variational.pip()?;
                let pip_0: f32 = pip.get(0)?.get(0)?.to_scalar()?;
                let pip_5: f32 = pip.get(5)?.get(0)?.to_scalar()?;
                println!(
                    "iter {}: loss = {:.4}, PIP[0] = {:.4}, PIP[5] = {:.4}",
                    i, loss_val, pip_0, pip_5
                );
            }
        }

        // Check PIPs - features 0 and 5 should have high PIPs
        let pip = model.variational.pip()?;
        let pip_0: f32 = pip.get(0)?.get(0)?.to_scalar()?;
        let pip_5: f32 = pip.get(5)?.get(0)?.to_scalar()?;

        // Mean PIP of other features
        let mut other_sum = 0.0f32;
        for j in 0..p {
            if j != 0 && j != 5 {
                let val: f32 = pip.get(j)?.get(0)?.to_scalar()?;
                other_sum += val;
            }
        }
        let other_mean = other_sum / (p - 2) as f32;

        println!("\nPosterior Inclusion Probabilities (REINFORCE):");
        println!("  PIP[0] (true): {:.4}", pip_0);
        println!("  PIP[5] (true): {:.4}", pip_5);
        println!("  Others mean:   {:.4}", other_mean);

        // REINFORCE has higher variance - use relaxed threshold
        // True features should have notably higher PIPs than others
        assert!(
            pip_0 > other_mean * 1.5,
            "PIP[0] should be > 1.5x other mean, got {} vs {}",
            pip_0, other_mean
        );
        assert!(
            pip_5 > other_mean * 1.5,
            "PIP[5] should be > 1.5x other mean, got {} vs {}",
            pip_5, other_mean
        );

        Ok(())
    }
}
