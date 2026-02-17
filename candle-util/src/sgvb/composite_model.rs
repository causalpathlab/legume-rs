//! Composite model for multi-eta likelihoods.
//!
//! Composes multiple SGVB modules, each producing its own eta (linear predictor).
//! Useful for likelihoods that depend on multiple linear predictors, such as:
//! - Gaussian: η₁ for mean, η₂ for log-variance
//! - Negative binomial: η₁ for log-mean, η₂ for dispersion
//! - Zero-inflated models: η₁ for count, η₂ for zero-inflation

use candle_core::{Result, Tensor};

use super::regression_linear::LinearModelSGVB;
use super::traits::{
    AnalyticalKL, BlackBoxLikelihood, LocalReparamSample, Prior, VariationalDistribution,
};

/// Composite model that combines multiple SGVB modules.
///
/// Each module produces its own eta from potentially different
/// design matrices and variational distributions.
pub struct CompositeModel<V, P> {
    pub modules: Vec<LinearModelSGVB<V, P>>,
}

impl<V: VariationalDistribution, P: Prior> CompositeModel<V, P> {
    /// Create a composite model from a vector of modules.
    pub fn new(modules: Vec<LinearModelSGVB<V, P>>) -> Self {
        assert!(
            !modules.is_empty(),
            "CompositeModel requires at least one module"
        );
        Self { modules }
    }

    /// Number of modules (number of etas).
    pub fn num_modules(&self) -> usize {
        self.modules.len()
    }
}

/// Compute local reparameterization loss for a composite model.
///
/// All modules must use local reparameterization (same variational + prior types).
pub fn composite_local_reparam_loss<V, P, L>(
    model: &CompositeModel<V, P>,
    likelihood: &L,
    num_samples: usize,
    kl_weight: f64,
) -> Result<Tensor>
where
    V: VariationalDistribution,
    P: Prior + AnalyticalKL,
    L: BlackBoxLikelihood,
{
    let samples: Vec<LocalReparamSample> = model
        .modules
        .iter()
        .map(|m| m.local_reparam_sample(num_samples))
        .collect::<Result<_>>()?;

    samples_local_reparam_loss(&samples, likelihood, kl_weight)
}

/// Compute local reparameterization loss from pre-sampled outputs.
///
/// Allows mixing modules that were sampled separately using local reparameterization.
pub fn samples_local_reparam_loss<L>(
    samples: &[LocalReparamSample],
    likelihood: &L,
    kl_weight: f64,
) -> Result<Tensor>
where
    L: BlackBoxLikelihood,
{
    let etas: Vec<&Tensor> = samples.iter().map(|s| &s.eta).collect();
    let llik = likelihood.log_likelihood(&etas)?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    // Sum KLs across modules
    let mut total_kl = samples[0].kl.clone();
    for s in &samples[1..] {
        total_kl = (&total_kl + &s.kl)?;
    }

    // ELBO = E[log p(y|η)] − β·KL
    let elbo = (llik.mean(0)? - (total_kl * kl_weight)?)?;
    elbo.neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::{GaussianPrior, LinearRegressionSGVB, SGVBConfig};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Optimizer, VarBuilder, VarMap};

    /// Gaussian likelihood: y ~ N(η₁, exp(η₂))
    struct TestGaussianLikelihood {
        y: Tensor,
    }

    impl TestGaussianLikelihood {
        fn new(y: Tensor) -> Self {
            Self { y }
        }
    }

    impl BlackBoxLikelihood for TestGaussianLikelihood {
        fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
            assert!(etas.len() >= 2, "TestGaussianLikelihood requires 2 etas");
            let mu = etas[0]; // mean: (S, n, k)
            let log_var_raw = etas[1]; // log-variance: (S, n, k)

            // Clamp log_var to avoid numerical issues with exp()
            let log_var = log_var_raw.clamp(-10.0, 10.0)?;

            // log N(y; μ, exp(log_var)) = -0.5 * [log(2π) + log_var + (y-μ)²/exp(log_var)]
            let ln_2pi = (2.0 * std::f64::consts::PI).ln();

            let diff = mu.broadcast_sub(&self.y)?;
            let diff_sq = diff.sqr()?;
            let var = log_var.exp()?;
            let scaled_diff_sq = (diff_sq / &var)?;

            let log_prob = ((scaled_diff_sq + &log_var)? + ln_2pi)? * (-0.5);

            // Sum over (n, k) dimensions
            log_prob?.sum(2)?.sum(1)
        }
    }

    #[test]
    fn test_composite_model_construction() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 50;
        let p_mean = 10;
        let p_var = 5;
        let k = 2;

        // Different design matrices for mean and variance
        let x_mean = Tensor::randn(0f32, 1f32, (n, p_mean), &device)?;
        let x_var = Tensor::randn(0f32, 1f32, (n, p_var), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let prior_mean = GaussianPrior::new(vb.pp("prior_mean"), 1.0)?;
        let prior_var = GaussianPrior::new(vb.pp("prior_var"), 1.0)?;
        let config = SGVBConfig::default();

        let model_mean =
            LinearRegressionSGVB::new(vb.pp("mean"), x_mean, k, prior_mean, config.clone())?;
        let model_var =
            LinearRegressionSGVB::new(vb.pp("var"), x_var, k, prior_var, config.clone())?;

        let composite = CompositeModel::new(vec![model_mean, model_var]);

        assert_eq!(composite.num_modules(), 2);

        // Test local reparam sampling
        let samples: Vec<LocalReparamSample> = composite
            .modules
            .iter()
            .map(|m| m.local_reparam_sample(10))
            .collect::<Result<_>>()?;
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].eta.dims(), &[10, n, k]);
        assert_eq!(samples[1].eta.dims(), &[10, n, k]);

        Ok(())
    }

    #[test]
    fn test_heteroscedastic_regression() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 100;
        let p_mean = 5;
        let p_var = 3;
        let k = 1;

        // Design matrices
        let x_mean = Tensor::randn(0f32, 1f32, (n, p_mean), &device)?;
        let x_var = Tensor::randn(0f32, 1f32, (n, p_var), &device)?;

        // Generate heteroscedastic data:
        // true_mean = x_mean[:, 0] * 2.0
        // true_log_var = x_var[:, 0] * 0.5 (so variance varies with x_var)
        let true_mean = (x_mean.narrow(1, 0, 1)? * 2.0)?;
        let true_log_var = (x_var.narrow(1, 0, 1)? * 0.5)?;
        let true_std = (true_log_var.clone() / 2.0)?.exp()?; // sqrt(var) = exp(log_var/2)

        let noise = Tensor::randn(0f32, 1f32, (n, k), &device)?;
        let y = (true_mean + (noise * true_std)?)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let prior_mean = GaussianPrior::new(vb.pp("prior_mean"), 1.0)?;
        let prior_var = GaussianPrior::new(vb.pp("prior_var"), 1.0)?;
        let config = SGVBConfig::new(30);

        let model_mean =
            LinearRegressionSGVB::new(vb.pp("mean"), x_mean, k, prior_mean, config.clone())?;
        let model_var =
            LinearRegressionSGVB::new(vb.pp("var"), x_var, k, prior_var, config.clone())?;

        let composite = CompositeModel::new(vec![model_mean, model_var]);
        let likelihood = TestGaussianLikelihood::new(y);

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.01)?;

        for i in 0..300 {
            let loss = composite_local_reparam_loss(&composite, &likelihood, 30, 1.0)?;
            optimizer.backward_step(&loss)?;

            if i % 100 == 0 {
                let loss_val: f32 = loss.to_scalar()?;
                let elbo_val = -loss_val;
                println!(
                    "iter {:4}: loss = {:10.4}, ELBO = {:10.4}",
                    i, loss_val, elbo_val
                );
            }
        }

        // Check that mean model learned the signal (true=2.0, threshold relaxed for stochastic variability)
        let mean_coef = composite.modules[0].coef_mean()?;
        let mean_coef_0: f32 = mean_coef.get(0)?.get(0)?.to_scalar()?;
        println!("\nMean coef[0] (true=2.0): {:.4}", mean_coef_0);
        assert!(
            mean_coef_0 > 0.5,
            "Mean coef[0] should be > 0.5, got {}",
            mean_coef_0
        );

        Ok(())
    }
}
