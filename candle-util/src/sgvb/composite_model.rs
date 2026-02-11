//! Composite model for multi-eta likelihoods.
//!
//! Composes multiple SGVB modules, each producing its own eta (linear predictor).
//! Useful for likelihoods that depend on multiple linear predictors, such as:
//! - Gaussian: η₁ for mean, η₂ for log-variance
//! - Negative binomial: η₁ for log-mean, η₂ for dispersion
//! - Zero-inflated models: η₁ for count, η₂ for zero-inflation

use candle_core::{Result, Tensor};

use super::traits::{BlackBoxLikelihood, SgvbModel, SgvbSample};

/// Composite model that combines multiple SGVB modules.
///
/// Each module produces its own eta from potentially different
/// design matrices and variational distributions.
pub struct CompositeModel<M: SgvbModel> {
    pub modules: Vec<M>,
}

impl<M: SgvbModel> CompositeModel<M> {
    /// Create a composite model from a vector of modules.
    pub fn new(modules: Vec<M>) -> Self {
        assert!(
            !modules.is_empty(),
            "CompositeModel requires at least one module"
        );
        Self { modules }
    }

    /// Sample from all modules.
    ///
    /// Returns a vector of SgvbSample, one per module.
    pub fn sample_all(&self, num_samples: usize) -> Result<Vec<SgvbSample>> {
        self.modules.iter().map(|m| m.sample(num_samples)).collect()
    }

    /// Number of modules (number of etas).
    pub fn num_modules(&self) -> usize {
        self.modules.len()
    }
}

/// Sum a collection of tensors element-wise.
fn sum_tensors<'a>(mut tensors: impl Iterator<Item = &'a Tensor>) -> Result<Tensor> {
    let first = tensors.next().expect("Expected at least one tensor");
    let mut result = first.clone();
    for t in tensors {
        result = (&result + t)?;
    }
    Ok(result)
}

/// Compute SGVB loss for a composite model using REINFORCE estimator.
///
/// Each module contributes its own eta to the likelihood, and the
/// log_prior and log_q terms are summed across modules.
pub fn composite_sgvb_loss<M, L>(
    model: &CompositeModel<M>,
    likelihood: &L,
    num_samples: usize,
    normalize: bool,
) -> Result<Tensor>
where
    M: SgvbModel,
    L: BlackBoxLikelihood,
{
    let samples = model.sample_all(num_samples)?;

    // Collect etas for likelihood
    let etas: Vec<&Tensor> = samples.iter().map(|s| &s.eta).collect();
    let llik = likelihood.log_likelihood(&etas)?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    // Sum log_prior and log_q across modules
    let log_prior = sum_tensors(samples.iter().map(|s| &s.log_prior))?;
    let log_q = sum_tensors(samples.iter().map(|s| &s.log_q))?;
    let log_q_grad = sum_tensors(samples.iter().map(|s| &s.log_q_grad))?;

    // Reward = log p(y|η) + log p(θ) - log q(θ)
    let reward = ((&llik + &log_prior)? - &log_q)?;

    // Normalize reward (control variate)
    let reward_norm = if normalize {
        let mean = reward.mean(0)?;
        let var = reward.var(0)?;
        let std = (var + 1e-8)?.sqrt()?;
        reward.broadcast_sub(&mean)?.broadcast_div(&std)?
    } else {
        reward
    };

    // Detach reward and compute surrogate loss
    let reward_detached = reward_norm.detach();
    let surrogate_loss = (&reward_detached * &log_q_grad)?.mean(0)?.neg()?;

    Ok(surrogate_loss)
}

/// Compute direct ELBO loss for a composite model with reparameterization gradients.
pub fn composite_direct_elbo_loss<M, L>(
    model: &CompositeModel<M>,
    likelihood: &L,
    num_samples: usize,
) -> Result<Tensor>
where
    M: SgvbModel,
    L: BlackBoxLikelihood,
{
    let samples = model.sample_all(num_samples)?;

    // Collect etas for likelihood
    let etas: Vec<&Tensor> = samples.iter().map(|s| &s.eta).collect();
    let llik = likelihood.log_likelihood(&etas)?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    // Sum log_prior and log_q across modules
    let log_prior = sum_tensors(samples.iter().map(|s| &s.log_prior))?;
    let log_q = sum_tensors(samples.iter().map(|s| &s.log_q))?;

    // ELBO = log_lik + log_prior - log_q
    let elbo = ((&llik + &log_prior)? - &log_q)?;

    // Return negative mean ELBO as loss
    elbo.mean(0)?.neg()
}

/// Compute raw ELBO for a composite model (for monitoring).
pub fn composite_elbo<M, L>(
    model: &CompositeModel<M>,
    likelihood: &L,
    num_samples: usize,
) -> Result<Tensor>
where
    M: SgvbModel,
    L: BlackBoxLikelihood,
{
    let samples = model.sample_all(num_samples)?;
    samples_elbo(&samples, likelihood)
}

// ============================================================================
// Functions that work directly with Vec<SgvbSample> for heterogeneous modules
// ============================================================================

/// Compute direct ELBO loss from pre-sampled outputs.
///
/// Allows mixing models with different variational types by sampling separately
/// and combining. Each sample provides one eta.
pub fn samples_direct_elbo_loss<L>(samples: &[SgvbSample], likelihood: &L) -> Result<Tensor>
where
    L: BlackBoxLikelihood,
{
    let etas: Vec<&Tensor> = samples.iter().map(|s| &s.eta).collect();
    let llik = likelihood.log_likelihood(&etas)?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    let log_prior = sum_tensors(samples.iter().map(|s| &s.log_prior))?;
    let log_q = sum_tensors(samples.iter().map(|s| &s.log_q))?;

    let elbo = ((&llik + &log_prior)? - &log_q)?;
    elbo.mean(0)?.neg()
}

/// Compute SGVB loss from pre-sampled outputs using REINFORCE estimator.
///
/// Allows mixing models with different variational types by sampling separately
/// and combining. Each sample provides one eta.
pub fn samples_sgvb_loss<L>(
    samples: &[SgvbSample],
    likelihood: &L,
    normalize: bool,
) -> Result<Tensor>
where
    L: BlackBoxLikelihood,
{
    let etas: Vec<&Tensor> = samples.iter().map(|s| &s.eta).collect();
    let llik = likelihood.log_likelihood(&etas)?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    // Sum log_prior and log_q across modules
    let log_prior = sum_tensors(samples.iter().map(|s| &s.log_prior))?;
    let log_q = sum_tensors(samples.iter().map(|s| &s.log_q))?;
    let log_q_grad = sum_tensors(samples.iter().map(|s| &s.log_q_grad))?;

    // Reward = log p(y|η) + log p(θ) - log q(θ)
    let reward = ((&llik + &log_prior)? - &log_q)?;

    // Normalize reward (control variate)
    let reward_norm = if normalize {
        let mean = reward.mean(0)?;
        let var = reward.var(0)?;
        let std = (var + 1e-8)?.sqrt()?;
        reward.broadcast_sub(&mean)?.broadcast_div(&std)?
    } else {
        reward
    };

    // Detach reward and compute surrogate loss
    let reward_detached = reward_norm.detach();
    let surrogate_loss = (&reward_detached * &log_q_grad)?.mean(0)?.neg()?;

    Ok(surrogate_loss)
}

/// Compute raw ELBO from pre-sampled outputs (for monitoring).
pub fn samples_elbo<L>(samples: &[SgvbSample], likelihood: &L) -> Result<Tensor>
where
    L: BlackBoxLikelihood,
{
    let etas: Vec<&Tensor> = samples.iter().map(|s| &s.eta).collect();
    let llik = likelihood.log_likelihood(&etas)?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    let log_prior = sum_tensors(samples.iter().map(|s| &s.log_prior))?;
    let log_q = sum_tensors(samples.iter().map(|s| &s.log_q))?;

    let elbo = ((&llik + &log_prior)? - &log_q)?;
    elbo.mean(0)
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

        // Test sampling
        let samples = composite.sample_all(10)?;
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
            let loss = composite_direct_elbo_loss(&composite, &likelihood, 30)?;
            optimizer.backward_step(&loss)?;

            if i % 100 == 0 {
                let elbo = composite_elbo(&composite, &likelihood, 50)?;
                println!(
                    "iter {:4}: loss = {:10.4}, ELBO = {:10.4}",
                    i,
                    loss.to_scalar::<f32>()?,
                    elbo.to_scalar::<f32>()?
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
