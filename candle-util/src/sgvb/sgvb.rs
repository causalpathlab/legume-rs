use candle_core::{Result, Tensor};

use super::traits::{BlackBoxLikelihood, SgvbModel};

/// Configuration for SGVB estimator.
#[derive(Debug, Clone)]
pub struct SGVBConfig {
    /// Number of Monte Carlo samples S for gradient estimation
    pub num_samples: usize,
}

impl Default for SGVBConfig {
    fn default() -> Self {
        Self { num_samples: 10 }
    }
}

impl SGVBConfig {
    /// Create a new SGVB configuration.
    pub fn new(num_samples: usize) -> Self {
        Self { num_samples }
    }
}

/// Compute the SGVB surrogate loss using score function (REINFORCE) estimator.
///
/// The ELBO is: E_q[log p(y|η) + log p(θ) - log q(θ)]
///
/// We use the score function gradient estimator:
/// ∇ELBO ≈ E[(normalized_reward) * ∇log q(θ)]
///
/// With control variate: reward = (reward - mean) / std
///
/// # Arguments
/// * `model` - SGVB model that provides samples and log probabilities
/// * `likelihood` - Black-box likelihood p(y|η)
/// * `config` - SGVB configuration
///
/// # Returns
/// Surrogate loss (scalar) that when differentiated gives REINFORCE gradients
pub fn sgvb_loss<M, L>(model: &M, likelihood: &L, config: &SGVBConfig) -> Result<Tensor>
where
    M: SgvbModel,
    L: BlackBoxLikelihood,
{
    // 1. Sample from model: get η, log p(θ), log q(θ), and log q(θ) with gradients
    let sample = model.sample(config.num_samples)?;

    // 2. Compute log likelihood: log p(y|η)
    let llik = likelihood.log_likelihood(&[&sample.eta])?;
    let llik = if llik.rank() > 1 {
        llik.sum(1)? // Sum over observations if per-observation
    } else {
        llik
    };

    // 3. Reward = log p(y|η) + log p(θ) - log q(θ) (ELBO components)
    let reward = ((&llik + &sample.log_prior)? - &sample.log_q)?;

    // 4. Normalize reward (control variate)
    let mean = reward.mean(0)?;
    let var = reward.var(0)?;
    let std = (var + 1e-8)?.sqrt()?;
    let reward_norm = reward.broadcast_sub(&mean)?.broadcast_div(&std)?;

    // 5. DETACH reward from computation graph
    let reward_detached = reward_norm.detach();

    // 6. Surrogate loss = -mean(reward * log_q_grad)
    // Minimizing this surrogate loss maximizes the ELBO via REINFORCE
    let surrogate_loss = (&reward_detached * &sample.log_q_grad)?.mean(0)?.neg()?;

    Ok(surrogate_loss)
}

/// Compute the raw ELBO (for monitoring, not for gradients).
///
/// ELBO = E_q[log p(y|η) + log p(θ) - log q(θ)]
///
/// # Returns
/// Mean ELBO estimate over samples (scalar)
pub fn compute_elbo<M, L>(model: &M, likelihood: &L, num_samples: usize) -> Result<Tensor>
where
    M: SgvbModel,
    L: BlackBoxLikelihood,
{
    let sample = model.sample(num_samples)?;

    let llik = likelihood.log_likelihood(&[&sample.eta])?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    let elbo = ((&llik + &sample.log_prior)? - &sample.log_q)?;
    elbo.mean(0)
}

/// Compute direct ELBO loss with reparameterization gradients.
///
/// Unlike `sgvb_loss` which uses REINFORCE (score function estimator),
/// this computes -ELBO directly and allows gradients to flow through
/// the entire computation graph via reparameterization.
///
/// Use this for models like Susie where you need gradients to flow
/// through deterministic transformations (e.g., softmax for selection).
///
/// # Arguments
/// * `model` - SGVB model that provides samples and log probabilities
/// * `likelihood` - Black-box likelihood p(y|η)
/// * `num_samples` - Number of Monte Carlo samples
///
/// # Returns
/// Negative ELBO (scalar) - minimize this to maximize ELBO
pub fn direct_elbo_loss<M, L>(model: &M, likelihood: &L, num_samples: usize) -> Result<Tensor>
where
    M: SgvbModel,
    L: BlackBoxLikelihood,
{
    let sample = model.sample(num_samples)?;

    let llik = likelihood.log_likelihood(&[&sample.eta])?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    // ELBO = log_lik + log_prior - log_q
    let elbo = ((&llik + &sample.log_prior)? - &sample.log_q)?;

    // Return negative mean ELBO as loss
    elbo.mean(0)?.neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::{GaussianPrior, LinearRegressionSGVB};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    /// Simple Gaussian likelihood for testing
    struct GaussianLikelihood {
        y: Tensor,     // observations (n, k)
        ln_sigma: f64, // log noise std
    }

    impl GaussianLikelihood {
        fn new(y: Tensor, sigma: f64) -> Self {
            Self {
                y,
                ln_sigma: sigma.ln(),
            }
        }
    }

    impl BlackBoxLikelihood for GaussianLikelihood {
        fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
            // etas[0]: (S, n, k), y: (n, k)
            // log N(y; eta, sigma^2) = -0.5 * [(y - eta)^2 / sigma^2 + 2*ln(sigma) + ln(2pi)]
            let eta = etas[0];
            let sigma_sq = (2.0 * self.ln_sigma).exp();
            let ln_2pi = (2.0 * std::f64::consts::PI).ln();
            let const_term = 2.0 * self.ln_sigma + ln_2pi;

            let diff_sq = eta.broadcast_sub(&self.y)?.sqr()?;
            let log_prob = (((diff_sq / sigma_sq)? + const_term)? * (-0.5))?;

            // Sum over (n, k) dimensions, return (S,)
            log_prob.sum(2)?.sum(1)
        }
    }

    #[test]
    fn test_sgvb_loss_runs() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 20;
        let p = 5;
        let k = 2;

        // Create design matrix
        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        // Create observations
        let y = Tensor::randn(0f32, 1f32, (n, k), &device)?;

        // Create model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();
        let model = LinearRegressionSGVB::new(vb.pp("model"), x, k, prior, config.clone())?;

        // Create likelihood
        let likelihood = GaussianLikelihood::new(y, 1.0);

        // Compute loss
        let loss = sgvb_loss(&model, &likelihood, &config)?;

        assert!(loss.dims().is_empty()); // Scalar

        Ok(())
    }

    #[test]
    fn test_elbo_computation() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 20;
        let p = 5;
        let k = 2;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let y = Tensor::randn(0f32, 1f32, (n, k), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();
        let model = LinearRegressionSGVB::new(vb.pp("model"), x, k, prior, config)?;
        let likelihood = GaussianLikelihood::new(y, 1.0);

        let elbo = compute_elbo(&model, &likelihood, 100)?;

        assert!(elbo.dims().is_empty()); // Scalar

        Ok(())
    }
}
