use candle_core::{Result, Tensor};

use super::traits::{BlackBoxLikelihood, Prior, VariationalDistribution};

/// Configuration for SGVB estimator.
#[derive(Debug, Clone)]
pub struct SGVBConfig {
    /// Number of Monte Carlo samples S for gradient estimation
    pub num_samples: usize,
    /// Whether to normalize rewards (control variate)
    pub normalize: bool,
}

impl Default for SGVBConfig {
    fn default() -> Self {
        Self {
            num_samples: 10,
            normalize: true,
        }
    }
}

impl SGVBConfig {
    /// Create a new SGVB configuration.
    pub fn new(num_samples: usize, normalize: bool) -> Self {
        Self { num_samples, normalize }
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
/// * `variational` - Variational distribution q(θ)
/// * `likelihood` - Black-box likelihood p(y|η)
/// * `prior` - Prior distribution p(θ)
/// * `x_design` - Design matrix X, shape (n, p)
/// * `config` - SGVB configuration
///
/// # Returns
/// Surrogate loss (scalar) that when differentiated gives REINFORCE gradients
pub fn sgvb_loss<L, P, V>(
    variational: &V,
    likelihood: &L,
    prior: &P,
    x_design: &Tensor,
    config: &SGVBConfig,
) -> Result<Tensor>
where
    L: BlackBoxLikelihood,
    P: Prior,
    V: VariationalDistribution,
{
    // 1. Sample θ: (S, p, k) from q(θ)
    let (theta, _epsilon) = variational.sample(config.num_samples)?;

    // 2. Compute η = X @ θ: (S, n, k)
    // X is (n, p), θ is (S, p, k)
    // We need to batch matmul: unsqueeze X to (1, n, p), then matmul with (S, p, k) -> (S, n, k)
    let eta = x_design.unsqueeze(0)?.broadcast_matmul(&theta)?;

    // 3. Compute ELBO components (all should be detached for reward computation)
    // log p(y|η): shape (S,) or (S, n) - we'll sum to (S,)
    let llik = likelihood.log_likelihood(&[&eta])?;
    let llik = if llik.rank() > 1 {
        llik.sum(1)?  // Sum over observations if per-observation
    } else {
        llik
    };

    // log p(θ): shape (S,)
    let log_prior = prior.log_prob(&theta)?;

    // log q(θ): shape (S,)
    let log_q = variational.log_prob(&theta)?;

    // 4. Reward = log p(y|η) + log p(θ) - log q(θ) (ELBO components)
    let reward = ((&llik + &log_prior)? - &log_q)?;

    // 5. Normalize reward (control variate) if configured
    let reward_norm = if config.normalize {
        let mean = reward.mean(0)?;
        let var = reward.var(0)?;
        let std = (var + 1e-8)?.sqrt()?;
        reward.broadcast_sub(&mean)?.broadcast_div(&std)?
    } else {
        reward
    };

    // 6. DETACH reward from computation graph
    // The reward should not contribute gradients - only log_q should
    let reward_detached = reward_norm.detach();

    // 7. Recompute log_q WITH gradient tracking
    // We detach theta so gradients flow only through log_q's parameters
    let log_q_grad = variational.log_prob(&theta.detach())?;

    // 8. Surrogate loss = -mean(reward * log_q)
    // Minimizing this surrogate loss maximizes the ELBO via REINFORCE
    let surrogate_loss = (&reward_detached * &log_q_grad)?.mean(0)?.neg()?;

    Ok(surrogate_loss)
}

/// Compute the raw ELBO (for monitoring, not for gradients).
///
/// ELBO = E_q[log p(y|η) + log p(θ) - log q(θ)]
///
/// # Returns
/// Mean ELBO estimate over samples (scalar)
pub fn compute_elbo<L, P, V>(
    variational: &V,
    likelihood: &L,
    prior: &P,
    x_design: &Tensor,
    num_samples: usize,
) -> Result<Tensor>
where
    L: BlackBoxLikelihood,
    P: Prior,
    V: VariationalDistribution,
{
    let (theta, _) = variational.sample(num_samples)?;

    let eta = x_design.unsqueeze(0)?.broadcast_matmul(&theta)?;

    let llik = likelihood.log_likelihood(&[&eta])?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    let log_prior = prior.log_prob(&theta)?;
    let log_q = variational.log_prob(&theta)?;

    let elbo = ((&llik + &log_prior)? - &log_q)?;
    elbo.mean(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::{GaussianPrior, GaussianVariational};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    /// Simple Gaussian likelihood for testing
    struct GaussianLikelihood {
        y: Tensor,       // observations (n, k)
        ln_sigma: f64,   // log noise std
    }

    impl GaussianLikelihood {
        fn new(y: Tensor, sigma: f64) -> Self {
            Self { y, ln_sigma: sigma.ln() }
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

            let diff_sq = eta.broadcast_sub(&self.y)?.powf(2.0)?;
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

        // Create variational distribution
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let variational = GaussianVariational::new(vb.pp("var"), p, k)?;

        // Create prior
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;

        // Create likelihood
        let likelihood = GaussianLikelihood::new(y, 1.0);

        // Compute loss
        let config = SGVBConfig::default();
        let loss = sgvb_loss(&variational, &likelihood, &prior, &x, &config)?;

        assert!(loss.dims().is_empty());  // Scalar

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
        let variational = GaussianVariational::new(vb.pp("var"), p, k)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let likelihood = GaussianLikelihood::new(y, 1.0);

        let elbo = compute_elbo(&variational, &likelihood, &prior, &x, 100)?;

        assert!(elbo.dims().is_empty());  // Scalar

        Ok(())
    }
}
