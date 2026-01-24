use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::traits::VariationalDistribution;

/// Susie (Sum of Single Effects) variational distribution.
///
/// θ = Σ_l (α_l ⊙ β_l) where:
/// - α_l = softmax(logits_l, dim=p) - selection probabilities, shape (p, k)
/// - β_l ~ N(μ_l, σ_l²) - effect sizes, shape (p, k)
///
/// Each output dimension k has its own independent feature selection.
/// This can be used with LinearModelSGVB for Susie regression.
pub struct SusieVar {
    /// Selection logits, shape (L, p, k)
    logits: Tensor,
    /// Effect size means, shape (L, p, k)
    beta_mean: Tensor,
    /// Effect size log-stds, shape (L, p, k)
    beta_ln_std: Tensor,
    /// Number of components L
    num_components: usize,
}

impl SusieVar {
    /// Create a new Susie variational distribution.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `num_components` - Number of single-effect components L
    /// * `p` - Number of features
    /// * `k` - Number of output dimensions
    pub fn new(vb: VarBuilder, num_components: usize, p: usize, k: usize) -> Result<Self> {
        // Initialize logits to uniform (zeros -> equal softmax probs)
        // Shape: (L, p, k) - separate selection for each output dimension
        let logits = vb.get_with_hints(
            (num_components, p, k),
            "logits",
            candle_nn::Init::Const(0.0),
        )?;

        // Initialize beta_mean to small random values
        let beta_mean = vb.get_with_hints(
            (num_components, p, k),
            "beta_mean",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 },
        )?;

        // Initialize beta_ln_std to 0 (std = 1)
        let beta_ln_std = vb.get_with_hints(
            (num_components, p, k),
            "beta_ln_std",
            candle_nn::Init::Const(0.0),
        )?;

        Ok(Self {
            logits,
            beta_mean,
            beta_ln_std,
            num_components,
        })
    }

    /// Get selection probabilities α = softmax(logits, dim=1) for each component.
    /// Softmax is applied over the p dimension for each (l, k) pair.
    ///
    /// # Returns
    /// Selection probabilities, shape (L, p, k)
    pub fn alpha(&self) -> Result<Tensor> {
        // softmax over dim 1 (the p dimension)
        candle_nn::ops::softmax(&self.logits, 1)
    }

    /// Get log selection probabilities log(α) = log_softmax(logits, dim=1).
    ///
    /// # Returns
    /// Log selection probabilities, shape (L, p, k)
    pub fn log_alpha(&self) -> Result<Tensor> {
        candle_nn::ops::log_softmax(&self.logits, 1)
    }

    /// Get the posterior inclusion probabilities (PIPs) for each feature and output.
    /// PIP_{j,k} = 1 - Π_l (1 - α_{l,j,k})
    ///
    /// # Returns
    /// PIPs, shape (p, k)
    pub fn pip(&self) -> Result<Tensor> {
        let alpha = self.alpha()?; // (L, p, k)
        let one_minus_alpha = (1.0 - &alpha)?;
        let log_one_minus_alpha = one_minus_alpha.log()?;
        let sum_log = log_one_minus_alpha.sum(0)?; // (p, k)
        let prod = sum_log.exp()?;
        1.0 - prod
    }

    /// Get effect size means per component.
    ///
    /// # Returns
    /// Effect size means, shape (L, p, k)
    pub fn beta_mean(&self) -> &Tensor {
        &self.beta_mean
    }

    /// Get effect size standard deviations per component.
    ///
    /// # Returns
    /// Effect size stds, shape (L, p, k)
    pub fn beta_std(&self) -> Result<Tensor> {
        self.beta_ln_std.exp()
    }

    /// Get the device of the parameters.
    pub fn device(&self) -> &Device {
        self.logits.device()
    }

    /// Get the dtype of the parameters.
    pub fn dtype(&self) -> DType {
        self.logits.dtype()
    }

    /// Get number of components L.
    pub fn num_components(&self) -> usize {
        self.num_components
    }
}

impl VariationalDistribution for SusieVar {
    /// Sample θ = Σ_l (α_l ⊙ β_l) using reparameterization for β.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples S
    ///
    /// # Returns
    /// (theta, epsilon) where:
    /// - theta: shape (S, p, k) - combined effect
    /// - epsilon: shape (S, L, p, k) - noise used for β sampling
    fn sample(&self, num_samples: usize) -> Result<(Tensor, Tensor)> {
        let (l, p, k) = self.beta_mean.dims3()?;
        let device = self.beta_mean.device();
        let dtype = self.beta_mean.dtype();

        // 1. Get α = softmax(logits, dim=1): (L, p, k)
        let alpha = self.alpha()?;

        // 2. Sample ε ~ N(0, 1): (S, L, p, k)
        let epsilon = Tensor::randn(0f32, 1f32, (num_samples, l, p, k), device)?.to_dtype(dtype)?;

        // 3. β = μ + σ * ε: (S, L, p, k)
        let std = self.beta_ln_std.exp()?;
        let beta = self.beta_mean.unsqueeze(0)?.broadcast_add(&epsilon.broadcast_mul(&std)?)?;

        // 4. θ_l = α_l ⊙ β_l: (S, L, p, k)
        // α: (L, p, k) -> (1, L, p, k)
        let alpha_expanded = alpha.unsqueeze(0)?;
        let theta_l = alpha_expanded.broadcast_mul(&beta)?;

        // 5. θ = Σ_l θ_l: (S, p, k)
        let theta = theta_l.sum(1)?;

        Ok((theta, epsilon))
    }

    /// Compute log q(θ) including both β entropy and selection entropy.
    ///
    /// This includes:
    /// 1. Negative entropy of β: -H[q(β)]
    /// 2. Negative entropy of selection: Σ α * log(α)
    ///
    /// Including the selection entropy allows REINFORCE gradients to flow to logits.
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        let s = theta.dim(0)?;
        let device = theta.device();
        let dtype = theta.dtype();

        // 1. Negative entropy of β: -H[q(β)] = -0.5 * Σ (1 + ln(2π) + 2*ln(σ))
        let ln_2pi = Tensor::new((2.0 * std::f64::consts::PI).ln(), device)?.to_dtype(dtype)?;
        let two_ln_std = (&self.beta_ln_std * 2.0)?;
        let neg_entropy_beta = ((two_ln_std.broadcast_add(&ln_2pi)? + 1.0)? * (-0.5))?;
        let neg_entropy_beta_sum = neg_entropy_beta.sum(2)?.sum(1)?.sum(0)?;

        // 2. Negative entropy of selection: -H[α] = Σ α * log(α)
        // This term makes gradients flow to logits via REINFORCE
        let log_alpha = self.log_alpha()?; // (L, p, k)
        let alpha = self.alpha()?; // (L, p, k)
        let neg_entropy_alpha = (&alpha * &log_alpha)?.sum(2)?.sum(1)?.sum(0)?;

        // Combined negative entropy
        let neg_entropy = (&neg_entropy_beta_sum + &neg_entropy_alpha)?;

        // Broadcast to (S,)
        neg_entropy.broadcast_as((s,))
    }

    /// Get the mean of θ: E[θ] = Σ_l (α_l ⊙ μ_l)
    fn mean(&self) -> Result<Tensor> {
        self.theta_mean()
    }

    /// Get the variance of θ.
    /// Var[θ_{j,k}] = Σ_l [α_{l,j,k} * (σ²_{l,j,k} + μ²_{l,j,k}) - (α_{l,j,k} * μ_{l,j,k})²]
    fn var(&self) -> Result<Tensor> {
        let alpha = self.alpha()?; // (L, p, k)

        let mu = &self.beta_mean; // (L, p, k)
        let sigma_sq = (&self.beta_ln_std * 2.0)?.exp()?; // (L, p, k)

        // E[θ²] = Σ_l α_l * (σ² + μ²)
        let mu_sq = mu.powf(2.0)?;
        let second_moment_l = alpha.broadcast_mul(&(&sigma_sq + &mu_sq)?)?;
        let second_moment = second_moment_l.sum(0)?; // (p, k)

        // E[θ]² = (Σ_l α_l * μ)²
        let first_moment_l = alpha.broadcast_mul(mu)?;
        let first_moment = first_moment_l.sum(0)?; // (p, k)
        let first_moment_sq = first_moment.powf(2.0)?;

        // Var[θ] = E[θ²] - E[θ]²
        second_moment - first_moment_sq
    }
}

impl SusieVar {
    /// Get the actual mean of θ: E[θ] = Σ_l (α_l ⊙ μ_l)
    ///
    /// # Returns
    /// Mean of θ, shape (p, k)
    pub fn theta_mean(&self) -> Result<Tensor> {
        let alpha = self.alpha()?; // (L, p, k)
        let theta_l = alpha.broadcast_mul(&self.beta_mean)?; // (L, p, k)
        theta_l.sum(0) // (p, k)
    }

    /// Compute log q(β) for given β samples.
    ///
    /// # Arguments
    /// * `beta` - β samples, shape (S, L, p, k)
    ///
    /// # Returns
    /// Log probability, shape (S,)
    pub fn log_prob_beta(&self, beta: &Tensor) -> Result<Tensor> {
        let dtype = beta.dtype();
        let device = beta.device();
        let ln_2pi = Tensor::new((2.0 * std::f64::consts::PI).ln(), device)?.to_dtype(dtype)?;

        let std = self.beta_ln_std.exp()?;
        let diff = beta.broadcast_sub(&self.beta_mean)?;
        let normalized_sq = diff.powf(2.0)?.broadcast_div(&std.powf(2.0)?)?;

        let two_ln_std = (&self.beta_ln_std * 2.0)?;
        let log_prob_element =
            (normalized_sq.broadcast_add(&two_ln_std)?.broadcast_add(&ln_2pi)? * (-0.5))?;

        // Sum over L, p, k dimensions -> (S,)
        log_prob_element.sum(3)?.sum(2)?.sum(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_susie_variational_shapes() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let l = 3;
        let p = 20;
        let k = 2;
        let s = 10;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new(vb, l, p, k)?;

        // Check alpha shape - now (L, p, k)
        let alpha = susie.alpha()?;
        assert_eq!(alpha.dims(), &[l, p, k]);

        // Check PIP shape - now (p, k)
        let pip = susie.pip()?;
        assert_eq!(pip.dims(), &[p, k]);

        // Check sample shape
        let (theta, epsilon) = susie.sample(s)?;
        assert_eq!(theta.dims(), &[s, p, k]);
        assert_eq!(epsilon.dims(), &[s, l, p, k]);

        // Check theta_mean shape
        let theta_mean = susie.theta_mean()?;
        assert_eq!(theta_mean.dims(), &[p, k]);

        // Check var shape
        let var = susie.var()?;
        assert_eq!(var.dims(), &[p, k]);

        Ok(())
    }

    #[test]
    fn test_alpha_sums_to_one() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 3;
        let p = 20;
        let k = 2;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new(vb, l, p, k)?;

        let alpha = susie.alpha()?; // (L, p, k)
        let alpha_sum = alpha.sum(1)?; // Sum over p -> (L, k)

        for i in 0..l {
            for j in 0..k {
                let sum: f64 = alpha_sum.get(i)?.get(j)?.to_scalar()?;
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "Alpha should sum to 1 for l={}, k={}, got {}",
                    i, j, sum
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_susie_with_linear_model() -> Result<()> {
        use crate::sgvb::{compute_elbo, sgvb_loss, GaussianPrior, LinearModelSGVB, SGVBConfig};
        use crate::sgvb::traits::BlackBoxLikelihood;
        use candle_core::Tensor;

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 30;
        let p = 10;
        let k = 1;
        let l = 2;

        // Create design matrix and observations
        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let y = Tensor::randn(0f32, 1f32, (n, k), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        // Create Susie variational distribution
        let susie = SusieVar::new(vb.pp("susie"), l, p, k)?;

        // Create prior and config
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        // Combine into generic LinearModelSGVB
        let model = LinearModelSGVB::from_variational(susie, x, prior, config.clone());

        // Simple Gaussian likelihood
        struct GaussianLik { y: Tensor }
        impl BlackBoxLikelihood for GaussianLik {
            fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
                let eta = etas[0];
                let diff_sq = eta.broadcast_sub(&self.y)?.powf(2.0)?;
                let log_prob = (diff_sq * (-0.5))?;
                log_prob.sum(2)?.sum(1)
            }
        }
        let likelihood = GaussianLik { y };

        // Test that sgvb_loss works
        let loss = sgvb_loss(&model, &likelihood, &config)?;
        assert!(loss.dims().is_empty());

        // Test that compute_elbo works
        let elbo = compute_elbo(&model, &likelihood, 10)?;
        assert!(elbo.dims().is_empty());

        // Test model methods
        let eta_mean = model.eta_mean()?;
        assert_eq!(eta_mean.dims(), &[n, k]);

        let coef_mean = model.coef_mean()?;
        assert_eq!(coef_mean.dims(), &[p, k]);

        Ok(())
    }
}
