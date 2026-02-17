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
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
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
                                   // Clamp to avoid log(0) when alpha ≈ 1
        let one_minus_alpha = (1.0 - &alpha)?.clamp(1e-10, 1.0)?;
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
        let mu_sq = mu.sqr()?;
        let second_moment_l = alpha.broadcast_mul(&(&sigma_sq + &mu_sq)?)?;
        let second_moment = second_moment_l.sum(0)?; // (p, k)

        // E[θ]² = (Σ_l α_l * μ)²
        let first_moment_l = alpha.broadcast_mul(mu)?;
        let first_moment = first_moment_l.sum(0)?; // (p, k)
        let first_moment_sq = first_moment.sqr()?;

        // Var[θ] = E[θ²] - E[θ]², clamped to avoid negative values from numerical precision
        (second_moment - first_moment_sq)?.clamp(1e-8, f64::INFINITY)
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
        // Create scalar constant directly in f32 to avoid Metal F64 conversion issues
        let ln_2pi =
            Tensor::new((2.0 * std::f64::consts::PI).ln() as f32, device)?.to_dtype(dtype)?;

        let std = self.beta_ln_std.exp()?;
        let diff = beta.broadcast_sub(&self.beta_mean)?;
        let normalized_sq = diff.sqr()?.broadcast_div(&std.sqr()?)?;

        let two_ln_std = (&self.beta_ln_std * 2.0)?;
        let log_prob_element = (normalized_sq
            .broadcast_add(&two_ln_std)?
            .broadcast_add(&ln_2pi)?
            * (-0.5))?;

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

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new(vb, l, p, k)?;

        // Check alpha shape - now (L, p, k)
        let alpha = susie.alpha()?;
        assert_eq!(alpha.dims(), &[l, p, k]);

        // Check PIP shape - now (p, k)
        let pip = susie.pip()?;
        assert_eq!(pip.dims(), &[p, k]);

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
                    i,
                    j,
                    sum
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_susie_with_linear_model() -> Result<()> {
        use crate::sgvb::traits::BlackBoxLikelihood;
        use crate::sgvb::{local_reparam_loss, GaussianPrior, LinearModelSGVB, SGVBConfig};
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
        let model = LinearModelSGVB::from_variational(susie, x, prior, config);

        // Simple Gaussian likelihood
        struct GaussianLik {
            y: Tensor,
        }
        impl BlackBoxLikelihood for GaussianLik {
            fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
                let eta = etas[0];
                let diff_sq = eta.broadcast_sub(&self.y)?.sqr()?;
                let log_prob = (diff_sq * (-0.5))?;
                log_prob.sum(2)?.sum(1)
            }
        }
        let likelihood = GaussianLik { y };

        // Test that local_reparam_loss works
        let loss = local_reparam_loss(&model, &likelihood, 10, 1.0)?;
        assert!(loss.dims().is_empty());

        // Test model methods
        let eta_mean = model.eta_mean()?;
        assert_eq!(eta_mean.dims(), &[n, k]);

        let coef_mean = model.coef_mean()?;
        assert_eq!(coef_mean.dims(), &[p, k]);

        Ok(())
    }
}
