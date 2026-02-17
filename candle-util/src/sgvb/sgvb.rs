use candle_core::{Result, Tensor};

use super::regression_linear::LinearModelSGVB;
use super::traits::{AnalyticalKL, BlackBoxLikelihood, Prior, VariationalDistribution};

/// Configuration for SGVB estimator.
#[derive(Debug, Clone)]
pub struct SGVBConfig {
    /// Number of Monte Carlo samples S for gradient estimation
    pub num_samples: usize,
    /// KL divergence weight β for annealing (0 = no KL, 1 = full ELBO)
    pub kl_weight: f64,
}

impl Default for SGVBConfig {
    fn default() -> Self {
        Self {
            num_samples: 10,
            kl_weight: 1.0,
        }
    }
}

impl SGVBConfig {
    /// Create a new SGVB configuration.
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            kl_weight: 1.0,
        }
    }
}

/// Compute loss using the local reparameterization trick.
///
/// Instead of sampling θ in p-space (which accumulates O(p) variance),
/// samples η directly in n-space where variance is controlled.
///
/// loss = -E[log p(y|η)] + β·KL(q‖p)
///
/// where KL is computed analytically, not via MC samples.
pub fn local_reparam_loss<V, P, L>(
    model: &LinearModelSGVB<V, P>,
    likelihood: &L,
    num_samples: usize,
    kl_weight: f64,
) -> Result<Tensor>
where
    V: VariationalDistribution,
    P: Prior + AnalyticalKL,
    L: BlackBoxLikelihood,
{
    let sample = model.local_reparam_sample(num_samples)?;
    let llik = likelihood.log_likelihood(&[&sample.eta])?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    // ELBO = E[log p(y|η)] − β·KL
    let elbo = (llik.mean(0)? - (sample.kl * kl_weight)?)?;
    elbo.neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::{GaussianPrior, LinearRegressionSGVB};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Optimizer, VarBuilder, VarMap};

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
    fn test_local_reparam_loss_recovery() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 150;
        let p = 50;
        let k = 1;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        // y = X[:,0] * 2.0 + noise
        let true_coef = 2.0f64;
        let x_first = x.narrow(1, 0, 1)?;
        let noise = Tensor::randn(0f32, 0.5f32, (n, k), &device)?;
        let y = (x_first * true_coef)?.add(&noise)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let likelihood = GaussianLikelihood::new(y, 0.5);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::new(50);
        let model = LinearRegressionSGVB::new(vb.pp("model"), x, k, prior, config)?;

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.01)?;

        for i in 0..200 {
            let loss = local_reparam_loss(&model, &likelihood, 50, 1.0)?;
            optimizer.backward_step(&loss)?;

            if i % 50 == 0 {
                let loss_val: f32 = loss.to_scalar()?;
                println!("local_reparam iter {}: loss = {:.4}", i, loss_val);
            }
        }

        let coef_mean = model.coef_mean()?;
        let coef_first: f32 = coef_mean.get(0)?.get(0)?.to_scalar()?;

        let mut other_sum = 0.0f32;
        for i in 1..p {
            let val: f32 = coef_mean.get(i)?.get(0)?.to_scalar()?;
            other_sum += val.abs();
        }
        let other_mean = other_sum / (p - 1) as f32;

        println!("\nLocal reparam coefficients:");
        println!("  First (true={:.1}): {:.4}", true_coef, coef_first);
        println!("  Others mean abs: {:.4}", other_mean);

        assert!(
            coef_first > 1.0,
            "First coef should be > 1.0, got {}",
            coef_first
        );
        assert!(
            coef_first.abs() > other_mean * 2.0,
            "First coef should dominate others"
        );

        Ok(())
    }
}
