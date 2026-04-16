use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::traits::{IndependentGateVariational, VariationalDistribution};
use super::variational_susie::smoothed_sigmoid;

/// Spike-and-slab variational distribution with independent per-variable gates.
///
/// θ_j = γ_j · β_j where:
/// - γ_j ~ Bernoulli(π_j) — inclusion gate per variable
/// - β_j ~ N(μ_j, σ²_j) — effect size when included
///
/// Each variable is independently included or excluded. No component
/// structure (unlike SuSiE). Flat only — no multilevel, because
/// soft-collapsing correlated variables cancels their signal.
///
/// Uses affine-smoothed sigmoid for inclusion probabilities:
/// π_j = ε + (1-2ε) · σ(logit_j)
pub struct SpikeSlabVar {
    /// Inclusion logits, shape (p, k)
    inclusion_logits: Tensor,
    /// Effect size means, shape (p, k)
    beta_mean: Tensor,
    /// Effect size log-stds, shape (p, k)
    beta_ln_std: Tensor,
    /// Smoothing epsilon for sigmoid
    epsilon: f64,
}

impl SpikeSlabVar {
    /// Create a new spike-and-slab variational distribution.
    ///
    /// `epsilon` controls sigmoid smoothing (e.g. 0.01): π ∈ [ε, 1-ε].
    pub fn new(vb: VarBuilder, p: usize, k: usize, epsilon: f64) -> Result<Self> {
        let inclusion_logits =
            vb.get_with_hints((p, k), "inclusion_logits", candle_nn::Init::Const(0.0))?;

        let beta_mean = vb.get_with_hints(
            (p, k),
            "beta_mean",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;

        let beta_ln_std = vb.get_with_hints((p, k), "beta_ln_std", candle_nn::Init::Const(0.0))?;

        Ok(Self {
            inclusion_logits,
            beta_mean,
            beta_ln_std,
            epsilon,
        })
    }

    /// Posterior inclusion probabilities = inclusion_prob directly.
    pub fn pip(&self) -> Result<Tensor> {
        self.inclusion_prob()
    }

    /// Inclusion probabilities π = ε + (1-2ε) · σ(logit).
    pub fn inclusion_prob(&self) -> Result<Tensor> {
        smoothed_sigmoid(&self.inclusion_logits, self.epsilon)
    }

    pub fn device(&self) -> &Device {
        self.inclusion_logits.device()
    }

    pub fn dtype(&self) -> DType {
        self.inclusion_logits.dtype()
    }
}

impl VariationalDistribution for SpikeSlabVar {
    /// E[θ_j] = π_j · μ_j
    fn mean(&self) -> Result<Tensor> {
        let pi = self.inclusion_prob()?;
        pi.broadcast_mul(&self.beta_mean)
    }

    /// Var[θ_j] = π_j(σ² + μ²) - (π_j μ_j)²
    fn var(&self) -> Result<Tensor> {
        let pi = self.inclusion_prob()?;
        let mu = &self.beta_mean;
        let sigma_sq = (&self.beta_ln_std * 2.0)?.exp()?;

        let second_moment = pi.broadcast_mul(&(mu.sqr()? + &sigma_sq)?)?;
        let first_moment_sq = pi.broadcast_mul(mu)?.sqr()?;

        (second_moment - first_moment_sq)?.clamp(1e-8, f64::INFINITY)
    }
}

impl IndependentGateVariational for SpikeSlabVar {
    fn inclusion_prob(&self) -> Result<Tensor> {
        self.inclusion_prob()
    }

    fn effect_mean(&self) -> Result<Tensor> {
        Ok(self.beta_mean.clone())
    }

    fn effect_std(&self) -> Result<Tensor> {
        self.beta_ln_std.exp()
    }

    /// KL(Bernoulli(π) || Bernoulli(π₀)) summed over all (p, k).
    /// π₀ = prior_inclusion / p (expected number of inclusions = prior_inclusion * k).
    fn kl_bernoulli(&self, prior_inclusion: f64) -> Result<Tensor> {
        let pi = self.inclusion_prob()?;
        let one_minus_pi = (1.0 - &pi)?;

        let p = self.inclusion_logits.dim(0)? as f64;
        let pi_0 = (prior_inclusion / p).clamp(1e-10, 1.0 - 1e-10);
        let log_pi_0 = pi_0.ln();
        let log_1_minus_pi_0 = (1.0 - pi_0).ln();

        // KL = π(log π - log π₀) + (1-π)(log(1-π) - log(1-π₀))
        let term1 = (&pi * (pi.log()? - log_pi_0)?)?;
        let term2 = (&one_minus_pi * (one_minus_pi.log()? - log_1_minus_pi_0)?)?;
        (term1 + term2)?.sum_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{Optimizer, VarBuilder, VarMap};

    #[test]
    fn test_spike_slab_shapes() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let p = 20;
        let k = 2;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let ss = SpikeSlabVar::new(vb, p, k, 0.01)?;

        assert_eq!(ss.inclusion_prob()?.dims(), &[p, k]);
        assert_eq!(ss.pip()?.dims(), &[p, k]);
        assert_eq!(ss.mean()?.dims(), &[p, k]);
        assert_eq!(ss.var()?.dims(), &[p, k]);

        Ok(())
    }

    #[test]
    fn test_pip_equals_inclusion() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;
        let p = 10;
        let k = 1;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let ss = SpikeSlabVar::new(vb, p, k, 0.01)?;

        let pip = ss.pip()?;
        let inc = ss.inclusion_prob()?;
        let diff: f64 = (pip - inc)?.abs()?.sum_all()?.to_scalar()?;
        assert!(
            diff < 1e-10,
            "PIP should equal inclusion_prob, diff={}",
            diff
        );

        Ok(())
    }

    #[test]
    fn test_inclusion_bounds() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;
        let eps = 0.01;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let ss = SpikeSlabVar::new(vb, 10, 1, eps)?;

        let pi_vals: Vec<f64> = ss.inclusion_prob()?.flatten_all()?.to_vec1()?;
        for &v in &pi_vals {
            assert!(v >= eps && v <= 1.0 - eps, "π={} out of bounds", v);
        }

        Ok(())
    }

    #[test]
    fn test_sparse_recovery() -> Result<()> {
        use crate::sgvb::traits::BlackBoxLikelihood;
        use crate::sgvb::{FixedGaussianPrior, RegressionSGVB, SGVBConfig};

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 200;
        let p = 30;
        let k = 1;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let x_5 = x.narrow(1, 5, 1)?;
        let x_15 = x.narrow(1, 15, 1)?;
        let noise = Tensor::randn(0f32, 0.5f32, (n, k), &device)?;
        let y = ((x_5 * 2.0)? + (x_15 * 1.5)? + noise)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let ss = SpikeSlabVar::new(vb.pp("ss"), p, k, 0.01)?;
        let prior = FixedGaussianPrior::new(1.0);
        let config = SGVBConfig::new(30);
        let model = RegressionSGVB::from_variational(ss, x, prior, config);

        struct GaussianLik {
            y: Tensor,
        }
        impl BlackBoxLikelihood for GaussianLik {
            fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
                let eta = etas[0];
                let diff_sq = eta.broadcast_sub(&self.y)?.sqr()?;
                (diff_sq * (-0.5))?.sum(2)?.sum(1)
            }
        }
        let likelihood = GaussianLik { y };

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.05)?;

        for _ in 0..500 {
            let loss = crate::sgvb::local_reparam_loss(&model, &likelihood, 30, 1.0)?;
            optimizer.backward_step(&loss)?;
        }

        let pip = model.variational.pip()?;
        let pip_5: f32 = pip.get(5)?.get(0)?.to_scalar()?;
        let pip_15: f32 = pip.get(15)?.get(0)?.to_scalar()?;

        let mut other_sum = 0.0f32;
        for j in 0..p {
            if j != 5 && j != 15 {
                other_sum += pip.get(j)?.get(0)?.to_scalar::<f32>()?;
            }
        }
        let other_mean = other_sum / (p - 2) as f32;

        println!("Spike-slab PIPs:");
        println!("  PIP[5]  = {:.4}", pip_5);
        println!("  PIP[15] = {:.4}", pip_15);
        println!("  Others mean = {:.4}", other_mean);

        assert!(
            pip_5 > other_mean * 2.0,
            "PIP[5] ({:.4}) should be > 2x others ({:.4})",
            pip_5,
            other_mean
        );
        assert!(
            pip_15 > other_mean * 2.0,
            "PIP[15] ({:.4}) should be > 2x others ({:.4})",
            pip_15,
            other_mean
        );

        Ok(())
    }
}
