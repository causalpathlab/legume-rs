use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::susie_util::{kl_categorical_uniform, pip_from_alpha};
use super::traits::{ComponentVariational, VariationalDistribution};
use super::variational_susie::smoothed_sigmoid;

/// Bi-directional Susie (Sum of Single Effects) variational distribution.
///
/// θ[p,k] = Σ_l α_p[l,p] * π_k[l,k] * β[l,p,k]
///
/// where:
/// - α_p[l,:] = softmax over predictors (dim P) — single-effect selection with null
/// - π_k[l,:] = smoothed sigmoid per outcome (dim K) — independent trait inclusion
/// - β[l,p,k] ~ N(μ[l,p,k], σ[l,p,k]²) — per-feature effect size
///
/// The predictor axis uses softmax (enforcing single-effect per component),
/// while the outcome axis uses independent sigmoid gates (allowing each component
/// to affect multiple traits). This avoids outer-product PIP dilution: a causal
/// SNP can have PIP close to 1 on each affected trait independently.
pub struct BiSusieVar {
    /// Selection logits for predictors, shape (L, P) or (L, P+1) with null
    logits_predictor: Tensor,
    /// Inclusion logits for outcomes, shape (L, K) — independent sigmoid gates
    logits_outcome: Tensor,
    /// Effect size means per component, shape (L, P, K)
    beta_mean: Tensor,
    /// Effect size log-stds per component, shape (L, P, K)
    beta_ln_std: Tensor,
    /// Number of components L
    num_components: usize,
    /// Number of real predictors P (excluding null)
    num_predictors: usize,
    /// Number of outcomes K
    num_outcomes: usize,
    /// Whether null absorber is appended to the predictor softmax
    has_null: bool,
    /// Smoothing epsilon for outcome sigmoid: π ∈ [ε, 1-ε]
    gate_epsilon: f64,
}

impl BiSusieVar {
    /// Create a new BiSusie variational distribution (no null on predictor axis).
    pub fn new(
        vb: VarBuilder,
        num_components: usize,
        num_predictors: usize,
        num_outcomes: usize,
    ) -> Result<Self> {
        Self::new_inner(vb, num_components, num_predictors, num_outcomes, false)
    }

    /// Create a new BiSusie with null absorber on the predictor softmax.
    ///
    /// The predictor axis gets a (P+1)th null position (same as SusieVar).
    /// The outcome axis uses independent sigmoid gates — no null needed
    /// since each gate can independently go to ε ≈ 0.
    pub fn new_with_null(
        vb: VarBuilder,
        num_components: usize,
        num_predictors: usize,
        num_outcomes: usize,
    ) -> Result<Self> {
        Self::new_inner(vb, num_components, num_predictors, num_outcomes, true)
    }

    fn new_inner(
        vb: VarBuilder,
        num_components: usize,
        num_predictors: usize,
        num_outcomes: usize,
        has_null: bool,
    ) -> Result<Self> {
        let p_logits = num_predictors + has_null as usize;

        let logits_predictor = vb.get_with_hints(
            (num_components, p_logits),
            "logits_predictor",
            candle_nn::Init::Const(0.0),
        )?;

        let logits_outcome = vb.get_with_hints(
            (num_components, num_outcomes),
            "logits_outcome",
            candle_nn::Init::Const(0.0),
        )?;

        let beta_mean = vb.get_with_hints(
            (num_components, num_predictors, num_outcomes),
            "beta_mean",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;

        let beta_ln_std = vb.get_with_hints(
            (num_components, num_predictors, num_outcomes),
            "beta_ln_std",
            candle_nn::Init::Const(0.0),
        )?;

        Ok(Self {
            logits_predictor,
            logits_outcome,
            beta_mean,
            beta_ln_std,
            num_components,
            num_predictors,
            num_outcomes,
            has_null,
            gate_epsilon: 0.01,
        })
    }

    /// Predictor log-selection probabilities log(α_p), shape (L, P).
    /// Softmax with null excluded when present.
    pub fn log_alpha_predictor(&self) -> Result<Tensor> {
        let full = candle_nn::ops::log_softmax(&self.logits_predictor, 1)?;
        if self.has_null {
            full.narrow(1, 0, self.num_predictors)
        } else {
            Ok(full)
        }
    }

    /// Predictor selection probabilities α_p, shape (L, P).
    pub fn alpha_predictor(&self) -> Result<Tensor> {
        self.log_alpha_predictor()?.exp()
    }

    /// Outcome inclusion probabilities π_k, shape (L, K).
    /// Independent smoothed sigmoid per outcome.
    pub fn pi_outcome(&self) -> Result<Tensor> {
        smoothed_sigmoid(&self.logits_outcome, self.gate_epsilon)
    }

    /// Joint selection weights for each (predictor, outcome) pair per component.
    /// α_p[l,p] * π_k[l,k], shape (L, P, K).
    pub fn alpha_joint(&self) -> Result<Tensor> {
        let alpha_p = self.alpha_predictor()?; // (L, P)
        let pi_k = self.pi_outcome()?; // (L, K)
        alpha_p.unsqueeze(2)?.broadcast_mul(&pi_k.unsqueeze(1)?)
    }

    /// Posterior inclusion probabilities for each (predictor, outcome) pair.
    /// PIP[p,k] = 1 - Π_l (1 - α_p[l,p] * π_k[l,k])
    /// Returns shape (P, K).
    pub fn pip(&self) -> Result<Tensor> {
        pip_from_alpha(&self.alpha_joint()?)
    }

    pub fn beta_mean(&self) -> &Tensor {
        &self.beta_mean
    }

    pub fn beta_std(&self) -> Result<Tensor> {
        self.beta_ln_std.exp()
    }

    pub fn device(&self) -> &Device {
        self.logits_predictor.device()
    }

    pub fn dtype(&self) -> DType {
        self.logits_predictor.dtype()
    }

    pub fn num_components(&self) -> usize {
        self.num_components
    }

    pub fn num_predictors(&self) -> usize {
        self.num_predictors
    }

    pub fn num_outcomes(&self) -> usize {
        self.num_outcomes
    }

    /// E[θ[p,k]] = Σ_l α_p[l,p] * π_k[l,k] * μ[l,p,k]
    pub fn theta_mean(&self) -> Result<Tensor> {
        let joint = self.alpha_joint()?;
        joint.broadcast_mul(&self.beta_mean)?.sum(0)
    }

    /// Selection KL: categorical on predictor axis + Bernoulli on outcome axis.
    ///
    /// Predictor: KL(softmax(logits_p) || Uniform(P_logits))
    /// Outcome: Σ_l Σ_k KL(Bernoulli(π_k[l,k]) || Bernoulli(π₀))
    ///   where π₀ = prior_alpha / K
    pub fn kl_categorical(&self, prior_alpha: f64) -> Result<Tensor> {
        // Predictor axis: categorical KL (includes null if present)
        let kl_p = kl_categorical_uniform(&self.logits_predictor, prior_alpha)?;

        // Outcome axis: Bernoulli KL per (component, outcome)
        let pi = self.pi_outcome()?; // (L, K)
        let one_minus_pi = (1.0 - &pi)?;
        let k = self.num_outcomes as f64;
        let pi_0 = (prior_alpha / k).clamp(1e-10, 1.0 - 1e-10);
        let log_pi_0 = pi_0.ln();
        let log_1_minus_pi_0 = (1.0 - pi_0).ln();
        let term1 = (&pi * (pi.log()? - log_pi_0)?)?;
        let term2 = (&one_minus_pi * (one_minus_pi.log()? - log_1_minus_pi_0)?)?;
        let kl_k = (term1 + term2)?.sum_all()?;

        kl_p + kl_k
    }
}

impl ComponentVariational for BiSusieVar {
    fn alpha(&self) -> Result<Tensor> {
        self.alpha_joint()
    }
    fn beta_mean(&self) -> Result<Tensor> {
        Ok(self.beta_mean.clone())
    }
    fn beta_std(&self) -> Result<Tensor> {
        self.beta_std()
    }
    fn num_components(&self) -> usize {
        self.num_components
    }
}

impl VariationalDistribution for BiSusieVar {
    fn mean(&self) -> Result<Tensor> {
        self.theta_mean()
    }

    fn var(&self) -> Result<Tensor> {
        let joint = self.alpha_joint()?; // (L, P, K)

        let mu = &self.beta_mean;
        let sigma_sq = (&self.beta_ln_std * 2.0)?.exp()?;

        let mu_sq = mu.sqr()?;
        let second_moment_term = (&sigma_sq + &mu_sq)?;
        let second_moment = joint.broadcast_mul(&second_moment_term)?.sum(0)?;

        let first_moment = joint.broadcast_mul(mu)?.sum(0)?;
        let first_moment_sq = first_moment.sqr()?;

        (second_moment - first_moment_sq)?.clamp(1e-8, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_bisusie_with_linear_model() -> Result<()> {
        use crate::sgvb::traits::BlackBoxLikelihood;
        use crate::sgvb::{local_reparam_loss, GaussianPrior, RegressionSGVB, SGVBConfig};

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 50;
        let p = 6;
        let k = 8;
        let l = 3;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let y = Tensor::randn(0f32, 1f32, (n, k), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let bisusie = BiSusieVar::new(vb.pp("bisusie"), l, p, k)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = RegressionSGVB::from_variational(bisusie, x, prior, config);

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

        let loss = local_reparam_loss(&model, &likelihood, 10, 1.0)?;
        assert!(loss.dims().is_empty());

        let eta_mean = model.eta_mean()?;
        assert_eq!(eta_mean.dims(), &[n, k]);

        let coef_mean = model.coef_mean()?;
        assert_eq!(coef_mean.dims(), &[p, k]);

        let pip = model.variational.pip()?;
        assert_eq!(pip.dims(), &[p, k]);

        Ok(())
    }

    #[test]
    fn test_bisusie_shapes() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let l = 5;
        let p = 8;
        let k = 20;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let bisusie = BiSusieVar::new(vb, l, p, k)?;

        assert_eq!(bisusie.alpha_predictor()?.dims(), &[l, p]);
        assert_eq!(bisusie.pi_outcome()?.dims(), &[l, k]);
        assert_eq!(bisusie.alpha_joint()?.dims(), &[l, p, k]);
        assert_eq!(bisusie.pip()?.dims(), &[p, k]);

        assert_eq!(bisusie.theta_mean()?.dims(), &[p, k]);
        assert_eq!(bisusie.var()?.dims(), &[p, k]);

        Ok(())
    }

    #[test]
    fn test_predictor_alpha_sums_to_one() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 3;
        let p = 8;
        let k = 20;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let bisusie = BiSusieVar::new(vb, l, p, k)?;

        let alpha_p_sum = bisusie.alpha_predictor()?.sum(1)?;
        for i in 0..l {
            let sum_p: f64 = alpha_p_sum.get(i)?.to_scalar()?;
            assert!((sum_p - 1.0).abs() < 1e-5);
        }

        // Outcome axis uses sigmoid — each π ∈ [ε, 1-ε], does NOT sum to 1
        let pi_k = bisusie.pi_outcome()?;
        let pi_vals: Vec<f64> = pi_k.flatten_all()?.to_vec1()?;
        for &v in &pi_vals {
            assert!((0.01..=0.99).contains(&v), "π_k={} out of bounds", v);
        }

        Ok(())
    }
}
