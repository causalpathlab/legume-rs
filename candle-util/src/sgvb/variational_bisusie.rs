use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::traits::VariationalDistribution;
use crate::candle_aux_layers::sparsemax;

/// Selection function for BiSusieVar.
#[derive(Clone, Copy, Debug, Default)]
pub enum SelectionFn {
    /// Standard softmax - always positive probabilities
    #[default]
    Softmax,
    /// Sparsemax - can produce exact zeros for cleaner sparsity
    Sparsemax,
}

/// Bi-directional Susie (Sum of Single Effects) variational distribution.
///
/// θ[p,k] = Σ_l α_p[l,p] * α_k[l,k] * β[l]
///
/// where:
/// - α_p[l,:] = softmax/sparsemax over predictors (dim P) - each component selects one predictor
/// - α_k[l,:] = softmax/sparsemax over outcomes (dim K) - each component selects one outcome
/// - β[l] ~ N(μ[l], σ[l]²) - effect size per component
///
/// This creates an outer-product structure where each component l selects
/// a (predictor, outcome) pair, enforcing sparsity in both dimensions.
pub struct BiSusieVar {
    /// Selection logits for predictors, shape (L, P)
    logits_predictor: Tensor,
    /// Selection logits for outcomes, shape (L, K)
    logits_outcome: Tensor,
    /// Effect size means per component, shape (L,)
    beta_mean: Tensor,
    /// Effect size log-stds per component, shape (L,)
    beta_ln_std: Tensor,
    /// Number of components L
    num_components: usize,
    /// Number of predictors P
    num_predictors: usize,
    /// Number of outcomes K
    num_outcomes: usize,
    /// Selection function (softmax or sparsemax)
    selection_fn: SelectionFn,
}

impl BiSusieVar {
    /// Create a new Bi-directional Susie variational distribution with softmax.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `num_components` - Number of single-effect components L
    /// * `num_predictors` - Number of predictors P
    /// * `num_outcomes` - Number of outcomes K
    pub fn new(
        vb: VarBuilder,
        num_components: usize,
        num_predictors: usize,
        num_outcomes: usize,
    ) -> Result<Self> {
        Self::with_selection_fn(vb, num_components, num_predictors, num_outcomes, SelectionFn::Softmax)
    }

    /// Create a new Bi-directional Susie with sparsemax for exact zeros.
    pub fn with_sparsemax(
        vb: VarBuilder,
        num_components: usize,
        num_predictors: usize,
        num_outcomes: usize,
    ) -> Result<Self> {
        Self::with_selection_fn(vb, num_components, num_predictors, num_outcomes, SelectionFn::Sparsemax)
    }

    /// Create a new Bi-directional Susie with specified selection function.
    pub fn with_selection_fn(
        vb: VarBuilder,
        num_components: usize,
        num_predictors: usize,
        num_outcomes: usize,
        selection_fn: SelectionFn,
    ) -> Result<Self> {
        let logits_predictor = vb.get_with_hints(
            (num_components, num_predictors),
            "logits_predictor",
            candle_nn::Init::Const(0.0),
        )?;

        let logits_outcome = vb.get_with_hints(
            (num_components, num_outcomes),
            "logits_outcome",
            candle_nn::Init::Const(0.0),
        )?;

        let beta_mean = vb.get_with_hints(
            (num_components,),
            "beta_mean",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;

        let beta_ln_std = vb.get_with_hints(
            (num_components,),
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
            selection_fn,
        })
    }

    /// Get predictor selection probabilities α_p.
    pub fn alpha_predictor(&self) -> Result<Tensor> {
        match self.selection_fn {
            SelectionFn::Softmax => candle_nn::ops::softmax(&self.logits_predictor, 1),
            SelectionFn::Sparsemax => sparsemax(&self.logits_predictor),
        }
    }

    /// Get outcome selection probabilities α_k.
    pub fn alpha_outcome(&self) -> Result<Tensor> {
        match self.selection_fn {
            SelectionFn::Softmax => candle_nn::ops::softmax(&self.logits_outcome, 1),
            SelectionFn::Sparsemax => sparsemax(&self.logits_outcome),
        }
    }

    /// Get joint selection probabilities for each (predictor, outcome) pair per component.
    /// Returns shape (L, P, K)
    pub fn alpha_joint(&self) -> Result<Tensor> {
        let alpha_p = self.alpha_predictor()?;
        let alpha_k = self.alpha_outcome()?;
        let alpha_p_expanded = alpha_p.unsqueeze(2)?;
        let alpha_k_expanded = alpha_k.unsqueeze(1)?;
        alpha_p_expanded.broadcast_mul(&alpha_k_expanded)
    }

    /// Get the posterior inclusion probabilities (PIPs) for each (predictor, outcome) pair.
    /// PIP[p,k] = 1 - Π_l (1 - α_p[l,p] * α_k[l,k])
    /// Returns shape (P, K)
    pub fn pip(&self) -> Result<Tensor> {
        let joint = self.alpha_joint()?;
        let one_minus_joint = (1.0 - &joint)?.clamp(1e-10, 1.0)?;
        let log_one_minus = one_minus_joint.log()?;
        let sum_log = log_one_minus.sum(0)?;
        1.0 - sum_log.exp()?
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

    /// Get the mean of θ: E[θ[p,k]] = Σ_l α_p[l,p] * α_k[l,k] * μ[l]
    /// Returns shape (P, K)
    pub fn theta_mean(&self) -> Result<Tensor> {
        let joint = self.alpha_joint()?;
        let beta = self.beta_mean.unsqueeze(1)?.unsqueeze(2)?;
        joint.broadcast_mul(&beta)?.sum(0)
    }
}

impl VariationalDistribution for BiSusieVar {
    fn sample(&self, num_samples: usize) -> Result<(Tensor, Tensor)> {
        let device = self.device();
        let dtype = self.dtype();
        let p = self.num_predictors;
        let k = self.num_outcomes;

        let mean = self.theta_mean()?;
        let var = self.var()?;
        let std = (var + 1e-8)?.sqrt()?;

        let epsilon = Tensor::randn(0f32, 1f32, (num_samples, p, k), device)?.to_dtype(dtype)?;
        let theta = mean
            .unsqueeze(0)?
            .broadcast_add(&epsilon.broadcast_mul(&std)?)?;

        Ok((theta, epsilon))
    }

    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        let dtype = theta.dtype();
        let device = theta.device();
        let ln_2pi =
            Tensor::new((2.0 * std::f64::consts::PI).ln() as f32, device)?.to_dtype(dtype)?;

        let mean = self.theta_mean()?;
        let var = self.var()?;
        let log_var = (var.clone() + 1e-8)?.log()?;

        let diff = theta.broadcast_sub(&mean)?;
        let normalized_sq = diff.sqr()?.broadcast_div(&(var + 1e-8)?)?;

        let log_prob_element =
            (normalized_sq.broadcast_add(&log_var)?.broadcast_add(&ln_2pi)? * (-0.5))?;

        log_prob_element.sum(2)?.sum(1)
    }

    fn mean(&self) -> Result<Tensor> {
        self.theta_mean()
    }

    fn var(&self) -> Result<Tensor> {
        let joint = self.alpha_joint()?;

        let mu = &self.beta_mean;
        let sigma_sq = (&self.beta_ln_std * 2.0)?.exp()?;

        let mu_expanded = mu.unsqueeze(1)?.unsqueeze(2)?;
        let sigma_sq_expanded = sigma_sq.unsqueeze(1)?.unsqueeze(2)?;

        let mu_sq = mu_expanded.sqr()?;
        let second_moment_term = (&sigma_sq_expanded + &mu_sq)?;
        let second_moment_l = joint.broadcast_mul(&second_moment_term)?;
        let second_moment = second_moment_l.sum(0)?;

        let first_moment_l = joint.broadcast_mul(&mu_expanded)?;
        let first_moment = first_moment_l.sum(0)?;
        let first_moment_sq = first_moment.sqr()?;

        (second_moment - first_moment_sq)?.clamp(1e-8, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_bisusie_regression_recovers_sparse_mapping() -> Result<()> {
        use crate::sgvb::traits::BlackBoxLikelihood;
        use crate::sgvb::{direct_elbo_loss, GaussianPrior, LinearModelSGVB, SGVBConfig};
        use candle_nn::{AdamW, Optimizer};

        let device = Device::Cpu;
        let dtype = DType::F32;

        // Problem size: N observations, P predictors, K outcomes
        let n = 200;
        let p = 5; // predictors (e.g., cell types)
        let k = 10; // outcomes (e.g., topics)
        let l = 3; // number of true effects (and BiSusie components)

        // True sparse mapping: 3 (predictor, outcome) pairs
        // predictor 0 -> outcome 2, effect = 2.0
        // predictor 2 -> outcome 5, effect = 1.5
        // predictor 4 -> outcome 8, effect = 1.8
        let true_pairs = [(0usize, 2usize, 2.0f32), (2, 5, 1.5), (4, 8, 1.8)];

        // Build true theta matrix (P x K)
        let mut theta_true_data = vec![0.0f32; p * k];
        for &(pi, ki, effect) in &true_pairs {
            theta_true_data[pi * k + ki] = effect;
        }
        let theta_true = Tensor::from_vec(theta_true_data, (p, k), &device)?;

        // Generate design matrix X (N x P)
        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        // Generate Y = X @ theta_true + noise
        let y_mean = x.matmul(&theta_true)?;
        let noise = Tensor::randn(0f32, 0.5f32, (n, k), &device)?;
        let y = (&y_mean + &noise)?;

        // Create BiSusie model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let bisusie = BiSusieVar::new(vb.pp("bisusie"), l, p, k)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::new(20);

        let model = LinearModelSGVB::from_variational(bisusie, x.clone(), prior, config.clone());

        // Gaussian likelihood
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

        // Train
        let mut optimizer = AdamW::new_lr(varmap.all_vars(), 0.05)?;

        for _epoch in 0..300 {
            let loss = direct_elbo_loss(&model, &likelihood, config.num_samples)?;
            optimizer.backward_step(&loss)?;
        }

        // Check PIPs
        let pip = model.variational.pip()?; // (P, K)

        // For each true pair, PIP should be high
        for &(pi, ki, _) in &true_pairs {
            let pip_val: f32 = pip.get(pi)?.get(ki)?.to_scalar()?;
            assert!(
                pip_val > 0.5,
                "PIP for true pair ({}, {}) should be > 0.5, got {}",
                pi,
                ki,
                pip_val
            );
        }

        // Check that most other entries have low PIP
        let mut low_pip_count = 0;
        let mut total_null = 0;
        for pi in 0..p {
            for ki in 0..k {
                let is_true = true_pairs.iter().any(|&(tp, tk, _)| tp == pi && tk == ki);
                if !is_true {
                    total_null += 1;
                    let pip_val: f32 = pip.get(pi)?.get(ki)?.to_scalar()?;
                    if pip_val < 0.3 {
                        low_pip_count += 1;
                    }
                }
            }
        }
        let low_pip_ratio = low_pip_count as f32 / total_null as f32;
        assert!(
            low_pip_ratio > 0.8,
            "At least 80% of null entries should have PIP < 0.3, got {:.1}%",
            low_pip_ratio * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_bisusie_with_linear_model() -> Result<()> {
        use crate::sgvb::traits::BlackBoxLikelihood;
        use crate::sgvb::{compute_elbo, direct_elbo_loss, GaussianPrior, LinearModelSGVB, SGVBConfig};

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

        let model = LinearModelSGVB::from_variational(bisusie, x, prior, config.clone());

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

        // Test loss computation works
        let loss = direct_elbo_loss(&model, &likelihood, config.num_samples)?;
        assert!(loss.dims().is_empty());

        // Test ELBO computation
        let elbo = compute_elbo(&model, &likelihood, 10)?;
        assert!(elbo.dims().is_empty());

        // Test model outputs
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
        let s = 10;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let bisusie = BiSusieVar::new(vb, l, p, k)?;

        assert_eq!(bisusie.alpha_predictor()?.dims(), &[l, p]);
        assert_eq!(bisusie.alpha_outcome()?.dims(), &[l, k]);
        assert_eq!(bisusie.alpha_joint()?.dims(), &[l, p, k]);
        assert_eq!(bisusie.pip()?.dims(), &[p, k]);

        let (theta, epsilon) = bisusie.sample(s)?;
        assert_eq!(theta.dims(), &[s, p, k]);
        assert_eq!(epsilon.dims(), &[s, p, k]);

        assert_eq!(bisusie.theta_mean()?.dims(), &[p, k]);
        assert_eq!(bisusie.var()?.dims(), &[p, k]);

        Ok(())
    }

    #[test]
    fn test_alpha_sums_to_one() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 3;
        let p = 8;
        let k = 20;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let bisusie = BiSusieVar::new(vb, l, p, k)?;

        let alpha_p_sum = bisusie.alpha_predictor()?.sum(1)?;
        let alpha_k_sum = bisusie.alpha_outcome()?.sum(1)?;

        for i in 0..l {
            let sum_p: f64 = alpha_p_sum.get(i)?.to_scalar()?;
            let sum_k: f64 = alpha_k_sum.get(i)?.to_scalar()?;
            assert!((sum_p - 1.0).abs() < 1e-5);
            assert!((sum_k - 1.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_joint_sums_to_one() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 2;
        let p = 3;
        let k = 4;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let bisusie = BiSusieVar::new(vb, l, p, k)?;
        let joint = bisusie.alpha_joint()?;

        for i in 0..l {
            let slice = joint.get(i)?;
            let sum: f64 = slice.sum_all()?.to_scalar()?;
            assert!((sum - 1.0).abs() < 1e-5);
        }

        Ok(())
    }
}
