use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::susie_util::{kl_categorical_uniform, pip_from_alpha};
use super::traits::{ComponentVariational, VariationalDistribution};

/// Affine-smoothed sigmoid: ε + (1 - 2ε) · σ(x).
///
/// Maps logits to (ε, 1-ε), ensuring gradients never fully vanish.
/// At x=0, returns 0.5 regardless of ε.
pub fn smoothed_sigmoid(logits: &Tensor, epsilon: f64) -> Result<Tensor> {
    let scale = 1.0 - 2.0 * epsilon;
    (candle_nn::ops::sigmoid(logits)? * scale)? + epsilon
}

/// Per-component gate parameters for SusieVar.
struct GateParams {
    /// Gate logits, shape (L,)
    logits: Tensor,
    /// Smoothing epsilon: π_l = ε + (1-2ε) · σ(logit_l)
    epsilon: f64,
}

/// Susie (Sum of Single Effects) variational distribution.
///
/// θ = Σ_l γ_l · (α_l ⊙ β_l) where:
/// - γ_l ~ Bernoulli(π_l) - component gate (optional), scalar per component
/// - α_l = softmax(logits_l, dim=p) - selection probabilities, shape (p, k)
/// - β_l ~ N(μ_l, σ_l²) - effect sizes, shape (p, k)
///
/// When component gates are enabled, the effective selection probability is
/// π_l · α_{l,j,k}, allowing unused components to be pruned (π_l → ε).
/// The gate uses affine-smoothed sigmoid: π_l = ε + (1-2ε) · σ(logit_l)
/// to prevent vanishing gradients at saturation.
///
/// Each output dimension k has its own independent feature selection.
/// This can be used with RegressionSGVB for Susie regression.
pub struct SusieVar {
    /// Selection logits, shape (L, p_logits, k) where p_logits = p + has_null as usize.
    /// When `has_null`, position p is the null (no-effect) absorber.
    logits: Tensor,
    /// Effect size means, shape (L, p, k)
    beta_mean: Tensor,
    /// Effect size log-stds, shape (L, p, k)
    beta_ln_std: Tensor,
    /// Number of real features (excluding null)
    p: usize,
    /// Number of components L
    num_components: usize,
    /// Whether a null position is appended to the softmax
    has_null: bool,
    /// Per-component Bernoulli gate. None = all components always active.
    gate: Option<GateParams>,
}

impl SusieVar {
    /// Create a new Susie variational distribution (no component gates, no null).
    ///
    /// All L components are always active. Use [`new_gated`] to enable
    /// per-component Bernoulli gates for automatic component pruning.
    pub fn new(vb: VarBuilder, num_components: usize, p: usize, k: usize) -> Result<Self> {
        Self::new_inner(vb, num_components, p, k, None, false)
    }

    /// Create a new Susie variational distribution with a null absorber.
    ///
    /// # Why null?
    ///
    /// Standard SuSiE softmax forces `Σ_j α_j = 1` per component, so every
    /// component must select *some* SNP — even in LD blocks with no signal.
    /// With gradient-based SGVB, this causes false positives: noise
    /// correlations push α toward spurious concentration, and the Gaussian
    /// KL penalty alone isn't enough to prevent it.
    ///
    /// # How it works
    ///
    /// Appends a (p+1)th "null" position to the softmax logits. The null has
    /// no associated β parameters — selecting null means zero contribution to
    /// the linear predictor. In the ELBO:
    ///
    /// - **Signal blocks**: causal SNPs get strong, consistent likelihood
    ///   gradients that overcome the null. Mass concentrates on real SNPs.
    /// - **Null blocks**: no SNP gets consistent positive gradient. The null
    ///   is the lowest-cost option (no Gaussian KL for fitting noise), so
    ///   mass naturally flows there.
    ///
    /// `alpha()` and `pip()` return only the p real positions; the null mass
    /// is excluded. Use [`null_mass`] to inspect per-component null absorption.
    pub fn new_with_null(
        vb: VarBuilder,
        num_components: usize,
        p: usize,
        k: usize,
    ) -> Result<Self> {
        Self::new_inner(vb, num_components, p, k, None, true)
    }

    /// Create a new Susie variational distribution with per-component gates.
    ///
    /// Each component l has a Bernoulli gate γ_l with inclusion probability
    /// π_l = ε + (1-2ε) · σ(logit_l). The effective selection probability
    /// becomes π_l · α_{l,j,k}.
    ///
    /// Set L generously — unused components will be pruned (π_l → ε).
    ///
    /// `gate_epsilon` is the smoothing epsilon for sigmoid (e.g., 0.01).
    /// Ensures π ∈ [ε, 1-ε] so gradients never fully vanish.
    pub fn new_gated(
        vb: VarBuilder,
        num_components: usize,
        p: usize,
        k: usize,
        gate_epsilon: f64,
    ) -> Result<Self> {
        Self::new_inner(vb, num_components, p, k, Some(gate_epsilon), false)
    }

    /// Unified constructor with all options.
    ///
    /// `gate_epsilon`: `None` = ungated, `Some(ε)` = gated with smoothing ε.
    /// `has_null`: whether to append a null absorber position to the softmax.
    pub(crate) fn new_inner(
        vb: VarBuilder,
        num_components: usize,
        p: usize,
        k: usize,
        gate_epsilon: Option<f64>,
        has_null: bool,
    ) -> Result<Self> {
        let p_logits = p + has_null as usize;
        let logits = vb.get_with_hints(
            (num_components, p_logits, k),
            "logits",
            candle_nn::Init::Const(0.0),
        )?;

        let beta_mean = vb.get_with_hints(
            (num_components, p, k),
            "beta_mean",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;

        let beta_ln_std = vb.get_with_hints(
            (num_components, p, k),
            "beta_ln_std",
            candle_nn::Init::Const(0.0),
        )?;

        // Gate logits initialize to 0 → π = 0.5 at init
        let gate = match gate_epsilon {
            Some(eps) => Some(GateParams {
                logits: vb.get_with_hints(
                    (num_components,),
                    "component_logits",
                    candle_nn::Init::Const(0.0),
                )?,
                epsilon: eps,
            }),
            None => None,
        };

        Ok(Self {
            logits,
            beta_mean,
            beta_ln_std,
            p,
            num_components,
            has_null,
            gate,
        })
    }

    /// Compute KL divergence for the selection distribution.
    ///
    /// Returns the sum of:
    /// 1. Categorical KL: `Σ_l Σ_j Σ_k α̃_{l,j,k} (log α̃_{l,j,k} - log(prior_alpha/p))`
    ///    where α̃ is the softmax selection (before gating).
    /// 2. Bernoulli KL (if gated): `Σ_l KL(Bernoulli(π_l) || Bernoulli(0.5))`
    ///    using an uninformative Bernoulli(0.5) prior on component inclusion.
    ///
    /// `prior_alpha` controls the categorical prior: uniform `1/p` per SNP
    /// when `prior_alpha = 1`.
    ///
    /// Returns a scalar tensor.
    pub fn kl_categorical(&self, prior_alpha: f64) -> Result<Tensor> {
        let cat_kl = kl_categorical_uniform(&self.logits, prior_alpha)?;
        match &self.gate {
            None => Ok(cat_kl),
            Some(g) => cat_kl + Self::kl_bernoulli(g),
        }
    }

    /// KL(Bernoulli(π_l) || Bernoulli(0.5)) summed over components.
    ///
    /// Simplifies to: Σ_l [π log π + (1-π) log(1-π) + ln2]
    fn kl_bernoulli(gate: &GateParams) -> Result<Tensor> {
        let pi = smoothed_sigmoid(&gate.logits, gate.epsilon)?; // (L,)
        let one_minus_pi = (1.0 - &pi)?;
        let neg_entropy = ((&pi * pi.log()?)? + (&one_minus_pi * one_minus_pi.log()?)?)?;
        (neg_entropy + 2.0f64.ln())?.sum_all()
    }

    /// Get component inclusion probabilities π_l = ε + (1-2ε) · σ(logit_l).
    ///
    /// Returns shape (L,). Panics if not gated.
    pub fn component_pi(&self) -> Result<Tensor> {
        let g = self
            .gate
            .as_ref()
            .expect("component_pi called on ungated SusieVar");
        smoothed_sigmoid(&g.logits, g.epsilon)
    }

    /// Whether this SusieVar has per-component gates.
    pub fn is_gated(&self) -> bool {
        self.gate.is_some()
    }

    /// Get effective selection probabilities (gated α).
    ///
    /// - Ungated: α = softmax(logits, dim=p), shape (L, p, k)
    /// - Gated: α = π_l · softmax(logits, dim=p), shape (L, p, k)
    ///
    /// The softmax within each component still sums to 1 over p,
    /// but the per-component gate scales the whole distribution.
    pub fn alpha(&self) -> Result<Tensor> {
        let softmax = self.softmax_alpha()?;
        match &self.gate {
            None => Ok(softmax),
            Some(_) => {
                let pi = self.component_pi()?.unsqueeze(1)?.unsqueeze(2)?;
                softmax.broadcast_mul(&pi)
            }
        }
    }

    /// Get log effective selection probabilities.
    ///
    /// - Ungated: log_softmax(logits, dim=p)
    /// - Gated: log(π_l) + log_softmax(logits, dim=p)
    pub fn log_alpha(&self) -> Result<Tensor> {
        let log_sm = self.log_softmax_alpha()?;
        match &self.gate {
            None => Ok(log_sm),
            Some(_) => {
                let log_pi = self.component_pi()?.log()?.unsqueeze(1)?.unsqueeze(2)?;
                log_sm.broadcast_add(&log_pi)
            }
        }
    }

    /// Raw softmax selection probabilities for real features (before gating).
    /// Shape (L, p, k). Sums to ≤ 1 over p (< 1 when null absorbs mass).
    fn softmax_alpha(&self) -> Result<Tensor> {
        self.log_softmax_alpha()?.exp()
    }

    /// Raw log-softmax selection probabilities for real features (before gating).
    /// Shape (L, p, k).
    fn log_softmax_alpha(&self) -> Result<Tensor> {
        let full = self.log_softmax_full()?;
        if self.has_null {
            full.narrow(1, 0, self.p)
        } else {
            Ok(full)
        }
    }

    /// Full log-softmax over all positions including null.
    /// Shape (L, p_logits, k) where p_logits = p + has_null.
    fn log_softmax_full(&self) -> Result<Tensor> {
        candle_nn::ops::log_softmax(&self.logits, 1)
    }

    /// Whether this SusieVar has a null absorber position.
    pub fn has_null(&self) -> bool {
        self.has_null
    }

    /// Per-component probability mass on the null position.
    ///
    /// Returns shape (L, k). Panics if `!has_null`.
    pub fn null_mass(&self) -> Result<Tensor> {
        assert!(self.has_null, "null_mass called on SusieVar without null");
        self.log_softmax_full()?
            .narrow(1, self.p, 1)?
            .exp()?
            .squeeze(1) // (L, k)
    }

    /// Get the posterior inclusion probabilities (PIPs) for each feature and output.
    /// PIP_{j,k} = 1 - Π_l (1 - α_{l,j,k})
    ///
    /// # Returns
    /// PIPs, shape (p, k)
    pub fn pip(&self) -> Result<Tensor> {
        pip_from_alpha(&self.alpha()?)
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

impl ComponentVariational for SusieVar {
    fn alpha(&self) -> Result<Tensor> {
        self.alpha()
    }
    fn beta_mean(&self) -> Result<Tensor> {
        Ok(self.beta_mean().clone())
    }
    fn beta_std(&self) -> Result<Tensor> {
        self.beta_std()
    }
    fn num_components(&self) -> usize {
        self.num_components()
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
    use candle_nn::{Optimizer, VarBuilder, VarMap};

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
        use crate::sgvb::{local_reparam_loss, GaussianPrior, RegressionSGVB, SGVBConfig};
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

        // Combine into generic RegressionSGVB
        let model = RegressionSGVB::from_variational(susie, x, prior, config);

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

    // --- Gated SusieVar tests ---

    #[test]
    fn test_smoothed_sigmoid_bounds() -> Result<()> {
        let device = Device::Cpu;
        // Test at extreme logits
        let logits = Tensor::from_vec(vec![-100.0f64, -10.0, 0.0, 10.0, 100.0], (5,), &device)?;
        let eps = 0.01;
        let pi = smoothed_sigmoid(&logits, eps)?;
        let vals: Vec<f64> = pi.to_vec1()?;

        for &v in &vals {
            assert!(v >= eps, "π should be >= ε={}, got {}", eps, v);
            assert!(
                v <= 1.0 - eps,
                "π should be <= 1-ε={}, got {}",
                1.0 - eps,
                v
            );
        }
        // At logit=0, sigmoid=0.5, so π = ε + (1-2ε)*0.5 = 0.5
        assert!(
            (vals[2] - 0.5).abs() < 1e-10,
            "π at logit=0 should be 0.5, got {}",
            vals[2]
        );

        Ok(())
    }

    #[test]
    fn test_gated_susie_shapes() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let l = 3;
        let p = 20;
        let k = 2;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new_gated(vb, l, p, k, 0.01)?;

        assert!(susie.is_gated());
        assert_eq!(susie.alpha()?.dims(), &[l, p, k]);
        assert_eq!(susie.pip()?.dims(), &[p, k]);
        assert_eq!(susie.component_pi()?.dims(), &[l]);

        Ok(())
    }

    #[test]
    fn test_gated_alpha_scaled_by_pi() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 3;
        let p = 20;
        let k = 1;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new_gated(vb, l, p, k, 0.01)?;

        let alpha = susie.alpha()?; // (L, p, k)
        let pi = susie.component_pi()?; // (L,)
        let softmax_alpha = susie.softmax_alpha()?; // (L, p, k)

        // Gated α should equal π_l · softmax_α
        for comp in 0..l {
            let pi_l: f64 = pi.get(comp)?.to_scalar()?;
            let alpha_sum: f64 = alpha.get(comp)?.sum_all()?.to_scalar()?;
            let softmax_sum: f64 = softmax_alpha.get(comp)?.sum_all()?.to_scalar()?;

            // softmax sums to k (=1 per output dim), so gated α sums to π_l * k
            assert!(
                (alpha_sum - pi_l * k as f64).abs() < 1e-5,
                "Gated α should sum to π_l * k = {:.4}, got {:.4}",
                pi_l * k as f64,
                alpha_sum
            );
            assert!(
                (softmax_sum - k as f64).abs() < 1e-5,
                "Softmax α should sum to k = {}, got {:.4}",
                k,
                softmax_sum
            );
        }

        Ok(())
    }

    #[test]
    fn test_gated_kl_nonnegative() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 5;
        let p = 50;
        let k = 1;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new_gated(vb, l, p, k, 0.01)?;
        let kl: f64 = susie.kl_categorical(1.0)?.to_scalar()?;

        // KL divergence is non-negative
        assert!(kl >= -1e-10, "KL should be non-negative, got {}", kl);

        Ok(())
    }

    #[test]
    fn test_gated_gradient_flow() -> Result<()> {
        use crate::sgvb::traits::BlackBoxLikelihood;
        use crate::sgvb::{local_reparam_loss, GaussianPrior, RegressionSGVB, SGVBConfig};
        use candle_core::Tensor;

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 30;
        let p = 10;
        let k = 1;
        let l = 4; // intentionally more components than true effects

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let y = Tensor::randn(0f32, 1f32, (n, k), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new_gated(vb.pp("susie"), l, p, k, 0.01)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = RegressionSGVB::from_variational(susie, x, prior, config);

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

        // Verify loss is finite and gradient step works
        let loss = local_reparam_loss(&model, &likelihood, 10, 1.0)?;
        assert!(loss.to_scalar::<f32>()?.is_finite());

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.01)?;
        optimizer.backward_step(&loss)?;

        let loss2 = local_reparam_loss(&model, &likelihood, 10, 1.0)?;
        let l1: f32 = loss.to_scalar()?;
        let l2: f32 = loss2.to_scalar()?;
        assert!(
            (l1 - l2).abs() > 1e-8,
            "Loss should change after gradient step: {} vs {}",
            l1,
            l2
        );

        Ok(())
    }

    #[test]
    fn test_gated_sparse_recovery() -> Result<()> {
        use crate::sgvb::traits::BlackBoxLikelihood;
        use crate::sgvb::{local_reparam_loss, FixedGaussianPrior, RegressionSGVB, SGVBConfig};
        use candle_core::Tensor;

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 200;
        let p = 20;
        let k = 1;
        let l = 6; // way more than 1 true effect

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let x_5 = x.narrow(1, 5, 1)?;
        let noise = Tensor::randn(0f32, 0.3f32, (n, k), &device)?;
        let y = ((x_5 * 3.0)? + noise)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new_gated(vb.pp("susie"), l, p, k, 0.01)?;
        let prior = FixedGaussianPrior::new(0.5);
        let config = SGVBConfig::new(30);

        let model = RegressionSGVB::from_variational(susie, x, prior, config);

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
            let loss = local_reparam_loss(&model, &likelihood, 30, 1.0)?;
            optimizer.backward_step(&loss)?;
        }

        // π values may stay near 0.5 with Bernoulli(0.5) prior — that's OK.
        // The important thing is that PIP concentrates on the true variable.
        let pi = model.variational.component_pi()?;
        let pi_vals: Vec<f32> = pi.to_vec1()?;
        println!("Component π values: {:?}", pi_vals);

        // All π should be in [ε, 1-ε]
        for &v in &pi_vals {
            assert!(v >= 0.01 && v <= 0.99, "π out of bounds: {}", v);
        }

        // Check PIP concentrates on feature 5
        let pip = model.variational.pip()?;
        let pip_5: f32 = pip.get(5)?.get(0)?.to_scalar()?;
        let mut other_sum = 0.0f32;
        for j in 0..p {
            if j != 5 {
                other_sum += pip.get(j)?.get(0)?.to_scalar::<f32>()?;
            }
        }
        let other_mean = other_sum / (p - 1) as f32;
        println!("PIP[5] = {:.4}, other mean PIP = {:.4}", pip_5, other_mean);
        assert!(
            pip_5 > other_mean * 2.0,
            "PIP[5] ({:.4}) should be > 2x other mean ({:.4})",
            pip_5,
            other_mean
        );

        Ok(())
    }

    #[test]
    fn test_ungated_unchanged() -> Result<()> {
        // Verify that ungated SusieVar behaves exactly as before
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 2;
        let p = 10;
        let k = 1;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new(vb, l, p, k)?;
        assert!(!susie.is_gated());

        // Alpha should sum to 1 (no gating)
        let alpha = susie.alpha()?;
        let alpha_sum: f64 = alpha.get(0)?.sum_all()?.to_scalar()?;
        assert!(
            (alpha_sum - 1.0).abs() < 1e-5,
            "Ungated alpha should sum to 1, got {}",
            alpha_sum
        );

        // KL should be purely categorical (no Bernoulli term)
        let kl: f64 = susie.kl_categorical(1.0)?.to_scalar()?;
        assert!(kl.is_finite());
        assert!(kl >= -1e-10);

        Ok(())
    }
}
