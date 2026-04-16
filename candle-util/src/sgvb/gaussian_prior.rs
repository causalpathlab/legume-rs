use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::traits::{AnalyticalKL, Prior};

/// Maximum value for ln(τ) to prevent numerical overflow.
/// ln(100) ≈ 4.6, so τ is capped at ~100.
const MAX_LN_TAU: f64 = 4.6;

/// Learnable Gaussian prior p(θ) = N(0, τ²I)
///
/// The prior scale τ is a learnable parameter stored as ln(τ).
/// τ is capped at exp(MAX_LN_TAU) ≈ 100 to prevent numerical issues.
pub struct GaussianPrior {
    /// Log scale parameter ln(τ)
    ln_tau: Tensor,
}

impl GaussianPrior {
    /// Create a new Gaussian prior with learnable scale.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `init_tau` - Initial value for τ (will be stored as ln(τ))
    ///
    /// # Returns
    /// Initialized GaussianPrior
    pub fn new(vb: VarBuilder, init_tau: f32) -> Result<Self> {
        let ln_tau_init = init_tau.ln();
        let ln_tau = vb.get_with_hints((), "ln_tau", candle_nn::Init::Const(ln_tau_init as f64))?;
        Ok(Self { ln_tau })
    }

    /// Get the prior scale τ = exp(clamp(ln_tau)).
    pub fn tau(&self) -> Result<f32> {
        let clamped = self.ln_tau.clamp(-MAX_LN_TAU, MAX_LN_TAU)?;
        // Move to CPU for dtype conversion (Metal doesn't support F64)
        let val: f32 = clamped
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .exp()?
            .to_scalar()?;
        Ok(val)
    }

    /// Get the device of the parameters.
    pub fn device(&self) -> &Device {
        self.ln_tau.device()
    }

    /// Get the dtype of the parameters.
    pub fn dtype(&self) -> DType {
        self.ln_tau.dtype()
    }
}

impl Prior for GaussianPrior {
    /// Compute log p(θ) = sum over all elements of log N(θ; 0, τ²)
    ///
    /// log N(θ; 0, τ²) = -0.5 * [θ²/τ² + 2*ln(τ) + ln(2π)]
    ///
    /// # Arguments
    /// * `theta` - Parameter samples, shape (S, p, k)
    ///
    /// # Returns
    /// Log prior probability, shape (S,)
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        let dtype = theta.dtype();
        let device = theta.device();
        // Create scalar constants directly in the target dtype to avoid Metal F64 conversion issues
        let ln_2pi =
            Tensor::new((2.0 * std::f64::consts::PI).ln() as f32, device)?.to_dtype(dtype)?;

        // Clamp ln_tau to prevent overflow
        let ln_tau_clamped = self
            .ln_tau
            .clamp(-MAX_LN_TAU, MAX_LN_TAU)?
            .to_dtype(dtype)?;

        // τ = exp(ln_tau_clamped)
        let tau = ln_tau_clamped.exp()?;
        let tau_sq = tau.sqr()?;

        // θ²/τ²: shape (S, p, k)
        let theta_sq_normalized = theta.sqr()?.broadcast_div(&tau_sq)?;

        // 2*ln(τ) + ln(2π) is a scalar, broadcast to all elements
        let const_term = (ln_tau_clamped * 2.0)?.broadcast_add(&ln_2pi)?;

        // log p = -0.5 * [θ²/τ² + 2*ln(τ) + ln(2π)]
        let log_prob_element = (theta_sq_normalized.broadcast_add(&const_term)? * (-0.5))?;

        // Sum over dimensions 1 and 2 (p and k)
        log_prob_element.sum(2)?.sum(1)
    }
}

impl AnalyticalKL for GaussianPrior {
    /// KL(N(μ, σ²) ‖ N(0, τ²)) = Σ [ln(τ/σ) + (σ² + μ²)/(2τ²) − 0.5]
    fn kl_from_gaussian(&self, mean: &Tensor, var: &Tensor) -> Result<Tensor> {
        let dtype = mean.dtype();

        let ln_tau_clamped = self
            .ln_tau
            .clamp(-MAX_LN_TAU, MAX_LN_TAU)?
            .to_dtype(dtype)?;
        let tau = ln_tau_clamped.exp()?;
        let tau_sq = tau.sqr()?;

        // ln(τ) - 0.5·ln(var)
        let ln_var = (var + 1e-8)?.log()?;
        let log_term = ln_tau_clamped.broadcast_sub(&(ln_var * 0.5)?)?;

        // (var + mean²) / (2τ²)
        let quad_term = ((var + mean.sqr()?)?.broadcast_div(&tau_sq)? * 0.5)?;

        // KL per element = ln(τ/σ) + (σ² + μ²)/(2τ²) - 0.5
        let kl_elements = ((log_term + quad_term)? - 0.5)?;

        // Sum over all (p, k) elements
        kl_elements.sum_all()
    }
}

/// Fixed (non-learnable) Gaussian prior p(θ) = N(0, τ²I)
pub struct FixedGaussianPrior {
    /// Fixed scale parameter τ
    tau: f32,
}

impl FixedGaussianPrior {
    /// Create a new fixed Gaussian prior.
    ///
    /// # Arguments
    /// * `tau` - Prior scale τ
    pub fn new(tau: f32) -> Self {
        Self { tau }
    }

    /// Get the prior scale τ.
    pub fn tau(&self) -> f32 {
        self.tau
    }
}

impl Prior for FixedGaussianPrior {
    /// Compute log p(θ) = sum over all elements of log N(θ; 0, τ²)
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        let ln_2pi: f64 = (2.0 * std::f64::consts::PI).ln();
        let ln_tau: f64 = (self.tau as f64).ln();
        let tau_sq: f64 = (self.tau as f64).powi(2);

        // θ²/τ²: shape (S, p, k)
        let theta_sq_normalized = (theta.sqr()? / tau_sq)?;

        // 2*ln(τ) + ln(2π)
        let const_term = 2.0 * ln_tau + ln_2pi;

        // log p = -0.5 * [θ²/τ² + 2*ln(τ) + ln(2π)]
        let log_prob_element = ((theta_sq_normalized + const_term)? * (-0.5))?;

        // Sum over dimensions 1 and 2 (p and k)
        log_prob_element.sum(2)?.sum(1)
    }
}

impl AnalyticalKL for FixedGaussianPrior {
    /// KL(N(μ, σ²) ‖ N(0, τ²)) = Σ [ln(τ/σ) + (σ² + μ²)/(2τ²) − 0.5]
    fn kl_from_gaussian(&self, mean: &Tensor, var: &Tensor) -> Result<Tensor> {
        let ln_tau: f64 = (self.tau as f64).ln();
        let tau_sq: f64 = (self.tau as f64).powi(2);

        let ln_var = (var + 1e-8)?.log()?;
        let log_term = (ln_var * (-0.5))? + ln_tau;

        let quad_term = (((var + mean.sqr()?)? / tau_sq)? * 0.5)?;

        let kl_elements = ((log_term? + quad_term)? - 0.5)?;
        kl_elements.sum_all()
    }
}

/// Mixture-of-Gaussians prior (ash-style): p(β) = Σ_m w_m · N(β; 0, τ²_m).
///
/// Uses a fixed variance grid with learnable mixture weights. The near-zero
/// component acts as a soft spike, absorbing polygenic signal that would
/// otherwise inflate PIPs on individual SNPs through LD.
///
/// The analytical KL uses the identity:
///   KL(N(μ,σ²) ‖ Σ_m w_m N(0,τ²_m))
///     = -Σ_j log Σ_m w_m · N(μ_j; 0, σ²_j + τ²_m)
///       + Σ_j [-0.5 log(2πσ²_j) - 0.5]
///     = const(q) - Σ_j logsumexp_m [log w_m + log N(μ_j; 0, σ²_j + τ²_m)]
///
/// In practice we compute per-element:
///   kl_j = min over mixture of [KL(q_j ‖ N(0,τ²_m)) - log w_m]
/// using the soft-min (negative logsumexp) form, which is exact.
pub struct MixtureGaussianPrior {
    /// Variance grid τ²_m, fixed. Length M.
    tau_sq: Vec<f64>,
    /// Mixture weight logits (learnable), shape (M,).
    weight_logits: Tensor,
}

impl MixtureGaussianPrior {
    /// Create a mixture-of-Gaussians prior with a geometric variance grid.
    ///
    /// Grid: one near-zero component (τ² = 1e-10) plus `num_grid - 1` points
    /// geometrically spaced from `tau_sq_min` to `tau_sq_max`.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for learnable weight logits
    /// * `num_grid` - Number of mixture components M (including the spike)
    /// * `tau_sq_min` - Smallest non-spike variance (e.g., 0.001)
    /// * `tau_sq_max` - Largest variance (e.g., 1.0)
    pub fn new(vb: VarBuilder, num_grid: usize, tau_sq_min: f64, tau_sq_max: f64) -> Result<Self> {
        assert!(num_grid >= 2, "need at least 2 grid points");
        let mut tau_sq = Vec::with_capacity(num_grid);
        tau_sq.push(1e-10); // near point-mass at zero
        let log_min = tau_sq_min.ln();
        let log_max = tau_sq_max.ln();
        for i in 0..(num_grid - 1) {
            let t = i as f64 / (num_grid - 2).max(1) as f64;
            tau_sq.push((log_min + t * (log_max - log_min)).exp());
        }

        let weight_logits = vb.get_with_hints(
            (num_grid,),
            "mix_weight_logits",
            candle_nn::Init::Const(0.0), // uniform init
        )?;

        Ok(Self {
            tau_sq,
            weight_logits,
        })
    }

    /// Create from an explicit variance grid.
    pub fn from_grid(vb: VarBuilder, tau_sq: Vec<f64>) -> Result<Self> {
        let m = tau_sq.len();
        assert!(m >= 1, "need at least 1 grid point");
        let weight_logits =
            vb.get_with_hints((m,), "mix_weight_logits", candle_nn::Init::Const(0.0))?;
        Ok(Self {
            tau_sq,
            weight_logits,
        })
    }

    /// Get learned mixture weights w_m = softmax(logits).
    pub fn weights(&self) -> Result<Tensor> {
        candle_nn::ops::softmax(&self.weight_logits, 0)
    }

    /// Get the variance grid.
    pub fn tau_sq_grid(&self) -> &[f64] {
        &self.tau_sq
    }

    /// Number of mixture components.
    pub fn num_components(&self) -> usize {
        self.tau_sq.len()
    }
}

impl Prior for MixtureGaussianPrior {
    /// log p(θ) = Σ_j log Σ_m w_m · N(θ_j; 0, τ²_m)
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        let dtype = theta.dtype();
        let device = theta.device();
        let ln_2pi =
            Tensor::new((2.0 * std::f64::consts::PI).ln() as f32, device)?.to_dtype(dtype)?;

        let log_w = candle_nn::ops::log_softmax(&self.weight_logits, 0)?.to_dtype(dtype)?;

        // For each grid point m, compute log [w_m · N(θ; 0, τ²_m)] per element
        let mut log_components = Vec::with_capacity(self.tau_sq.len());
        for (m, &tau2) in self.tau_sq.iter().enumerate() {
            let ln_tau2 = tau2.ln();
            // log N(θ; 0, τ²) = -0.5 [θ²/τ² + ln(τ²) + ln(2π)]
            let log_n = ((theta.sqr()? / tau2)?.broadcast_add(&ln_2pi)? + ln_tau2)? * (-0.5);
            let log_w_m = log_w.get(m)?;
            log_components.push(log_n?.broadcast_add(&log_w_m)?);
        }

        // Stack to (M, S, p, k), logsumexp over M -> (S, p, k), sum over (p, k) -> (S,)
        let lse = Tensor::stack(&log_components, 0)?.log_sum_exp(0)?;
        lse.sum(2)?.sum(1)
    }
}

impl AnalyticalKL for MixtureGaussianPrior {
    /// KL(N(μ, σ²) ‖ Σ_m w_m N(0, τ²_m))
    ///
    /// = -logsumexp_m [ log w_m - KL(N(μ,σ²) ‖ N(0,τ²_m)) ]
    ///   + const  (the const cancels: it's H(q) which appears in both terms)
    ///
    /// Equivalently, per element j:
    ///   = -logsumexp_m [ log w_m + log N(μ_j; 0, σ²_j + τ²_m) ] - H(q_j)
    ///
    /// We use the direct form: for each m, compute the Gaussian KL to N(0, τ²_m),
    /// then combine via -logsumexp(-kl + log_w).
    fn kl_from_gaussian(&self, mean: &Tensor, var: &Tensor) -> Result<Tensor> {
        let dtype = mean.dtype();
        let log_w = candle_nn::ops::log_softmax(&self.weight_logits, 0)?.to_dtype(dtype)?;
        let ln_var = (var + 1e-8)?.log()?;

        // For each grid point, compute -KL(q ‖ N(0,τ²_m)) + log w_m per element
        let mut neg_kl_plus_logw = Vec::with_capacity(self.tau_sq.len());
        for (m, &tau2) in self.tau_sq.iter().enumerate() {
            let ln_tau = (tau2.sqrt()).ln();
            // KL_m per element = ln(τ_m/σ) + (σ² + μ²)/(2τ²_m) - 0.5
            let log_term = ((&ln_var * (-0.5))? + ln_tau)?;
            let quad_term = (((var + mean.sqr()?)? / tau2)? * 0.5)?;
            let kl_m = ((log_term + quad_term)? - 0.5)?;

            let log_w_m = log_w.get(m)?;
            // -kl_m + log_w_m
            neg_kl_plus_logw.push(kl_m.neg()?.broadcast_add(&log_w_m)?);
        }

        // Stack to (M, p, k), logsumexp over dim 0 -> (p, k), negate and sum
        let lse = Tensor::stack(&neg_kl_plus_logw, 0)?.log_sum_exp(0)?;
        lse.neg()?.sum_all()
    }
}

/// Dispatch enum wrapping either a fixed single-Gaussian or mixture-of-Gaussians prior.
///
/// Implements `Prior + AnalyticalKL` by delegation, so `RegressionSGVB<V, PriorKind>`
/// works without duplicating model variants.
pub enum PriorKind {
    Fixed(FixedGaussianPrior),
    Mixture(MixtureGaussianPrior),
}

impl Prior for PriorKind {
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor> {
        match self {
            Self::Fixed(p) => p.log_prob(theta),
            Self::Mixture(p) => p.log_prob(theta),
        }
    }
}

impl AnalyticalKL for PriorKind {
    fn kl_from_gaussian(&self, mean: &Tensor, var: &Tensor) -> Result<Tensor> {
        match self {
            Self::Fixed(p) => p.kl_from_gaussian(mean, var),
            Self::Mixture(p) => p.kl_from_gaussian(mean, var),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_tau_value() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let init_tau = 2.0f32;
        let prior = GaussianPrior::new(vb, init_tau)?;

        let tau = prior.tau()?;
        assert!(
            (tau - init_tau).abs() < 1e-5,
            "Expected {}, got {}",
            init_tau,
            tau
        );

        Ok(())
    }

    #[test]
    fn test_log_prob_shape() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let s = 10;
        let p = 5;
        let k = 3;

        let prior = GaussianPrior::new(vb, 1.0)?;
        let theta = Tensor::randn(0f32, 1f32, (s, p, k), &Device::Cpu)?;
        let log_prob = prior.log_prob(&theta)?;

        assert_eq!(log_prob.dims(), &[s]);

        Ok(())
    }

    #[test]
    fn test_fixed_prior_log_prob() -> Result<()> {
        let prior = FixedGaussianPrior::new(1.0);

        let s = 5;
        let p = 2;
        let k = 2;

        // Theta at zero
        let theta = Tensor::zeros((s, p, k), DType::F64, &Device::Cpu)?;
        let log_prob = prior.log_prob(&theta)?;

        // Expected: -0.5 * ln(2π) * (p * k) per sample
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() * (p * k) as f64;

        for i in 0..s {
            let actual: f64 = log_prob.get(i)?.to_scalar()?;
            assert!(
                (actual - expected).abs() < 1e-5,
                "Expected {}, got {}",
                expected,
                actual
            );
        }

        Ok(())
    }

    #[test]
    fn test_mixture_prior_grid() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let prior = MixtureGaussianPrior::new(vb, 10, 0.001, 1.0)?;

        assert_eq!(prior.num_components(), 10);
        assert!(
            prior.tau_sq_grid()[0] < 1e-9,
            "first component should be near-zero spike"
        );
        assert!(
            (prior.tau_sq_grid()[1] - 0.001).abs() < 1e-6,
            "second should be tau_sq_min"
        );
        let last = *prior.tau_sq_grid().last().unwrap();
        assert!(
            (last - 1.0).abs() < 1e-6,
            "last should be tau_sq_max, got {}",
            last
        );

        // Weights should be uniform at init
        let w = prior.weights()?;
        let w_vals: Vec<f64> = w.to_vec1()?;
        for &v in &w_vals {
            assert!((v - 0.1).abs() < 1e-5, "expected uniform 0.1, got {}", v);
        }
        Ok(())
    }

    #[test]
    fn test_mixture_prior_log_prob_shape() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let prior = MixtureGaussianPrior::new(vb, 5, 0.01, 1.0)?;

        let s = 8;
        let p = 4;
        let k = 2;
        let theta = Tensor::randn(0f64, 1f64, (s, p, k), &Device::Cpu)?;
        let log_prob = prior.log_prob(&theta)?;
        assert_eq!(log_prob.dims(), &[s]);

        // log prob should be finite and negative
        let vals: Vec<f64> = log_prob.to_vec1()?;
        for &v in &vals {
            assert!(v.is_finite(), "log_prob should be finite");
            assert!(v < 0.0, "log_prob should be negative");
        }
        Ok(())
    }

    #[test]
    fn test_mixture_reduces_to_single_gaussian() -> Result<()> {
        // With a single-component mixture, KL should match FixedGaussianPrior
        let tau = 0.5f64;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let mixture = MixtureGaussianPrior::from_grid(vb, vec![tau * tau])?;
        let fixed = FixedGaussianPrior::new(tau as f32);

        let p = 20;
        let k = 1;
        let mean = Tensor::randn(0f64, 0.3f64, (p, k), &Device::Cpu)?;
        let var = (Tensor::ones((p, k), DType::F64, &Device::Cpu)? * 0.2)?;

        let kl_mix: f64 = mixture.kl_from_gaussian(&mean, &var)?.to_scalar()?;
        let kl_fixed: f64 = fixed
            .kl_from_gaussian(&mean.to_dtype(DType::F64)?, &var.to_dtype(DType::F64)?)?
            .to_scalar()?;

        assert!(
            (kl_mix - kl_fixed).abs() < 1e-4,
            "Single-component mixture KL ({}) should match fixed ({})",
            kl_mix,
            kl_fixed
        );
        Ok(())
    }
}
