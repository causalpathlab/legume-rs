use candle_core::{Result, Tensor};

/// Black-box likelihood trait.
/// The likelihood function is treated as a black box - no gradients flow through it.
pub trait BlackBoxLikelihood {
    /// Evaluate log p(y|η) - NO gradients through this
    ///
    /// # Arguments
    /// * `etas` - Slice of linear predictor tensors, each shape (S, n, k) for S samples
    ///
    /// # Returns
    /// Log-likelihood values, shape (S,) summed over observations, or (S, n) per observation
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor>;
}

/// Variational distribution trait for local reparameterization.
pub trait VariationalDistribution {
    /// Get the variational mean μ.
    ///
    /// # Returns
    /// Mean tensor, shape (p, k)
    fn mean(&self) -> Result<Tensor>;

    /// Get the variational variance σ².
    ///
    /// # Returns
    /// Variance tensor, shape (p, k)
    fn var(&self) -> Result<Tensor>;
}

/// Prior distribution trait.
pub trait Prior {
    /// Compute log p(θ) under the prior.
    ///
    /// # Arguments
    /// * `theta` - Parameter samples, shape (S, p, k)
    ///
    /// # Returns
    /// Log prior probability, shape (S,) summed over parameter dimensions
    fn log_prob(&self, theta: &Tensor) -> Result<Tensor>;
}

/// Trait for priors that support analytical KL divergence from a Gaussian q.
///
/// Used by the local reparameterization trick to avoid sampling in p-space.
pub trait AnalyticalKL {
    /// KL(N(mean, diag(var)) || prior), summed over all (p, k) elements.
    ///
    /// # Arguments
    /// * `mean` - Variational mean, shape (p, k)
    /// * `var` - Variational variance, shape (p, k)
    ///
    /// # Returns
    /// Scalar tensor with the KL divergence
    fn kl_from_gaussian(&self, mean: &Tensor, var: &Tensor) -> Result<Tensor>;
}

/// Variational distribution with L mixture components, each with selection
/// probabilities and per-component effect sizes.
///
/// Extends `VariationalDistribution` with the structure needed by the
/// multilevel regression model.
pub trait ComponentVariational: VariationalDistribution {
    /// Selection probabilities per component, shape (L, p, k).
    fn alpha(&self) -> Result<Tensor>;
    /// Effect size means per component, shape (L, p, k).
    /// May be computed (e.g. broadcast from scalar effects), hence `Result<Tensor>`.
    fn beta_mean(&self) -> Result<Tensor>;
    /// Effect size standard deviations per component, shape (L, p, k).
    fn beta_std(&self) -> Result<Tensor>;
    /// Number of mixture components L.
    fn num_components(&self) -> usize;
}

/// Variational distribution with independent per-variable Bernoulli gates
/// (spike-and-slab). Each variable is included/excluded independently.
///
/// Unlike `ComponentVariational` (SuSiE), there is no component structure —
/// just per-variable inclusion indicators and effect sizes.
pub trait IndependentGateVariational: VariationalDistribution {
    /// Per-variable inclusion probabilities, shape (p, k).
    fn inclusion_prob(&self) -> Result<Tensor>;
    /// Per-variable effect size means, shape (p, k).
    fn effect_mean(&self) -> Result<Tensor>;
    /// Per-variable effect size stds, shape (p, k).
    fn effect_std(&self) -> Result<Tensor>;
    /// KL divergence of Bernoulli gates from prior.
    fn kl_bernoulli(&self, prior_inclusion: f64) -> Result<Tensor>;
}

/// Trait for models that support local reparameterization sampling.
pub trait LocalReparamModel {
    /// Sample η in n-space using the local reparameterization trick.
    fn forward(&self, num_samples: usize) -> Result<LocalReparamSample>;
}

/// Sample output from local reparameterization (sampling in n-space, not p-space).
pub struct LocalReparamSample {
    /// Linear predictor samples, shape (S, n, k) — sampled in n-space
    pub eta: Tensor,
    /// Analytical KL divergence, scalar
    pub kl: Tensor,
}
