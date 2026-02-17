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

/// Sample output from local reparameterization (sampling in n-space, not p-space).
pub struct LocalReparamSample {
    /// Linear predictor samples, shape (S, n, k) — sampled in n-space
    pub eta: Tensor,
    /// Analytical KL divergence, scalar
    pub kl: Tensor,
}
