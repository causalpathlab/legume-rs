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

/// Variational distribution trait for reparameterized sampling.
pub trait VariationalDistribution {
    /// Sample using reparameterization trick.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples S to draw
    ///
    /// # Returns
    /// Tuple of (samples, epsilon) where:
    /// - samples: shape (S, p, k) sampled parameters
    /// - epsilon: shape (S, p, k) standard normal noise used for reparameterization
    fn sample(&self, num_samples: usize) -> Result<(Tensor, Tensor)>;

    /// Compute log q(θ|φ) for the variational distribution.
    ///
    /// # Arguments
    /// * `samples` - Parameter samples, shape (S, p, k)
    ///
    /// # Returns
    /// Log probability values, shape (S,) summed over parameter dimensions
    fn log_prob(&self, samples: &Tensor) -> Result<Tensor>;

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

/// Sample output from an SGVB model containing all components needed for REINFORCE.
pub struct SgvbSample {
    /// Linear predictor samples, shape (S, n, k)
    pub eta: Tensor,
    /// Log prior probability log p(θ), shape (S,)
    pub log_prior: Tensor,
    /// Log variational probability log q(θ) for reward computation (detached), shape (S,)
    pub log_q: Tensor,
    /// Log variational probability log q(θ) with gradients for surrogate loss, shape (S,)
    pub log_q_grad: Tensor,
}

/// SGVB model trait - encapsulates variational distribution, prior, and model structure.
pub trait SgvbModel {
    /// Sample and compute all ELBO components except likelihood.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples S to draw
    ///
    /// # Returns
    /// SgvbSample containing eta, log_prior, log_q, and log_q_grad
    fn sample(&self, num_samples: usize) -> Result<SgvbSample>;
}
