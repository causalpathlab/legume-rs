//! Stochastic Gradient Variational Bayes (SGVB) module.
//!
//! Implements Black Box Variational Inference using the score function (REINFORCE)
//! estimator with Gaussian reparameterization and control variates.
//!
//! # Key characteristics
//!
//! - Likelihood is black-box (no gradients through it)
//! - Score function gradient: `∇ELBO ≈ E[(normalized_reward) * ∇log q(θ)]`
//! - Control variate: `(reward - mean) / std`
//! - Linear model: `η = X * θ` where `θ ~ q(θ) = N(μ, σ)`
//!
//! # Example
//!
//! ```ignore
//! use candle_util::sgvb::{LinearRegressionSGVB, GaussianPrior, SGVBConfig, sgvb_loss, BlackBoxLikelihood};
//!
//! // Define your black-box likelihood
//! struct MyLikelihood { /* ... */ }
//! impl BlackBoxLikelihood for MyLikelihood {
//!     fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
//!         // Your likelihood computation here
//!     }
//! }
//!
//! // Create the model (encapsulates variational distribution, prior, design matrix)
//! let model = LinearRegressionSGVB::new(vb, x_design, k, prior, config.clone())?;
//!
//! // Training loop
//! for _ in 0..num_iters {
//!     let loss = sgvb_loss(&model, &likelihood, &config)?;
//!     // optimizer.backward_step(&loss)?;
//! }
//!
//! // Get posterior mean predictions
//! let predictions = model.eta_mean()?;
//! ```

mod gaussian;
mod linear_regression_model;
mod prior;
mod sgvb;
mod traits;

pub use gaussian::GaussianVariational;
pub use linear_regression_model::LinearRegressionSGVB;
pub use prior::{FixedGaussianPrior, GaussianPrior};
pub use sgvb::{compute_elbo, sgvb_loss, SGVBConfig};
pub use traits::{BlackBoxLikelihood, Prior, SgvbModel, SgvbSample, VariationalDistribution};
