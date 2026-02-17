//! Stochastic Gradient Variational Bayes (SGVB) module.
//!
//! Implements variational inference using the local reparameterization trick
//! with analytical KL divergence and antithetic sampling.
//!
//! # Key characteristics
//!
//! - Likelihood is black-box (no gradients through it)
//! - Local reparameterization: samples η in n-space instead of θ in p-space
//! - Analytical KL divergence (no MC estimation needed)
//! - Antithetic sampling for variance reduction
//! - Linear model: `η = X * θ` where `θ ~ q(θ) = N(μ, σ)`
//!
//! # Example
//!
//! ```ignore
//! use candle_util::sgvb::{LinearRegressionSGVB, GaussianPrior, SGVBConfig, local_reparam_loss, BlackBoxLikelihood};
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
//!     let loss = local_reparam_loss(&model, &likelihood, config.num_samples, 1.0)?;
//!     // optimizer.backward_step(&loss)?;
//! }
//!
//! // Get posterior mean predictions
//! let predictions = model.eta_mean()?;
//! ```

pub mod cavi_susie;
mod composite_model;
mod gaussian_prior;
pub mod likelihood;
mod regression_linear;
#[allow(clippy::module_inception)]
mod sgvb;
mod traits;
pub mod variant_tree;
mod variational_bisusie;
mod variational_gaussian;
mod variational_io;
mod variational_multilevel_susie;
mod variational_susie;

pub use composite_model::{
    composite_local_reparam_loss, samples_local_reparam_loss, CompositeModel,
};
pub use gaussian_prior::{FixedGaussianPrior, GaussianPrior};
pub use likelihood::{
    estimate_kappa_mle, l2_normalize_dim, lgamma_approx, log_bessel_i, suggest_kappa_init,
    vmf_log_normalizer, FixedGaussianLikelihood, GaussianLikelihood, NegativeBinomialLikelihood,
    OffsetPoissonLikelihood, PoissonLikelihood, VmfFixedKappaLikelihood,
};
pub use regression_linear::{LinearModelSGVB, LinearRegressionSGVB};
pub use sgvb::{local_reparam_loss, SGVBConfig};
pub use traits::{
    AnalyticalKL, BlackBoxLikelihood, LocalReparamSample, Prior, VariationalDistribution,
};
pub use variational_bisusie::BiSusieVar;
pub use variational_gaussian::GaussianVar;
pub use variational_io::{SparseVariationalOutput, VariationalOutput};
pub use variational_multilevel_susie::MultiLevelSusieVar;
pub use variational_susie::SusieVar;
