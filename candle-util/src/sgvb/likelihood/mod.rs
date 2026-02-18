//! Likelihood functions for SGVB inference.
//!
//! This module provides various likelihood functions that implement
//! the `BlackBoxLikelihood` trait for use with SGVB models.

mod gaussian;
mod negbinom;
mod poisson;
mod rss;
mod vmf;

pub use gaussian::{FixedGaussianLikelihood, GaussianLikelihood};
pub use negbinom::{lgamma_approx, NegativeBinomialLikelihood};
pub use poisson::{OffsetPoissonLikelihood, PoissonLikelihood};
pub use rss::{RssLikelihood, RssSvd};
pub use vmf::{
    estimate_kappa_mle, l2_normalize_dim, log_bessel_i, suggest_kappa_init, vmf_log_normalizer,
    VmfFixedKappaLikelihood,
};
