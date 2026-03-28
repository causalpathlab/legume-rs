//! MCMC engine: core sampling primitives, traits, and generic runner.
//!
//! - [`McmcModel`]: Trait for pluggable MCMC models.
//! - [`run_mcmc`], [`run_mcmc_parallel`]: Generic MCMC loop with warmup/thin/collection.
//! - [`elliptical_slice_step`]: Single ESS transition.
//! - [`EssSampler`]: Legacy ESS-specific runner (prefer [`run_mcmc`] for new models).
//! - [`McmcChain`]: Collected posterior samples with summary statistics.

mod chain;
mod ess;
mod model;
mod runner;
pub mod traits;

pub use chain::McmcChain;
pub use ess::{elliptical_slice_step, EssSampler};
pub use model::McmcModel;
pub use runner::{run_mcmc, run_mcmc_parallel, McmcConfig};
pub use traits::{EssParam, EssParamSummary};
