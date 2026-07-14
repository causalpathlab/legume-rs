//! MCMC engine: core sampling primitives, traits, and generic runner.
//!
//! - [`McmcModel`]: Trait for pluggable MCMC models.
//! - [`run_mcmc`], [`run_mcmc_parallel`]: Generic MCMC loop with warmup/thin/collection.
//! - [`elliptical_slice_step`]: Single ESS transition.
//! - [`EssSampler`]: Legacy ESS-specific runner (prefer [`run_mcmc`] for new models).
//! - [`McmcChain`]: Collected posterior samples with summary statistics.
//! - [`ess`], [`mcse_proportion`]: How much a chain's draws are actually worth.
//!
//! Note that "ESS" is two different things in the MCMC literature. Here, the *elliptical
//! slice sampler* lives in `elliptical_slice` and [`ess`] means *effective sample size* —
//! the names say which, so neither has to be guessed at.

mod chain;
pub mod diagnostics;
mod elliptical_slice;
mod model;
mod runner;
pub mod traits;

pub use chain::McmcChain;
pub use diagnostics::{ess, mcse_proportion};
pub use elliptical_slice::{elliptical_slice_step, EssSampler};
pub use model::McmcModel;
pub use runner::{run_mcmc, run_mcmc_parallel, McmcConfig};
pub use traits::{EssParam, EssParamSummary};
