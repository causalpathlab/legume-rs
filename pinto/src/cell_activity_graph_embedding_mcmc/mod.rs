//! `pinto cage-mcmc` — Bayesian ESS variant of `pinto cage`.
//!
//! Same data pipeline and same per-pair Bernoulli logistic likelihood
//! as `cage`, but the parameters (`e_cell`, `e_gene`, γ, `b_cell`) are
//! sampled rather than optimized. Backend is pure-CPU `nalgebra` (no
//! Candle, no autograd). Per-sweep we cycle one Elliptical Slice
//! Sampling step over each parameter block; the log-likelihood
//! evaluation parallelizes over gene chunks via rayon — the
//! embarrassingly-parallel axis.

pub mod args;
pub mod fit;
pub mod loglik;
pub mod model;

pub use args::CageMcmcArgs;
pub use fit::fit_cage_mcmc;
