//! Hierarchical Stochastic Block Model (HSBM) for graph community detection.
//!
//! A drop-in alternative to Leiden clustering that infers hierarchical
//! community structure using a Bayesian stochastic block model with a
//! binary tree prior. Inference is performed via Variational EM with
//! collapsed Gibbs sampling (E-step) and candle-based autodiff (M-step).
//!
//! # Model
//!
//! Poisson (Gamma-Poisson conjugate) with optional degree correction.
//!
//! # References
//!
//! Park & Bader (2017). "Fast and reliable inference algorithm for
//! hierarchical stochastic block models." arXiv:1711.05150.

#![deny(missing_docs)]
#![deny(warnings)]

/// Binary tree data structure with O(1) lowest-common-ancestor queries
pub mod btree;

/// Poisson (Gamma-Poisson) score functions for CPU and candle
pub mod model;

/// Sufficient statistics: Z*A*Z' edge counts, cluster sizes, volumes
pub mod sufficient_stats;

/// Collapsed Gibbs sampler (E-step)
pub mod gibbs;

/// Variational EM outer loop with candle-based M-step
pub mod variational;

#[cfg(test)]
mod test;

pub use variational::{Hsblock, HsbmOptions};
