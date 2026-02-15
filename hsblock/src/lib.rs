//! Hierarchical Stochastic Block Model (HSBM) for graph community detection.
//!
//! A drop-in alternative to Leiden clustering that infers hierarchical
//! community structure using a Bayesian stochastic block model with a
//! binary tree prior. Inference is performed via collapsed Gibbs sampling,
//! with Poisson rates analytically integrated out via Gamma-Poisson conjugacy.
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

/// Poisson (Gamma-Poisson) score functions (CPU)
pub mod model;

/// Sufficient statistics: Z*A*Z' edge counts, cluster sizes, volumes
pub mod sufficient_stats;

/// Collapsed Gibbs sampler
pub mod gibbs;

/// Collapsed Gibbs inference loop
pub mod inference;

#[cfg(test)]
mod test;

pub use btree::GammaPoissonParam;
pub use inference::{Hsblock, HsbmOptions};
