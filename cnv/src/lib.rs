//! Copy number variation detection from single-cell expression data.
//!
//! Pipeline (per-sample HMM):
//! 1. [`hmm`] — core HMM primitives (forward-backward, Viterbi, EM).
//! 2. [`kmeans_init`] — kmeans+BIC for choosing K and seeding emission params.
//! 3. [`per_sample`] — top-level per-topic / per-sample HMM driver with
//!    iterative reference refinement.
//!
//! Ploidy is not identifiable from expression alone; states represent relative
//! CN (loss/neutral/gain or finer with K=5/6).

pub mod genome_order;
pub mod hmm;
pub mod kmeans_init;
pub mod per_sample;
