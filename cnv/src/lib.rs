//! Copy number variation detection from single-cell expression data.
//!
//! Pipeline:
//! 1. [`genomic_coarsening`] — Greedy bottom-up merging of adjacent genes by
//!    correlation on `log(mu_resid)` profiles
//! 2. [`gibbs_hmm`] — Mixture of HMMs with blocked Gibbs sampling (ESS for
//!    emission params, forward-backward for states)
//! 3. [`hmm`] — Core HMM primitives (forward-backward, Viterbi, shared Viterbi)
//!
//! Ploidy is not identifiable from expression alone; states represent relative CN
//! (loss/neutral/gain).

pub mod coarsening_tree;
pub mod detect;
pub mod factorial_tree;
pub mod genome_order;
pub mod genomic_coarsening;
pub mod gibbs_hmm;
pub mod hmm;
