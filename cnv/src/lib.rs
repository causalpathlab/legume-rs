//! Copy number variation detection from single-cell expression data.
//!
//! Pipeline:
//! 1. [`gene_loci`] — GFF gene loci for `data-beans` row names.
//! 2. [`infercnv`] — scalable inferCNV-style log-ratio: reference subtraction
//!    and chromosome-bounded window smoothing (`O(G)` per column).
//! 3. [`cell_profile`] — streamed per-cell inferCNV profiles from backends
//!    to a genomic-interval backend (the `cnv infercnv` binary).
//! 4. [`clone_call`] — chromosome-arm sketch, cluster, donor-purity +
//!    spatial-structure test → stratum `0` (mixable) vs donor-private clones.
//! 5. [`hmm`] — core HMM primitives (forward-backward, Viterbi, EM).
//! 6. [`kmeans_init`] — kmeans+BIC for choosing K and seeding emission params.
//! 7. [`per_sample`] — top-level per-topic / per-sample HMM driver with
//!    iterative reference refinement.
//!
//! Ploidy is not identifiable from expression alone; states represent relative
//! CN (loss/neutral/gain or finer with K=5/6).

pub mod cell_profile;
pub mod clone_call;
pub mod gene_loci;
pub mod genome_order;
pub mod hmm;
pub mod infercnv;
pub mod kmeans_init;
pub mod per_sample;
