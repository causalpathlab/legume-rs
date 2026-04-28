//! Flat-K cell community model for spatial transcriptomics.
//!
//! Ports `link_community` from edges to cells: each cell u gets a single
//! community label z_u ∈ {0..K-1} via collapsed Gibbs on a Poisson DC-SBM
//! over per-cell gene profiles. Multi-level graph coarsening (spatial KNN +
//! cosine similarity on expression) runs Gibbs on super-cells first, then
//! transfers labels to fine resolution and refines with memoized
//! component-partitioned EM Gibbs + greedy.

pub mod fit;
pub mod gibbs;
pub mod hybrid;
pub mod model;
pub mod profiles;
