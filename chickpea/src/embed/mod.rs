//! Joint multi-modal graph-embedding (SIMBA-inspired, sparse, count-NCE).
//!
//! Discriminative joint embedding of features and cells in a single
//! shared `H`-dim space. Trained via count-noise-contrastive estimation
//! (Gutmann & Hyvärinen 2010; mechanically a GloVe-style log-bilinear
//! count factorization with NB-Fisher feature weights for housekeeping
//! downweighting) on sketch-coarsened pseudobulk pseudographs.
//!
//! One relation: `(feature, cell)` over the unified panel — each input
//! file's rows concatenate into a shared feature axis; cell barcode is
//! the primary key for `E_cell`, so any barcode appearing in multiple
//! files shares one embedding row. Cross-modal alignment falls out of
//! the shared cell rows; modality (RNA / ATAC / protein / …) is purely
//! a naming convention from the user.
//!
//! Sparsity is preserved end-to-end: triplet streams from `SparseIoVec`
//! are sampled directly; no dense pseudobulk is materialized.

pub mod coarsen;
pub mod data;
pub mod eval;
pub mod fit;
pub mod loss;
pub mod model;
pub mod training;
