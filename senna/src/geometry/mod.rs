//! Geometric and numerical routines used across senna — dimensionality
//! reduction (PHATE, t-SNE), pseudobulk similarity construction, cell-level
//! layout refinement, and vMF topic-to-celltype assignment. These are pure
//! functions on `nalgebra` matrices with no I/O and no CLI dependencies, so
//! they can be consumed freely by `postprocess::*` (visualize, annotate) and
//! by training-time helpers.

pub(crate) mod cell_layout;
pub(crate) mod phate;
pub(crate) mod similarity;
pub(crate) mod tsne;
pub(crate) mod vmf;
