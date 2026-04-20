//! Geometric and numerical routines used across senna — dimensionality
//! reduction (PHATE, t-SNE), pseudobulk similarity construction, and
//! cell-level layout refinement. Pure functions on `nalgebra` matrices
//! with no I/O and no CLI dependencies, so they can be consumed freely
//! by `postprocess::*` (layout, annotate) and by training-time
//! helpers like `topic::anchor_prior`.

pub(crate) mod cell_layout;
pub(crate) mod phate;
pub(crate) mod similarity;
pub(crate) mod tsne;
pub(crate) mod umap;
