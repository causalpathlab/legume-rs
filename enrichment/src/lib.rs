//! Bipartite gene-set enrichment for annotating trained topic / SVD / cluster
//! models against marker-gene databases.
//!
//! Given a group × gene profile matrix (topics' β, cluster centroids, SVD
//! loadings) and a marker-gene membership matrix (gene × celltype), compute a
//! K × C significance table via GSEA-style weighted KS enrichment scores with
//! Efron–Tibshirani restandardization against a pseudobulk-level permutation
//! null. The FDR-sparse Q matrix is a portable "celltype lens" — swap marker
//! DBs, recompute Q; train β once, annotate any dataset via θ · Q.

pub mod cellproj;
pub mod es;
pub mod fdr;
pub mod markers;
pub mod null;
pub mod orchestrate;
pub mod q_matrix;
pub mod specificity;

pub use cellproj::{label_cells, LabelWithConfidence};
pub use es::{rank_descending, weighted_ks_es};
pub use fdr::bh_fdr;
pub use orchestrate::{annotate, AnnotateConfig, AnnotateOutputs, GroupInputs};
pub use specificity::{compute_specificity, SpecificityMode};

pub type Mat = nalgebra::DMatrix<f32>;
