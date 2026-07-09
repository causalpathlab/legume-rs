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
pub mod ontology;
pub mod ontology_enrich;
pub mod orchestrate;
pub mod q_matrix;
pub mod specificity;
pub mod treebh;

/// The label every annotator in the workspace writes for a cell, community or
/// trajectory node it declined to call.
///
/// It is a **wire format**: it lands in `*.annot.parquet` and
/// `trajectory_annotation.parquet`, and downstream readers (`faba plot`, `senna
/// plot`) branch on it to keep a non-call out of the palette and off the figure.
/// Producer and consumer live in different crates and different processes, so the
/// two must not drift — reference this const rather than retyping the string.
pub const UNASSIGNED_LABEL: &str = "unassigned";

pub use cellproj::{label_cells, LabelWithConfidence};
pub use es::{rank_descending, weighted_ks_es};
pub use fdr::bh_fdr;
pub use ontology::{
    annotate_ontology_core, parse_label_map, OntologyAccess, OntologyParams, OntologyScore,
};
pub use ontology_enrich::{ontology_module_score, OntologyModuleScore};
pub use orchestrate::{annotate, AnnotateConfig, AnnotateOutputs, GroupInputs};
pub use specificity::{compute_specificity, SpecificityMode};

pub type Mat = nalgebra::DMatrix<f32>;
