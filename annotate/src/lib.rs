//! Cell-type annotation library.
//!
//! This crate owns the annotation implementations (enrichment, projection,
//! ontology) and reads only plain file paths and loaded matrices. Run-manifest
//! loading, resolution, and mutation stay in `senna` — see
//! `senna::annotate_manifest` for the adapters that resolve a `run.senna.json`
//! into the input structs here and write the returned artifact paths back.
#![allow(
    clippy::wildcard_imports,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::struct_field_names
)]

pub mod args;
pub mod by_enrichment;
pub mod by_projection;
mod go_signature;
pub mod inputs;
pub mod mat_io;
pub mod ontology;
pub mod outputs;

pub use args::{AnnotateArgs, AnnotateOntologyArgs, AnnotateProjectionArgs};
pub use by_enrichment::{plan as plan_enrichment, run as annotate_by_enrichment, EnrichmentPlan};
pub use by_projection::{run as annotate_by_projection, ProjectionInputs};
pub use inputs::{load_cluster_labels, EnrichmentInputs};
pub use ontology::run as annotate_ontology;
pub use outputs::{clean_outputs, AnnotationOutputs, ENRICHMENT_OUTPUT_SUFFIXES};
