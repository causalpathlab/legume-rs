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

pub mod annotate;
pub mod assoc;
pub mod carried_rows;
pub mod cluster;
pub mod cluster_aggregation;
pub mod embed_common;
pub mod lineage;
pub mod lineage_plot;
#[path = "gem/marker_embedding.rs"]
pub mod marker_embedding;
pub mod marker_support;
pub mod multiome_layout;
pub mod output_helpers;
pub mod pb_reference;
pub mod principal_graph;
pub mod pseudotime;
pub mod run_manifest;
pub mod senna_input;
