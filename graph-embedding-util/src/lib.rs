//! `graph-embedding` — count-NCE bipartite-graph embedding over
//! (cell, feature) edges.
//!
//! Modality-agnostic discriminative embedding: features and cells are
//! embedded into a single H-dimensional space and scored bilinearly,
//! optimized via NEG-style noise-contrastive estimation. Negatives are
//! drawn within each batch so the model can't earn signal by separating
//! cells along technical-batch confounders.
//!
//! Consumed by the `senna gbe` subcommand. The library has no clap or
//! run-manifest deps — callers translate their CLI arguments into
//! [`FitConfig`] and own the output naming.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::many_single_char_names
)]

pub mod cell_projection;
pub mod coarsen;
pub mod data;
pub mod eval;
pub mod feature_network;
pub mod feature_qc;
pub mod fit;
pub mod loss;
pub mod model;
pub mod null_call;
pub mod postprocess;
pub mod progress;
pub mod training;
pub mod type_annotation;

pub use auxiliary_data::feature_names::FeatureNameKind;
pub use data::{load_unified_data, validate_multiome_groups, LoadUnifiedArgs, UnifiedData};
pub use data_beans_alg::refine_multilevel::RefineParams;
pub use eval::{
    embedding_col_names, save_embedding, save_outputs, save_outputs_named,
    write_feature_coembedding, EmbeddingFileNames, OutputContext,
};
pub use feature_qc::{hvg_feature_qc, FeatureQcConfig, FeatureQcResult};
pub use fit::{
    fit, load_feature_network, CalibrationDiag, CalibrationKind, CellLineage, FeatFactorSpec,
    FeatureNetworkArgs, FeatureProjection, FeatureProjectionConfig, FitConfig, FitOutput,
    LineageQc, PbLevelVelocity, DEFAULT_PROJECTION_CALIB_RIDGE, DEFAULT_PROJECTION_RIDGE,
};
pub use model::JointEmbedModel;
pub use postprocess::{cell_clusters, feature_coembedding};

/// Graceful-stop on Ctrl+C. Lives in `matrix-util` so the annotation crates *below* this one
/// (`enrichment`, which owns the raw-count marker bootstrap) can share the same flag and the same
/// interrupt-safe replicate combinators — a process must have exactly one SIGINT handler, and
/// `ctrlc::set_handler` panics on a second registration. Re-exported here so `crate::stop::…` and
/// `graph_embedding_util::stop::…` keep resolving for the callers that already use them.
pub use matrix_util::stop;
pub use matrix_util::stop::{setup_stop_handler, stop_flag};
