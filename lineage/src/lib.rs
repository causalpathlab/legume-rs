//! Lineage family (lineage, lineage_plot, assoc, pseudotime).
//!
//! This crate owns the trajectory implementations and reads only plain
//! file paths and loaded matrices. Run-manifest loading, resolution, and
//! mutation stay in `senna` — see `senna::lineage_manifest` for the
//! adapters that resolve a `run.senna.json` into the input structs here.
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

pub mod assoc;
pub mod lineage;
pub mod lineage_plot;
pub mod mat_io;
pub mod pseudotime;
