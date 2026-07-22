//! `faba all` — the end-to-end pipeline that chains the per-modality
//! subcommands over one set of BAM files.
//!
//! Steps shared with the standalone subcommands (BAM index checks, mito QC,
//! mass enrichment) live in [`crate::quant`], which the standalone
//! entries also use.

/// The `faba all` command-line surface.
pub mod args;
/// The `faba all` run. Binary entry: [`run::run_pipeline`].
pub mod run;
/// The per-modality steps, in run order.
mod steps;
