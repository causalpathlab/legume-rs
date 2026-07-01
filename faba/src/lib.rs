//! Library surface for `faba`.
//!
//! The crate ships primarily as the `faba` binary (see `main.rs`), but the
//! `gem` subsystem (joint gene-count + RNA-modification embedding) is also
//! exposed here as a library so its integration tests can live under
//! `tests/` and drive the real `model` + `train` + `sampling` stack through
//! the public API.

/// Canonical feature-name (sparse-matrix row) convention shared by every
/// modality: `{gene}/{modality}/{subunit}/{channel}` (channel innermost; subunit
/// optional). Producers format rows with `feature_row`; consumers split them with
/// `parse_feature_row`.
pub mod feature_name;

pub mod gem;

/// Single-sample editing statistics (beta-binomial p-values, Benjamini-Hochberg
/// FDR). Exposed so the pure-function tests can live under `tests/`.
pub mod hypothesis_tests;
