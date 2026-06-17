//! Library surface for `faba`.
//!
//! The crate ships primarily as the `faba` binary (see `main.rs`), but the
//! `gem` subsystem (joint gene-count + RNA-modification embedding) is also
//! exposed here as a library so its integration tests can live under
//! `tests/` and drive the real `model` + `train` + `sampling` stack through
//! the public API.

pub mod gem;

/// Single-sample editing statistics (beta-binomial p-values, Benjamini-Hochberg
/// FDR). Exposed so the pure-function tests can live under `tests/`.
pub mod hypothesis_tests;
