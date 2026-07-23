//! Library surface for `faba`.
//!
//! The crate ships primarily as the `faba` binary (see `main.rs`), but the
//! `gem` subsystem (joint spliced + unspliced gene-count embedding) is also
//! exposed here as a library so its integration tests can live under
//! `tests/` and drive the real `model` + `train` + `sampling` stack through
//! the public API.

/// Canonical feature-name (sparse-matrix row) convention shared by every
/// modality: `{gene}/{modality}/{subunit}/{channel}` (channel innermost; subunit
/// optional). Producers format rows with `feature_row`; consumers split them with
/// `parse_feature_row`.
pub mod feature_name;

pub mod gem;

/// `{out}.gem.json` — the one place a consumer can ask which faba program
/// produced a prefix. Written by every gem-family producer, read by every
/// consumer that would otherwise have to guess.
pub mod manifest;

/// `faba gem-encoder` — masked generative embedding of the nascent→mature
/// transition (`u + δ → s`). Binary entries: [`gem_encoder::run::run_gem_encoder`]
/// and [`gem_encoder::args::GemEncoderArgs`].
pub mod gem_encoder;

/// Single-sample editing statistics (beta-binomial p-values, Benjamini-Hochberg
/// FDR). Exposed so the pure-function tests can live under `tests/`.
pub mod hypothesis_tests;
