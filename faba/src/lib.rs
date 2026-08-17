//! Library surface for `faba`.
//!
//! The crate ships primarily as the `faba` binary (see `main.rs`), but the
//! `gem` subsystem (joint spliced + unspliced gene-count embedding) is also
//! exposed here as a library so its integration tests can live under
//! `tests/` and drive the real `model` + `train` + `sampling` stack through
//! the public API.

pub mod gem;

/// `{out}.gem.json` — the one place a consumer can ask which faba program
/// produced a prefix. Written by every gem-family producer, read by every
/// consumer that would otherwise have to guess.
pub mod manifest;

/// `faba gem-encoder` — masked generative embedding of the nascent→mature
/// transition (`u + δ → s`). Binary entries: [`gem_encoder::run::run_gem_encoder`]
/// and [`gem_encoder::args::GemEncoderArgs`].
pub mod gem_encoder;

/// The two-sample and single-sample statistics the editing caller is built on:
/// beta-binomial and Fisher-exact p-values, and the log odds ratio. Exposed so
/// the pure-function tests can live under `tests/`.
///
/// Deliberately NO FDR adjustment. Neighbouring conversion sites are covered by
/// the same reads, and a read converted at one site is evidence against its
/// unconverted neighbour, so the dependence is not even reliably positive and
/// Benjamini-Hochberg's assumption fails. Editing selects on a marginal p-value
/// and claims no FDR guarantee. Callers whose units genuinely are independent
/// use [`matrix_util::hypothesis::benjamini_hochberg`].
pub mod hypothesis_tests;
