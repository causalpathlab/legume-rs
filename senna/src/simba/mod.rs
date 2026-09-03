//! `senna simba`: the faithful SIMBA baseline (see `graph_embedding_util::simba`).
//!
//! Thin by design: the recipe lives in the shared crate; this module owns the
//! command line, the same loader / cell QC / HVG plumbing as `bge`, and the
//! bge-shaped artifacts (`cell_embedding`, `feature_embedding`,
//! `feature_loading`, `h0..h{D-1}` columns, `senna.json`) so one comparison
//! script reads every arm.

mod args;
mod run;

pub use args::SimbaArgs;
pub use run::fit_simba;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
