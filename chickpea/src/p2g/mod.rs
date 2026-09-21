//! peak-to-gene subcommand.
//!
//! The old rSVD embedding + SuSiE-RSS + GhostKnockoff / TMLE path has been
//! removed. The intended pipeline trains peak/gene embeddings with
//! `graph-embedding-util`, embeds cells, clusters, and refines peak→gene within
//! each cluster, writing E2G-like parquet. That association path is not wired
//! yet — see `todo.md`.

pub mod run;

mod input;

pub use run::{run_peak_to_gene, PeakToGeneArgs};
