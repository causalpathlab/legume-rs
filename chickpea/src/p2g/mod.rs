//! peak-to-gene subcommand.
//!
//! Planned pipeline (see `chickpea/todo.md` and
//! `docs/superpowers/plans/2026-09-21-chickpea-ge-util-p2g.md`):
//! 1. rough ABC / pb co-occurrence map ([`abc_map`])
//! 2. train peak/gene embeds with `graph-embedding-util` ([`embed_ge`])
//! 3. embed cells → cluster → refine within cluster ([`cluster`], [`refine`])
//! 4. write E2G-like parquet ([`parquet_out`])
//!
//! The old rSVD / SuSiE / GhostKnockoff / LOCO-TMLE path has been removed.

pub mod run;

mod abc_map;
mod cluster;
mod embed_ge;
mod input;
mod parquet_out;
mod refine;

pub use run::{run_peak_to_gene, PeakToGeneArgs};
