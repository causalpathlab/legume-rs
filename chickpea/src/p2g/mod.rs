//! peak-to-gene subcommand.
//!
//! Workflow (see `chickpea/todo.md`):
//! 1. data-beans multilevel pb collapse (+ optional batch adjustment)
//! 2. rough ABC / pb co-occurrence map ([`abc_map`])
//! 3. train peak/gene embeds with `graph-embedding-util` ([`embed_ge`])
//! 4. embed pb samples → cluster → refine within cluster ([`cluster`], [`refine`])
//! 5. write E2G-like parquet ([`parquet_out`])
//!
//! The old rSVD / SuSiE / GhostKnockoff / LOCO-TMLE path has been removed.

pub mod run;

mod abc_map;
mod cluster;
mod embed_ge;
mod input;
mod parquet_out;
mod refine;
mod workflow;

pub use run::{run_peak_to_gene, PeakToGeneArgs};
