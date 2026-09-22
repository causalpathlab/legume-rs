//! peak-to-gene subcommand.
//!
//! Workflow (see `chickpea/todo.md`):
//! 1. data-beans multilevel pb collapse (+ optional batch adjustment)
//! 2. score peak–gene links on the pb profiles, Pearson or ABC ([`link_map`])
//! 3. train peak/gene embeds with `graph-embedding-util` hier ([`embed_ge`])
//! 4. embed pb samples → cluster → recompute links within cluster ([`cluster`], [`refine`])
//! 5. write E2G-like parquet ([`parquet_out`])

pub mod run;

pub mod cells;
pub mod cluster;
pub mod embed_ge;
pub mod gene_activity;
pub mod input;
pub mod link_map;
pub mod module_init;
pub mod parquet_out;
pub mod pb_levels;
pub mod refine;
pub mod workflow;

pub use run::{run_peak_to_gene, PeakToGeneArgs};
