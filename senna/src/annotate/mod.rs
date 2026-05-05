//! `senna annotate` — bipartite enrichment annotation pipeline.
//!
//! Thin shell around the `enrichment` crate. Reads a run manifest
//! (`run.senna.json`) produced by any training subcommand, loads the matching
//! β / θ / pb_gene / pb_latent artifacts, parses a marker-gene TSV, runs the
//! enrichment `annotate()` pipeline, and writes:
//!
//! - `{out}.annotation.parquet` — N × C cell posterior
//! - `{out}.argmax.tsv` — per-cell label + max probability
//! - `{out}.topic_celltype_q.parquet` — K × C FDR-sparse softmax Q
//! - `{out}.topic_celltype_es.parquet` — K × C diagnostic (ES, es_std, p, q)
//!
//! The manifest is updated in place with an `annotate` section pointing at
//! the newly-written artifacts.

mod args;
pub mod inputs;
mod run;

pub use args::AnnotateArgs;
pub use run::annotate_run;
