//! `senna gem` — joint gene-count embedding over the shared
//! `graph_embedding_util` engine: `senna bge`'s driver run over every feature
//! row of a gene-count matrix (rows = features, no modality split).
//!
//! Each row is `{gene}/count/{spliced|unspliced}`, matched across input files
//! by exact name (the row itself IS the join key). The former per-gene
//! β-sharing factorization and its analytic splice-velocity readout are gone
//! with the composite engine that produced them; a future task reintroduces
//! spliced/unspliced as explicit **tracks** on top of this same driver.

pub mod args;
/// Per-gene pooling of the feature axis for HVG ranking (spliced + unspliced
/// rows of a gene share one entry). Replaces the deleted `rows` module for
/// the one thing gem's HVG selection still needs.
pub(crate) mod hvg;
/// Loading gem's co-embedded **feature** embedding (`{out}.feature_embedding.parquet`)
/// for the marker-space nearest-centroid call in `senna annotate-by-projection` / `senna lineage` —
/// the metric-compatible table, not β. See the module docs for why β/θ can't be used.
pub mod marker_embedding;
/// The `senna gem` run: joint gene-count embedding over the shared
/// `graph_embedding_util` engine (bge, over every feature row). Binary entry: [`run::run_gem_embedding`].
pub mod run;
pub mod sample_id;
