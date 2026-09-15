//! `senna gem` — joint gene-count embedding over the shared
//! `graph_embedding_util` engine: `senna bge`'s driver run over every feature
//! row of a gene-count matrix (rows = features, no modality split).
//!
//! Each row is `{gene}/count/{spliced|unspliced}`, matched across input files
//! by exact name (the row itself IS the join key). The former per-gene
//! β-sharing factorization and its analytic splice-velocity readout are gone
//! with the composite engine that produced them; a future task reintroduces
//! spliced/unspliced as explicit **tracks** on top of this same driver.

pub(crate) mod args;
/// Pooled per-gene HVG projection weights over a [`tracks::TrackPlan`]
/// (every track of a gene shares one selection decision, weight lands only
/// on the base row). Replaces the deleted `rows` module for the one thing
/// gem's HVG selection still needs.
pub(crate) mod hvg;
/// Multi-file input resolution and loading: matches `--modality` files to
/// gene files by sample id, loads them into one `UnifiedData`, and assigns
/// the [`tracks::TrackPlan`].
pub(crate) mod load;
/// Loading gem's co-embedded **feature** embedding (`{out}.feature_embedding.parquet`)
/// for the marker-space nearest-centroid call in `senna annotate-by-projection` / `senna lineage` —
/// the metric-compatible table, not β. See the module docs for why β/θ can't be used.
pub mod marker_embedding;
/// The `senna gem` run: joint gene-count embedding over the shared
/// `graph_embedding_util` engine (bge, over every feature row). Binary entry: [`run::run_gem_embedding`].
pub mod run;
pub mod sample_id;
/// Row-grammar track assignment: turns a gem feature axis (gene counts plus
/// any `--modality` files) into a [`tracks::TrackPlan`] /
/// `graph_embedding_util::fit::TrackSpec`.
pub(crate) mod tracks;
