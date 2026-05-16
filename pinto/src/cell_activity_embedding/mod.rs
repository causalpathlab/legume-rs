//! `pinto cage` — activity-gated cell-graph embedding.
//!
//! Learns per-cell embeddings on the spatial cell-cell graph by visiting
//! each gene: every gene defines a per-cell activity vector that gates a
//! shared multi-scale cell-cell hierarchy. NCE updates are summed over a
//! small per-gene per-level learnable gate `α[g, ℓ]`. Embedding-only —
//! no count decoder.

pub mod args;
pub mod fit;
pub mod gene_chain_sampler;
pub mod gene_gating;

pub use args::CellActivityEmbeddingArgs;
pub use fit::fit_cell_activity_embedding;
