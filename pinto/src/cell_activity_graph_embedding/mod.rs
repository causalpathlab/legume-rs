//! `pinto cage` — activity-gated cell-graph embedding.
//!
//! Learns per-cell embeddings on the spatial cell-cell graph by visiting
//! each gene: every gene defines a per-cell activity vector that gates a
//! shared multi-scale cell-cell hierarchy. Embedding-only — no count decoder.
//!
//! The cell axis is collapsed into super-cells per coarsening level and one
//! shared SVD basis is fit over them ([`pb_basis`]); that basis warm-starts the
//! trained PB table. Chain levels differ only in their negative pools.

pub mod args;
pub mod fit;
pub mod gene_chain_sampler;
pub mod gene_gating;
pub mod loss;
pub mod pair_projection;
pub mod pb_basis;
pub mod pb_frame;
pub mod pretrained;

#[cfg(test)]
mod gene_gating_tests;
#[cfg(test)]
mod pb_frame_tests;

#[cfg(test)]
mod loss_tests;

#[cfg(test)]
mod pretrained_tests;

#[cfg(test)]
mod tests;

pub use args::CellActivityGraphEmbeddingArgs;
pub use fit::fit_cell_activity_graph_embedding;
