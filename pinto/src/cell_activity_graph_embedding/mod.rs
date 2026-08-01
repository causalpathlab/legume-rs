//! `pinto cage` — activity-gated cell-graph embedding.
//!
//! Learns per-cell embeddings on the spatial cell-cell graph by visiting
//! each gene: every gene defines a per-cell activity vector that gates a
//! shared multi-scale cell-cell hierarchy. Embedding-only — no count decoder.
//!
//! Gene selection is SAMPLED, not learned ([`selection`]): the cell axis is
//! collapsed into super-cells, one shared basis is fit to give the sampler a
//! frozen `D`-dim cell side, and geu's per-dim block Gibbs samples `z_gh`
//! against a pseudobulk Poisson.
//!
//! The selection reaches SGD as a GATE: `pip` is installed frozen
//! ([`JointEmbedModel::install_gate_pip`]) and a fresh `z ~ Bern(pip)` is drawn
//! once per epoch, so each epoch trains a different sub-network. `pip` is
//! re-estimated every `--selection-refresh-epochs` against the live embedding,
//! since the cold chain saw an SVD basis the model never uses.
//!
//! `e_feat` is randomly initialized, deliberately NOT set to `E[z·β]`: that
//! already carries the sampler's shrinkage, so gating it too would apply the
//! same selection twice. The two are alternatives, not complements —
//! `feature_posterior_mean.parquet` ships `E[z·β]` for downstream use.
//!
//! A learned variational spike-and-slab gate was tried first and removed: it
//! did not select at either end of its initialization. Chain levels differ only
//! in their negative pools.

pub mod args;
pub mod fit;
pub mod gene_chain_sampler;
pub mod gene_gating;
pub mod loss;
pub mod pair_projection;
pub mod selection;

#[cfg(test)]
mod loss_tests;

#[cfg(test)]
mod tests;

pub use args::CellActivityGraphEmbeddingArgs;
pub use fit::fit_cell_activity_graph_embedding;
