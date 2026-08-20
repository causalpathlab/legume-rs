use crate::data::indexed::SparseEdgeBatch;
use candle_core::{Result, Tensor};

/// Indexed encoder: takes packed `(indices, values)` from the adaptive feature
/// window. Each cell carries its own top-K feature ids and values; the
/// encoder gathers feature embeddings at those ids and pools by value, so
/// no `[N, S]` is ever materialized.
pub trait IndexedEncoderT {
    /// Forward pass with packed indexed input.
    ///
    /// # Arguments
    /// * `indices` - [N, K] u32 in [0, D); per-cell top-K feature ids
    /// * `values` - [N, K] f32; per-cell raw values at those ids
    /// * `values_null` - [N, K] f32 (optional); μ_residual gathered at the same ids
    /// * `values_mean` - [N, K] f32 (optional); per-gene mean rate `μ_d` gathered
    ///   at the same ids. Composes with `values_null` as a multiplicative
    ///   count-rate divisor inside `anscombe_lite`, so the encoder sees the
    ///   Anscombe-stabilized biological deviation from each gene's typical
    ///   rate under the prevailing batch.
    /// * `sparse_edges` - pre-normalised per-cell sub-adjacency edges
    ///   (pre-built by [`crate::data::indexed::IndexedInMemoryData::minibatch_sparse_edges`]),
    ///   supplied when the encoder owns a graph-diffusion block.
    /// * `train` - whether to use dropout/batchnorm
    ///
    /// # Returns `(log_z_nk, kl_loss_n)`
    /// * `log_z_nk` - [N, K_topics] log-probabilities on the simplex
    /// * `kl_loss_n` - [N] per-sample KL divergence
    fn forward_indexed_t(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        sparse_edges: Option<&SparseEdgeBatch>,
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

    fn dim_latent(&self) -> usize;
}
