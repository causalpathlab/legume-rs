use candle_core::{Result, Tensor};

/// Indexed encoder: takes (indices, values) input from adaptive feature window.
///
/// Instead of processing all D features densely, operates on a union of per-sample
/// top-K feature indices (S << D), using embedding lookup for feature aggregation.
pub trait IndexedEncoderT {
    /// Forward pass with indexed input.
    ///
    /// # Arguments
    /// * `union_indices` - [S] u32, sorted union of per-sample top-K feature indices
    /// * `indexed_x` - [N, S] f32, input values at union positions
    /// * `indexed_x_null` - [N, S] f32, optional batch correction at union positions
    /// * `train` - whether to use dropout/batchnorm
    ///
    /// # Returns `(log_z_nk, kl_loss_n)`
    /// * `log_z_nk` - [N, K] log-probabilities on the simplex
    /// * `kl_loss_n` - [N] per-sample KL divergence
    fn forward_indexed_t(
        &self,
        union_indices: &Tensor,
        indexed_x: &Tensor,
        indexed_x_null: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

    fn dim_latent(&self) -> usize;
}

/// Indexed decoder: computes likelihood only at selected feature positions.
///
/// The dictionary [K, D] is sliced to [K, S] via `index_select` on the union indices,
/// reducing the intermediate tensor from N×K×D to N×K×S. The softmax normalization
/// is computed over all D features, so gradients flow to all logits through the
/// partition function even when only S positions are selected.
pub trait IndexedDecoderT {
    /// Forward pass at indexed feature positions.
    ///
    /// # Arguments
    /// * `log_z_nk` - [N, K] log-probabilities from encoder
    /// * `union_indices` - [S] u32, sorted union of per-sample top-K feature indices
    /// * `indexed_x` - [N, S] observed data at union positions
    ///
    /// # Returns `(log_recon_ns, llik_n)`
    /// * `log_recon_ns` - [N, S] log-reconstruction at indexed positions
    /// * `llik_n` - [N] per-sample log-likelihood
    fn forward_indexed(
        &self,
        log_z_nk: &Tensor,
        union_indices: &Tensor,
        indexed_x: &Tensor,
    ) -> Result<(Tensor, Tensor)>;

    /// Get the full dictionary matrix [D, K] in log-space
    fn get_dictionary(&self) -> Result<Tensor>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;

    /// Build a lightweight log-likelihood closure for ESS evaluation.
    ///
    /// Pre-computes the dictionary slice for the given union indices,
    /// avoiding repeated softmax and index_select across ESS iterations.
    #[allow(clippy::type_complexity)]
    fn build_ess_llik<'a>(
        &'a self,
        union_indices: &'a Tensor,
        indexed_x: &'a Tensor,
    ) -> Result<Box<dyn Fn(&Tensor) -> Result<Tensor> + 'a>> {
        use crate::candle_loss_functions::topic_likelihood;
        use candle_nn::ops;

        let log_dict_dk = self.get_dictionary()?.detach();
        let log_dict_sk = log_dict_dk.index_select(union_indices, 0)?;
        let beta_ks = log_dict_sk.t()?.exp()?.contiguous()?; // [K, S]
        let x_clamped = indexed_x.clamp(0.0, f64::INFINITY)?;

        Ok(Box::new(move |z_nk: &Tensor| {
            let z = ops::log_softmax(z_nk, 1)?.exp()?;
            let recon = z.matmul(&beta_ks)?; // [N, S]
            topic_likelihood(&x_clamped, &recon)
        }))
    }
}
