use crate::candle_model_traits::EssLlikFn;
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
    /// # Arguments
    /// * `log_q_s` - [1, S] log selection frequency for importance weighting
    fn forward_indexed(
        &self,
        log_z_nk: &Tensor,
        union_indices: &Tensor,
        indexed_x: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<(Tensor, Tensor)>;

    /// Forward pass using a pre-computed log-dictionary slice [K, S].
    ///
    /// Uses matmul `z @ exp(beta)` instead of logsumexp to reduce
    /// autograd graph depth (1 matmul vs 4 broadcast ops).
    fn forward_indexed_with_log_beta(
        &self,
        log_z_nk: &Tensor,
        log_beta_ks: &Tensor,
        indexed_x: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let z_nk = log_z_nk.exp()?; // [N, K] — log-probs to probs
        let beta_ks = log_beta_ks.exp()?; // [K, S]
        let recon_ns = z_nk.matmul(&beta_ks)?; // [N, S]
        let log_recon_ns = (recon_ns + 1e-8)?.log()?;

        let llik = indexed_x
            .clamp(0.0, f64::INFINITY)?
            .mul(&log_recon_ns)?
            .sum(indexed_x.rank() - 1)?;

        Ok((log_recon_ns, llik))
    }

    /// Compute the log-dictionary slice [K, S] at union indices.
    ///
    /// Returns `(log_beta_ks, beta_ks_detached)` — the live slice for gradient
    /// and the detached+exp'd version for ESS.
    fn prepare_dictionary_slice(
        &self,
        union_indices: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let log_beta_ks = self.get_conditional_log_beta_ks(union_indices, log_q_s)?;
        let beta_ks = log_beta_ks.detach().exp()?;
        Ok((log_beta_ks, beta_ks))
    }

    /// Get the importance-weighted conditional log-dictionary [K, S].
    ///
    /// Default slices the full [D, K] dictionary. Override for O(K·S)
    /// conditional softmax with importance weighting.
    fn get_conditional_log_beta_ks(
        &self,
        union_indices: &Tensor,
        _log_q_s: &Tensor,
    ) -> Result<Tensor> {
        let log_dict_dk = self.get_dictionary()?;
        let log_dict_sk = log_dict_dk.index_select(union_indices, 0)?;
        log_dict_sk.t()?.contiguous()
    }

    /// Get the full dictionary matrix [D, K] in log-space (O(K·D) softmax).
    ///
    /// Used for output/evaluation. Prefer `get_conditional_log_beta_ks` for training.
    fn get_dictionary(&self) -> Result<Tensor>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;

    /// Build a lightweight log-likelihood closure for ESS evaluation.
    ///
    /// Pre-computes the dictionary slice for the given union indices,
    /// avoiding repeated softmax and index_select across ESS iterations.
    fn build_ess_llik<'a>(
        &'a self,
        union_indices: &'a Tensor,
        indexed_x: &'a Tensor,
        log_q_s: &Tensor,
        topic_smoothing: f64,
    ) -> Result<EssLlikFn<'a>> {
        let log_beta_ks = self
            .get_conditional_log_beta_ks(union_indices, log_q_s)?
            .detach();
        let beta_ks = log_beta_ks.exp()?.contiguous()?; // [K, S]
        Self::build_ess_llik_from_beta(beta_ks, indexed_x, topic_smoothing, self.dim_latent())
    }

    /// Build ESS log-likelihood closure from a pre-computed β [K, S] matrix.
    ///
    /// Use this when the caller has already computed the dictionary slice
    /// (e.g., shared with the decoder forward pass) to avoid redundant
    /// log_softmax over [K, D].
    fn build_ess_llik_from_beta<'a>(
        beta_ks: Tensor,
        indexed_x: &'a Tensor,
        topic_smoothing: f64,
        dim_latent: usize,
    ) -> Result<EssLlikFn<'a>> {
        use candle_nn::ops;

        let x_pos = indexed_x.clamp(0.0, f64::INFINITY)?;
        let k = dim_latent as f64;

        Ok(Box::new(move |z_nk: &Tensor| {
            let mut z = ops::softmax(z_nk, 1)?;
            if topic_smoothing > 0.0 {
                z = ((z * (1.0 - topic_smoothing))? + topic_smoothing / k)?;
            }
            let recon = z.matmul(&beta_ks)?; // [N, S]
            x_pos.mul(&(recon + 1e-8)?.log()?)?.sum(x_pos.rank() - 1)
        }))
    }
}
