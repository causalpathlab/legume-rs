use crate::candle_model_traits::EssLlikFn;
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
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

    fn dim_latent(&self) -> usize;
}

/// Indexed decoder: computes the importance-weighted conditional softmax
/// over the per-batch union `[S]` and gathers the per-cell likelihood at
/// each cell's `K` selected positions. The dense `[N, S]` reconstruction
/// is built and consumed inside the forward pass — never returned.
pub trait IndexedDecoderT {
    /// Forward pass at the per-batch union; returns per-cell log-likelihood.
    ///
    /// # Arguments
    /// * `log_z_nk` - [N, K_topics] log-probabilities from encoder
    /// * `union_indices` - [S] u32; sorted union of per-cell top-K ids
    /// * `scatter_pos` - [N, K] u32 in [0, S); per-cell positions of values in the union
    /// * `values` - [N, K] f32; per-cell observed counts (in scatter_pos order)
    /// * `values_weight` - [N, K] f32 (optional); per-position multiplicative
    ///   weight on the `(value+1).log()` likelihood term. Use to apply
    ///   NB-Fisher info weights so housekeeping observations contribute
    ///   less to the loss. `None` reduces to the bare `(value+1).log()`.
    /// * `log_q_s` - [1, S] log selection frequency for importance weighting
    ///
    /// # Returns
    /// * `llik_n` - [N] per-cell log-likelihood
    fn forward_indexed(
        &self,
        log_z_nk: &Tensor,
        union_indices: &Tensor,
        scatter_pos: &Tensor,
        values: &Tensor,
        values_weight: Option<&Tensor>,
        log_q_s: &Tensor,
    ) -> Result<Tensor>;

    /// Forward pass with a pre-computed `log β [K, S]`.
    ///
    /// Materializes `recon [N, S]` inside the function via `z @ exp(β)` and
    /// gathers the per-cell likelihood at `scatter_pos` — `[N, S]` does
    /// not escape the call.
    fn forward_indexed_with_log_beta(
        &self,
        log_z_nk: &Tensor,
        log_beta_ks: &Tensor,
        scatter_pos: &Tensor,
        values: &Tensor,
        values_weight: Option<&Tensor>,
    ) -> Result<Tensor> {
        let z_nk = log_z_nk.exp()?; // [N, K_topics]
        let beta_ks = log_beta_ks.exp()?; // [K, S]
        let recon_ns = z_nk.matmul(&beta_ks)?; // [N, S]
        let log_recon_ns = (recon_ns + 1e-8)?.log()?;
        let log_recon_at = log_recon_ns.gather(scatter_pos, 1)?; // [N, K]
                                                                 // Log1p-weighted likelihood: log(1+x) compresses dynamic range so
                                                                 // marker counts contribute comparably to housekeeping. NB-Fisher
                                                                 // weights, when supplied, further attenuate housekeeping
                                                                 // observations so β doesn't get pulled toward modeling them.
        let weight = (values + 1.0)?.log()?;
        let weight = match values_weight {
            Some(w) => weight.mul(w)?,
            None => weight,
        };
        let llik = weight.mul(&log_recon_at)?.sum(values.rank() - 1)?;
        Ok(llik)
    }

    /// Compute `(log β_ks_live, β_ks_detached)` — the live slice for
    /// gradient and the detached + exp'd version for ESS.
    fn prepare_dictionary_slice(
        &self,
        union_indices: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let log_beta_ks = self.get_conditional_log_beta_ks(union_indices, log_q_s)?;
        let beta_ks = log_beta_ks.detach().exp()?;
        Ok((log_beta_ks, beta_ks))
    }

    /// Importance-weighted conditional log-dictionary `[K, S]`.
    ///
    /// Default slices the full `[D, K]` dictionary. Override for `O(K·S)`
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

    /// Get the full dictionary matrix `[D, K]` in log-space (`O(K·D)`
    /// softmax). Used for output / evaluation.
    fn get_dictionary(&self) -> Result<Tensor>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;

    /// Build a lightweight per-cell log-likelihood closure for ESS.
    ///
    /// Pre-computes the `[K, S]` dictionary slice once, then the closure
    /// gathers per-cell likelihood from the same packed inputs the
    /// decoder consumes during training.
    fn build_ess_llik<'a>(
        &'a self,
        union_indices: &'a Tensor,
        scatter_pos: &'a Tensor,
        values: &'a Tensor,
        values_weight: Option<&'a Tensor>,
        log_q_s: &Tensor,
        topic_smoothing: f64,
    ) -> Result<EssLlikFn<'a>> {
        let log_beta_ks = self
            .get_conditional_log_beta_ks(union_indices, log_q_s)?
            .detach();
        let beta_ks = log_beta_ks.exp()?.contiguous()?; // [K, S]
        Self::build_ess_llik_from_beta(
            beta_ks,
            scatter_pos,
            values,
            values_weight,
            topic_smoothing,
            self.dim_latent(),
        )
    }

    /// Same as [`build_ess_llik`] but takes a pre-computed β `[K, S]`.
    fn build_ess_llik_from_beta<'a>(
        beta_ks: Tensor,
        scatter_pos: &'a Tensor,
        values: &'a Tensor,
        values_weight: Option<&'a Tensor>,
        topic_smoothing: f64,
        dim_latent: usize,
    ) -> Result<EssLlikFn<'a>> {
        use candle_nn::ops;

        let v_pos = values.clamp(0.0, f64::INFINITY)?;
        let weight_base = match values_weight {
            Some(w) => v_pos.mul(w)?,
            None => v_pos,
        };
        let k = dim_latent as f64;

        Ok(Box::new(move |z_nk: &Tensor| {
            let mut z = ops::softmax(z_nk, 1)?;
            if topic_smoothing > 0.0 {
                z = ((z * (1.0 - topic_smoothing))? + topic_smoothing / k)?;
            }
            let recon = z.matmul(&beta_ks)?; // [N, S]
            let log_recon_at = (recon + 1e-8)?.log()?.gather(scatter_pos, 1)?; // [N, K]
            weight_base.mul(&log_recon_at)?.sum(weight_base.rank() - 1)
        }))
    }
}
