use crate::traits::indexed::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// Embedded indexed topic-model decoder.
///
/// β factorizes as `log_softmax_d(α · ρᵀ)` where:
///   - α `[K, H]` (`topic_embeddings`) is learnable in the decoder's scope.
///   - ρ `[D, H]` (`feature_embeddings`) is the **same** tensor handle held
///     by the encoder. Cloning it into the decoder preserves the underlying
///     `Var`, so optimizer updates from either path land on the same
///     parameter — exactly the ETM tying (Dieng et al., 2020).
///
/// The full `[K, D]` dictionary is never materialized on the gradient path:
/// the per-batch slice goes through `ρ[union, :] [S, H]` → `α · ρ_Sᵀ [K, S]`,
/// then the importance-weighted conditional softmax over `S`.
pub struct EmbeddedTopicDecoder {
    n_features: usize,
    n_topics: usize,
    embedding_dim: usize,
    topic_embeddings: Tensor,
    feature_embeddings: Tensor,
}

impl EmbeddedTopicDecoder {
    /// Construct with a shared feature embedding handle.
    ///
    /// `feature_embeddings` is expected to be the encoder's `[D, H]` table —
    /// pass `encoder.feature_embeddings().clone()`. The clone is a cheap
    /// `Arc` bump that preserves the `Var` backing the tensor.
    pub fn new(n_topics: usize, feature_embeddings: Tensor, vs: VarBuilder) -> Result<Self> {
        let dims = feature_embeddings.dims();
        if dims.len() != 2 {
            candle_core::bail!(
                "EmbeddedTopicDecoder: feature_embeddings must be 2-D [D, H], got {:?}",
                dims
            );
        }
        let n_features = dims[0];
        let embedding_dim = dims[1];

        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let topic_embeddings =
            vs.get_with_hints((n_topics, embedding_dim), "topic.embeddings", init_ws)?;

        Ok(Self {
            n_features,
            n_topics,
            embedding_dim,
            topic_embeddings,
            feature_embeddings,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Topic embeddings `α [K, H]`.
    pub fn topic_embeddings(&self) -> &Tensor {
        &self.topic_embeddings
    }

    /// Shared feature embeddings `ρ [D, H]`.
    pub fn feature_embeddings(&self) -> &Tensor {
        &self.feature_embeddings
    }

    /// Full `[K, D]` pre-softmax logits `α · ρᵀ`. Used by the anchor-prior
    /// cross-entropy penalty, which softmaxes along D internally.
    pub fn full_logits_kd(&self) -> Result<Tensor> {
        self.topic_embeddings.matmul(&self.feature_embeddings.t()?)
    }
}

impl IndexedDecoderT for EmbeddedTopicDecoder {
    fn forward_indexed(
        &self,
        log_z_nk: &Tensor,
        union_indices: &Tensor,
        scatter_pos: &Tensor,
        values: &Tensor,
        values_weight: Option<&Tensor>,
        log_q_s: &Tensor,
    ) -> Result<Tensor> {
        let log_beta_ks = self.get_conditional_log_beta_ks(union_indices, log_q_s)?;
        self.forward_indexed_with_log_beta(
            log_z_nk,
            &log_beta_ks,
            scatter_pos,
            values,
            values_weight,
        )
    }

    /// Importance-weighted conditional log-softmax over the per-batch union.
    ///
    /// Computes `log_softmax_S(α · ρ_Sᵀ − log_q_s)` — Jean et al. (2015)
    /// importance correction for top-K sampled softmax. `O(K·S·H)` work and
    /// no `[K, D]` materialization.
    fn get_conditional_log_beta_ks(
        &self,
        union_indices: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<Tensor> {
        let rho_sh = self.feature_embeddings.index_select(union_indices, 0)?; // [S, H]
        let w_ks = self.topic_embeddings.matmul(&rho_sh.t()?)?; // [K, S]
        let w_ks = w_ks.broadcast_sub(log_q_s)?;
        ops::log_softmax(&w_ks, w_ks.rank() - 1)
    }

    /// Full `[D, K]` log-β for output / evaluation. Builds the `[K, D]`
    /// logits and softmaxes along D once — only called outside the training
    /// hot path.
    fn get_dictionary(&self) -> Result<Tensor> {
        let logits_kd = self.full_logits_kd()?;
        let log_beta_kd = ops::log_softmax(&logits_kd, logits_kd.rank() - 1)?;
        log_beta_kd.transpose(0, 1)?.contiguous()
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    /// Build the decoder with a hand-crafted ρ, then check that
    /// `get_dictionary()` and a full-vocabulary `get_conditional_log_beta_ks`
    /// (log_q_s = 0, union = 0..D) agree — internal consistency between the
    /// per-batch slice and the dense output path.
    #[test]
    fn test_indexed_full_vocab_matches_dictionary() {
        let n_features = 10;
        let n_topics = 3;
        let embedding_dim = 5;
        let device = Device::Cpu;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let rho = vb
            .pp("enc")
            .get_with_hints(
                (n_features, embedding_dim),
                "feature.embeddings",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )
            .unwrap();
        let dec = EmbeddedTopicDecoder::new(n_topics, rho.clone(), vb.pp("dec")).unwrap();

        let all_idx: Vec<u32> = (0..n_features as u32).collect();
        let union_indices = Tensor::from_vec(all_idx, (n_features,), &device).unwrap();
        let log_q_s = Tensor::zeros((1, n_features), candle_core::DType::F32, &device).unwrap();

        let log_beta_ks = dec
            .get_conditional_log_beta_ks(&union_indices, &log_q_s)
            .unwrap(); // [K, D]
        let log_beta_dk_ref = dec.get_dictionary().unwrap(); // [D, K]
        let log_beta_ks_from_dict = log_beta_dk_ref.t().unwrap(); // [K, D]

        let a: Vec<Vec<f32>> = log_beta_ks.to_vec2().unwrap();
        let b: Vec<Vec<f32>> = log_beta_ks_from_dict.to_vec2().unwrap();
        for (ar, br) in a.iter().zip(b.iter()) {
            for (av, bv) in ar.iter().zip(br.iter()) {
                assert!((av - bv).abs() < 1e-5, "slice={} dict={}", av, bv);
            }
        }
    }

    /// Forward pass on a feature subset produces finite per-cell llik and
    /// the per-batch softmax sums to 1 along S.
    #[test]
    fn test_indexed_subset_forward_finite() {
        let n_features = 12;
        let n_topics = 4;
        let embedding_dim = 6;
        let n_samples = 3;
        let device = Device::Cpu;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let rho = vb
            .pp("enc")
            .get_with_hints(
                (n_features, embedding_dim),
                "feature.embeddings",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )
            .unwrap();
        let dec = EmbeddedTopicDecoder::new(n_topics, rho, vb.pp("dec")).unwrap();

        let logits = Tensor::randn(0.0f32, 1.0, (n_samples, n_topics), &device).unwrap();
        let log_z = ops::log_softmax(&logits, 1).unwrap();

        let subset: Vec<u32> = vec![1, 3, 5, 7, 9];
        let s = subset.len();
        let union_indices = Tensor::from_vec(subset, (s,), &device).unwrap();

        // Per-cell positions in the union — same K=S for simplicity.
        let scatter_row: Vec<u32> = (0..s as u32).collect();
        let scatter_flat: Vec<u32> = (0..n_samples)
            .flat_map(|_| scatter_row.iter().copied())
            .collect();
        let scatter_pos = Tensor::from_vec(scatter_flat, (n_samples, s), &device).unwrap();

        let values = Tensor::randn(0.0f32, 1.0, (n_samples, s), &device)
            .unwrap()
            .abs()
            .unwrap();
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, &device).unwrap();

        // Slice sums to 1 along S.
        let log_beta_ks = dec
            .get_conditional_log_beta_ks(&union_indices, &log_q_s)
            .unwrap();
        let row_sum_exp = log_beta_ks.exp().unwrap().sum(1).unwrap();
        let sums: Vec<f32> = row_sum_exp.to_vec1().unwrap();
        for s in &sums {
            assert!((s - 1.0).abs() < 1e-4, "row sum {} != 1", s);
        }

        let llik = dec
            .forward_indexed(
                &log_z,
                &union_indices,
                &scatter_pos,
                &values,
                None,
                &log_q_s,
            )
            .unwrap();
        let lv: Vec<f32> = llik.to_vec1().unwrap();
        for &v in &lv {
            assert!(v.is_finite(), "non-finite llik {}", v);
        }
    }

    /// Decoder gradients flow into ρ (proving the encoder/decoder share the
    /// same Var). Build ρ, wrap it as a Var, construct the decoder around
    /// the Var's tensor, compute a llik, backward, and check ρ has a grad.
    #[test]
    fn test_decoder_grad_reaches_feature_embeddings() {
        use candle_core::Var;
        let n_features = 6;
        let n_topics = 2;
        let embedding_dim = 4;
        let device = Device::Cpu;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Build ρ as a Var inside the VarMap (mimicking the encoder).
        let rho_init = Tensor::randn(0.0f32, 1.0, (n_features, embedding_dim), &device).unwrap();
        let rho_var = Var::from_tensor(&rho_init).unwrap();
        let rho_tensor = rho_var.as_tensor().clone();
        let dec = EmbeddedTopicDecoder::new(n_topics, rho_tensor, vb.pp("dec")).unwrap();

        let log_z = ops::log_softmax(
            &Tensor::randn(0.0f32, 1.0, (2, n_topics), &device).unwrap(),
            1,
        )
        .unwrap();
        let subset: Vec<u32> = vec![0, 1, 2, 3];
        let s = subset.len();
        let union_indices = Tensor::from_vec(subset, (s,), &device).unwrap();
        let scatter_row: Vec<u32> = (0..s as u32).collect();
        let scatter_flat: Vec<u32> = (0..2).flat_map(|_| scatter_row.iter().copied()).collect();
        let scatter_pos = Tensor::from_vec(scatter_flat, (2, s), &device).unwrap();
        let values = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (2, s),
            &device,
        )
        .unwrap();
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, &device).unwrap();

        let llik = dec
            .forward_indexed(
                &log_z,
                &union_indices,
                &scatter_pos,
                &values,
                None,
                &log_q_s,
            )
            .unwrap();
        let loss = llik.mean_all().unwrap().neg().unwrap();
        let grads = loss.backward().unwrap();
        assert!(
            grads.get(rho_var.as_tensor()).is_some(),
            "no gradient reached shared feature embeddings"
        );
    }
}
