use crate::candle_aux_linear::*;
use crate::candle_decoder_topic::{bgm_log_reconstruction, BgmState};
use crate::candle_indexed_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// Indexed topic decoder: computes likelihood only at selected feature positions
/// via `index_select`, while the softmax normalizer still covers all D features.
pub struct IndexedTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
}

impl IndexedTopicDecoder {
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        // No per-feature bias: only top-K features are observed per sample,
        // so rarely-selected features would get no corrective gradient on their bias.
        let dictionary = log_softmax_linear_nobias(n_topics, n_features, vs.pp("dictionary"))?;
        Ok(Self {
            n_features,
            n_topics,
            dictionary,
        })
    }

    pub fn dictionary(&self) -> &SoftmaxLinear {
        &self.dictionary
    }
}

impl IndexedDecoderT for IndexedTopicDecoder {
    fn forward_indexed(
        &self,
        log_z_nk: &Tensor,
        union_indices: &Tensor,
        indexed_x: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let log_beta_ks = self
            .dictionary
            .biased_weight_ks_conditional(union_indices, log_q_s)?;
        self.forward_indexed_with_log_beta(log_z_nk, &log_beta_ks, indexed_x)
    }

    fn get_conditional_log_beta_ks(
        &self,
        union_indices: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<Tensor> {
        self.dictionary
            .biased_weight_ks_conditional(union_indices, log_q_s)
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.dictionary.weight_dk()
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

/// Indexed BGM topic decoder: `p_ng = π_n · hk_g + (1-π_n) · Σ_k z_nk β_kg`
/// evaluated on the union feature indices.
pub struct BgmIndexedTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
    bgm: BgmState,
}

impl BgmIndexedTopicDecoder {
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear_nobias(n_topics, n_features, vs.pp("dictionary"))?;
        let bgm = BgmState::new(n_features, &vs)?;
        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            bgm,
        })
    }

    pub fn pi(&self) -> Result<f32> {
        self.bgm.pi()
    }

    /// `log hk` re-normalized over the union indices `S`, with the same
    /// importance-correction (`- log_q_s`) applied to `log β`.
    fn log_bgm_conditional_s(&self, union_indices: &Tensor, log_q_s: &Tensor) -> Result<Tensor> {
        let log_bgm_s = self
            .bgm
            .log_bgm_1d
            .index_select(union_indices, 1)?
            .broadcast_sub(log_q_s)?;
        ops::log_softmax(&log_bgm_s, log_bgm_s.rank() - 1)
    }
}

impl IndexedDecoderT for BgmIndexedTopicDecoder {
    fn forward_indexed(
        &self,
        log_z_nk: &Tensor,
        union_indices: &Tensor,
        indexed_x: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let log_beta_ks = self
            .dictionary
            .biased_weight_ks_conditional(union_indices, log_q_s)?;
        let log_beta_tilde_ns = {
            let z_nk = log_z_nk.exp()?;
            let beta_ks = log_beta_ks.exp()?;
            (z_nk.matmul(&beta_ks)? + 1e-8)?.log()?
        };
        let log_bgm_1s = self.log_bgm_conditional_s(union_indices, log_q_s)?;
        let log_recon_ns =
            bgm_log_reconstruction(&log_beta_tilde_ns, &log_bgm_1s, &self.bgm.logit_pi)?;

        let llik = indexed_x
            .clamp(0.0, f64::INFINITY)?
            .mul(&log_recon_ns)?
            .sum(indexed_x.rank() - 1)?;
        Ok((log_recon_ns, llik))
    }

    fn get_conditional_log_beta_ks(
        &self,
        union_indices: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<Tensor> {
        self.dictionary
            .biased_weight_ks_conditional(union_indices, log_q_s)
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.dictionary.weight_dk()
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
    use crate::candle_decoder_topic::MultinomTopicDecoder;
    use crate::candle_model_traits::DecoderModuleT;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_indexed_matches_dense_all_features() {
        let n_features = 10;
        let n_topics = 3;
        let n_samples = 4;
        let device = Device::Cpu;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let dense_decoder = MultinomTopicDecoder::new(n_features, n_topics, vb.pp("dec")).unwrap();
        let indexed_decoder = IndexedTopicDecoder::new(n_features, n_topics, vb.pp("dec")).unwrap();

        let logits = Tensor::randn(0.0f32, 1.0, (n_samples, n_topics), &device).unwrap();
        let log_z = candle_nn::ops::log_softmax(&logits, 1).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (n_samples, n_features), &device)
            .unwrap()
            .abs()
            .unwrap();

        // Dense forward
        let (_recon_dense, llik_dense) = dense_decoder
            .forward_with_llik(&log_z, &x, &|_, _| unreachable!())
            .unwrap();

        // Indexed forward with ALL features and uniform log_q (should match dense exactly)
        let all_indices: Vec<u32> = (0..n_features as u32).collect();
        let union_indices = Tensor::from_vec(all_indices, (n_features,), &device).unwrap();
        let log_q_s = Tensor::zeros((1, n_features), candle_core::DType::F32, &device).unwrap();
        let (log_recon_indexed, llik_indexed) = indexed_decoder
            .forward_indexed(&log_z, &union_indices, &x, &log_q_s)
            .unwrap();

        let llik_dense_vals: Vec<f32> = llik_dense.to_vec1().unwrap();
        let llik_indexed_vals: Vec<f32> = llik_indexed.to_vec1().unwrap();

        for (d, i) in llik_dense_vals.iter().zip(llik_indexed_vals.iter()) {
            assert!((d - i).abs() < 1e-4, "dense={} vs indexed={}", d, i);
        }

        let recon_dense_log = dense_decoder.dictionary().forward_log(&log_z).unwrap();
        let recon_dense_vals: Vec<Vec<f32>> = recon_dense_log.to_vec2().unwrap();
        let recon_indexed_vals: Vec<Vec<f32>> = log_recon_indexed.to_vec2().unwrap();

        for (row_d, row_i) in recon_dense_vals.iter().zip(recon_indexed_vals.iter()) {
            for (d, i) in row_d.iter().zip(row_i.iter()) {
                assert!((d - i).abs() < 1e-4, "recon dense={} vs indexed={}", d, i);
            }
        }
    }

    #[test]
    fn test_indexed_subset_features() {
        let n_features = 10;
        let n_topics = 3;
        let n_samples = 2;
        let device = Device::Cpu;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let indexed_decoder = IndexedTopicDecoder::new(n_features, n_topics, vb.pp("dec")).unwrap();

        let logits = Tensor::randn(0.0f32, 1.0, (n_samples, n_topics), &device).unwrap();
        let log_z = candle_nn::ops::log_softmax(&logits, 1).unwrap();

        let x_full = Tensor::randn(0.0f32, 1.0, (n_samples, n_features), &device)
            .unwrap()
            .abs()
            .unwrap();

        // Select a subset of features
        let subset: Vec<u32> = vec![1, 3, 5, 7];
        let s = subset.len();
        let union_indices = Tensor::from_vec(subset.clone(), (s,), &device).unwrap();
        let x_indexed = x_full.index_select(&union_indices, 1).unwrap();

        // Indexed forward with uniform log_q (conditional softmax)
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, &device).unwrap();
        let (log_recon_ns, llik_indexed) = indexed_decoder
            .forward_indexed(&log_z, &union_indices, &x_indexed, &log_q_s)
            .unwrap();

        // log_recon should be valid (finite, <= 0 for log-probabilities)
        let recon_vals: Vec<Vec<f32>> = log_recon_ns.to_vec2().unwrap();
        for row in &recon_vals {
            for &v in row {
                assert!(v.is_finite(), "non-finite log_recon: {}", v);
                assert!(v <= 0.0 + 1e-6, "log_recon > 0: {}", v);
            }
        }

        // llik should be finite and negative (log-likelihood of counts)
        let llik_vals: Vec<f32> = llik_indexed.to_vec1().unwrap();
        for &v in &llik_vals {
            assert!(v.is_finite(), "non-finite llik: {}", v);
        }
    }
}
