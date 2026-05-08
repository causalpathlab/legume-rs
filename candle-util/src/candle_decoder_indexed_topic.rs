use crate::candle_aux_linear::*;
use crate::candle_indexed_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Indexed topic-model decoder.
///
/// Builds the importance-weighted conditional softmax over the per-batch
/// union `[S]`, materializes `recon [N, S] = z @ exp(β)` inside the
/// forward pass, and gathers the per-cell likelihood at each cell's `K`
/// positions before returning. The `[N, S]` reconstruction is scoped to
/// the call — never returned to the caller.
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
        scatter_pos: &Tensor,
        values: &Tensor,
        values_weight: Option<&Tensor>,
        log_q_s: &Tensor,
    ) -> Result<Tensor> {
        let log_beta_ks = self
            .dictionary
            .biased_weight_ks_conditional(union_indices, log_q_s)?;
        self.forward_indexed_with_log_beta(
            log_z_nk,
            &log_beta_ks,
            scatter_pos,
            values,
            values_weight,
        )
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

    /// Round-trip: with `S = D` and every cell selecting all features, the
    /// packed indexed decoder llik must match the dense decoder llik.
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

        // Dense forward.
        let (_recon_dense, llik_dense) = dense_decoder
            .forward_with_llik(&log_z, &x, &|_, _| unreachable!())
            .unwrap();

        // Packed indexed forward: union = all features; scatter_pos[i, k] = k.
        let all_indices: Vec<u32> = (0..n_features as u32).collect();
        let union_indices = Tensor::from_vec(all_indices, (n_features,), &device).unwrap();
        let scatter_row: Vec<u32> = (0..n_features as u32).collect();
        let scatter_flat: Vec<u32> = (0..n_samples)
            .flat_map(|_| scatter_row.iter().copied())
            .collect();
        let scatter_pos = Tensor::from_vec(scatter_flat, (n_samples, n_features), &device).unwrap();
        let log_q_s = Tensor::zeros((1, n_features), candle_core::DType::F32, &device).unwrap();

        let llik_indexed = indexed_decoder
            .forward_indexed(&log_z, &union_indices, &scatter_pos, &x, None, &log_q_s)
            .unwrap();

        let llik_dense_vals: Vec<f32> = llik_dense.to_vec1().unwrap();
        let llik_indexed_vals: Vec<f32> = llik_indexed.to_vec1().unwrap();

        for (d, i) in llik_dense_vals.iter().zip(llik_indexed_vals.iter()) {
            assert!((d - i).abs() < 1e-4, "dense={} vs indexed={}", d, i);
        }
    }

    /// With a feature subset, llik must be finite and well-formed.
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

        // Pick a subset; both cells use the same K positions for simplicity.
        let subset: Vec<u32> = vec![1, 3, 5, 7];
        let s = subset.len();
        let union_indices = Tensor::from_vec(subset.clone(), (s,), &device).unwrap();
        let values = x_full.index_select(&union_indices, 1).unwrap(); // [N, K=4]

        // Both cells select positions [0, 1, 2, 3] in the union — same K=S.
        let scatter_row: Vec<u32> = (0..s as u32).collect();
        let scatter_flat: Vec<u32> = (0..n_samples)
            .flat_map(|_| scatter_row.iter().copied())
            .collect();
        let scatter_pos = Tensor::from_vec(scatter_flat, (n_samples, s), &device).unwrap();

        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, &device).unwrap();
        let llik = indexed_decoder
            .forward_indexed(
                &log_z,
                &union_indices,
                &scatter_pos,
                &values,
                None,
                &log_q_s,
            )
            .unwrap();

        let llik_vals: Vec<f32> = llik.to_vec1().unwrap();
        for &v in &llik_vals {
            assert!(v.is_finite(), "non-finite llik: {}", v);
        }
    }
}
