use crate::candle_aux_linear::*;
use crate::candle_indexed_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Indexed topic model decoder.
///
/// Computes likelihood only at selected feature positions by slicing the
/// dictionary [K, D] to [K, S] via `index_select`, reducing the intermediate
/// tensor from N×K×D to N×K×S. The softmax normalization is computed over
/// all D features, so gradients flow to all logits through the partition function.
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
    ) -> Result<(Tensor, Tensor)> {
        // [K, D] log-softmax dictionary (normalized over all D features)
        let log_beta_kd = self.dictionary.biased_weight_kd()?;
        // [K, S] — select only the indexed features
        let log_beta_ks = log_beta_kd.index_select(union_indices, 1)?;

        // logsumexp over K: log(Σ_k z_nk * β_ks) → [N, S]
        let log_h = log_z_nk.unsqueeze(2)?; // [N, K, 1]
        let log_w = log_beta_ks.unsqueeze(0)?; // [1, K, S]
        let log_terms = log_h.broadcast_add(&log_w)?; // [N, K, S]
        let log_recon_ns = log_terms.log_sum_exp(1)?; // [N, S]

        // llik = Σ_s x_ns * log(recon_ns)
        let llik = indexed_x
            .clamp(0.0, f64::INFINITY)?
            .mul(&log_recon_ns)?
            .sum(indexed_x.rank() - 1)?; // [N]

        Ok((log_recon_ns, llik))
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
    use crate::candle_decoder_topic::TopicDecoder;
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

        let dense_decoder = TopicDecoder::new(n_features, n_topics, vb.pp("dec")).unwrap();
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

        // Indexed forward with ALL features (should match dense exactly)
        let all_indices: Vec<u32> = (0..n_features as u32).collect();
        let union_indices = Tensor::from_vec(all_indices, (n_features,), &device).unwrap();
        let (log_recon_indexed, llik_indexed) = indexed_decoder
            .forward_indexed(&log_z, &union_indices, &x)
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

        let dense_decoder = TopicDecoder::new(n_features, n_topics, vb.pp("dec")).unwrap();
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

        // Indexed forward
        let (log_recon_ns, llik_indexed) = indexed_decoder
            .forward_indexed(&log_z, &union_indices, &x_indexed)
            .unwrap();

        // Dense forward — extract same positions for comparison
        let log_recon_dense = dense_decoder.dictionary().forward_log(&log_z).unwrap();
        let log_recon_dense_subset = log_recon_dense.index_select(&union_indices, 1).unwrap();

        // Log-reconstructions at selected positions should match
        let indexed_vals: Vec<Vec<f32>> = log_recon_ns.to_vec2().unwrap();
        let dense_vals: Vec<Vec<f32>> = log_recon_dense_subset.to_vec2().unwrap();

        for (row_i, row_d) in indexed_vals.iter().zip(dense_vals.iter()) {
            for (i, d) in row_i.iter().zip(row_d.iter()) {
                assert!((i - d).abs() < 1e-4, "indexed={} vs dense_subset={}", i, d);
            }
        }

        // Indexed llik should match dense llik computed at same subset positions
        let dense_llik_at_subset = x_indexed
            .clamp(0.0, f64::INFINITY)
            .unwrap()
            .mul(&log_recon_dense_subset)
            .unwrap()
            .sum(1)
            .unwrap();

        let llik_indexed_vals: Vec<f32> = llik_indexed.to_vec1().unwrap();
        let llik_dense_subset_vals: Vec<f32> = dense_llik_at_subset.to_vec1().unwrap();

        for (i, d) in llik_indexed_vals.iter().zip(llik_dense_subset_vals.iter()) {
            assert!(
                (i - d).abs() < 1e-4,
                "indexed llik={} vs dense subset llik={}",
                i,
                d
            );
        }
    }
}
