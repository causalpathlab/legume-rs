#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_indexed_model_traits::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// Marker-guided topic decoder.
///
/// Biases the dictionary logits with a marker membership matrix so that
/// topics naturally align with marker-defined cell types during training:
///
///   log β_kd = log_softmax(W_kd + (M_da @ A_ak)^T)
///
/// - `M_da` [D, A]: fixed binary marker membership matrix (genes × cell types)
/// - `A_ak` [A, K]: learnable alignment matrix (cell types → topics)
pub struct MarkerGuidedTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
    marker_da: Tensor,  // [D, A] fixed
    alignment: Tensor,  // [A, K] learnable
}

impl MarkerGuidedTopicDecoder {
    /// Create a marker-guided topic decoder.
    ///
    /// # Arguments
    /// * `n_features` - number of features (genes) D
    /// * `n_topics` - number of topics K
    /// * `marker_da` - [D, A] fixed binary marker membership matrix
    /// * `vs` - variable builder for learnable parameters
    pub fn new(
        n_features: usize,
        n_topics: usize,
        marker_da: Tensor,
        vs: VarBuilder,
    ) -> Result<Self> {
        let dictionary = log_softmax_linear(n_topics, n_features, vs.pp("dictionary"))?;

        let n_annots = marker_da.dim(1)?;
        let alignment = vs.get_with_hints(
            (n_annots, n_topics),
            "marker_alignment",
            candle_nn::init::ZERO,
        )?;

        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            marker_da,
            alignment,
        })
    }

    /// Get the learned alignment matrix A_ak [A, K]
    pub fn get_alignment_matrix(&self) -> Result<Tensor> {
        Ok(self.alignment.clone())
    }

    /// Compute marker bias: (M_da @ A_ak)^T → [K, D]
    fn marker_bias_kd(&self) -> Result<Tensor> {
        let bias_dk = self.marker_da.matmul(&self.alignment)?; // [D, K]
        bias_dk.t() // [K, D]
    }

    /// Compute biased log-dictionary: log_softmax(raw_logits + marker_bias)
    fn biased_log_beta_kd(&self) -> Result<Tensor> {
        let raw_kd = self.dictionary.raw_biased_logits_kd()?;
        let marker_kd = self.marker_bias_kd()?;
        let combined_kd = (&raw_kd + &marker_kd)?;
        ops::log_softmax(&combined_kd, combined_kd.rank() - 1)
    }

    /// Log-space forward with marker bias applied to dictionary.
    fn forward_log_biased(&self, log_h_nk: &Tensor) -> Result<Tensor> {
        let log_beta_kd = self.biased_log_beta_kd()?;

        let log_beta_kd = match *log_h_nk.dims() {
            [b1, b2, _, _] => log_beta_kd.broadcast_left((b1, b2))?,
            [bsize, _, _] => log_beta_kd.broadcast_left(bsize)?,
            _ => log_beta_kd,
        };

        let log_h = log_h_nk.unsqueeze(2)?; // [N, K, 1]
        let log_w = log_beta_kd.unsqueeze(0)?; // [1, K, D]
        let log_terms = log_h.broadcast_add(&log_w)?; // [N, K, D]
        log_terms.log_sum_exp(1) // [N, D]
    }
}

impl DecoderModuleT for MarkerGuidedTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        self.forward_log_biased(z_nk)?.exp()
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.biased_log_beta_kd()?.transpose(0, 1)
    }

    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &Tensor,
        _llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let log_recon_nd = self.forward_log_biased(z_nk)?;
        let recon_nd = log_recon_nd.exp()?;

        let llik = x_nd
            .clamp(0.0, f64::INFINITY)?
            .mul(&log_recon_nd)?
            .sum(x_nd.rank() - 1)?;

        Ok((recon_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

/// Marker-guided indexed topic decoder.
///
/// Same marker bias applied to the dictionary, but computes likelihood
/// only at selected feature positions (indexed variant).
pub struct MarkerGuidedIndexedTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
    marker_da: Tensor,  // [D, A] fixed
    alignment: Tensor,  // [A, K] learnable
}

impl MarkerGuidedIndexedTopicDecoder {
    pub fn new(
        n_features: usize,
        n_topics: usize,
        marker_da: Tensor,
        vs: VarBuilder,
    ) -> Result<Self> {
        let dictionary = log_softmax_linear_nobias(n_topics, n_features, vs.pp("dictionary"))?;

        let n_annots = marker_da.dim(1)?;
        let alignment = vs.get_with_hints(
            (n_annots, n_topics),
            "marker_alignment",
            candle_nn::init::ZERO,
        )?;

        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            marker_da,
            alignment,
        })
    }

    pub fn get_alignment_matrix(&self) -> Result<Tensor> {
        Ok(self.alignment.clone())
    }

    /// Compute full biased log-dictionary [K, D]
    fn biased_log_beta_kd(&self) -> Result<Tensor> {
        let bias_dk = self.marker_da.matmul(&self.alignment)?;
        let bias_kd = bias_dk.t()?;
        let raw_kd = self.dictionary.raw_biased_logits_kd()?;
        let combined_kd = (&raw_kd + &bias_kd)?;
        ops::log_softmax(&combined_kd, combined_kd.rank() - 1)
    }
}

impl IndexedDecoderT for MarkerGuidedIndexedTopicDecoder {
    fn forward_indexed(
        &self,
        log_z_nk: &Tensor,
        union_indices: &Tensor,
        indexed_x: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let log_beta_kd = self.biased_log_beta_kd()?;
        let log_beta_ks = log_beta_kd.index_select(union_indices, 1)?;

        let log_h = log_z_nk.unsqueeze(2)?;
        let log_w = log_beta_ks.unsqueeze(0)?;
        let log_terms = log_h.broadcast_add(&log_w)?;
        let log_recon_ns = log_terms.log_sum_exp(1)?;

        let llik = indexed_x
            .clamp(0.0, f64::INFINITY)?
            .mul(&log_recon_ns)?
            .sum(indexed_x.rank() - 1)?;

        Ok((log_recon_ns, llik))
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        let log_beta_kd = self.biased_log_beta_kd()?;
        log_beta_kd.transpose(0, 1)
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
    use candle_nn::{AdamW, Optimizer, VarMap};

    /// Test: marker-guided decoder recovers correct topic-celltype alignment
    /// on overlapping topics that share most of their gene distribution.
    ///
    /// Setup: 3 topics, 30 genes.
    /// - Genes 0-9: shared background (all topics have equal weight)
    /// - Genes 10-14: markers for cell type 0 (topic 0 has high weight)
    /// - Genes 15-19: markers for cell type 1 (topic 1 has high weight)
    /// - Genes 20-24: markers for cell type 2 (topic 2 has high weight)
    /// - Genes 25-29: more shared background
    ///
    /// With markers, the decoder bias forces each topic to align with its markers.
    #[test]
    fn test_marker_decoder_recovers_alignment_on_overlapping_topics() {
        let n_features = 30;
        let n_topics = 3;
        let n_annots = 3;
        let n_samples = 200;
        let device = Device::Cpu;

        let (_, x_nd, log_z_nk, marker_da) =
            build_overlapping_topic_data(n_features, n_topics, n_annots, n_samples, &device);

        // === Train marker-guided decoder ===
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let marker_decoder =
            MarkerGuidedTopicDecoder::new(n_features, n_topics, marker_da.clone(), vb).unwrap();

        let mut adam = AdamW::new_lr(varmap.all_vars(), 1e-2).unwrap();

        let mut prev_loss = f32::INFINITY;
        for epoch in 0..300 {
            let (_, llik) = marker_decoder
                .forward_with_llik(&log_z_nk, &x_nd, &|_, _| unreachable!())
                .unwrap();
            let loss = llik.neg().unwrap().mean_all().unwrap();
            adam.backward_step(&loss).unwrap();

            let loss_val = loss.to_scalar::<f32>().unwrap();
            if epoch > 50 {
                assert!(
                    loss_val < prev_loss + 0.5,
                    "loss not decreasing: {} -> {}",
                    prev_loss,
                    loss_val
                );
            }
            prev_loss = loss_val;
        }

        // === Check alignment matrix ===
        let alignment = marker_decoder
            .get_alignment_matrix()
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        let mut aligned_topics: Vec<usize> = Vec::new();
        for a in 0..n_annots {
            let max_k = alignment[a]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            aligned_topics.push(max_k);
        }

        // Each cell type should align to a distinct topic
        let mut sorted = aligned_topics.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            n_annots,
            "Not all cell types aligned to distinct topics: {:?}",
            aligned_topics
        );

        // === Check dictionary has elevated marker genes ===
        let dict = marker_decoder.get_dictionary().unwrap();
        let dict_vals: Vec<Vec<f32>> = dict.to_vec2().unwrap();

        for a in 0..n_annots {
            let k = aligned_topics[a];
            let marker_start = 10 + a * 5;

            let marker_avg: f32 = (marker_start..marker_start + 5)
                .map(|d| dict_vals[d][k])
                .sum::<f32>()
                / 5.0;

            for other_k in 0..n_topics {
                if other_k == k {
                    continue;
                }
                let other_avg: f32 = (marker_start..marker_start + 5)
                    .map(|d| dict_vals[d][other_k])
                    .sum::<f32>()
                    / 5.0;

                assert!(
                    marker_avg > other_avg,
                    "Cell type {} markers (genes {}-{}): aligned topic {} avg={:.4} \
                     but topic {} avg={:.4}",
                    a, marker_start, marker_start + 4, k, marker_avg, other_k, other_avg
                );
            }
        }
    }

    /// Test: plain decoder still trains on the same data (sanity check).
    #[test]
    fn test_plain_decoder_trains_on_same_data() {
        let n_features = 30;
        let n_topics = 3;
        let n_annots = 3;
        let n_samples = 200;
        let device = Device::Cpu;

        let (_, x_nd, log_z_nk, _) =
            build_overlapping_topic_data(n_features, n_topics, n_annots, n_samples, &device);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let plain_decoder = TopicDecoder::new(n_features, n_topics, vb).unwrap();

        let mut adam = AdamW::new_lr(varmap.all_vars(), 1e-2).unwrap();

        let first_loss = {
            let (_, llik) = plain_decoder
                .forward_with_llik(&log_z_nk, &x_nd, &|_, _| unreachable!())
                .unwrap();
            llik.neg().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap()
        };

        for _epoch in 0..300 {
            let (_, llik) = plain_decoder
                .forward_with_llik(&log_z_nk, &x_nd, &|_, _| unreachable!())
                .unwrap();
            let loss = llik.neg().unwrap().mean_all().unwrap();
            adam.backward_step(&loss).unwrap();
        }

        let final_loss = {
            let (_, llik) = plain_decoder
                .forward_with_llik(&log_z_nk, &x_nd, &|_, _| unreachable!())
                .unwrap();
            llik.neg().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap()
        };

        assert!(
            final_loss < first_loss,
            "Plain decoder should decrease loss: {} -> {}",
            first_loss, final_loss
        );
    }

    /// Test: marker-guided indexed decoder produces valid output and
    /// matching dictionary shape to the dense variant.
    #[test]
    fn test_marker_indexed_matches_dense_shape() {
        let n_features = 15;
        let n_topics = 3;
        let n_annots = 3;
        let n_samples = 4;
        let device = Device::Cpu;

        let mut m_data = vec![0.0f32; n_features * n_annots];
        for a in 0..n_annots {
            let start = a * 5;
            for d in start..(start + 5).min(n_features) {
                m_data[d * n_annots + a] = 1.0;
            }
        }
        let marker_da = Tensor::from_vec(m_data, (n_features, n_annots), &device).unwrap();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let dense_decoder = MarkerGuidedTopicDecoder::new(
            n_features, n_topics, marker_da.clone(), vb.pp("dec"),
        ).unwrap();
        let indexed_decoder = MarkerGuidedIndexedTopicDecoder::new(
            n_features, n_topics, marker_da, vb.pp("dec"),
        ).unwrap();

        let logits = Tensor::randn(0.0f32, 1.0, (n_samples, n_topics), &device).unwrap();
        let log_z = ops::log_softmax(&logits, 1).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (n_samples, n_features), &device)
            .unwrap().abs().unwrap();

        let (_, llik_dense) = dense_decoder
            .forward_with_llik(&log_z, &x, &|_, _| unreachable!()).unwrap();

        let all_indices: Vec<u32> = (0..n_features as u32).collect();
        let union_indices = Tensor::from_vec(all_indices, (n_features,), &device).unwrap();
        let (_, llik_indexed) = indexed_decoder
            .forward_indexed(&log_z, &union_indices, &x).unwrap();

        // Both produce valid negative log-likelihoods
        let dense_vals: Vec<f32> = llik_dense.to_vec1().unwrap();
        let indexed_vals: Vec<f32> = llik_indexed.to_vec1().unwrap();
        for (d, i) in dense_vals.iter().zip(indexed_vals.iter()) {
            assert!(*d < 0.0, "dense llik should be negative: {}", d);
            assert!(*i < 0.0, "indexed llik should be negative: {}", i);
        }

        assert_eq!(dense_decoder.get_dictionary().unwrap().dims(), &[n_features, n_topics]);
        assert_eq!(indexed_decoder.get_dictionary().unwrap().dims(), &[n_features, n_topics]);
    }

    /// Helper: build overlapping-topic test data.
    /// Returns (gt_dict [K*D flat], x_nd tensor, log_z_nk tensor, marker_da tensor).
    fn build_overlapping_topic_data(
        n_features: usize,
        n_topics: usize,
        n_annots: usize,
        n_samples: usize,
        device: &Device,
    ) -> (Vec<f32>, Tensor, Tensor, Tensor) {
        let mut gt_dict = vec![0.0f32; n_topics * n_features];
        for k in 0..n_topics {
            for d in 0..10 { gt_dict[k * n_features + d] = 1.0; }
            for d in 25..n_features { gt_dict[k * n_features + d] = 1.0; }
            let ms = 10 + k * 5;
            for d in ms..(ms + 5) { gt_dict[k * n_features + d] = 5.0; }
            for ok in 0..n_topics {
                if ok != k {
                    for d in ms..(ms + 5) { gt_dict[ok * n_features + d] += 0.1; }
                }
            }
        }
        for k in 0..n_topics {
            let s: f32 = gt_dict[k * n_features..(k + 1) * n_features].iter().sum();
            for d in 0..n_features { gt_dict[k * n_features + d] /= s; }
        }

        let total = 100.0f32;
        let mut x_data = vec![0.0f32; n_samples * n_features];
        let mut z_data = vec![0.0f32; n_samples * n_topics];
        for i in 0..n_samples {
            let t = i % n_topics;
            z_data[i * n_topics + t] = 1.0;
            for d in 0..n_features {
                x_data[i * n_features + d] = gt_dict[t * n_features + d] * total;
            }
        }

        let x_nd = Tensor::from_vec(x_data, (n_samples, n_features), device).unwrap();
        let z_nk = Tensor::from_vec(z_data, (n_samples, n_topics), device).unwrap();
        let log_z_nk = ops::log_softmax(&z_nk, 1).unwrap();

        let mut m_data = vec![0.0f32; n_features * n_annots];
        for a in 0..n_annots {
            let ms = 10 + a * 5;
            for d in ms..(ms + 5) { m_data[d * n_annots + a] = 1.0; }
        }
        let marker_da = Tensor::from_vec(m_data, (n_features, n_annots), device).unwrap();

        (gt_dict, x_nd, log_z_nk, marker_da)
    }
}
