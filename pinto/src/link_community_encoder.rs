#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//! Trainable amortized encoder for link community assignments.
//!
//! A 1-hidden-layer MLP trained on Gibbs outputs at each coarsening level,
//! implemented with candle for autograd.
//!
//! Architecture:
//!   input: y / s (size-factor-normalized profile, M dims)
//!   hidden: ReLU(W1 · input + b1)     [H dims]
//!   output: softmax(W2 · hidden + b2) [K dims]

use crate::link_community_model::{LinkCommunityClassifier, LinkProfileStore};
use candle_util::candle_core::{DType, Device, Tensor, Var};
use candle_util::candle_nn::{
    self, linear, Activation, Linear, Module, Optimizer, VarBuilder, VarMap,
};

/// Trainable MLP encoder for link community prediction.
pub struct LinkCommunityEncoder {
    /// candle variable map (owns parameters).
    varmap: VarMap,
    /// First linear layer (M → H).
    layer1: Linear,
    /// Second linear layer (H → K).
    layer2: Linear,
    /// Input dimension.
    m: usize,
    /// Hidden dimension.
    h: usize,
    /// Output dimension.
    k: usize,
    /// Device (CPU).
    device: Device,
}

impl LinkCommunityEncoder {
    /// Create with random initialization.
    pub fn new(m: usize, hidden_dim: usize, k: usize) -> candle_util::candle_core::Result<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let layer1 = linear(m, hidden_dim, vb.pp("layer1"))?;
        let layer2 = linear(hidden_dim, k, vb.pp("layer2"))?;

        Ok(LinkCommunityEncoder {
            varmap,
            layer1,
            layer2,
            m,
            h: hidden_dim,
            k,
            device,
        })
    }

    /// Initialize from the analytic classifier (warm start).
    ///
    /// Sets layer1 to identity-like pass-through and layer2 to log-rates,
    /// so the MLP starts close to the linear classifier's predictions.
    pub fn from_classifier(
        classifier: &LinkCommunityClassifier,
        hidden_dim: usize,
    ) -> candle_util::candle_core::Result<Self> {
        let m = classifier.m;
        let k = classifier.k;
        let h = hidden_dim;
        let device = Device::Cpu;

        // Build W1 [H × M]: identity-like for first min(H,M) units
        let mut w1_data = vec![0.0f32; h * m];
        for i in 0..h.min(m) {
            w1_data[i * m + i] = 1.0;
        }
        let w1_tensor = Tensor::from_vec(w1_data, (h, m), &device)?;
        let b1_tensor = Tensor::zeros((h,), DType::F32, &device)?;

        // Build W2 [K × H]: log-rates from classifier
        let mut w2_data = vec![0.0f32; k * h];
        for c in 0..k {
            for j in 0..h {
                if j < m {
                    w2_data[c * h + j] = classifier.log_rates[c * m + j] as f32;
                }
            }
        }
        let w2_tensor = Tensor::from_vec(w2_data, (k, h), &device)?;

        // b2 [K]: log_prior - rate_totals
        let b2_data: Vec<f32> = (0..k)
            .map(|c| (classifier.log_prior[c] - classifier.rate_totals[c]) as f32)
            .collect();
        let b2_tensor = Tensor::from_vec(b2_data, (k,), &device)?;

        // Create VarMap with our tensors
        let varmap = VarMap::new();
        {
            let mut data = varmap.data().lock().unwrap();
            data.insert("layer1.weight".to_string(), Var::from_tensor(&w1_tensor)?);
            data.insert("layer1.bias".to_string(), Var::from_tensor(&b1_tensor)?);
            data.insert("layer2.weight".to_string(), Var::from_tensor(&w2_tensor)?);
            data.insert("layer2.bias".to_string(), Var::from_tensor(&b2_tensor)?);
        }

        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let layer1 = linear(m, h, vb.pp("layer1"))?;
        let layer2 = linear(h, k, vb.pp("layer2"))?;

        Ok(LinkCommunityEncoder {
            varmap,
            layer1,
            layer2,
            m,
            h,
            k,
            device,
        })
    }

    /// Forward pass: profiles [N × M] → logits [N × K].
    fn forward(&self, x: &Tensor) -> candle_util::candle_core::Result<Tensor> {
        let h = self.layer1.forward(x)?;
        let h = Activation::Relu.forward(&h)?;
        self.layer2.forward(&h)
    }

    /// Build log1p-normalized input tensor [N × M] from profiles.
    ///
    /// Applies log1p(y_g / s_e) to compress dynamic range.
    fn build_input_tensor(&self, profiles: &LinkProfileStore) -> Tensor {
        let n = profiles.n_edges;
        let m = profiles.m;
        let mut input = vec![0.0f32; n * m];
        for e in 0..n {
            let sf_inv = if profiles.size_factors[e] > 0.0 {
                1.0 / profiles.size_factors[e]
            } else {
                1.0
            };
            let row = profiles.profile(e);
            let base = e * m;
            for g in 0..m {
                input[base + g] = (row[g] * sf_inv).ln_1p();
            }
        }
        Tensor::from_vec(input, (n, m), &self.device).unwrap()
    }

    /// Predict community assignments for all edges, batched.
    pub fn predict_labels(&self, profiles: &LinkProfileStore) -> Vec<usize> {
        let x = self.build_input_tensor(profiles);
        let logits = self.forward(&x).unwrap();
        let argmax = logits.argmax(1).unwrap();
        let labels: Vec<u32> = argmax.to_vec1().unwrap();
        labels.iter().map(|&l| l as usize).collect()
    }

    /// Train the encoder on edge profiles with hard labels from Gibbs.
    ///
    /// Uses Adam optimizer with cross-entropy loss (full-batch).
    /// Returns the final average loss.
    pub fn train(
        &mut self,
        profiles: &LinkProfileStore,
        labels: &[usize],
        epochs: usize,
        lr: f64,
    ) -> f64 {
        let n = profiles.n_edges;
        debug_assert_eq!(labels.len(), n);

        let x = self.build_input_tensor(profiles);
        let target: Vec<u32> = labels.iter().map(|&l| l as u32).collect();
        let target = Tensor::from_vec(target, (n,), &self.device).unwrap();

        let mut opt = candle_nn::AdamW::new_lr(self.varmap.all_vars(), lr).unwrap();

        let mut final_loss = 0.0;

        for _epoch in 0..epochs {
            let logits = self.forward(&x).unwrap();
            let log_probs = candle_nn::ops::log_softmax(&logits, 1).unwrap();
            let loss = candle_nn::loss::nll(&log_probs, &target).unwrap();

            final_loss = loss.to_scalar::<f32>().unwrap() as f64;
            opt.backward_step(&loss).unwrap();
        }

        final_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link_community_model::LinkCommunityStats;

    fn make_planted(n_edges: usize, m: usize, k: usize) -> (LinkProfileStore, Vec<usize>) {
        let mut profiles = vec![0.0f32; n_edges * m];
        let mut labels = vec![0usize; n_edges];
        for e in 0..n_edges {
            let c = e % k;
            labels[e] = c;
            for g in 0..m {
                let signal = if g % k == c { 10.0 } else { 1.0 };
                profiles[e * m + g] = signal;
            }
        }
        (LinkProfileStore::new(profiles, n_edges, m), labels)
    }

    #[test]
    fn test_encoder_trains_on_planted() {
        let k = 3;
        let m = 9;
        let n = 300;
        let h = 16;

        let (store, labels) = make_planted(n, m, k);
        let mut encoder = LinkCommunityEncoder::new(m, h, k).unwrap();

        let pred_before = encoder.predict_labels(&store);
        let acc_before = pred_before
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &t)| p == t)
            .count();

        let loss = encoder.train(&store, &labels, 50, 0.01);

        let pred_after = encoder.predict_labels(&store);
        let acc_after = pred_after
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &t)| p == t)
            .count();

        assert!(
            acc_after > acc_before || acc_before == n,
            "Training should improve: before={}/{}, after={}/{}, loss={:.4}",
            acc_before,
            n,
            acc_after,
            n,
            loss,
        );
        assert!(
            acc_after >= n - 10,
            "Should recover planted: {}/{}, loss={:.4}",
            acc_after,
            n,
            loss,
        );
    }

    #[test]
    fn test_encoder_warm_start_from_classifier() {
        let k = 3;
        let m = 6;
        let n = 60;
        let h = 12;

        let (store, labels) = make_planted(n, m, k);
        let stats = LinkCommunityStats::from_profiles(&store, k, &labels);
        let classifier = LinkCommunityClassifier::from_stats(&stats, 1.0, 1.0);

        let encoder = LinkCommunityEncoder::from_classifier(&classifier, h).unwrap();
        let pred = encoder.predict_labels(&store);
        let acc = pred
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &t)| p == t)
            .count();

        assert_eq!(
            acc, n,
            "Warm-started encoder should match classifier: {}/{}",
            acc, n
        );
    }

    #[test]
    fn test_encoder_generalizes_to_noisy() {
        let k = 2;
        let m = 8;
        let n_train = 200;
        let n_test = 100;
        let h = 16;

        let (train_store, train_labels) = make_planted(n_train, m, k);
        let mut encoder = LinkCommunityEncoder::new(m, h, k).unwrap();
        encoder.train(&train_store, &train_labels, 50, 0.01);

        let mut test_profiles = vec![0.0f32; n_test * m];
        let mut test_labels = vec![0usize; n_test];
        for e in 0..n_test {
            let c = e % k;
            test_labels[e] = c;
            for g in 0..m {
                let signal = if g % k == c { 7.0 } else { 3.0 };
                test_profiles[e * m + g] = signal;
            }
        }
        let test_store = LinkProfileStore::new(test_profiles, n_test, m);

        let pred = encoder.predict_labels(&test_store);
        let acc = pred
            .iter()
            .zip(test_labels.iter())
            .filter(|(&p, &t)| p == t)
            .count();

        assert!(
            acc >= n_test - 5,
            "Encoder should generalize: {}/{}",
            acc,
            n_test,
        );
    }
}
