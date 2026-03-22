//! Trainable amortized Siamese encoder for link community assignments.
//!
//! Both module-based and indexed encoders share the same edge-level architecture:
//!   node_i → embed → h_i [H]
//!   node_j → embed → h_j [H]
//!   h_edge = [h_i + h_j ; |h_i - h_j|] → FC layers → K logits
//!
//! They differ only in how nodes are embedded:
//!   - Module: per-node module profile [M] → Linear → h [H]
//!   - Indexed: per-node top-K genes → embedding lookup → weighted agg → h [H]

use crate::link_community::model::{LinkCommunityClassifier, LinkProfileStore};
use crate::util::node_indexed::build_node_minibatch;
use candle_util::candle_aux_layers::{stack_relu_linear, StackLayers};
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_indexed_data_loader::IndexedSample;
use candle_util::candle_nn::{self, linear, Linear, Module, Optimizer, VarBuilder, VarMap};

// ---------------------------------------------------------------------------
// Legacy per-edge-profile encoder (used by module path)
// ---------------------------------------------------------------------------

/// Per-edge-profile MLP encoder (non-Siamese, operates on pre-built edge profiles).
pub struct LinkCommunityEncoder {
    varmap: VarMap,
    layer1: Linear,
    layer2: Linear,
    device: Device,
}

impl LinkCommunityEncoder {
    pub fn from_classifier(
        classifier: &LinkCommunityClassifier,
        hidden_dim: usize,
    ) -> candle_util::candle_core::Result<Self> {
        use candle_util::candle_core::Var;

        let m = classifier.m;
        let k = classifier.k;
        let h = hidden_dim;
        let device = Device::Cpu;

        let mut w1_data = vec![0.0f32; h * m];
        for i in 0..h.min(m) {
            w1_data[i * m + i] = 1.0;
        }
        let w1_tensor = Tensor::from_vec(w1_data, (h, m), &device)?;
        let b1_tensor = Tensor::zeros((h,), DType::F32, &device)?;

        let mut w2_data = vec![0.0f32; k * h];
        for c in 0..k {
            for j in 0..h.min(m) {
                w2_data[c * h + j] = classifier.log_rates[c * m + j] as f32;
            }
        }
        let w2_tensor = Tensor::from_vec(w2_data, (k, h), &device)?;
        let b2_data: Vec<f32> = (0..k)
            .map(|c| (classifier.log_prior[c] - classifier.rate_totals[c]) as f32)
            .collect();
        let b2_tensor = Tensor::from_vec(b2_data, (k,), &device)?;

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
        Ok(Self {
            varmap,
            layer1,
            layer2,
            device,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_util::candle_core::Result<Tensor> {
        let h = self.layer1.forward(x)?;
        let h = candle_nn::Activation::Relu.forward(&h)?;
        self.layer2.forward(&h)
    }

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

    pub fn predict_labels(&self, profiles: &LinkProfileStore) -> Vec<usize> {
        let x = self.build_input_tensor(profiles);
        let logits = self.forward(&x).unwrap();
        let labels: Vec<u32> = logits.argmax(1).unwrap().to_vec1().unwrap();
        labels.iter().map(|&l| l as usize).collect()
    }

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
        let target = Tensor::from_vec(
            labels.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            (n,),
            &self.device,
        )
        .unwrap();
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

// ---------------------------------------------------------------------------
// Trait: node embedding
// ---------------------------------------------------------------------------

/// How to embed nodes into a latent space for edge-level classification.
pub trait NodeEmbedder {
    type NodeData: ?Sized;

    /// Embed a batch of nodes → `[n_nodes, H]`.
    fn embed_nodes(
        &self,
        data: &Self::NodeData,
        node_indices: &[usize],
        device: &Device,
    ) -> candle_util::candle_core::Result<Tensor>;
}

// ---------------------------------------------------------------------------
// Generic Siamese link encoder
// ---------------------------------------------------------------------------

/// Siamese link community encoder parameterised by a `NodeEmbedder`.
///
/// Edge representation = `[h_left + h_right; |h_left - h_right|]` → FC → K.
pub struct SiameseLinkEncoder<E: NodeEmbedder> {
    pub varmap: VarMap,
    pub embedder: E,
    fc: StackLayers<Linear>,
    device: Device,
}

/// Pre-built edge index tensors for reuse across epochs.
struct EdgeTensors {
    all_nodes: Vec<usize>,
    left_t: Tensor,
    right_t: Tensor,
}

fn build_edge_tensors(edges: &[(usize, usize)], device: &Device) -> EdgeTensors {
    let n_edges = edges.len();
    let (all_nodes, node_to_pos) = build_node_lookup(edges);
    let left_t = Tensor::from_vec(
        edges
            .iter()
            .map(|&(l, _)| node_to_pos[l] as u32)
            .collect::<Vec<_>>(),
        (n_edges,),
        device,
    )
    .unwrap();
    let right_t = Tensor::from_vec(
        edges
            .iter()
            .map(|&(_, r)| node_to_pos[r] as u32)
            .collect::<Vec<_>>(),
        (n_edges,),
        device,
    )
    .unwrap();
    EdgeTensors {
        all_nodes,
        left_t,
        right_t,
    }
}

impl<E: NodeEmbedder> SiameseLinkEncoder<E> {
    fn forward_edge(
        &self,
        h_left: &Tensor,
        h_right: &Tensor,
    ) -> candle_util::candle_core::Result<Tensor> {
        let sum_part = (h_left + h_right)?;
        let diff_part = (h_left - h_right)?.abs()?;
        let pooled = Tensor::cat(&[&sum_part, &diff_part], 1)?;
        self.fc.forward(&pooled)
    }

    fn compute_logits(
        &self,
        node_data: &E::NodeData,
        et: &EdgeTensors,
    ) -> candle_util::candle_core::Result<Tensor> {
        let h_all = self
            .embedder
            .embed_nodes(node_data, &et.all_nodes, &self.device)?;
        let h_left = h_all.index_select(&et.left_t, 0)?;
        let h_right = h_all.index_select(&et.right_t, 0)?;
        self.forward_edge(&h_left, &h_right)
    }

    /// Train on edges with Gibbs labels (full-batch, Adam + cross-entropy).
    pub fn train_on_edges(
        &mut self,
        node_data: &E::NodeData,
        edges: &[(usize, usize)],
        labels: &[usize],
        epochs: usize,
        lr: f64,
    ) -> f64 {
        let n_edges = edges.len();
        debug_assert_eq!(labels.len(), n_edges);

        let et = build_edge_tensors(edges, &self.device);
        let target = Tensor::from_vec(
            labels.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            (n_edges,),
            &self.device,
        )
        .unwrap();

        let mut opt = candle_nn::AdamW::new_lr(self.varmap.all_vars(), lr).unwrap();
        let mut final_loss = 0.0;

        for _epoch in 0..epochs {
            let logits = self.compute_logits(node_data, &et).unwrap();
            let log_probs = candle_nn::ops::log_softmax(&logits, 1).unwrap();
            let loss = candle_nn::loss::nll(&log_probs, &target).unwrap();
            final_loss = loss.to_scalar::<f32>().unwrap() as f64;
            opt.backward_step(&loss).unwrap();
        }

        final_loss
    }

    /// Predict community labels for edges.
    pub fn predict_edges(&self, node_data: &E::NodeData, edges: &[(usize, usize)]) -> Vec<usize> {
        let et = build_edge_tensors(edges, &self.device);
        let logits = self.compute_logits(node_data, &et).unwrap();
        let out: Vec<u32> = logits.argmax(1).unwrap().to_vec1().unwrap();
        out.iter().map(|&l| l as usize).collect()
    }
}

fn build_node_lookup(edges: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut all_nodes: Vec<usize> = edges.iter().flat_map(|&(l, r)| [l, r]).collect();
    all_nodes.sort_unstable();
    all_nodes.dedup();
    let max_node = all_nodes.last().copied().unwrap_or(0);
    let mut lookup = vec![usize::MAX; max_node + 1];
    for (pos, &node) in all_nodes.iter().enumerate() {
        lookup[node] = pos;
    }
    (all_nodes, lookup)
}

fn build_fc_stack(
    embedding_dim: usize,
    layers: &[usize],
    k: usize,
    vb: VarBuilder,
) -> candle_util::candle_core::Result<StackLayers<Linear>> {
    let last_hidden = layers.last().copied().unwrap_or(2 * embedding_dim);
    let intermediates = if layers.len() > 1 {
        &layers[..layers.len() - 1]
    } else {
        &[]
    };
    let mut fc = stack_relu_linear(2 * embedding_dim, last_hidden, intermediates, vb.pp("fc"))?;
    fc.push(linear(last_hidden, k, vb.pp("classify"))?);
    Ok(fc)
}

// ---------------------------------------------------------------------------
// Indexed node embedder
// ---------------------------------------------------------------------------

/// Embeds nodes using adaptive top-K gene selection + learnable embeddings.
pub struct IndexedNodeEmbedder {
    feature_embeddings: Tensor, // [G, H]
    n_features: usize,
}

impl NodeEmbedder for IndexedNodeEmbedder {
    type NodeData = [IndexedSample];

    fn embed_nodes(
        &self,
        samples: &[IndexedSample],
        node_indices: &[usize],
        device: &Device,
    ) -> candle_util::candle_core::Result<Tensor> {
        let (union_idx, indexed_x) =
            build_node_minibatch(samples, node_indices, self.n_features, device)
                .map_err(|e| candle_util::candle_core::Error::Msg(e.to_string()))?;

        let e_sh = self.feature_embeddings.index_select(&union_idx, 0)?;
        let lx_ns = (indexed_x + 1.0)?.log()?;
        let denom = lx_ns.sum_keepdim(1)?;
        let normalized = (lx_ns.broadcast_div(&denom)? * (self.n_features as f64))?;
        normalized.matmul(&e_sh)
    }
}

pub type IndexedLinkEncoder = SiameseLinkEncoder<IndexedNodeEmbedder>;

impl IndexedLinkEncoder {
    pub fn new_indexed(
        n_features: usize,
        embedding_dim: usize,
        layers: &[usize],
        k: usize,
    ) -> candle_util::candle_core::Result<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let feature_embeddings =
            vb.get_with_hints((n_features, embedding_dim), "feature.embeddings", init_ws)?;

        let fc = build_fc_stack(embedding_dim, layers, k, vb.clone())?;

        Ok(Self {
            varmap,
            embedder: IndexedNodeEmbedder {
                feature_embeddings,
                n_features,
            },
            fc,
            device,
        })
    }
}

// ---------------------------------------------------------------------------
// Legacy API aliases (used by fit_srt_link_community.rs)
// ---------------------------------------------------------------------------

pub type IndexedLinkCommunityEncoder = IndexedLinkEncoder;

impl IndexedLinkCommunityEncoder {
    pub fn new(
        n_features: usize,
        embedding_dim: usize,
        layers: &[usize],
        k: usize,
    ) -> candle_util::candle_core::Result<Self> {
        Self::new_indexed(n_features, embedding_dim, layers, k)
    }

    pub fn train_on_edges_indexed(
        &mut self,
        node_samples: &[IndexedSample],
        edges: &[(usize, usize)],
        labels: &[usize],
        epochs: usize,
        lr: f64,
    ) -> f64 {
        self.train_on_edges(node_samples, edges, labels, epochs, lr)
    }

    pub fn predict_edges_indexed(
        &self,
        node_samples: &[IndexedSample],
        edges: &[(usize, usize)],
    ) -> Vec<usize> {
        self.predict_edges(node_samples, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_util::candle_indexed_data_loader::top_k_indices;

    struct ModuleNodeEmbedder {
        proj: Linear,
        m: usize,
        h: usize,
    }

    struct NodeProfileData {
        profiles: Vec<f32>,
        size_factors: Vec<f32>,
        n_nodes: usize,
        m: usize,
    }

    impl NodeEmbedder for ModuleNodeEmbedder {
        type NodeData = NodeProfileData;

        fn embed_nodes(
            &self,
            data: &NodeProfileData,
            node_indices: &[usize],
            device: &Device,
        ) -> candle_util::candle_core::Result<Tensor> {
            let n = node_indices.len();
            let m = data.m;
            let mut input = vec![0.0f32; n * m];
            for (i, &node) in node_indices.iter().enumerate() {
                let sf_inv = if data.size_factors[node] > 0.0 {
                    1.0 / data.size_factors[node]
                } else {
                    1.0
                };
                let src = &data.profiles[node * m..(node + 1) * m];
                let dst = &mut input[i * m..(i + 1) * m];
                for g in 0..m {
                    dst[g] = (src[g] * sf_inv).ln_1p();
                }
            }
            let x = Tensor::from_vec(input, (n, m), device)?;
            self.proj.forward(&x)
        }
    }

    type ModuleLinkEncoder = SiameseLinkEncoder<ModuleNodeEmbedder>;

    impl ModuleLinkEncoder {
        fn new_module(
            m: usize,
            layers: &[usize],
            k: usize,
        ) -> candle_util::candle_core::Result<Self> {
            let device = Device::Cpu;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

            let h = layers.first().copied().unwrap_or(m);
            let proj = linear(m, h, vb.pp("node_proj"))?;
            let fc = build_fc_stack(h, layers, k, vb.clone())?;

            Ok(Self {
                varmap,
                embedder: ModuleNodeEmbedder { proj, m, h },
                fc,
                device,
            })
        }
    }

    #[test]
    fn test_indexed_encoder_trains_on_planted() {
        let k = 3;
        let n_genes = 30;
        let n_nodes = 60;
        let n_edges = 200;
        let emb_dim = 16;
        let hidden = 32;
        let ctx = 15;

        let mut node_samples = Vec::with_capacity(n_nodes);
        for node in 0..n_nodes {
            let community = node % k;
            let mut values = vec![1.0f32; n_genes];
            for g in 0..n_genes {
                if g % k == community {
                    values[g] = 10.0;
                }
            }
            let (indices, vals) = top_k_indices(&values, ctx);
            node_samples.push(IndexedSample {
                indices,
                values: vals,
            });
        }

        let mut edges = Vec::with_capacity(n_edges);
        let mut labels = Vec::with_capacity(n_edges);
        for e in 0..n_edges {
            let left = (e * 7) % n_nodes;
            let right = {
                let r = (e * 13 + 3) % n_nodes;
                if r == left {
                    (r + 1) % n_nodes
                } else {
                    r
                }
            };
            edges.push((left, right));
            labels.push(left % k);
        }

        let mut encoder = IndexedLinkCommunityEncoder::new(n_genes, emb_dim, &[hidden], k).unwrap();
        let loss = encoder.train_on_edges_indexed(&node_samples, &edges, &labels, 100, 0.01);
        let pred = encoder.predict_edges_indexed(&node_samples, &edges);

        let acc = pred
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &t)| p == t)
            .count();
        assert!(
            acc >= n_edges * 8 / 10,
            "Indexed encoder should recover planted: {}/{}, loss={:.4}",
            acc,
            n_edges,
            loss,
        );
    }

    #[test]
    fn test_module_encoder_trains_on_planted() {
        let k = 3;
        let m = 9;
        let n_nodes = 60;
        let n_edges = 200;

        let mut profiles = vec![0.0f32; n_nodes * m];
        let mut size_factors = vec![0.0f32; n_nodes];
        for node in 0..n_nodes {
            let community = node % k;
            let base = node * m;
            for g in 0..m {
                let val = if g % k == community { 10.0 } else { 1.0 };
                profiles[base + g] = val;
                size_factors[node] += val;
            }
        }
        let node_data = NodeProfileData {
            profiles,
            size_factors,
            n_nodes,
            m,
        };

        let mut edges = Vec::with_capacity(n_edges);
        let mut labels = Vec::with_capacity(n_edges);
        for e in 0..n_edges {
            let left = (e * 7) % n_nodes;
            let right = {
                let r = (e * 13 + 3) % n_nodes;
                if r == left {
                    (r + 1) % n_nodes
                } else {
                    r
                }
            };
            edges.push((left, right));
            labels.push(left % k);
        }

        let mut encoder = ModuleLinkEncoder::new_module(m, &[16], k).unwrap();
        let loss = encoder.train_on_edges(&node_data, &edges, &labels, 100, 0.01);
        let pred = encoder.predict_edges(&node_data, &edges);

        let acc = pred
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &t)| p == t)
            .count();
        assert!(
            acc >= n_edges * 8 / 10,
            "Module encoder should recover planted: {}/{}, loss={:.4}",
            acc,
            n_edges,
            loss,
        );
    }
}
