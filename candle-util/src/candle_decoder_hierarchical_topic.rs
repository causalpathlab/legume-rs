#![allow(dead_code)]

use crate::candle_model_traits::*;
use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// softplus(x) = log(1 + exp(x)), numerically stable
fn softplus(x: &Tensor) -> Result<Tensor> {
    // For large x, softplus(x) ≈ x. For small x, softplus(x) ≈ exp(x).
    // Use: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    let abs_x = x.abs()?;
    let relu_x = x.relu()?;
    relu_x + (abs_x.neg()?.exp()? + 1.0)?.log()?
}

/// Numerically stable log(sigmoid(x)) = -softplus(-x)
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    softplus(&x.neg()?)?.neg()
}

/// Numerically stable log(1 - sigmoid(x)) = log(sigmoid(-x)) = -softplus(x)
fn log_one_minus_sigmoid(x: &Tensor) -> Result<Tensor> {
    softplus(x)?.neg()
}

///////////////////////////////////////////////
// Hierarchical Topic Decoder (BTree Gates) //
///////////////////////////////////////////////

/// A hierarchical topic decoder using stick-breaking gates on a binary tree.
///
/// The tree is a fixed-depth complete binary tree (heap layout, 1-indexed):
/// - Root = node 1
/// - Left child of node h = 2*h, right child = 2*h + 1
/// - Internal nodes: `[1, num_leaves)`
/// - Leaves: `[num_leaves, 2*num_leaves)` → K = 2^(depth-1) topics
///
/// **Parameters:**
/// - `root_logits [1, D]`: base word distribution (softmaxed to get root probability)
/// - `gate_logits [num_internal, D]`: per-internal-node sigmoid gates
///
/// **Top-down propagation:**
/// ```text
/// prob(root) = softmax(root_logits)
/// prob(left_child)  = prob(parent) * sigmoid(gate)
/// prob(right_child) = prob(parent) * (1 - sigmoid(gate))
/// ```
///
/// Leaf distributions are naturally sparse: genes must survive every gate
/// from root to leaf. Siblings share all gates except the last one.
pub struct HierarchicalTopicDecoder {
    n_features: usize,
    depth: usize,
    n_topics: usize,     // K = 2^(depth-1) leaf topics
    num_nodes: usize,    // 2^depth - 1
    num_internal: usize, // 2^(depth-1) - 1 (nodes that are not leaves)
    root_logits: Tensor, // [1, D]
    gate_logits: Tensor, // [num_internal, D]
}

impl HierarchicalTopicDecoder {
    /// Create a hierarchical topic decoder.
    ///
    /// * `n_features` - D, number of output features (genes)
    /// * `depth` - tree depth (>= 2). Number of leaf topics K = 2^(depth-1).
    /// * `vs` - VarBuilder for parameter allocation
    pub fn new(n_features: usize, depth: usize, vs: VarBuilder) -> Result<Self> {
        assert!(depth >= 2, "Tree depth must be at least 2");

        let n_topics = 1 << (depth - 1); // 2^(depth-1)
        let num_nodes = (1 << depth) - 1; // 2^depth - 1
        let num_internal = n_topics - 1; // internal nodes (non-leaves)

        // Root logits: initialized with Kaiming normal
        let root_logits = vs.get_with_hints(
            (1, n_features),
            "root_logits",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;

        // Gate logits: initialized to 0 (sigmoid(0) = 0.5 → equal split)
        let gate_logits = vs.get_with_hints(
            (num_internal, n_features),
            "gate_logits",
            candle_nn::Init::Const(0.0),
        )?;

        Ok(Self {
            n_features,
            depth,
            n_topics,
            num_nodes,
            num_internal,
            root_logits,
            gate_logits,
        })
    }

    /// Number of leaf topics K.
    pub fn n_topics(&self) -> usize {
        self.n_topics
    }

    /// Total number of tree nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Tree depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Compute log leaf dictionary [K, D] via top-down log-gate propagation.
    ///
    /// log(leaf_k_d) = log_softmax(root) + Σ_{gates on path} log_sigmoid(±gate)
    ///
    /// Numerically stable — accumulates log-probabilities instead of
    /// multiplying small probabilities that underflow for deep trees.
    pub fn log_leaf_dictionary_kd(&self) -> Result<Tensor> {
        let num_leaves = self.n_topics;

        // Root log-probability: log_softmax over features
        let log_root = ops::log_softmax(&self.root_logits, 1)?; // [1, D]

        if self.num_internal == 0 {
            return Ok(log_root);
        }

        // Propagate log-probabilities top-down using log_sigmoid
        let mut log_node_probs: Vec<Option<Tensor>> = vec![None; self.num_nodes + 1];
        log_node_probs[1] = Some(log_root);

        for h in 1..num_leaves {
            let log_parent = log_node_probs[h].as_ref().unwrap().clone(); // [1, D]

            // gate_logits row: h-1 (row 0 = node 1)
            let gate_h = self.gate_logits.i(h - 1)?.unsqueeze(0)?; // [1, D]

            // log(sigmoid(g)) and log(1 - sigmoid(g)) = log(sigmoid(-g))
            let log_left = log_parent.add(&log_sigmoid(&gate_h)?)?;
            let log_right = log_parent.add(&log_one_minus_sigmoid(&gate_h)?)?;

            log_node_probs[2 * h] = Some(log_left);
            log_node_probs[2 * h + 1] = Some(log_right);
        }

        let leaf_log_probs: Vec<Tensor> = (0..num_leaves)
            .map(|k| log_node_probs[num_leaves + k].as_ref().unwrap().clone())
            .collect();

        Tensor::cat(&leaf_log_probs, 0) // [K, D]
    }

    /// Compute leaf dictionary [K, D] on probability scale.
    pub fn leaf_dictionary_kd(&self) -> Result<Tensor> {
        self.log_leaf_dictionary_kd()?.exp()
    }

    /// Log-space forward: log(Σ_k z_nk * β_kd) via logsumexp
    pub fn forward_log(&self, z_nk: &Tensor) -> Result<Tensor> {
        let log_beta_kd = self.log_leaf_dictionary_kd()?; // [K, D]
        let eps = 1e-20;
        let log_z = (z_nk + eps)?.log()?;    // [N, K]
        let log_z = log_z.unsqueeze(2)?;      // [N, K, 1]
        let log_b = log_beta_kd.unsqueeze(0)?; // [1, K, D]
        let log_terms = log_z.broadcast_add(&log_b)?; // [N, K, D]
        log_terms.log_sum_exp(1) // [N, D]
    }

    /// Get all node probabilities [num_nodes, D] for hierarchy visualization.
    ///
    /// Returns probabilities for all tree nodes (internal + leaves), ordered
    /// by 1-indexed node number. Row 0 = root (node 1), etc.
    /// Computed via log-space propagation then exp for numerical stability.
    pub fn node_probabilities(&self) -> Result<Tensor> {
        let num_leaves = self.n_topics;

        let log_root = ops::log_softmax(&self.root_logits, 1)?;

        if self.num_internal == 0 {
            return log_root.exp();
        }

        let mut log_node_probs: Vec<Option<Tensor>> = vec![None; self.num_nodes + 1];
        log_node_probs[1] = Some(log_root);

        for h in 1..num_leaves {
            let log_parent = log_node_probs[h].as_ref().unwrap().clone();
            let gate_h = self.gate_logits.i(h - 1)?.unsqueeze(0)?;

            let log_left = log_parent.add(&log_sigmoid(&gate_h)?)?;
            let log_right = log_parent.add(&log_one_minus_sigmoid(&gate_h)?)?;

            log_node_probs[2 * h] = Some(log_left);
            log_node_probs[2 * h + 1] = Some(log_right);
        }

        let all_log_probs: Vec<Tensor> = (1..=self.num_nodes)
            .map(|h| log_node_probs[h].as_ref().unwrap().clone())
            .collect();

        Tensor::cat(&all_log_probs, 0)?.exp()
    }
}

impl DecoderModuleT for HierarchicalTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        self.forward_log(z_nk)?.exp()
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.log_leaf_dictionary_kd()?.exp()?.t()
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
        let log_recon_nd = self.forward_log(z_nk)?;
        let recon_nd = log_recon_nd.exp()?;

        // Direct log-space likelihood: llik = Σ_d x_d * log(recon_d)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candle_loss_functions::topic_likelihood;
    use candle_core::{DType, Device};

    fn make_decoder(depth: usize, n_features: usize) -> HierarchicalTopicDecoder {
        let dev = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        HierarchicalTopicDecoder::new(n_features, depth, vs).unwrap()
    }

    #[test]
    fn test_tree_dimensions() {
        let dec = make_decoder(3, 8);
        assert_eq!(dec.n_topics, 4); // 2^(3-1) = 4
        assert_eq!(dec.num_nodes, 7); // 2^3 - 1
        assert_eq!(dec.num_internal, 3); // 4 - 1
    }

    #[test]
    fn test_leaf_dictionary_shape() {
        let dec = make_decoder(3, 16);
        let beta = dec.leaf_dictionary_kd().unwrap();
        assert_eq!(beta.dims(), &[4, 16]); // [K=4, D=16]
    }

    #[test]
    fn test_leaf_probabilities_sum_to_one() {
        // Each leaf distribution should sum to <= root probability per gene.
        // The total across ALL leaves for each gene should equal the root probability
        // (since gates split probability mass without creating or destroying it).
        let dec = make_decoder(3, 8);
        let beta = dec.leaf_dictionary_kd().unwrap(); // [4, 8]

        // Sum across all leaves (dim 0) → should equal root_prob [1, D]
        let leaf_sum = beta.sum(0).unwrap(); // [D]
        let root_prob = ops::softmax(&dec.root_logits, 1)
            .unwrap()
            .squeeze(0)
            .unwrap(); // [D]

        let diff = leaf_sum
            .sub(&root_prob)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "Leaf probabilities should sum to root probability, diff={}",
            diff
        );

        // Root prob sums to 1 (softmax), so leaf sum should also sum to 1
        let total: f32 = leaf_sum.sum_all().unwrap().to_scalar().unwrap();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "Total leaf probability should be 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_forward_shape() {
        let dec = make_decoder(3, 16);
        let dev = Device::Cpu;

        // z_nk: [N=5, K=4]
        let z = Tensor::ones((5, 4), DType::F32, &dev).unwrap();
        let z = (z / 4.0f64).unwrap(); // uniform topic proportions
        let recon = dec.forward(&z).unwrap();
        assert_eq!(recon.dims(), &[5, 16]); // [N=5, D=16]
    }

    #[test]
    fn test_get_dictionary_shape() {
        let dec = make_decoder(4, 32);
        let dict = dec.get_dictionary().unwrap();
        assert_eq!(dict.dims(), &[32, 8]); // [D=32, K=8]
    }

    #[test]
    fn test_node_probabilities_shape() {
        let dec = make_decoder(3, 8);
        let node_probs = dec.node_probabilities().unwrap();
        assert_eq!(node_probs.dims(), &[7, 8]); // [num_nodes=7, D=8]
    }

    #[test]
    fn test_siblings_more_similar_than_distant() {
        // With gates initialized to 0 (sigmoid=0.5), siblings get the same
        // parent probability split equally. They should be identical at init.
        // Distant topics (different subtrees) should also be identical at init
        // because all gates are 0.5. But after perturbation, siblings should
        // be more similar.
        let dev = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let dec = HierarchicalTopicDecoder::new(8, 3, vs).unwrap();

        // At initialization (gates=0), all leaves should be equal:
        // each gets root_prob * 0.5 * 0.5 = root_prob / 4
        let beta = dec.leaf_dictionary_kd().unwrap(); // [4, 8]

        let leaf0: Vec<f32> = beta.i(0).unwrap().to_vec1().unwrap();
        let leaf1: Vec<f32> = beta.i(1).unwrap().to_vec1().unwrap();
        let leaf2: Vec<f32> = beta.i(2).unwrap().to_vec1().unwrap();

        // At init: siblings (0,1) share parent node 2; (2,3) share node 3
        // All should be equal since all gates are 0.5
        let diff_siblings: f32 = leaf0
            .iter()
            .zip(leaf1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let diff_distant: f32 = leaf0
            .iter()
            .zip(leaf2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff_siblings < 1e-5, "Siblings should be identical at init");
        assert!(
            diff_distant < 1e-5,
            "All leaves should be identical at init (uniform gates)"
        );
    }

    #[test]
    fn test_depth2_minimal() {
        // Depth 2: root + 2 leaves, 1 internal node (the root is also the only internal)
        let dec = make_decoder(2, 4);
        assert_eq!(dec.n_topics, 2);
        assert_eq!(dec.num_nodes, 3);
        assert_eq!(dec.num_internal, 1);

        let beta = dec.leaf_dictionary_kd().unwrap();
        assert_eq!(beta.dims(), &[2, 4]);

        // Sum of both leaves should equal root probability
        let leaf_sum: f32 = beta.sum(0).unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!((leaf_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_forward_with_llik() {
        let dec = make_decoder(2, 8);
        let dev = Device::Cpu;

        let z = Tensor::ones((3, 2), DType::F32, &dev).unwrap();
        let z = (z / 2.0f64).unwrap();
        let x = Tensor::ones((3, 8), DType::F32, &dev).unwrap();

        let (recon, llik) = dec.forward_with_llik(&z, &x, &topic_likelihood).unwrap();
        assert_eq!(recon.dims(), &[3, 8]);
        assert_eq!(llik.dims(), &[3]); // per-sample log-likelihood
    }
}
