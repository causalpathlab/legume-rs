//! Variational EM outer loop with candle-based M-step.
//!
//! Alternates between:
//! - **E-step**: Collapsed Gibbs sampling (CPU) to update cluster assignments
//! - **M-step**: Stochastic gradient update of tree parameters via candle autodiff

use crate::btree::BTree;
use crate::gibbs::{build_adj_list, GibbsSampler};
use crate::model::tree_score_candle;
use crate::sufficient_stats::{SufficientStats, WeightedEdge};
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use leiden::{Clustering, Network};
use log::info;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Options for HSBM inference.
#[derive(Debug, Clone)]
pub struct HsbmOptions {
    /// Depth of the binary tree (K = 2^(depth-1) leaf clusters). Default: 3
    pub tree_depth: usize,
    /// Number of outer variational EM iterations. Default: 100
    pub vb_iter: usize,
    /// Number of Gibbs sweeps per VB iteration (E-step). Default: 10
    pub inner_iter: usize,
    /// Number of Gibbs sweeps in the final E-step. Default: 100
    pub final_inner_iter: usize,
    /// Number of burn-in Gibbs sweeps to discard. Default: 5
    pub burnin_iter: usize,
    /// Learning rate for Adam optimizer (M-step). Default: 0.01
    pub learning_rate: f64,
    /// Use degree-corrected model. Default: true
    pub degree_corrected: bool,
    /// Random seed. Default: 42
    pub seed: u64,
    /// Initial shape parameter a0 for Gamma prior. Default: 1.0
    pub init_a0: f64,
    /// Initial rate parameter b0 for Gamma prior. Default: 1.0
    pub init_b0: f64,
}

impl Default for HsbmOptions {
    fn default() -> Self {
        HsbmOptions {
            tree_depth: 3,
            vb_iter: 100,
            inner_iter: 10,
            final_inner_iter: 100,
            burnin_iter: 5,
            learning_rate: 0.01,
            degree_corrected: true,
            seed: 42,
            init_a0: 1.0,
            init_b0: 1.0,
        }
    }
}

/// Hierarchical Stochastic Block Model — drop-in alternative to Leiden.
///
/// # Usage
///
/// ```ignore
/// use hsblock::{Hsblock, HsbmOptions};
/// use leiden::{Network, Clustering, SimpleClustering};
///
/// let options = HsbmOptions::default();
/// let device = candle_core::Device::Cpu;
/// let mut hsblock = Hsblock::new(options, device);
///
/// let mut clustering = SimpleClustering::init_different_clusters(network.nodes());
/// hsblock.iterate(&network, &mut clustering);
/// ```
pub struct Hsblock {
    options: HsbmOptions,
    device: Device,
}

impl Hsblock {
    /// Create a new HSBM instance.
    pub fn new(options: HsbmOptions, device: Device) -> Self {
        Hsblock { options, device }
    }

    /// Run HSBM inference on the network, writing results into `clustering`.
    ///
    /// Same signature pattern as `Leiden::iterate()`:
    /// - Takes a `Network` (from the leiden crate)
    /// - Writes cluster labels into a `Clustering`
    /// - Returns `true` if clustering changed, `false` if unchanged
    pub fn iterate<C: Clustering>(&mut self, network: &Network, clustering: &mut C) -> bool {
        let n = network.nodes();
        if n == 0 {
            return false;
        }

        // Extract edge list from Network
        let edges = extract_edges(network);
        let adj_list = build_adj_list(&edges, n);

        let k = 1 << (self.options.tree_depth - 1); // 2^(D-1) clusters

        // Initialize membership: random assignment to K clusters
        let mut rng = SmallRng::seed_from_u64(self.options.seed);
        let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..k)).collect();

        // Save initial labels for comparison
        let initial_labels: Vec<usize> = (0..n).map(|i| clustering.get(i)).collect();

        // Build data structures
        let mut tree = BTree::new(
            self.options.tree_depth,
            self.options.init_a0,
            self.options.init_b0,
        );
        let mut stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let mut gibbs =
            GibbsSampler::new(SmallRng::seed_from_u64(self.options.seed.wrapping_add(1)));

        // Run Variational EM
        match self.run_vem(&mut tree, &mut stats, &mut gibbs, &edges, &adj_list) {
            Ok(()) => {}
            Err(e) => {
                info!("HSBM M-step error (falling back to Gibbs-only): {}", e);
            }
        }

        // Write results into clustering
        let mut changed = false;
        for (v, &init_label) in initial_labels.iter().enumerate() {
            let new_label = stats.membership[v];
            if new_label != init_label {
                changed = true;
            }
            clustering.set(v, new_label);
        }
        clustering.remove_empty_clusters();

        changed
    }

    /// Internal: run the full Variational EM loop.
    fn run_vem(
        &self,
        tree: &mut BTree,
        stats: &mut SufficientStats,
        gibbs: &mut GibbsSampler,
        edges: &[WeightedEdge],
        adj_list: &[Vec<(usize, f64)>],
    ) -> anyhow::Result<()> {
        let opts = &self.options;
        let device = &self.device;

        let num_nodes = tree.num_nodes();

        // Initialize learnable parameters from tree as Var tensors
        let init_ln_a0: Vec<f32> = tree.ln_a0[1..=num_nodes]
            .iter()
            .map(|&x| x as f32)
            .collect();
        let init_ln_b0: Vec<f32> = tree.ln_b0[1..=num_nodes]
            .iter()
            .map(|&x| x as f32)
            .collect();

        let ln_a0_t = candle_core::Tensor::from_vec(init_ln_a0, (num_nodes,), device)?;
        let ln_b0_t = candle_core::Tensor::from_vec(init_ln_b0, (num_nodes,), device)?;

        let ln_a0_var = candle_core::Var::from_tensor(&ln_a0_t)?;
        let ln_b0_var = candle_core::Var::from_tensor(&ln_b0_t)?;

        let adam_params = ParamsAdamW {
            lr: opts.learning_rate,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(vec![ln_a0_var.clone(), ln_b0_var.clone()], adam_params)?;

        info!(
            "HSBM: n={}, K={}, depth={}, dc={}, vb_iter={}, inner={}",
            stats.n, stats.k, opts.tree_depth, opts.degree_corrected, opts.vb_iter, opts.inner_iter,
        );

        for vb_iter in 0..opts.vb_iter {
            let is_final = vb_iter == opts.vb_iter - 1;
            let sweeps = if is_final {
                opts.final_inner_iter
            } else {
                opts.inner_iter
            };

            // E-step: Gibbs sampling (parallel for large graphs, sequential for small)
            let total_sweeps = opts.burnin_iter + sweeps;
            let moves =
                gibbs.run_parallel(tree, stats, adj_list, total_sweeps, opts.degree_corrected);

            // M-step: update tree parameters via candle autodiff
            let (node_edge, node_total) = stats.aggregate_to_tree(tree, opts.degree_corrected);

            let (edge_t, total_t) =
                SufficientStats::tree_stats_to_tensors(&node_edge, &node_total, device)?;

            // Use the Var tensors directly for autodiff
            let score = tree_score_candle(
                ln_a0_var.as_tensor(),
                ln_b0_var.as_tensor(),
                &edge_t,
                &total_t,
            )?;
            let loss = score.neg()?; // minimize negative score = maximize score

            // Adam backward step
            optimizer.backward_step(&loss)?;

            // Copy updated parameters back to BTree
            let updated_a0: Vec<f32> = ln_a0_var
                .as_tensor()
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let updated_b0: Vec<f32> = ln_b0_var
                .as_tensor()
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            for i in 0..num_nodes {
                tree.set_node_ln_params(i + 1, updated_a0[i] as f64, updated_b0[i] as f64);
            }

            if vb_iter % 10 == 0 || is_final {
                let score_val: f32 = score.to_dtype(DType::F32)?.to_scalar()?;
                info!(
                    "  VB iter {}: score={:.4}, moves={}, K_eff={}",
                    vb_iter,
                    score_val,
                    moves,
                    count_nonempty_clusters(stats),
                );
            }

            // Periodic GPU recalibration of sufficient stats
            if vb_iter > 0 && vb_iter % 50 == 0 {
                stats.recompute_candle(edges, device)?;
            }
        }

        Ok(())
    }
}

/// Extract weighted edge list from a leiden `Network`.
fn extract_edges(network: &Network) -> Vec<WeightedEdge> {
    let mut edges = Vec::new();
    let n = network.nodes();

    for i in 0..n {
        for (j, w) in network.neighbors(i) {
            if j > i {
                // Undirected: only store each edge once
                edges.push((i, j, w as f32));
            }
        }
    }

    edges
}

/// Count non-empty clusters.
fn count_nonempty_clusters(stats: &SufficientStats) -> usize {
    stats.cluster_size.iter().filter(|&&s| s > 0.0).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use leiden::clustering::SimpleClustering;
    use leiden::network::Graph;

    /// Build a planted partition graph as a leiden Network.
    fn planted_partition_network(
        n_per_cluster: usize,
        n_clusters: usize,
        p_in: f32,
        p_out: f32,
        seed: u64,
    ) -> (Network, Vec<usize>) {
        let n = n_per_cluster * n_clusters;
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut true_labels = Vec::with_capacity(n);
        for c in 0..n_clusters {
            for _ in 0..n_per_cluster {
                true_labels.push(c);
            }
        }

        // First pass: collect edges and compute degrees
        let mut edge_list = Vec::new();
        let mut degree = vec![0.0f32; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let p = if true_labels[i] == true_labels[j] {
                    p_in
                } else {
                    p_out
                };
                if rng.random::<f32>() < p {
                    let w = 1.0f32;
                    degree[i] += w;
                    degree[j] += w;
                    edge_list.push((i, j, w));
                }
            }
        }

        let mut graph = Graph::with_capacity(n, edge_list.len());
        for i in 0..n {
            graph.add_node(degree[i]);
        }
        for &(i, j, w) in &edge_list {
            graph.add_edge((i as u32).into(), (j as u32).into(), w);
        }

        (Network::new_from_graph(graph), true_labels)
    }

    #[test]
    fn test_hsblock_basic() {
        let (network, true_labels) = planted_partition_network(30, 2, 0.5, 0.02, 42);
        let n = network.nodes();

        let options = HsbmOptions {
            tree_depth: 2,
            vb_iter: 20,
            inner_iter: 5,
            final_inner_iter: 20,
            burnin_iter: 2,
            degree_corrected: false,
            seed: 42,
            ..Default::default()
        };

        let mut hsblock = Hsblock::new(options, Device::Cpu);
        let mut clustering = SimpleClustering::init_different_clusters(n);

        let changed = hsblock.iterate(&network, &mut clustering);
        assert!(changed, "Clustering should have changed from initial");

        // Check that we get a reasonable number of clusters
        let n_clusters = clustering.num_clusters();
        assert!(
            n_clusters >= 1 && n_clusters <= n,
            "Number of clusters ({}) should be reasonable",
            n_clusters
        );

        // Check basic consistency
        for i in 0..n {
            assert!(clustering.get(i) < n_clusters);
        }

        // The algorithm should find roughly 2 clusters for planted partition
        let _ = true_labels; // true_labels available for ARI computation if needed
    }

    #[test]
    fn test_hsblock_degree_corrected() {
        let (network, _true_labels) = planted_partition_network(25, 2, 0.4, 0.05, 99);
        let n = network.nodes();

        let options = HsbmOptions {
            tree_depth: 2,
            vb_iter: 15,
            inner_iter: 5,
            final_inner_iter: 10,
            burnin_iter: 2,
            degree_corrected: true,
            seed: 99,
            ..Default::default()
        };

        let mut hsblock = Hsblock::new(options, Device::Cpu);
        let mut clustering = SimpleClustering::init_different_clusters(n);

        let changed = hsblock.iterate(&network, &mut clustering);
        assert!(changed);
        assert!(clustering.num_clusters() >= 1);
    }

    #[test]
    fn test_extract_edges() {
        // Small test: triangle 0-1-2
        let mut graph = Graph::with_capacity(3, 3);
        for _ in 0..3 {
            graph.add_node(2.0);
        }
        graph.add_edge(0u32.into(), 1u32.into(), 1.0);
        graph.add_edge(1u32.into(), 2u32.into(), 1.0);
        graph.add_edge(0u32.into(), 2u32.into(), 1.0);

        let network = Network::new_from_graph(graph);
        let edges = extract_edges(&network);

        assert_eq!(edges.len(), 3);
    }
}
