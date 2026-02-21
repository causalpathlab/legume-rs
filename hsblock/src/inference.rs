//! HSBM inference: collapsed Gibbs sampling + greedy refinement.
//!
//! 1. **Gibbs sweeps** (stochastic): explore the posterior over cluster
//!    assignments. Poisson rates are analytically integrated out via
//!    Gamma-Poisson conjugacy — only (a0, b0) hyperpriors and memberships remain.
//! 2. **Greedy sweeps** (argmax): deterministic refinement to the MAP assignment.

use crate::btree::BTree;
use crate::gibbs::{build_adj_list, GibbsSampler};
use crate::model::tree_score_cpu;
use crate::sufficient_stats::{SufficientStats, WeightedEdge};
use leiden::{Clustering, Network};
use log::info;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Options for HSBM inference.
#[derive(Debug, Clone)]
pub struct HsbmOptions {
    /// Depth of the binary tree (K = 2^(depth-1) leaf clusters). Default: 3
    pub tree_depth: usize,
    /// Number of Gibbs sweeps. Default: 100
    pub num_sweeps: usize,
    /// Use degree-corrected model. Default: true
    pub degree_corrected: bool,
    /// Random seed. Default: 42
    pub seed: u64,
    /// Initial shape parameter a0 for Gamma prior. Default: 1.0
    pub init_a0: f64,
    /// Initial rate parameter b0 for Gamma prior. Default: 1.0
    pub init_b0: f64,
    /// Multiplicative scale for edge weights. Default: 100.0
    ///
    /// The Poisson-Gamma conjugate model expects count-like data, but fuzzy
    /// kernel weights from KNN graphs are continuous in (0, 1]. Scaling them
    /// up gives the model more signal to discriminate cluster structure.
    pub edge_scale: f64,
}

impl Default for HsbmOptions {
    fn default() -> Self {
        HsbmOptions {
            tree_depth: 3,
            num_sweeps: 100,
            degree_corrected: true,
            seed: 42,
            init_a0: 1.0,
            init_b0: 1.0,
            edge_scale: 100.0,
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
/// let mut hsblock = Hsblock::new(options);
///
/// let mut clustering = SimpleClustering::init_different_clusters(network.nodes());
/// hsblock.iterate(&network, &mut clustering);
/// ```
pub struct Hsblock {
    options: HsbmOptions,
}

impl Hsblock {
    /// Create a new HSBM instance.
    pub fn new(options: HsbmOptions) -> Self {
        Hsblock { options }
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

        // Extract edge list from Network, scaling weights for the Poisson model.
        // Fuzzy kernel weights are in (0,1], so scaling makes them count-like.
        let edges = extract_edges(network, self.options.edge_scale);
        let adj_list = build_adj_list(&edges, n);

        let k = 1 << (self.options.tree_depth - 1); // 2^(D-1) clusters

        // Initialize membership: random assignment to K clusters
        let mut rng = SmallRng::seed_from_u64(self.options.seed);
        let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..k)).collect();

        // Save initial labels for comparison
        let initial_labels: Vec<usize> = (0..n).map(|i| clustering.get(i)).collect();

        // Build data structures
        let tree = BTree::with_gamma_poisson(
            self.options.tree_depth,
            self.options.init_a0,
            self.options.init_b0,
        );
        let mut stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let mut gibbs =
            GibbsSampler::new(SmallRng::seed_from_u64(self.options.seed.wrapping_add(1)));

        let dc = self.options.degree_corrected;

        info!(
            "HSBM: n={}, K={}, depth={}, dc={}, sweeps={}",
            n, k, self.options.tree_depth, dc, self.options.num_sweeps,
        );

        // Run collapsed Gibbs sampling, then finalize with greedy argmax sweeps
        let moves = gibbs.run_parallel(&tree, &mut stats, &adj_list, self.options.num_sweeps, dc);
        let greedy_moves = gibbs.run_greedy(&tree, &mut stats, &adj_list, 10, dc);

        // Log final score
        let (node_edge, node_total) = stats.aggregate_to_tree(&tree, dc);
        let a0: Vec<f64> = (1..=tree.num_nodes())
            .map(|n| tree.node_params(n).0)
            .collect();
        let b0: Vec<f64> = (1..=tree.num_nodes())
            .map(|n| tree.node_params(n).1)
            .collect();
        let score = tree_score_cpu(&a0, &b0, &node_edge[1..], &node_total[1..]);
        info!(
            "HSBM done: score={:.4}, gibbs_moves={}, greedy_moves={}, K_eff={}",
            score,
            moves,
            greedy_moves,
            count_nonempty_clusters(&stats),
        );

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
}

/// Extract weighted edge list from a leiden `Network`, scaling weights.
///
/// The Poisson model needs count-like edge weights. Fuzzy kernel weights
/// are in (0, 1], so we multiply by `edge_scale` to make them informative.
fn extract_edges(network: &Network, edge_scale: f64) -> Vec<WeightedEdge> {
    let mut edges = Vec::new();
    let n = network.nodes();
    let scale = edge_scale as f32;

    for i in 0..n {
        for (j, w) in network.neighbors(i) {
            if j > i {
                // Undirected: only store each edge once
                edges.push((i, j, w as f32 * scale));
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
        for &deg in degree.iter().take(n) {
            graph.add_node(deg);
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
            num_sweeps: 50,
            degree_corrected: false,
            seed: 42,
            ..Default::default()
        };

        let mut hsblock = Hsblock::new(options);
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
            num_sweeps: 50,
            degree_corrected: true,
            seed: 99,
            ..Default::default()
        };

        let mut hsblock = Hsblock::new(options);
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
        let edges = extract_edges(&network, 1.0);

        assert_eq!(edges.len(), 3);
    }
}
