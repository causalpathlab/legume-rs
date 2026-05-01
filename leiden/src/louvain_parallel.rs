use crate::parallel_local_moving::ParallelLocalMoving;
use crate::{Clustering, Network};
use rustc_hash::FxHashSet as HashSet;

/// Perform the Louvain clustering algorithm
pub struct ParallelLouvain {
    local_moving: ParallelLocalMoving,
}

impl ParallelLouvain {
    /// Initialize the Leiden algorithm with the given resolution and randomness parameters.
    #[must_use]
    pub fn new(resolution: f64) -> ParallelLouvain {
        ParallelLouvain {
            local_moving: ParallelLocalMoving::new(resolution),
        }
    }

    /// Iterate the Louvain algorithm a single level
    pub fn iterate_one_level<C: Clustering + Clone + Send + Sync + Default>(
        &mut self,
        n: &Network,
        c: &mut C,
    ) -> bool {
        self.local_moving.iterate(n, c)
    }

    /// Build a Louvain-compatible network from a list of adjacencies
    ///
    /// # Panics
    /// If an edge endpoint refers to a node index ≥ `n_nodes`, or if a node
    /// index in the constructed graph cannot be retrieved.
    pub fn build_network<I: Iterator<Item = (u32, u32)>>(n_nodes: usize, adjacency: I) -> Network {
        let mut network = Network::with_capacity(n_nodes);
        let mut node_indices = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            node_indices.push(network.add_node(1.0));
        }
        let mut seen: Vec<HashSet<u32>> = vec![HashSet::default(); n_nodes];
        let mut node_weights = vec![0.0; n_nodes];
        for (i, j) in adjacency {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let i_ = i as usize;
            let j_ = j as usize;
            if seen[i_].insert(j) {
                network.add_edge(i_, j_, 1.0);
                node_weights[i_] += 1.0;
                node_weights[j_] += 1.0;
            }
        }
        for &i in &node_indices {
            *network.node_weight_mut(i) = node_weights[i];
        }
        network
    }
}
