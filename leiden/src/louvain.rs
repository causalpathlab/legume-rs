use crate::standard_local_moving::StandardLocalMoving;
use crate::{Clustering, Network};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rustc_hash::FxHashSet as HashSet;

/// Perform the Louvain clustering algorithm
pub struct Louvain {
    rng: SmallRng,
    local_moving: StandardLocalMoving,
}

/// Default resolution for Louvain
pub const DEFAULT_RESOLUTION: f64 = 1.0;

impl Louvain {
    /// Initialize the Leiden algorithm with the given resolution and randomness parameters.
    /// An optional random seed can be supplied, otherwise a seed of 0 will be used.
    #[must_use]
    pub fn new(resolution: f64, seed: Option<usize>) -> Louvain {
        let seed = seed.unwrap_or_default() as u64;

        Louvain {
            rng: SmallRng::seed_from_u64(seed),
            local_moving: StandardLocalMoving::new(resolution),
        }
    }

    /// Iterate the Louvain algorithm a single level
    pub fn iterate_one_level<C: Clustering>(&mut self, n: &Network, c: &mut C) -> bool {
        self.local_moving.iterate(n, c, &mut self.rng)
    }

    /// Iterate the Louvain algorithm one step. Returns true if cluster labels were updated, otherwise returns false.
    pub fn iterate<C: Clustering>(&mut self, n: &Network, c: &mut C) -> bool {
        // Update the clustering by moving individual nodes between clusters.
        let mut update = self.local_moving.iterate(n, c, &mut self.rng);

        if c.num_clusters() == n.nodes() {
            return update;
        }

        // Create an aggregate network based on the refined clustering of
        // the non-aggregate network.
        let reduced_n = n.create_reduced_network(c);

        // Create one-cluster-per-node clustering
        let mut reduced_clusters = C::init_different_clusters(reduced_n.nodes());

        update |= self.iterate(&reduced_n, &mut reduced_clusters);

        c.merge_clusters(&reduced_clusters);

        update
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
