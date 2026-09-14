use crate::fast_local_moving::FastLocalMoving;
use crate::local_merging::LocalMerging;
use crate::{Clustering, Network, SimpleClustering, ZeroVec};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Perform the Leiden clustering algorithm
pub struct Leiden {
    resolution: f64,
    randomness: f64,

    seed: u64,
    /// Drives the sequential local moving.
    rng: SmallRng,
    /// Top-level `iterate` calls so far, and the recursion depth within one:
    /// together with the subnetwork index they key the refinement's streams.
    epoch: u64,
    depth: u64,

    local_moving: FastLocalMoving,
    num_nodes_per_cluster_reduced_network: Vec<usize>,
}

impl Leiden {
    /// Initialize the Leiden algorithm with the given resolution and randomness parameters.
    /// An optional random seed can be supplied, otherwise a seed of 0 will be used.
    #[must_use]
    pub fn new(resolution: f64, randomness: f64, seed: Option<usize>) -> Leiden {
        let seed = seed.unwrap_or_default() as u64;

        Leiden {
            resolution,
            randomness,
            seed,
            rng: SmallRng::seed_from_u64(seed),
            epoch: 0,
            depth: 0,
            local_moving: FastLocalMoving::new(resolution),
            num_nodes_per_cluster_reduced_network: Vec::new(),
        }
    }

    /// The stream that refines subnetwork `i` at the current epoch and depth:
    /// a function of the seed and the position alone, so the refinement is
    /// the same however rayon schedules the subnetworks.
    fn refinement_rng(&self, i: usize) -> SmallRng {
        let mut h = self.seed;
        for salt in [self.epoch, self.depth, i as u64] {
            h = (h ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)).rotate_left(23);
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        }
        SmallRng::seed_from_u64(h)
    }

    /// Iterate the Leiden algorithm one step. Returns true if cluster labels were updated, otherwise returns false.
    pub fn iterate<C: Clustering>(&mut self, n: &Network, c: &mut C) -> bool {
        if self.depth == 0 {
            self.epoch += 1;
        }

        // Update the clustering by moving individual nodes between clusters.
        let mut update = self.local_moving.iterate(n, c, &mut self.rng);

        if c.num_clusters() == n.nodes() {
            return update;
        }

        let subnetworks = n.create_subnetworks(c);

        let nodes_per_cluster = c.nodes_per_cluster();

        // clear clustering
        c.clear();

        self.num_nodes_per_cluster_reduced_network
            .zero_len(subnetworks.len());
        let mut cluster_counter = 0;

        // Refine every subnetwork in parallel: they are independent, and each
        // draws from its own seeded stream.
        let sub_clusterings: Vec<SimpleClustering> = subnetworks
            .par_iter()
            .enumerate()
            .map_init(
                || LocalMerging::new(self.randomness, self.resolution),
                |local_merging, (i, sub)| local_merging.run(sub, &mut self.refinement_rng(i)),
            )
            .collect();

        for (i, sub_clustering) in sub_clusterings.iter().enumerate() {
            for (j, &node) in nodes_per_cluster[i]
                .iter()
                .enumerate()
                .take(subnetworks[i].nodes())
            {
                c.set(node, cluster_counter + sub_clustering.get(j));
            }

            cluster_counter += sub_clustering.num_clusters();
            self.num_nodes_per_cluster_reduced_network[i] = sub_clustering.num_clusters();
        }
        c.remove_empty_clusters();

        // Create an aggregate network based on the refined clustering of
        // the non-aggregate network.
        let reduced_n = n.create_reduced_network(c);

        // Create an initial clustering for the aggregate network based on the
        // non-refined clustering of the non-aggregate network.
        let mut clusters_reduced_network = vec![0; c.num_clusters()];

        let mut i = 0;
        for (j, num_nodes) in self
            .num_nodes_per_cluster_reduced_network
            .iter()
            .enumerate()
        {
            for cluster in clusters_reduced_network.iter_mut().skip(i).take(*num_nodes) {
                *cluster = j;
            }
            i += num_nodes;
        }

        let mut clustering_reduced_network = C::new_from_labels(&clusters_reduced_network);

        // Recursively apply the algorithm to the aggregate network,
        // starting from the initial clustering created for this network.
        self.depth += 1;
        update |= self.iterate(&reduced_n, &mut clustering_reduced_network);
        self.depth -= 1;

        // Update the clustering of the non-aggregate network so that it
        // coincides with the final clustering obtained for the aggregate
        // network.
        c.merge_clusters(&clustering_reduced_network);

        update
    }
}
