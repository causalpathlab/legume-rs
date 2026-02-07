use crate::knn_match::ColumnDict;

use dashmap::DashMap;
use indicatif::ParallelProgressIterator;
use log::info;
use nalgebra::DMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use rayon::prelude::*;

const DEFAULT_BLOCK_SIZE: usize = 1000;

pub struct KnnGraph {
    /// Symmetric CSC adjacency matrix (n_nodes x n_nodes)
    pub adjacency: CscMatrix<f32>,
    /// Sorted edge list (i < j), deduplicated
    pub edges: Vec<(usize, usize)>,
    /// Edge distances/weights, parallel to `edges`
    pub distances: Vec<f32>,
    /// Number of nodes
    pub n_nodes: usize,
}

pub struct KnnGraphArgs {
    pub knn: usize,
    pub block_size: usize,
}

impl KnnGraph {
    /// Build a KNN graph from column vectors.
    ///
    /// * `points` - transposed coordinate matrix (d x n), where each column is a point
    /// * `args` - KNN graph construction parameters
    pub fn from_columns(points: &DMatrix<f32>, args: KnnGraphArgs) -> anyhow::Result<KnnGraph> {
        let nn = points.ncols();
        let points_vec = points.column_iter().collect::<Vec<_>>();
        let names = (0..nn).collect::<Vec<_>>();

        let dict = ColumnDict::from_dvector_views(points_vec, names);
        Self::build_from_dict(dict, nn, &args)
    }

    /// Build a KNN graph from row vectors (cells × features).
    ///
    /// * `data` - matrix (n x d), where each row is a point
    /// * `args` - KNN graph construction parameters
    pub fn from_rows(data: &DMatrix<f32>, args: KnnGraphArgs) -> anyhow::Result<KnnGraph> {
        let transposed = data.transpose();
        Self::from_columns(&transposed, args)
    }

    fn build_from_dict(
        dict: ColumnDict<usize>,
        nn: usize,
        args: &KnnGraphArgs,
    ) -> anyhow::Result<KnnGraph> {
        let nquery = (args.knn + 1).min(nn).max(2);

        let jobs = create_jobs(nn, args.block_size);
        let njobs = jobs.len() as u64;

        /////////////////////////////////////////////////////////////////
        // step 1: searching nearest neighbours                        //
        /////////////////////////////////////////////////////////////////

        let triplets: DashMap<(usize, usize), f32> = DashMap::new();

        jobs.into_par_iter()
            .progress_count(njobs)
            .try_for_each(|(lb, ub)| -> anyhow::Result<()> {
                for i in lb..ub {
                    let (_indices, _distances) = dict.search_others(&i, nquery)?;
                    for (j, d_ij) in _indices.into_iter().zip(_distances) {
                        triplets.insert((i, j), d_ij);
                    }
                }
                Ok(())
            })?;

        info!("{} triplets by kNN matching", triplets.len());

        if triplets.is_empty() {
            return Err(anyhow::anyhow!("empty triplets"));
        }

        //////////////////////////////////////////////////
        // step 2: reciprocal filtering (i <-> j edges) //
        //////////////////////////////////////////////////

        let mut edges: Vec<((usize, usize), f32)> = triplets
            .par_iter()
            .filter_map(|entry| {
                let &(i, j) = entry.key();
                if i < j && triplets.contains_key(&(j, i)) {
                    Some(((i, j), *entry.value()))
                } else {
                    None
                }
            })
            .collect();

        edges.par_sort_by_key(|&(ij, _)| ij);
        edges.dedup();

        info!("{} triplets after reciprocal matching", edges.len());

        ///////////////////////////////////////////////
        // step 3: construct sparse network backbone //
        ///////////////////////////////////////////////

        let mut coo = CooMatrix::new(nn, nn);
        for &((i, j), v) in edges.iter() {
            coo.push(i, j, v);
            coo.push(j, i, v);
        }

        let adjacency = CscMatrix::from(&coo);

        let (edge_pairs, distances): (Vec<_>, Vec<_>) = edges.into_iter().unzip();

        Ok(KnnGraph {
            adjacency,
            edges: edge_pairs,
            distances,
            n_nodes: nn,
        })
    }

    /// Get neighbors of a node from the CSC adjacency matrix
    pub fn neighbors(&self, node: usize) -> &[usize] {
        let offsets = self.adjacency.col_offsets();
        let start = offsets[node];
        let end = offsets[node + 1];
        &self.adjacency.row_indices()[start..end]
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tight clusters of 5 points each in 2D, well separated
    fn two_cluster_matrix() -> DMatrix<f32> {
        DMatrix::from_row_slice(
            10,
            2,
            &[
                // Cluster A near origin
                0.0, 0.0, //
                0.1, 0.0, //
                0.0, 0.1, //
                0.1, 0.1, //
                0.05, 0.05, //
                // Cluster B far away
                10.0, 10.0, //
                10.1, 10.0, //
                10.0, 10.1, //
                10.1, 10.1, //
                10.05, 10.05, //
            ],
        )
    }

    #[test]
    fn test_from_rows_basic() {
        let data = two_cluster_matrix();
        let graph = KnnGraph::from_rows(
            &data,
            KnnGraphArgs {
                knn: 4,
                block_size: 100,
            },
        )
        .unwrap();

        assert_eq!(graph.num_nodes(), 10);
        assert!(graph.num_edges() > 0);
        assert_eq!(graph.edges.len(), graph.distances.len());

        // All edges should be (i < j)
        for &(i, j) in &graph.edges {
            assert!(i < j, "Edge ({}, {}) not canonical", i, j);
        }

        // All distances should be non-negative
        for &d in &graph.distances {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_from_columns_equivalent_to_from_rows() {
        let data = two_cluster_matrix();
        let transposed = data.transpose();

        let g_rows = KnnGraph::from_rows(
            &data,
            KnnGraphArgs {
                knn: 3,
                block_size: 100,
            },
        )
        .unwrap();

        let g_cols = KnnGraph::from_columns(
            &transposed,
            KnnGraphArgs {
                knn: 3,
                block_size: 100,
            },
        )
        .unwrap();

        // Both should represent the same graph structure
        assert_eq!(g_rows.num_nodes(), g_cols.num_nodes());
        // HNSW is approximate, so edge sets may differ slightly;
        // just check the counts are close
        let diff = (g_rows.num_edges() as i64 - g_cols.num_edges() as i64).unsigned_abs();
        assert!(
            diff <= 2,
            "Edge counts differ too much: {} vs {}",
            g_rows.num_edges(),
            g_cols.num_edges()
        );
    }

    #[test]
    fn test_two_clusters_no_cross_edges() {
        let data = two_cluster_matrix();
        let graph = KnnGraph::from_rows(
            &data,
            KnnGraphArgs {
                knn: 4,
                block_size: 100,
            },
        )
        .unwrap();

        // With k=4 and well-separated clusters, no edges should cross clusters
        for &(i, j) in &graph.edges {
            let same_cluster = (i < 5 && j < 5) || (i >= 5 && j >= 5);
            assert!(
                same_cluster,
                "Cross-cluster edge ({}, {}) found between well-separated clusters",
                i, j
            );
        }
    }

    #[test]
    fn test_neighbors_symmetric() {
        let data = two_cluster_matrix();
        let graph = KnnGraph::from_rows(
            &data,
            KnnGraphArgs {
                knn: 3,
                block_size: 100,
            },
        )
        .unwrap();

        // Adjacency should be symmetric: if i is neighbor of j, j is neighbor of i
        for node in 0..graph.num_nodes() {
            for &neighbor in graph.neighbors(node) {
                let reverse_neighbors = graph.neighbors(neighbor);
                assert!(
                    reverse_neighbors.contains(&node),
                    "Node {} has neighbor {} but not vice versa",
                    node,
                    neighbor
                );
            }
        }
    }

    #[test]
    fn test_adjacency_dimensions() {
        let data = two_cluster_matrix();
        let graph = KnnGraph::from_rows(
            &data,
            KnnGraphArgs {
                knn: 3,
                block_size: 100,
            },
        )
        .unwrap();

        assert_eq!(graph.adjacency.nrows(), 10);
        assert_eq!(graph.adjacency.ncols(), 10);
    }

    #[test]
    fn test_create_jobs_helper() {
        let jobs = create_jobs(10, 3);
        assert_eq!(jobs, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);

        let jobs = create_jobs(6, 3);
        assert_eq!(jobs, vec![(0, 3), (3, 6)]);

        let jobs = create_jobs(1, 100);
        assert_eq!(jobs, vec![(0, 1)]);

        // block_size=0 should fall back to DEFAULT_BLOCK_SIZE
        let jobs = create_jobs(5, 0);
        assert_eq!(jobs, vec![(0, 5)]);
    }
}

fn create_jobs(ntot: usize, block_size: usize) -> Vec<(usize, usize)> {
    let block_size = if block_size == 0 {
        DEFAULT_BLOCK_SIZE
    } else {
        block_size
    };
    let nblock = ntot.div_ceil(block_size);
    (0..nblock)
        .map(|block| {
            let lb = block * block_size;
            let ub = ((block + 1) * block_size).min(ntot);
            (lb, ub)
        })
        .collect()
}
