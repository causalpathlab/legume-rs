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
    /// If true, keep only reciprocal edges (i→j AND j→i).
    /// If false, keep union edges (i→j OR j→i), using min distance.
    pub reciprocal: bool,
}

fn median_f32(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
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

        jobs.into_par_iter().progress_count(njobs).try_for_each(
            |(lb, ub)| -> anyhow::Result<()> {
                for i in lb..ub {
                    let (_indices, _distances) = dict.search_others(&i, nquery)?;
                    for (j, d_ij) in _indices.into_iter().zip(_distances) {
                        triplets.insert((i, j), d_ij);
                    }
                }
                Ok(())
            },
        )?;

        info!("{} triplets by kNN matching", triplets.len());

        if triplets.is_empty() {
            return Err(anyhow::anyhow!("empty triplets"));
        }

        ///////////////////////////////////////////////////
        // step 2: edge filtering (reciprocal or union) //
        ///////////////////////////////////////////////////

        let mut edges: Vec<((usize, usize), f32)> = if args.reciprocal {
            // Intersection: keep (i,j) only if both i→j and j→i exist
            triplets
                .par_iter()
                .filter_map(|entry| {
                    let &(i, j) = entry.key();
                    if i < j && triplets.contains_key(&(j, i)) {
                        Some(((i, j), *entry.value()))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            // Union: keep (i,j) if either i→j or j→i exists, min distance
            triplets
                .par_iter()
                .filter_map(|entry| {
                    let &(i, j) = entry.key();
                    if i < j {
                        let d_ij = *entry.value();
                        let d_ji = triplets.get(&(j, i)).map(|e| *e).unwrap_or(d_ij);
                        Some(((i, j), d_ij.min(d_ji)))
                    } else if i > j && !triplets.contains_key(&(j, i)) {
                        // Only (i→j) exists with i > j; emit as canonical (j, i)
                        Some(((j, i), *entry.value()))
                    } else {
                        None
                    }
                })
                .collect()
        };

        edges.par_sort_by_key(|&(ij, _)| ij);
        edges.dedup();

        info!(
            "{} edges after {} matching",
            edges.len(),
            if args.reciprocal {
                "reciprocal"
            } else {
                "union"
            }
        );

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

    /// Convert distances to similarity weights using an exponential kernel:
    /// `w = exp(-d / σ)` where σ = median distance.
    ///
    /// Returns weights parallel to `self.edges`, all in (0, 1].
    /// Consistent with the softmax(-d) pattern used in counterfactual
    /// inference (data-beans-alg) but with a global bandwidth.
    pub fn exp_kernel_weights(&self) -> Vec<f32> {
        if self.distances.is_empty() {
            return Vec::new();
        }
        let sigma = median_f32(&self.distances);
        let sigma = if sigma <= 0.0 { 1.0 } else { sigma };
        info!("exp_kernel_weights: σ (median distance) = {:.4}", sigma);
        self.distances.iter().map(|&d| (-d / sigma).exp()).collect()
    }

    /// Adaptive-bandwidth kernel weights with local connectivity.
    ///
    /// Per-point sigma calibration (originated in t-SNE, van der Maaten
    /// & Hinton 2008) ensures every node has the same effective number
    /// of neighbors, preventing isolated singletons in sparse regions.
    /// The rho subtraction and fuzzy-union symmetrization follow UMAP
    /// (McInnes et al. 2018), matching the scanpy default for Leiden.
    ///
    /// Algorithm:
    /// 1. rho_i = distance to nearest neighbor (local connectivity)
    /// 2. sigma_i via binary search: sum_j exp(-(d_ij - rho_i)/sigma_i) = log2(k)
    /// 3. Directed weight: w(i→j) = exp(-(d_ij - rho_i) / sigma_i)
    /// 4. Symmetrize: w_sym = w(i→j) + w(j→i) - w(i→j) * w(j→i)
    ///
    /// Returns weights parallel to `self.edges`, all in (0, 1].
    pub fn fuzzy_kernel_weights(&self) -> Vec<f32> {
        if self.distances.is_empty() {
            return Vec::new();
        }

        let offsets = self.adjacency.col_offsets();
        let row_indices = self.adjacency.row_indices();
        let values = self.adjacency.values();

        // Step 1-2: compute rho and sigma per node
        let mut rho = vec![0.0f32; self.n_nodes];
        let mut sigma = vec![1.0f32; self.n_nodes];

        for i in 0..self.n_nodes {
            let start = offsets[i];
            let end = offsets[i + 1];
            let dists: Vec<f32> = (start..end).map(|idx| values[idx]).collect();

            if dists.is_empty() {
                continue;
            }

            // rho = distance to nearest neighbor
            rho[i] = dists.iter().cloned().fold(f32::INFINITY, f32::min);

            // Binary search for sigma: target = log2(k)
            let target = (dists.len() as f32).log2();
            sigma[i] = smooth_knn_sigma(&dists, rho[i], target);
        }

        // Step 3-4: compute directed weights and symmetrize per edge
        let mut weights = Vec::with_capacity(self.edges.len());

        for &(i, j) in &self.edges {
            // directed weight i → j
            let d_ij = self.edge_distance_directed(offsets, row_indices, values, i, j);
            let w_ij = directed_umap_weight(d_ij, rho[i], sigma[i]);

            // directed weight j → i
            let d_ji = self.edge_distance_directed(offsets, row_indices, values, j, i);
            let w_ji = directed_umap_weight(d_ji, rho[j], sigma[j]);

            // fuzzy union: P(at least one edge) = P(A) + P(B) - P(A)*P(B)
            let w_sym = w_ij + w_ji - w_ij * w_ji;
            weights.push(w_sym);
        }

        weights
    }

    /// Look up the distance from node `from` to node `to` in the CSC adjacency.
    fn edge_distance_directed(
        &self,
        offsets: &[usize],
        row_indices: &[usize],
        values: &[f32],
        from: usize,
        to: usize,
    ) -> f32 {
        let start = offsets[from];
        let end = offsets[from + 1];
        for idx in start..end {
            if row_indices[idx] == to {
                return values[idx];
            }
        }
        f32::INFINITY
    }
}

/// Binary search for per-point sigma (UMAP's smooth_knn_dist).
///
/// Finds sigma such that: sum_j exp(-max(0, d_j - rho) / sigma) = target
fn smooth_knn_sigma(dists: &[f32], rho: f32, target: f32) -> f32 {
    const TOLERANCE: f32 = 1e-5;
    const MAX_ITER: usize = 64;

    let mean_dist: f32 = dists.iter().sum::<f32>() / dists.len().max(1) as f32;
    let min_sigma = 1e-3 * mean_dist;

    let mut lo = 0.0f32;
    let mut hi = f32::INFINITY;
    let mut mid = 1.0f32;

    for _ in 0..MAX_ITER {
        let mut psum = 0.0f32;
        for &d in dists {
            let gap = d - rho;
            if gap > 0.0 {
                psum += (-gap / mid).exp();
            } else {
                psum += 1.0;
            }
        }

        if (psum - target).abs() < TOLERANCE {
            break;
        }

        if psum > target {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            if hi.is_infinite() {
                mid *= 2.0;
            } else {
                mid = (lo + hi) / 2.0;
            }
        }
    }

    mid.max(min_sigma)
}

/// Compute a single directed UMAP membership weight.
fn directed_umap_weight(d: f32, rho: f32, sigma: f32) -> f32 {
    if d.is_infinite() || sigma <= 0.0 {
        return 0.0;
    }
    let gap = d - rho;
    if gap <= 0.0 {
        1.0
    } else {
        (-gap / sigma).exp()
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
                reciprocal: true,
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
                reciprocal: true,
            },
        )
        .unwrap();

        let g_cols = KnnGraph::from_columns(
            &transposed,
            KnnGraphArgs {
                knn: 3,
                block_size: 100,
                reciprocal: true,
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
                reciprocal: true,
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
                reciprocal: true,
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
                reciprocal: true,
            },
        )
        .unwrap();

        assert_eq!(graph.adjacency.nrows(), 10);
        assert_eq!(graph.adjacency.ncols(), 10);
    }

    #[test]
    fn test_exp_kernel_weights() {
        let data = two_cluster_matrix();
        let graph = KnnGraph::from_rows(
            &data,
            KnnGraphArgs {
                knn: 4,
                block_size: 100,
                reciprocal: true,
            },
        )
        .unwrap();

        let weights = graph.exp_kernel_weights();
        assert_eq!(weights.len(), graph.num_edges());

        // All weights should be in (0, 1]
        for &w in &weights {
            assert!(w > 0.0, "Weight {} should be > 0", w);
            assert!(w <= 1.0, "Weight {} should be <= 1", w);
        }

        // Median edge gets exp(-1) ≈ 0.37; closer edges get higher weights
        let mean_w: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        assert!(
            mean_w > 0.2 && mean_w < 0.9,
            "Mean weight {} should be in a reasonable range",
            mean_w
        );
    }

    #[test]
    fn test_fuzzy_kernel_weights() {
        let data = two_cluster_matrix();
        let graph = KnnGraph::from_rows(
            &data,
            KnnGraphArgs {
                knn: 4,
                block_size: 100,
                reciprocal: false, // union, like scanpy default
            },
        )
        .unwrap();

        let weights = graph.fuzzy_kernel_weights();
        assert_eq!(weights.len(), graph.num_edges());

        // All weights should be in (0, 1]
        for &w in &weights {
            assert!(w > 0.0, "Weight {} should be > 0", w);
            assert!(w <= 1.0, "Weight {} should be <= 1", w);
        }

        // With UMAP weights, no edge should be near zero (local sigma adapts)
        let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            min_w > 0.01,
            "Min fuzzy weight {} is too small; local sigma should prevent near-zero weights",
            min_w
        );
    }

    #[test]
    fn test_smooth_knn_sigma() {
        // 5 distances, rho = 0.1 (nearest neighbor)
        let dists = [0.1, 0.2, 0.3, 0.5, 1.0];
        let rho = 0.1;
        let target = (5.0f32).log2(); // log2(k)

        let sigma = super::smooth_knn_sigma(&dists, rho, target);
        assert!(sigma > 0.0, "sigma should be positive");

        // Verify the sigma achieves the target
        let psum: f32 = dists
            .iter()
            .map(|&d| {
                let gap = d - rho;
                if gap > 0.0 {
                    (-gap / sigma).exp()
                } else {
                    1.0
                }
            })
            .sum();

        assert!(
            (psum - target).abs() < 0.1,
            "psum {:.3} should be close to target {:.3}",
            psum,
            target
        );
    }

    #[test]
    fn test_median_f32() {
        assert_eq!(median_f32(&[1.0, 3.0, 2.0]), 2.0);
        assert_eq!(median_f32(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median_f32(&[5.0]), 5.0);
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
