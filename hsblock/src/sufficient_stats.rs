//! Sufficient statistics for the hierarchical stochastic block model.
//!
//! Tracks the K×K edge count matrix `C = Z * A * Z'`, cluster sizes,
//! and cluster volumes (degree sums) needed for the Poisson model.
//!
//! Supports O(degree) incremental updates when a single vertex moves
//! (Gibbs E-step), plus full recomputation for periodic recalibration.

use crate::btree::BTree;

/// Sufficient statistics for the HSBM.
///
/// All arrays use cluster indices in `0..k`.
#[derive(Debug, Clone)]
pub struct SufficientStats {
    /// Number of vertices
    pub n: usize,
    /// Number of clusters (K = num_leaves of the tree)
    pub k: usize,
    /// K×K edge count matrix, flattened row-major: `edge_counts[i * k + j]`
    pub edge_counts: Vec<f64>,
    /// Cluster sizes: number of vertices in each cluster
    pub cluster_size: Vec<f64>,
    /// Cluster volumes: sum of vertex degrees in each cluster (for degree correction)
    pub cluster_volume: Vec<f64>,
    /// Membership: vertex -> cluster assignment
    pub membership: Vec<usize>,
    /// Per-vertex degree (sum of edge weights)
    pub vertex_degree: Vec<f64>,
}

/// A weighted edge (i, j, weight)
pub type WeightedEdge = (usize, usize, f32);

impl SufficientStats {
    /// Build sufficient statistics from an edge list and initial cluster labels.
    ///
    /// * `edges` - Weighted edge list `(i, j, weight)` (undirected: provide each edge once)
    /// * `n` - Number of vertices
    /// * `k` - Number of clusters
    /// * `labels` - Initial cluster assignment for each vertex (length n)
    pub fn from_edges(edges: &[WeightedEdge], n: usize, k: usize, labels: &[usize]) -> Self {
        assert_eq!(labels.len(), n);
        let mut edge_counts = vec![0.0; k * k];
        let mut cluster_size = vec![0.0; k];
        let mut cluster_volume = vec![0.0; k];
        let mut vertex_degree = vec![0.0; n];

        // Compute vertex degrees
        for &(i, j, w) in edges {
            vertex_degree[i] += w as f64;
            vertex_degree[j] += w as f64;
        }

        // Compute cluster sizes and volumes
        for v in 0..n {
            let c = labels[v];
            cluster_size[c] += 1.0;
            cluster_volume[c] += vertex_degree[v];
        }

        // Compute edge counts between clusters
        for &(i, j, w) in edges {
            let ci = labels[i];
            let cj = labels[j];
            let w = w as f64;
            edge_counts[ci * k + cj] += w;
            if ci != cj {
                edge_counts[cj * k + ci] += w;
            }
        }

        SufficientStats {
            n,
            k,
            edge_counts,
            cluster_size,
            cluster_volume,
            membership: labels.to_vec(),
            vertex_degree,
        }
    }

    /// Get edge count between clusters `ci` and `cj`.
    #[inline]
    pub fn edge_stat(&self, ci: usize, cj: usize) -> f64 {
        self.edge_counts[ci * self.k + cj]
    }

    /// Get the "total" statistic for the pair (ci, cj).
    ///
    /// * Standard model: `cluster_size[ci] * cluster_size[cj]`
    ///   (or `n*(n-1)/2` for self-pairs)
    /// * Degree-corrected: `cluster_volume[ci] * cluster_volume[cj]`
    #[inline]
    pub fn total_stat(&self, ci: usize, cj: usize, degree_corrected: bool) -> f64 {
        if degree_corrected {
            let vi = self.cluster_volume[ci];
            let vj = self.cluster_volume[cj];
            if ci == cj {
                // Self-pair: analogous to n*(n-1)/2 in non-DC.
                // Use vol^2 / 2 to avoid double-counting undirected pairs.
                vi * vj / 2.0
            } else {
                vi * vj
            }
        } else if ci == cj {
            let s = self.cluster_size[ci];
            s * (s - 1.0) / 2.0
        } else {
            self.cluster_size[ci] * self.cluster_size[cj]
        }
    }

    /// Incrementally update statistics when moving `vertex` from `old_c` to `new_c`.
    ///
    /// * `vertex` - The vertex being moved
    /// * `old_c` - Previous cluster of vertex
    /// * `new_c` - New cluster of vertex
    /// * `neighbors` - Slice of `(neighbor_vertex, edge_weight)` for vertex
    pub fn delta_move(
        &mut self,
        vertex: usize,
        old_c: usize,
        new_c: usize,
        neighbors: &[(usize, f64)],
    ) {
        if old_c == new_c {
            return;
        }

        let deg = self.vertex_degree[vertex];

        // Update cluster sizes
        self.cluster_size[old_c] -= 1.0;
        self.cluster_size[new_c] += 1.0;

        // Update cluster volumes
        self.cluster_volume[old_c] -= deg;
        self.cluster_volume[new_c] += deg;

        // Update edge counts: for each neighbor, adjust the block counts
        for &(nbr, w) in neighbors {
            let nc = self.membership[nbr];

            // Remove contribution of edge (vertex, nbr) from old_c
            self.edge_counts[old_c * self.k + nc] -= w;
            if old_c != nc {
                self.edge_counts[nc * self.k + old_c] -= w;
            }

            // Add contribution of edge (vertex, nbr) to new_c
            self.edge_counts[new_c * self.k + nc] += w;
            if new_c != nc {
                self.edge_counts[nc * self.k + new_c] += w;
            }
        }

        // Update membership
        self.membership[vertex] = new_c;
    }

    /// Aggregate K×K edge counts and totals into per-tree-node statistics.
    ///
    /// For each cluster pair (ci, cj), the relevant tree node is `lca(ci, cj)`.
    /// This function sums edge_stat and total_stat for all pairs sharing the same LCA node.
    ///
    /// Returns `(node_edge_stats, node_total_stats)` vectors indexed by 1-indexed tree node.
    pub fn aggregate_to_tree<P: Clone>(
        &self,
        tree: &BTree<P>,
        degree_corrected: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let num_nodes = tree.num_nodes();
        let mut node_edge = vec![0.0; num_nodes + 1]; // 1-indexed
        let mut node_total = vec![0.0; num_nodes + 1];

        for ci in 0..self.k {
            for cj in ci..self.k {
                let lca = tree.lca(ci, cj);
                let e = self.edge_stat(ci, cj);
                let t = self.total_stat(ci, cj, degree_corrected);

                if ci == cj {
                    node_edge[lca] += e;
                    node_total[lca] += t;
                } else {
                    // Each unordered pair (ci, cj) with ci < cj
                    node_edge[lca] += e;
                    node_total[lca] += t;
                }
            }
        }

        (node_edge, node_total)
    }

    /// Full recomputation of sufficient statistics from edge list.
    ///
    /// This is useful for periodic recalibration to avoid floating-point drift
    /// from many incremental delta_move updates.
    pub fn recompute(&mut self, edges: &[WeightedEdge]) {
        let n = self.n;
        let k = self.k;

        self.edge_counts = vec![0.0; k * k];
        self.cluster_size = vec![0.0; k];
        self.cluster_volume = vec![0.0; k];

        for v in 0..n {
            let c = self.membership[v];
            self.cluster_size[c] += 1.0;
            self.cluster_volume[c] += self.vertex_degree[v];
        }

        for &(i, j, w) in edges {
            let ci = self.membership[i];
            let cj = self.membership[j];
            let w = w as f64;
            self.edge_counts[ci * k + cj] += w;
            if ci != cj {
                self.edge_counts[cj * k + ci] += w;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::btree::BTree;

    fn simple_graph() -> (Vec<WeightedEdge>, usize) {
        // Triangle: 0-1-2-0, plus edge 3-4
        // Two natural clusters: {0,1,2} and {3,4}
        let edges = vec![
            (0, 1, 1.0f32),
            (1, 2, 1.0),
            (0, 2, 1.0),
            (3, 4, 1.0),
            (1, 3, 0.5), // weak bridge
        ];
        (edges, 5)
    }

    #[test]
    fn test_from_edges() {
        let (edges, n) = simple_graph();
        let labels = vec![0, 0, 0, 1, 1]; // perfect clustering
        let stats = SufficientStats::from_edges(&edges, n, 2, &labels);

        assert_eq!(stats.cluster_size[0], 3.0);
        assert_eq!(stats.cluster_size[1], 2.0);

        // Within-cluster 0: edges 0-1, 1-2, 0-2 = 3 edges (weight 3.0)
        assert!((stats.edge_stat(0, 0) - 3.0).abs() < 1e-10);

        // Within-cluster 1: edge 3-4 = 1 edge (weight 1.0)
        assert!((stats.edge_stat(1, 1) - 1.0).abs() < 1e-10);

        // Between clusters: edge 1-3 (weight 0.5)
        assert!((stats.edge_stat(0, 1) - 0.5).abs() < 1e-10);
        assert!((stats.edge_stat(1, 0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_delta_move_consistency() {
        let (edges, n) = simple_graph();
        let labels = vec![0, 0, 0, 1, 1];
        let mut stats = SufficientStats::from_edges(&edges, n, 2, &labels);

        // Collect neighbors for vertex 1
        let mut nbrs = Vec::new();
        for &(i, j, w) in &edges {
            if i == 1 {
                nbrs.push((j, w as f64));
            }
            if j == 1 {
                nbrs.push((i, w as f64));
            }
        }

        // Move vertex 1 from cluster 0 to cluster 1
        stats.delta_move(1, 0, 1, &nbrs);

        // Verify by full recomputation
        let new_labels: Vec<usize> = stats.membership.clone();
        let fresh = SufficientStats::from_edges(&edges, n, 2, &new_labels);

        for ci in 0..2 {
            for cj in 0..2 {
                assert!(
                    (stats.edge_stat(ci, cj) - fresh.edge_stat(ci, cj)).abs() < 1e-10,
                    "Mismatch at ({}, {}): delta={}, fresh={}",
                    ci,
                    cj,
                    stats.edge_stat(ci, cj),
                    fresh.edge_stat(ci, cj)
                );
            }
        }
    }

    #[test]
    fn test_aggregate_to_tree() {
        let (edges, n) = simple_graph();
        let labels = vec![0, 0, 0, 1, 1];
        let stats = SufficientStats::from_edges(&edges, n, 2, &labels);

        let tree = BTree::with_gamma_poisson(2, 1.0, 1.0);
        let (node_edge, _node_total) = stats.aggregate_to_tree(&tree, false);

        // Depth 2 tree: root=1, leaves=2,3
        // lca(0,0) = leaf 2, lca(1,1) = leaf 3, lca(0,1) = root 1
        assert!(node_edge[1] > 0.0); // between-cluster edges at root
        assert!(node_edge[2] > 0.0 || node_edge[3] > 0.0); // within-cluster edges at leaves
    }

    #[test]
    fn test_recompute() {
        let (edges, n) = simple_graph();
        let labels = vec![0, 0, 0, 1, 1];
        let mut stats = SufficientStats::from_edges(&edges, n, 2, &labels);

        // Save original
        let orig_counts = stats.edge_counts.clone();

        // Recompute from scratch
        stats.recompute(&edges);

        for i in 0..4 {
            assert!(
                (stats.edge_counts[i] - orig_counts[i]).abs() < 1e-10,
                "Mismatch at {}: recomputed={}, orig={}",
                i,
                stats.edge_counts[i],
                orig_counts[i]
            );
        }
    }
}
