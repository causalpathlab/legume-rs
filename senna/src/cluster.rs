//! Clustering utilities and methods for single-cell data
//!
//! Supports multiple clustering algorithms on latent representations
//! (topic proportions, SVD embeddings, etc.)

use crate::embed_common::*;
use hsblock::{Hsblock, HsbmOptions};
use leiden::clustering::SimpleClustering;
use leiden::Clustering;
use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::knn_graph::{self, KnnGraph, KnnGraphArgs};
use matrix_util::traits::MatOps;

/// Clustering method
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterMethod {
    /// K-means clustering
    KMeans,
    /// Leiden clustering (graph-based)
    Leiden,
    /// Hierarchical Stochastic Block Model (graph-based)
    Hsblock,
}

/// Clustering result
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Cluster assignment for each cell (cell index -> cluster id)
    pub labels: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
}

impl ClusterResult {
    /// Get cluster size distribution
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut counts = vec![0; self.n_clusters];
        for &label in &self.labels {
            if label < self.n_clusters {
                counts[label] += 1;
            }
        }
        counts
    }

    /// Remove clusters smaller than `min_size`.
    ///
    /// Cells in removed clusters get label `usize::MAX` (written as NaN in output).
    /// Remaining clusters are renumbered contiguously.
    pub fn remove_small_clusters(&mut self, min_size: usize) {
        let sizes = self.cluster_sizes();

        // Build old→new label map; small clusters map to usize::MAX
        let mut new_label = vec![usize::MAX; self.n_clusters];
        let mut next = 0;
        for (old, &sz) in sizes.iter().enumerate() {
            if sz >= min_size {
                new_label[old] = next;
                next += 1;
            }
        }

        let n_removed = self.n_clusters - next;
        if n_removed > 0 {
            let n_cells_removed: usize = sizes.iter().filter(|&&s| s < min_size).sum();
            for label in self.labels.iter_mut() {
                *label = new_label[*label];
            }
            info!(
                "Removed {} cluster(s) with < {} cells ({} cells unassigned)",
                n_removed, min_size, n_cells_removed
            );
            self.n_clusters = next;
        }
    }

    /// Get cluster assignment histogram as ASCII, showing up to
    /// `max_show` largest clusters sorted by size (descending).
    pub fn histogram_ascii(&self, max_width: usize, max_show: usize) -> String {
        let sizes = self.cluster_sizes();

        // Sort non-empty clusters by size descending
        let mut ranked: Vec<(usize, usize)> = sizes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0)
            .map(|(id, &s)| (id, s))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));

        let n_total = ranked.len();
        let n_show = max_show.min(n_total);
        let max_size = ranked.first().map(|&(_, s)| s).unwrap_or(1);

        let mut lines = Vec::new();
        lines.push(format!(
            "Cluster assignments ({} cells, {} clusters):",
            self.labels.len(),
            n_total
        ));
        lines.push(String::new());

        for &(cluster_id, size) in ranked.iter().take(n_show) {
            let pct = 100.0 * size as f64 / self.labels.len() as f64;
            let bar_len = ((size as f64 / max_size as f64) * max_width as f64) as usize;
            let bar = "█".repeat(bar_len.max(1));

            lines.push(format!(
                "  Cluster {:3}  {:>6} cells ({:>5.1}%)  {}",
                cluster_id, size, pct, bar
            ));
        }

        if n_total > n_show {
            let hidden_cells: usize = ranked[n_show..].iter().map(|&(_, s)| s).sum();
            let hidden_pct = 100.0 * hidden_cells as f64 / self.labels.len() as f64;
            lines.push(format!(
                "  ... and {} more clusters ({} cells, {:.1}%)",
                n_total - n_show,
                hidden_cells,
                hidden_pct
            ));
        }

        lines.join("\n")
    }
}

/// Run k-means clustering on latent representation (cells × features)
pub fn kmeans_clustering(latent: &Mat, k: usize, max_iter: usize) -> anyhow::Result<ClusterResult> {
    if k == 0 {
        anyhow::bail!("Number of clusters must be > 0");
    }

    let n = latent.nrows();
    if k > n {
        anyhow::bail!(
            "Number of clusters ({}) exceeds number of samples ({})",
            k,
            n
        );
    }

    info!(
        "K-means: {} cells x {} features, k={}, max_iter={}",
        n,
        latent.ncols(),
        k,
        max_iter
    );

    let args = KmeansArgs {
        num_clusters: k,
        max_iter,
    };

    // Cluster rows (cells)
    let labels = latent.kmeans_rows(args);

    let result = ClusterResult {
        labels,
        n_clusters: k,
    };

    let sizes = result.cluster_sizes();
    let min_size = sizes.iter().copied().min().unwrap_or(0);
    let max_size = sizes.iter().copied().max().unwrap_or(0);
    info!(
        "K-means done: {} clusters, cluster sizes min={} max={}",
        k, min_size, max_size
    );

    Ok(result)
}

/// Run Leiden community detection on latent representation (cells × features)
///
/// * `target_clusters` - if Some, binary-search resolution to approximate this count
/// * `resolution` - starting (or fixed) resolution for CPM
pub fn leiden_clustering(
    latent: &Mat,
    knn: usize,
    resolution: f64,
    target_clusters: Option<usize>,
    seed: Option<u64>,
) -> anyhow::Result<ClusterResult> {
    let n = latent.nrows();
    let d = latent.ncols();
    if n < 2 {
        anyhow::bail!("Need at least 2 cells for Leiden clustering");
    }

    info!(
        "Leiden: {} cells x {} features, knn={}, seed={:?}",
        n, d, knn, seed
    );

    // Step 1: Column-wise z-score standardization then build KNN graph
    let mut latent_z = latent.clone();
    latent_z.scale_columns_inplace();

    info!("Building KNN graph (k={}) for {} cells ...", knn, n);
    let graph = KnnGraph::from_rows(
        &latent_z,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;

    let mean_degree = if graph.num_nodes() > 0 {
        2.0 * graph.num_edges() as f64 / graph.num_nodes() as f64
    } else {
        0.0
    };
    info!(
        "KNN graph: {} nodes, {} edges (mean degree {:.1})",
        graph.num_nodes(),
        graph.num_edges(),
        mean_degree
    );

    // Check connected components
    let n_components = count_components(&graph);
    info!("KNN graph has {} connected component(s)", n_components);

    // Step 2: Convert to Leiden Network with modularity objective
    let ln = graph.to_leiden_network();
    let resolution_scaled = ln.scale_resolution(resolution);
    info!(
        "Modularity resolution={:.4} → scaled={:.6e}, total_edge_weight={:.1}",
        resolution, resolution_scaled, ln.total_edge_weight
    );

    // Step 3: Run Leiden — with optional resolution tuning
    let seed_val = seed.map(|s| s as usize);

    let labels = if let Some(target_k) = target_clusters {
        info!(
            "Auto-tuning resolution to target ~{} clusters ...",
            target_k
        );
        knn_graph::tune_leiden_resolution(&ln.network, n, target_k, resolution_scaled, seed_val)
    } else {
        info!(
            "Running Leiden at scaled resolution={:.6e} ...",
            resolution_scaled
        );
        knn_graph::run_leiden(&ln.network, n, resolution_scaled, seed_val)
    };

    let mut compact = labels;
    knn_graph::compact_labels(&mut compact);
    let n_clusters = compact.iter().copied().max().unwrap_or(0) + 1;
    let result = ClusterResult {
        labels: compact,
        n_clusters,
    };

    let sizes = result.cluster_sizes();
    let min_size = sizes.iter().copied().min().unwrap_or(0);
    let max_size = sizes.iter().copied().max().unwrap_or(0);
    info!(
        "Leiden done: {} clusters, cluster sizes min={} max={} median={}",
        result.n_clusters,
        min_size,
        max_size,
        {
            let mut s = sizes.clone();
            s.sort();
            s[s.len() / 2]
        }
    );

    Ok(result)
}

/// Run HSBM community detection on latent representation (cells × features)
///
/// Uses the Hierarchical Stochastic Block Model as a drop-in alternative to Leiden.
/// Infers hierarchical community structure via collapsed Gibbs sampling with
/// Poisson rates analytically integrated out (Gamma-Poisson conjugacy).
///
/// * `tree_depth` - Depth of the binary tree (K = 2^(depth-1) leaf clusters). Default: 3
/// * `degree_corrected` - Whether to use degree-corrected model. Default: true
pub fn hsblock_clustering(
    latent: &Mat,
    knn: usize,
    tree_depth: usize,
    degree_corrected: bool,
    edge_scale: f64,
    seed: Option<u64>,
) -> anyhow::Result<ClusterResult> {
    let n = latent.nrows();
    let d = latent.ncols();
    if n < 2 {
        anyhow::bail!("Need at least 2 cells for HSBM clustering");
    }

    let k = 1 << (tree_depth - 1);
    info!(
        "HSBM: {} cells x {} features, knn={}, depth={} (K={}), dc={}, seed={:?}",
        n, d, knn, tree_depth, k, degree_corrected, seed
    );

    // Step 1: Column-wise z-score standardization then build KNN graph
    let mut latent_z = latent.clone();
    latent_z.scale_columns_inplace();

    info!("Building KNN graph (k={}) for {} cells ...", knn, n);
    let graph = KnnGraph::from_rows(
        &latent_z,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;

    let mean_degree = if graph.num_nodes() > 0 {
        2.0 * graph.num_edges() as f64 / graph.num_nodes() as f64
    } else {
        0.0
    };
    info!(
        "KNN graph: {} nodes, {} edges (mean degree {:.1})",
        graph.num_nodes(),
        graph.num_edges(),
        mean_degree
    );

    // Step 2: Convert to leiden::Network
    let ln = graph.to_leiden_network();
    let network = ln.network;

    // Step 3: Run HSBM
    let options = HsbmOptions {
        tree_depth,
        degree_corrected,
        edge_scale,
        seed: seed.unwrap_or(42),
        ..Default::default()
    };

    let mut hsblock = Hsblock::new(options);
    let mut clustering = SimpleClustering::init_different_clusters(n);

    let changed = hsblock.iterate(&network, &mut clustering);
    info!(
        "HSBM done: {} clusters, changed={}",
        clustering.num_clusters(),
        changed
    );

    let labels: Vec<usize> = (0..n).map(|i| clustering.get(i)).collect();
    let result = ClusterResult {
        n_clusters: clustering.num_clusters(),
        labels,
    };

    let sizes = result.cluster_sizes();
    let min_size = sizes.iter().copied().min().unwrap_or(0);
    let max_size = sizes.iter().copied().max().unwrap_or(0);
    info!(
        "HSBM done: {} clusters, cluster sizes min={} max={} median={}",
        result.n_clusters,
        min_size,
        max_size,
        {
            let mut s = sizes.clone();
            s.sort();
            s[s.len() / 2]
        }
    );

    Ok(result)
}

/// Count connected components in the KNN graph using BFS.
fn count_components(graph: &KnnGraph) -> usize {
    let n = graph.num_nodes();
    let mut visited = vec![false; n];
    let mut n_components = 0;

    for start in 0..n {
        if visited[start] {
            continue;
        }
        n_components += 1;
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            for &neighbor in graph.neighbors(node) {
                if !visited[neighbor] {
                    stack.push(neighbor);
                }
            }
        }
    }

    n_components
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_N_PER_CLUSTER: usize = 50;
    const TEST_N_GROUPS: usize = 3;
    const TEST_N: usize = TEST_N_GROUPS * TEST_N_PER_CLUSTER;

    /// 3 well-separated clusters of 50 points each in 3D
    fn three_cluster_latent() -> Mat {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = SmallRng::seed_from_u64(42);
        let noise = Normal::new(0.0f32, 0.05).unwrap();
        let dim = 3;

        // Centers are far apart relative to noise
        let centers: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]];

        let mut data = Mat::zeros(TEST_N, dim);
        for (c, center) in centers.iter().enumerate() {
            for i in 0..TEST_N_PER_CLUSTER {
                let row = c * TEST_N_PER_CLUSTER + i;
                for (d, &val) in center.iter().enumerate() {
                    data[(row, d)] = val + noise.sample(&mut rng);
                }
            }
        }
        data
    }

    const TEST_KNN: usize = 15;

    #[test]
    fn test_leiden_valid_output() {
        let latent = three_cluster_latent();
        let result = leiden_clustering(&latent, TEST_KNN, 1.0, None, Some(0)).unwrap();

        assert_eq!(result.labels.len(), TEST_N);
        assert!(result.n_clusters > 0);
        assert!(result.n_clusters <= TEST_N);

        // All labels should be in [0, n_clusters)
        for &label in &result.labels {
            assert!(label < result.n_clusters);
        }
    }

    #[test]
    fn test_leiden_no_cross_cluster_labels() {
        let latent = three_cluster_latent();
        let result = leiden_clustering(&latent, TEST_KNN, 1.0, None, Some(0)).unwrap();

        // Each ground-truth group should NOT share labels with other groups.
        // Collect the set of labels for each ground-truth group.
        let mut group_labels: Vec<std::collections::HashSet<usize>> = Vec::new();
        for c in 0..TEST_N_GROUPS {
            let start = c * TEST_N_PER_CLUSTER;
            let end = start + TEST_N_PER_CLUSTER;
            let labels: std::collections::HashSet<usize> =
                result.labels[start..end].iter().copied().collect();
            group_labels.push(labels);
        }

        // No label should appear in two different ground-truth groups
        for i in 0..TEST_N_GROUPS {
            for j in (i + 1)..TEST_N_GROUPS {
                let overlap: Vec<_> = group_labels[i].intersection(&group_labels[j]).collect();
                assert!(
                    overlap.is_empty(),
                    "Ground-truth groups {} and {} share labels: {:?}",
                    i,
                    j,
                    overlap
                );
            }
        }
    }

    #[test]
    fn test_leiden_reasonable_output_with_seed() {
        let latent = three_cluster_latent();
        let r1 = leiden_clustering(&latent, TEST_KNN, 1.0, None, Some(123)).unwrap();

        // HNSW parallel_insert is non-deterministic, so exact label
        // reproducibility is not guaranteed. Check structure instead.
        assert_eq!(r1.labels.len(), TEST_N);
        assert!(
            r1.n_clusters >= TEST_N_GROUPS,
            "Should find at least {} clusters",
            TEST_N_GROUPS
        );
        assert!(
            r1.n_clusters <= TEST_N / 2,
            "Should not be overly fragmented"
        );
    }

    #[test]
    fn test_leiden_resolution_effect() {
        let latent = three_cluster_latent();

        // Very low resolution → fewer clusters (tends toward 1 big cluster)
        let low_res = leiden_clustering(&latent, TEST_KNN, 0.1, None, Some(0)).unwrap();
        // Very high resolution → more clusters (tends toward singletons)
        let high_res = leiden_clustering(&latent, TEST_KNN, 5.0, None, Some(0)).unwrap();

        assert!(
            low_res.n_clusters <= high_res.n_clusters,
            "Lower resolution ({} clusters) should produce <= clusters than higher ({} clusters)",
            low_res.n_clusters,
            high_res.n_clusters
        );
    }

    #[test]
    fn test_leiden_too_few_cells() {
        let latent = Mat::from_row_slice(1, 2, &[1.0, 2.0]);
        let result = leiden_clustering(&latent, 5, 1.0, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_leiden_cluster_sizes_sum() {
        let latent = three_cluster_latent();
        let result = leiden_clustering(&latent, TEST_KNN, 1.0, None, Some(0)).unwrap();

        let sizes = result.cluster_sizes();
        let total: usize = sizes.iter().sum();
        assert_eq!(total, TEST_N, "Cluster sizes should sum to total cells");
    }

    #[test]
    fn test_leiden_resolution_tuning() {
        let latent = three_cluster_latent();
        let target_k = TEST_N_GROUPS; // 3

        let result = leiden_clustering(&latent, TEST_KNN, 0.1, Some(target_k), Some(42)).unwrap();

        assert_eq!(
            result.n_clusters, target_k,
            "Resolution tuning should find exactly {} clusters, got {}",
            target_k, result.n_clusters
        );
        assert_eq!(result.labels.len(), TEST_N);

        // All labels in [0, n_clusters)
        for &label in &result.labels {
            assert!(label < result.n_clusters);
        }

        // Cluster sizes should sum to total
        let total: usize = result.cluster_sizes().iter().sum();
        assert_eq!(total, TEST_N);
    }

    #[test]
    fn test_leiden_resolution_tuning_higher_k() {
        let latent = three_cluster_latent();
        let target_k = 6;

        let result = leiden_clustering(&latent, TEST_KNN, 0.5, Some(target_k), Some(42)).unwrap();

        // May not hit exactly 6, but should be close
        let diff = (result.n_clusters as isize - target_k as isize).unsigned_abs();
        assert!(
            diff <= 2,
            "Resolution tuning for k={} produced {} clusters (too far off)",
            target_k,
            result.n_clusters
        );
    }

    #[test]
    fn test_hsblock_valid_output() {
        let latent = three_cluster_latent();
        let result = hsblock_clustering(&latent, TEST_KNN, 3, false, 100.0, Some(42)).unwrap();

        assert_eq!(result.labels.len(), TEST_N);
        assert!(result.n_clusters > 0);
        assert!(result.n_clusters <= TEST_N);

        // All labels should be in [0, n_clusters)
        for &label in &result.labels {
            assert!(label < result.n_clusters);
        }

        // Cluster sizes should sum to total
        let total: usize = result.cluster_sizes().iter().sum();
        assert_eq!(total, TEST_N, "Cluster sizes should sum to total cells");
    }

    #[test]
    fn test_hsblock_degree_corrected() {
        let latent = three_cluster_latent();
        let result = hsblock_clustering(&latent, TEST_KNN, 2, true, 100.0, Some(99)).unwrap();

        assert_eq!(result.labels.len(), TEST_N);
        assert!(result.n_clusters > 0);

        for &label in &result.labels {
            assert!(label < result.n_clusters);
        }
    }

    #[test]
    fn test_hsblock_too_few_cells() {
        let latent = Mat::from_row_slice(1, 2, &[1.0, 2.0]);
        let result = hsblock_clustering(&latent, 5, 2, false, 100.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cluster_result_histogram() {
        let result = ClusterResult {
            labels: vec![0, 0, 0, 1, 1, 2],
            n_clusters: 3,
        };
        let hist = result.histogram_ascii(20, 100);
        assert!(hist.contains("6 cells"));
        assert!(hist.contains("3 clusters"));
    }
}
