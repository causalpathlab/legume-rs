//! Clustering utilities and methods for single-cell data
//!
//! Supports multiple clustering algorithms on latent representations
//! (topic proportions, SVD embeddings, etc.)

use crate::embed_common::*;
use leiden::clustering::SimpleClustering;
use leiden::leiden::Leiden;
use leiden::network::Graph;
use leiden::{Clustering, Network};
use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};

/// Clustering method
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterMethod {
    /// K-means clustering
    KMeans,
    /// Leiden clustering (graph-based)
    Leiden,
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

    /// Get cluster assignment histogram as ASCII
    pub fn histogram_ascii(&self, max_width: usize) -> String {
        let sizes = self.cluster_sizes();
        let max_size = *sizes.iter().max().unwrap_or(&1);

        let mut lines = Vec::new();
        lines.push(format!(
            "Cluster assignments ({} cells, {} clusters):",
            self.labels.len(),
            self.n_clusters
        ));
        lines.push(String::new());

        for (cluster_id, &size) in sizes.iter().enumerate() {
            if size == 0 {
                continue;
            }
            let pct = 100.0 * size as f64 / self.labels.len() as f64;
            let bar_len = ((size as f64 / max_size as f64) * max_width as f64) as usize;
            let bar = "█".repeat(bar_len.max(1));

            lines.push(format!(
                "  Cluster {:3}  {:>6} cells ({:>5.1}%)  {}",
                cluster_id, size, pct, bar
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

    let args = KmeansArgs {
        num_clusters: k,
        max_iter,
    };

    // Cluster rows (cells)
    let labels = latent.kmeans_rows(args);

    Ok(ClusterResult {
        labels,
        n_clusters: k,
    })
}

/// Run Leiden community detection on latent representation (cells × features)
///
/// 1. Build a KNN graph from the latent space
/// 2. Convert KNN edges into a Leiden `Network`
/// 3. Iterate the Leiden algorithm until convergence
/// 4. Extract cluster labels
pub fn leiden_clustering(
    latent: &Mat,
    knn: usize,
    resolution: f64,
    seed: Option<u64>,
) -> anyhow::Result<ClusterResult> {
    let n = latent.nrows();
    if n < 2 {
        anyhow::bail!("Need at least 2 cells for Leiden clustering");
    }

    // Step 1: Build KNN graph
    info!("Building KNN graph (k={}) for {} cells", knn, n);
    let graph = KnnGraph::from_rows(
        latent,
        KnnGraphArgs {
            knn,
            block_size: 1000,
        },
    )?;
    info!(
        "KNN graph: {} nodes, {} edges",
        graph.num_nodes(),
        graph.num_edges()
    );

    // Step 2: Convert to Leiden Network
    //
    // Convert distances to similarity weights: w = 1 / (1 + d)
    // Each node gets weight 1.0 (uniform node weights for CPM objective)
    let mut leiden_graph = Graph::with_capacity(n, graph.num_edges());
    for _ in 0..n {
        leiden_graph.add_node(1.0f32);
    }
    for (&(i, j), &dist) in graph.edges.iter().zip(graph.distances.iter()) {
        let weight = 1.0 / (1.0 + dist);
        leiden_graph.add_edge((i as u32).into(), (j as u32).into(), weight);
    }
    let network = Network::new_from_graph(leiden_graph);

    // Step 3: Run Leiden iterations
    let seed_val = seed.map(|s| s as usize);
    let mut leiden = Leiden::new(resolution, 0.01, seed_val);
    let mut clustering = SimpleClustering::init_different_clusters(n);

    let max_outer = 10;
    for iter in 0..max_outer {
        let updated = leiden.iterate(&network, &mut clustering);
        info!(
            "Leiden iteration {}: {} clusters",
            iter + 1,
            clustering.num_clusters()
        );
        if !updated {
            break;
        }
    }

    // Step 4: Extract labels
    let labels: Vec<usize> = (0..n).map(|i| clustering.get(i)).collect();
    let n_clusters = clustering.num_clusters();

    Ok(ClusterResult { labels, n_clusters })
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
        let result = leiden_clustering(&latent, TEST_KNN, 0.05, Some(0)).unwrap();

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
        let result = leiden_clustering(&latent, TEST_KNN, 0.05, Some(0)).unwrap();

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
                let overlap: Vec<_> = group_labels[i]
                    .intersection(&group_labels[j])
                    .collect();
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
    fn test_leiden_deterministic_with_seed() {
        let latent = three_cluster_latent();
        let r1 = leiden_clustering(&latent, TEST_KNN, 0.05, Some(123)).unwrap();
        let r2 = leiden_clustering(&latent, TEST_KNN, 0.05, Some(123)).unwrap();

        assert_eq!(r1.labels, r2.labels);
        assert_eq!(r1.n_clusters, r2.n_clusters);
    }

    #[test]
    fn test_leiden_resolution_effect() {
        let latent = three_cluster_latent();

        // Very low resolution → fewer clusters (tends toward 1 big cluster)
        let low_res = leiden_clustering(&latent, TEST_KNN, 0.001, Some(0)).unwrap();
        // Very high resolution → more clusters (tends toward singletons)
        let high_res = leiden_clustering(&latent, TEST_KNN, 10.0, Some(0)).unwrap();

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
        let result = leiden_clustering(&latent, 5, 1.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_leiden_cluster_sizes_sum() {
        let latent = three_cluster_latent();
        let result = leiden_clustering(&latent, TEST_KNN, 0.05, Some(0)).unwrap();

        let sizes = result.cluster_sizes();
        let total: usize = sizes.iter().sum();
        assert_eq!(total, TEST_N, "Cluster sizes should sum to total cells");
    }

    #[test]
    fn test_cluster_result_histogram() {
        let result = ClusterResult {
            labels: vec![0, 0, 0, 1, 1, 2],
            n_clusters: 3,
        };
        let hist = result.histogram_ascii(20);
        assert!(hist.contains("6 cells"));
        assert!(hist.contains("3 clusters"));
    }
}
