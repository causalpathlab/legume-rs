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
use matrix_util::traits::MatOps;

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
    //
    // Modularity quality increment: Δ = w_jl - γ · k_j · K_l / (2m)
    // The Leiden crate uses CPM form: Δ = w_jl - node_w · cluster_w · res
    // Setting node weights = degree and res = γ/(2m) gives modularity.
    let weights = graph.fuzzy_kernel_weights();
    info!("Converting KNN graph to Leiden network (modularity, fuzzy kernel weights) ...");

    // Compute node degrees (sum of edge weights per node)
    let mut node_degree = vec![0.0f32; n];
    let mut total_edge_weight = 0.0f64;
    let mut min_w = f32::MAX;
    let mut max_w = f32::MIN;
    for (&(i, j), &w) in graph.edges.iter().zip(weights.iter()) {
        min_w = min_w.min(w);
        max_w = max_w.max(w);
        node_degree[i] += w;
        node_degree[j] += w;
        total_edge_weight += w as f64;
    }

    let mut leiden_graph = Graph::with_capacity(n, graph.num_edges());
    for i in 0..n {
        leiden_graph.add_node(node_degree[i]);
    }
    for (&(i, j), &w) in graph.edges.iter().zip(weights.iter()) {
        leiden_graph.add_edge((i as u32).into(), (j as u32).into(), w);
    }
    let network = Network::new_from_graph(leiden_graph);

    // Scale resolution: modularity γ → CPM resolution = γ / (2m)
    let resolution_scaled = resolution / (2.0 * total_edge_weight);
    info!(
        "Edge weights: min={:.4}, max={:.4}, total={:.1}",
        min_w, max_w, total_edge_weight
    );
    info!(
        "Modularity resolution={:.4} → scaled={:.6e}",
        resolution, resolution_scaled
    );

    // Step 3: Run Leiden — with optional resolution tuning
    let seed_val = seed.map(|s| s as usize);

    let result = if let Some(target_k) = target_clusters {
        info!(
            "Auto-tuning resolution to target ~{} clusters ...",
            target_k
        );
        tune_leiden_resolution(&network, n, target_k, resolution_scaled, seed_val)?
    } else {
        info!("Running Leiden at scaled resolution={:.6e} ...", resolution_scaled);
        run_leiden(&network, n, resolution_scaled, seed_val)
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

/// Run Leiden at a fixed resolution and return the result.
fn run_leiden(
    network: &Network,
    n: usize,
    resolution: f64,
    seed: Option<usize>,
) -> ClusterResult {
    let mut leiden = Leiden::new(resolution, 0.01, seed);
    let mut clustering = SimpleClustering::init_different_clusters(n);

    let max_outer = 10;
    for iter in 0..max_outer {
        let updated = leiden.iterate(network, &mut clustering);
        info!(
            "  iteration {}: {} clusters{}",
            iter + 1,
            clustering.num_clusters(),
            if !updated { " (converged)" } else { "" }
        );
        if !updated {
            break;
        }
    }

    let labels: Vec<usize> = (0..n).map(|i| clustering.get(i)).collect();
    ClusterResult {
        labels,
        n_clusters: clustering.num_clusters(),
    }
}

/// Binary search on resolution to get close to `target_k` clusters.
///
/// Lower resolution → fewer clusters; higher → more clusters.
fn tune_leiden_resolution(
    network: &Network,
    n: usize,
    target_k: usize,
    initial_resolution: f64,
    seed: Option<usize>,
) -> anyhow::Result<ClusterResult> {
    let mut lo = 1e-6_f64;
    let mut hi = 10.0_f64;
    let mut best = run_leiden(network, n, initial_resolution, seed);
    let mut best_res = initial_resolution;

    info!(
        "  resolution={:.6} → {} clusters (target {})",
        initial_resolution, best.n_clusters, target_k
    );

    if best.n_clusters == target_k {
        return Ok(best);
    }

    // Adjust initial bounds based on first result
    if best.n_clusters > target_k {
        hi = initial_resolution;
    } else {
        lo = initial_resolution;
    }

    const MAX_SEARCH: usize = 20;

    for step in 0..MAX_SEARCH {
        let mid = (lo + hi) / 2.0;
        let result = run_leiden(network, n, mid, seed);

        info!(
            "  step {}: resolution={:.6} → {} clusters",
            step + 1,
            mid,
            result.n_clusters
        );

        // Binary search direction based on result at mid
        // (too many clusters → lower resolution, too few → raise it)
        if result.n_clusters > target_k {
            hi = mid;
        } else {
            lo = mid;
        }

        // Update best if closer to target
        let cur_diff = (result.n_clusters as isize - target_k as isize).unsigned_abs();
        let best_diff = (best.n_clusters as isize - target_k as isize).unsigned_abs();
        if cur_diff < best_diff {
            best = result;
            best_res = mid;
        }

        if best.n_clusters == target_k || (hi - lo) / hi.max(1e-10) < 1e-4 {
            break;
        }
    }

    info!(
        "  best resolution={:.6} → {} clusters (target {})",
        best_res, best.n_clusters, target_k
    );

    Ok(best)
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
    fn test_leiden_reasonable_output_with_seed() {
        let latent = three_cluster_latent();
        let r1 = leiden_clustering(&latent, TEST_KNN, 1.0, None, Some(123)).unwrap();

        // HNSW parallel_insert is non-deterministic, so exact label
        // reproducibility is not guaranteed. Check structure instead.
        assert_eq!(r1.labels.len(), TEST_N);
        assert!(r1.n_clusters >= TEST_N_GROUPS, "Should find at least {} clusters", TEST_N_GROUPS);
        assert!(r1.n_clusters <= TEST_N / 2, "Should not be overly fragmented");
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

        let result =
            leiden_clustering(&latent, TEST_KNN, 0.1, Some(target_k), Some(42)).unwrap();

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

        let result =
            leiden_clustering(&latent, TEST_KNN, 0.5, Some(target_k), Some(42)).unwrap();

        // May not hit exactly 6, but should be close
        let diff = (result.n_clusters as isize - target_k as isize).unsigned_abs();
        assert!(
            diff <= 2,
            "Resolution tuning for k={} produced {} clusters (too far off)",
            target_k, result.n_clusters
        );
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
