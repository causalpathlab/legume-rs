//! Clustering utilities and methods for single-cell data
//!
//! Supports multiple clustering algorithms on latent representations
//! (topic proportions, SVD embeddings, etc.)

use crate::embed_common::*;
use matrix_util::clustering::{Kmeans, KmeansArgs};

/// Clustering method
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterMethod {
    /// K-means clustering
    KMeans,
    /// Leiden clustering (graph-based)
    Leiden,
    /// Louvain clustering (graph-based)
    Louvain,
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
