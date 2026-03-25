//! Trait-generic module clustering for link community edge profiles.
//!
//! Provides a [`ModuleClusterer`] trait with K-means and Leiden implementations.
//! Used both for initial gene/gene-pair module discovery and for iterative
//! re-estimation in the outer EM loop.

use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::knn_graph::{self, KnnGraph, KnnGraphArgs};
use matrix_util::traits::MatOps;

type Mat = nalgebra::DMatrix<f32>;

// Re-export shared utility so callers don't need matrix_util directly.
pub use knn_graph::compact_labels;

/// Strategy for clustering items (genes or gene-pairs) into modules.
///
/// Input: `item_features` is an [items × features] matrix (e.g., gene
/// embeddings or gene-community rate vectors). Output: assignment vector
/// mapping each item to a module index.
pub trait ModuleClusterer: Send + Sync {
    fn cluster(&self, item_features: &Mat) -> Vec<usize>;
}

/// K-means module clustering.
pub struct KmeansModules {
    pub n_modules: usize,
    pub max_iter: usize,
}

impl ModuleClusterer for KmeansModules {
    fn cluster(&self, item_features: &Mat) -> Vec<usize> {
        item_features.kmeans_rows(KmeansArgs {
            num_clusters: self.n_modules,
            max_iter: self.max_iter,
        })
    }
}

/// Leiden community detection on a KNN graph of items.
///
/// Builds a KNN graph on the rows of the feature matrix, computes fuzzy
/// kernel weights, then runs Leiden with modularity objective. If
/// `target_modules` is set, binary-searches on resolution to approximate
/// the target number of clusters.
pub struct LeidenModules {
    pub knn: usize,
    pub resolution: f64,
    pub target_modules: Option<usize>,
    pub seed: Option<u64>,
}

impl ModuleClusterer for LeidenModules {
    fn cluster(&self, item_features: &Mat) -> Vec<usize> {
        let n = item_features.nrows();
        if n < 2 {
            return vec![0; n];
        }

        // Z-score standardize columns
        let mut z = item_features.clone();
        z.scale_columns_inplace();

        // Build KNN graph on rows
        let knn = self.knn.min(n - 1).max(1);
        let graph = KnnGraph::from_rows(
            &z,
            KnnGraphArgs {
                knn,
                block_size: 1000,
                reciprocal: false,
            },
        )
        .expect("KNN graph construction failed");

        // Convert to Leiden network and run
        let ln = graph.to_leiden_network();
        let resolution_scaled = ln.scale_resolution(self.resolution);
        let seed_val = self.seed.map(|s| s as usize);

        if let Some(target_k) = self.target_modules {
            knn_graph::tune_leiden_resolution(&ln.network, n, target_k, resolution_scaled, seed_val)
        } else {
            knn_graph::run_leiden(&ln.network, n, resolution_scaled, seed_val)
        }
    }
}
