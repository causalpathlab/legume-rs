//! Trait-generic module clustering for link community edge profiles.
//!
//! Provides a [`ModuleClusterer`] trait with K-means and Leiden implementations.
//! Used both for initial gene/gene-pair module discovery and for iterative
//! re-estimation in the outer EM loop.

use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::knn_graph;

type Mat = nalgebra::DMatrix<f32>;

// Re-export shared utility so callers don't need matrix_util directly.
pub use knn_graph::compact_labels;

/// Strategy for clustering items (gene pairs) into modules.
///
/// Input: `item_features` is an [items × features] matrix. Output:
/// assignment vector mapping each item to a module index.
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
