//! K-means clustering traits for matrices
//!
//! Provides traits for clustering rows or columns of matrices using the
//! `clustering` crate.

use nalgebra::DMatrix;

/// Arguments for k-means clustering
#[derive(Debug, Clone)]
pub struct KmeansArgs {
    /// Number of clusters
    pub num_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
}

impl Default for KmeansArgs {
    fn default() -> Self {
        Self {
            num_clusters: 1,
            max_iter: 100,
        }
    }
}

impl KmeansArgs {
    /// Create args with specified number of clusters
    pub fn with_clusters(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            ..Default::default()
        }
    }
}

/// Trait for k-means clustering on matrices
pub trait Kmeans {
    /// Cluster columns and return membership vector
    ///
    /// # Arguments
    /// * `args` - Clustering parameters
    ///
    /// # Returns
    /// Vector of cluster assignments, one per column
    fn kmeans_columns(&self, args: KmeansArgs) -> Vec<usize>;

    /// Cluster rows and return membership vector
    ///
    /// # Arguments
    /// * `args` - Clustering parameters
    ///
    /// # Returns
    /// Vector of cluster assignments, one per row
    fn kmeans_rows(&self, args: KmeansArgs) -> Vec<usize>;
}

impl<T> Kmeans for DMatrix<T>
where
    T: Clone + Sync + Send,
    Vec<T>: clustering::Elem,
{
    fn kmeans_columns(&self, args: KmeansArgs) -> Vec<usize> {
        if args.num_clusters <= 1 || self.ncols() == 0 {
            return vec![0; self.ncols()];
        }

        let data: Vec<Vec<T>> = self
            .column_iter()
            .map(|x| x.iter().cloned().collect())
            .collect();

        let clust = clustering::kmeans(args.num_clusters, &data, args.max_iter);
        clust.membership
    }

    fn kmeans_rows(&self, args: KmeansArgs) -> Vec<usize> {
        if args.num_clusters <= 1 || self.nrows() == 0 {
            return vec![0; self.nrows()];
        }

        let data: Vec<Vec<T>> = self
            .row_iter()
            .map(|x| x.iter().cloned().collect())
            .collect();

        let clust = clustering::kmeans(args.num_clusters, &data, args.max_iter);
        clust.membership
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_columns_single_cluster() {
        let mat = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let args = KmeansArgs::with_clusters(1);
        let membership = mat.kmeans_columns(args);

        assert_eq!(membership.len(), 4);
        assert!(membership.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_kmeans_columns_two_clusters() {
        // Create data with 2 clear clusters
        let mat = DMatrix::from_row_slice(
            2,
            6,
            &[
                0.0, 0.1, 0.2, 10.0, 10.1, 10.2, // row 0
                0.0, 0.1, 0.0, 10.0, 10.1, 10.2, // row 1
            ],
        );

        let args = KmeansArgs::with_clusters(2);
        let membership = mat.kmeans_columns(args);

        assert_eq!(membership.len(), 6);

        // First 3 columns should be in same cluster
        assert_eq!(membership[0], membership[1]);
        assert_eq!(membership[1], membership[2]);

        // Last 3 columns should be in same cluster
        assert_eq!(membership[3], membership[4]);
        assert_eq!(membership[4], membership[5]);

        // Two groups should be different
        assert_ne!(membership[0], membership[3]);
    }

    #[test]
    fn test_kmeans_rows() {
        // Create data with 2 clear row clusters
        let mat = DMatrix::from_row_slice(
            4,
            2,
            &[
                0.0, 0.0, // row 0 - cluster A
                0.1, 0.1, // row 1 - cluster A
                10.0, 10.0, // row 2 - cluster B
                10.1, 10.1, // row 3 - cluster B
            ],
        );

        let args = KmeansArgs::with_clusters(2);
        let membership = mat.kmeans_rows(args);

        assert_eq!(membership.len(), 4);

        // First 2 rows should be in same cluster
        assert_eq!(membership[0], membership[1]);

        // Last 2 rows should be in same cluster
        assert_eq!(membership[2], membership[3]);

        // Two groups should be different
        assert_ne!(membership[0], membership[2]);
    }

    #[test]
    fn test_kmeans_empty_matrix() {
        let mat: DMatrix<f32> = DMatrix::zeros(0, 0);

        let col_membership = mat.kmeans_columns(KmeansArgs::with_clusters(2));
        let row_membership = mat.kmeans_rows(KmeansArgs::with_clusters(2));

        assert!(col_membership.is_empty());
        assert!(row_membership.is_empty());
    }
}
