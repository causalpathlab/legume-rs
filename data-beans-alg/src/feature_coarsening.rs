use crate::random_projection::binary_sort_columns;
use log::info;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

type CscMat = nalgebra_sparse::CscMatrix<f32>;

/// Maps D fine features to d coarse meta-features and back.
#[derive(Clone, Serialize, Deserialize)]
pub struct FeatureCoarsening {
    /// For each original feature, the coarse group index it belongs to.
    pub fine_to_coarse: Vec<usize>,
    /// For each coarse group, the list of original feature indices.
    pub coarse_to_fine: Vec<Vec<usize>>,
    /// Number of coarse meta-features (d).
    pub num_coarse: usize,
}

impl FeatureCoarsening {
    /// Aggregate columns of an [N, D] matrix → [N, d] by summing
    /// features within each coarse group.
    pub fn aggregate_columns_nd(&self, data_nd: &DMatrix<f32>) -> DMatrix<f32> {
        let n = data_nd.nrows();
        let mut out = DMatrix::<f32>::zeros(n, self.num_coarse);
        // Group fine columns by coarse target to keep output column hot in cache
        for (c, fine_indices) in self.coarse_to_fine.iter().enumerate() {
            let mut col = out.column_mut(c);
            for &fine in fine_indices {
                col += data_nd.column(fine);
            }
        }
        out
    }

    /// Aggregate rows of a [D, S] matrix → [d, S] by summing
    /// features within each coarse group.
    pub fn aggregate_rows_ds(&self, data_ds: &DMatrix<f32>) -> DMatrix<f32> {
        let s = data_ds.ncols();
        let mut out = DMatrix::<f32>::zeros(self.num_coarse, s);
        // Iterate columns (contiguous in column-major) in inner loop
        for j in 0..s {
            let src_col = data_ds.column(j);
            let mut dst_col = out.column_mut(j);
            for (fine, &coarse) in self.fine_to_coarse.iter().enumerate() {
                dst_col[coarse] += src_col[fine];
            }
        }
        out
    }

    /// Expand log-probability dictionary [d, K] → [D, K].
    ///
    /// For fine feature `f` in group `c` (size `g`):
    ///   `expanded[f, k] = coarse[c, k] - ln(g)`
    ///
    /// After exponentiation, probabilities split evenly within each group:
    ///   `β[f, k] = β_coarse[c, k] / g`
    pub fn expand_log_dict_dk(&self, log_dict_dk: &DMatrix<f32>, d_fine: usize) -> DMatrix<f32> {
        let k = log_dict_dk.ncols();
        let mut expanded = DMatrix::<f32>::zeros(d_fine, k);
        // Iterate columns (contiguous in column-major) in outer loop
        for kk in 0..k {
            let src_col = log_dict_dk.column(kk);
            let mut dst_col = expanded.column_mut(kk);
            for (c, fine_indices) in self.coarse_to_fine.iter().enumerate() {
                let val = src_col[c] - (fine_indices.len() as f32).ln();
                for &f in fine_indices {
                    dst_col[f] = val;
                }
            }
        }
        expanded
    }

    /// Aggregate a sparse [D, n] CSC matrix → dense [d, n] by summing
    /// rows within each coarse group. Efficient: O(nnz) work.
    pub fn aggregate_sparse_csc(&self, data_dn: &CscMat) -> DMatrix<f32> {
        let n = data_dn.ncols();
        let mut out = DMatrix::<f32>::zeros(self.num_coarse, n);
        for j in 0..n {
            let col = data_dn.col(j);
            for (&row, &val) in col.row_indices().iter().zip(col.values().iter()) {
                out[(self.fine_to_coarse[row], j)] += val;
            }
        }
        out
    }
}

/// Build a feature coarsening from a data-dependent sketch.
///
/// Uses the collapsed pseudobulk data [D, S] to group co-expressed
/// features via binary hashing (SVD + binarization).
///
/// # Arguments
/// * `data_ds` - feature sketch matrix [D, S] (e.g. posterior mean of collapsed data)
/// * `max_features` - target maximum number of coarse features
pub fn compute_feature_coarsening(
    data_ds: &DMatrix<f32>,
    max_features: usize,
) -> anyhow::Result<FeatureCoarsening> {
    let d = data_ds.nrows();
    let s = data_ds.ncols();

    // sort_dim such that 2^sort_dim ≈ max_features
    let sort_dim = (max_features as f64).log2().ceil() as usize;
    let sort_dim = sort_dim.min(s); // can't use more dimensions than samples

    info!(
        "Feature coarsening: {} features, {} samples, sort_dim={}",
        d, s, sort_dim
    );

    // binary_sort_columns expects (feature × items): K × N
    // We want to sort D features using S-dimensional profiles.
    // Pass data_ds transposed: S × D (S features describing D items)
    let data_sd = data_ds.transpose();
    let codes = binary_sort_columns(&data_sd, sort_dim)?;

    // Group features by binary code
    let max_code = codes.iter().max().copied().unwrap_or(0);
    let mut coarse_to_fine: Vec<Vec<usize>> = vec![Vec::new(); max_code + 1];
    for (f, &code) in codes.iter().enumerate() {
        coarse_to_fine[code].push(f);
    }

    // Remove empty groups and reindex
    coarse_to_fine.retain(|v| !v.is_empty());
    let num_coarse = coarse_to_fine.len();

    let mut fine_to_coarse = vec![0usize; d];
    for (c, fine_indices) in coarse_to_fine.iter().enumerate() {
        for &f in fine_indices {
            fine_to_coarse[f] = c;
        }
    }

    info!(
        "Feature coarsening: {} → {} meta-features (target {})",
        d, num_coarse, max_features
    );

    Ok(FeatureCoarsening {
        fine_to_coarse,
        coarse_to_fine,
        num_coarse,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_aggregate_rows_sums_match() {
        // 6 features, 3 samples
        let data = DMatrix::from_row_slice(
            6,
            3,
            &[
                1.0, 2.0, 3.0, // feature 0
                4.0, 5.0, 6.0, // feature 1
                7.0, 8.0, 9.0, // feature 2
                10.0, 11.0, 12.0, // feature 3
                13.0, 14.0, 15.0, // feature 4
                16.0, 17.0, 18.0, // feature 5
            ],
        );

        let fc = FeatureCoarsening {
            fine_to_coarse: vec![0, 0, 1, 1, 2, 2],
            coarse_to_fine: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            num_coarse: 3,
        };

        let agg = fc.aggregate_rows_ds(&data);
        assert_eq!(agg.nrows(), 3);
        assert_eq!(agg.ncols(), 3);

        // Group 0: features 0+1
        assert_relative_eq!(agg[(0, 0)], 5.0);
        assert_relative_eq!(agg[(0, 1)], 7.0);
        assert_relative_eq!(agg[(0, 2)], 9.0);

        // Group 1: features 2+3
        assert_relative_eq!(agg[(1, 0)], 17.0);

        // Group 2: features 4+5
        assert_relative_eq!(agg[(2, 0)], 29.0);

        // Column sums should be preserved
        let orig_col_sum: f32 = data.column(0).iter().sum();
        let agg_col_sum: f32 = agg.column(0).iter().sum();
        assert_relative_eq!(orig_col_sum, agg_col_sum);
    }

    #[test]
    fn test_aggregate_columns_nd() {
        // 2 samples, 4 features → 2 groups
        let data = DMatrix::from_row_slice(
            2,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, // sample 0
                5.0, 6.0, 7.0, 8.0, // sample 1
            ],
        );

        let fc = FeatureCoarsening {
            fine_to_coarse: vec![0, 0, 1, 1],
            coarse_to_fine: vec![vec![0, 1], vec![2, 3]],
            num_coarse: 2,
        };

        let agg = fc.aggregate_columns_nd(&data);
        assert_eq!(agg.nrows(), 2);
        assert_eq!(agg.ncols(), 2);
        assert_relative_eq!(agg[(0, 0)], 3.0); // 1+2
        assert_relative_eq!(agg[(0, 1)], 7.0); // 3+4
        assert_relative_eq!(agg[(1, 0)], 11.0); // 5+6
        assert_relative_eq!(agg[(1, 1)], 15.0); // 7+8
    }

    #[test]
    fn test_expand_logits_preserves_probabilities() {
        // 2 topics, 3 coarse features → expand to 6 fine features
        // Groups: {0,1}, {2,3}, {4,5}
        let logits = DMatrix::from_row_slice(
            3,
            2,
            &[
                -1.2, -0.8, // coarse 0
                -0.5, -1.5, // coarse 1
                -1.0, -1.0, // coarse 2
            ],
        );

        let fc = FeatureCoarsening {
            fine_to_coarse: vec![0, 0, 1, 1, 2, 2],
            coarse_to_fine: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            num_coarse: 3,
        };

        let expanded = fc.expand_log_dict_dk(&logits, 6);
        assert_eq!(expanded.nrows(), 6);
        assert_eq!(expanded.ncols(), 2);

        let ln2 = 2.0f32.ln();

        // For each topic, sum of exp(expanded) within each group
        // should equal exp(coarse logit)
        for k in 0..2 {
            for (c, group) in fc.coarse_to_fine.iter().enumerate() {
                let coarse_prob: f32 = logits[(c, k)].exp();
                let fine_sum: f32 = group.iter().map(|&f| expanded[(f, k)].exp()).sum();
                assert_relative_eq!(fine_sum, coarse_prob, epsilon = 1e-6);
            }
        }

        // Each fine feature in a group of size 2 gets logit - ln(2)
        assert_relative_eq!(expanded[(0, 0)], -1.2 - ln2, epsilon = 1e-6);
        assert_relative_eq!(expanded[(1, 0)], -1.2 - ln2, epsilon = 1e-6);
    }

    #[test]
    fn test_compute_feature_coarsening() {
        use matrix_util::traits::SampleOps;

        // Create a D×S matrix with D=500 features, S=50 samples
        let d = 500;
        let s = 50;
        let data = DMatrix::<f32>::rnorm(d, s);

        let fc = compute_feature_coarsening(&data, 50).unwrap();

        // All features should be assigned
        assert_eq!(fc.fine_to_coarse.len(), d);

        // Coarse features should be reasonable
        assert!(fc.num_coarse > 0);
        assert!(fc.num_coarse <= 64); // 2^6 = 64 max for sort_dim=6

        // Every fine feature should appear in exactly one coarse group
        let mut counts = vec![0usize; d];
        for group in &fc.coarse_to_fine {
            for &f in group {
                counts[f] += 1;
            }
        }
        assert!(counts.iter().all(|&c| c == 1));

        // fine_to_coarse should be consistent with coarse_to_fine
        for (c, group) in fc.coarse_to_fine.iter().enumerate() {
            for &f in group {
                assert_eq!(fc.fine_to_coarse[f], c);
            }
        }
    }

    #[test]
    fn test_integration_coarsen_expand_roundtrip() {
        // Simulated data: D=500, N=300, K=5, max_features=50
        use matrix_util::traits::SampleOps;

        let d = 500;
        let s = 100;
        let k = 5;

        // Create pseudobulk sketch
        let sketch = DMatrix::<f32>::rnorm(d, s);
        let fc = compute_feature_coarsening(&sketch, 50).unwrap();

        // Create a fake log-dictionary at coarse resolution
        let coarse_logits = DMatrix::<f32>::rnorm(fc.num_coarse, k);

        // Expand to fine resolution
        let expanded = fc.expand_log_dict_dk(&coarse_logits, d);
        assert_eq!(expanded.nrows(), d);
        assert_eq!(expanded.ncols(), k);

        // For each topic, sum of exp(expanded) within each group
        // should equal exp(coarse logit)
        for kk in 0..k {
            for (c, group) in fc.coarse_to_fine.iter().enumerate() {
                let coarse_val = coarse_logits[(c, kk)].exp();
                let fine_sum: f32 = group.iter().map(|&f| expanded[(f, kk)].exp()).sum();
                assert_relative_eq!(fine_sum, coarse_val, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_skip_when_d_small() {
        // If D <= max_features, coarsening should still work but produce ~D groups
        use matrix_util::traits::SampleOps;
        let d = 30;
        let s = 20;
        let data = DMatrix::<f32>::rnorm(d, s);
        let fc = compute_feature_coarsening(&data, 50).unwrap();
        // Should produce fewer groups than D (binary hashing)
        assert!(fc.num_coarse <= d);
        assert!(fc.num_coarse > 0);
    }
}
