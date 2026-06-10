use crate::random_projection::binary_sort_columns;
use log::debug;
use matrix_util::dmatrix_util::build_columns_par;
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
        build_columns_par(n, self.num_coarse, |c, col| {
            for &fine in &self.coarse_to_fine[c] {
                let src = data_nd.column(fine);
                for (dst, src_v) in col.iter_mut().zip(src.iter()) {
                    *dst += *src_v;
                }
            }
        })
    }

    /// Aggregate rows of a [D, S] matrix → [d, S] by summing
    /// features within each coarse group.
    pub fn aggregate_rows_ds(&self, data_ds: &DMatrix<f32>) -> DMatrix<f32> {
        let s = data_ds.ncols();
        build_columns_par(self.num_coarse, s, |j, col| {
            let src_col = data_ds.column(j);
            for (fine, &coarse) in self.fine_to_coarse.iter().enumerate() {
                col[coarse] += src_col[fine];
            }
        })
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
        build_columns_par(d_fine, k, |kk, col| {
            let src_col = log_dict_dk.column(kk);
            for (c, fine_indices) in self.coarse_to_fine.iter().enumerate() {
                let val = src_col[c] - (fine_indices.len() as f32).ln();
                for &f in fine_indices {
                    col[f] = val;
                }
            }
        })
    }

    /// Aggregate a sparse [D, n] CSC matrix → dense [d, n] by summing
    /// rows within each coarse group. Efficient: O(nnz) work.
    pub fn aggregate_sparse_csc(&self, data_dn: &CscMat) -> DMatrix<f32> {
        let n = data_dn.ncols();
        build_columns_par(self.num_coarse, n, |j, col| {
            let src = data_dn.col(j);
            for (&row, &val) in src.row_indices().iter().zip(src.values().iter()) {
                col[self.fine_to_coarse[row]] += val;
            }
        })
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

    // Generic helper — also used for cell coarsening etc. Keep at debug;
    // callers (cell coarsening, multilevel, chickpea topic) emit their own
    // axis-specific log line.
    debug!(
        "binary-sort coarsening: {} items, {} sketch dims, sort_dim={}",
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

    debug!(
        "binary-sort coarsening: {} → {} groups (target {})",
        d, num_coarse, max_features
    );

    Ok(FeatureCoarsening {
        fine_to_coarse,
        coarse_to_fine,
        num_coarse,
    })
}

#[cfg(test)]
#[path = "feature_coarsening_tests.rs"]
mod tests;
