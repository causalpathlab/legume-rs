//! Per-gene weighting helpers used by reporting / annotation pipelines.
//!
//! `compute_nb_fisher_weights` derives bounded `(0, 1]` weights from the
//! NB dispersion trend (see `nb_dispersion`) — high-mean / high-dispersion
//! genes get attenuated, low-mean genes recover `w_g = 1` in the Poisson
//! limit. The same formula is applied inside DC-Poisson refinement so
//! clustering and reporting stay consistent.
//!
//! `apply_gene_weights` row-scales a `(num_genes × K)` matrix in place.
//!
//! Originally lived in `pinto/src/util/common.rs`; moved here so senna /
//! chickpea / other consumers don't have to duplicate the implementation.

use crate::nb_dispersion::DispersionTrend;
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_util::sparse_stat::SparseRunningStatistics;
use matrix_util::traits::RunningStatOps;
use matrix_util::utils::generate_minibatch_intervals;

/// Fit the NB mean-variance trend on `data_vec` and return per-gene
/// Fisher-info weights in row order (same as `data_vec.row_names()`).
pub fn compute_nb_fisher_weights(
    data_vec: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();
    let jobs = generate_minibatch_intervals(n_cells, n_genes, block_size);

    let mut stats = SparseRunningStatistics::<f32>::new(n_genes);
    for &(lb, ub) in &jobs {
        let csc = data_vec.read_columns_csc(lb..ub)?;
        stats.add_csc(&csc);
    }

    let trend = DispersionTrend::from_sparse_stats(&stats);
    let means = stats.mean();
    let sums = stats.sum();
    let total_mass: f64 = sums.iter().map(|&s| s as f64).sum();
    let avg_s = if n_cells > 0 {
        (total_mass / n_cells as f64) as f32
    } else {
        1.0
    };
    let inv_total = if total_mass > 0.0 {
        1.0 / total_mass as f32
    } else {
        0.0
    };

    Ok((0..n_genes)
        .map(|g| trend.fisher_weight(sums[g] * inv_total, avg_s, means[g]))
        .collect())
}

/// Scale each row of `sum_gk` by its corresponding entry of `weights`.
/// `weights.len()` must equal `sum_gk.nrows()`.
pub fn apply_gene_weights(sum_gk: &mut nalgebra::DMatrix<f32>, weights: &[f32]) {
    debug_assert_eq!(weights.len(), sum_gk.nrows());
    for (g, &w) in weights.iter().enumerate() {
        sum_gk.row_mut(g).scale_mut(w);
    }
}
