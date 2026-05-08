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

use crate::feature_coarsening::FeatureCoarsening;
use crate::nb_dispersion::DispersionTrend;
use crate::sparse_streaming::streaming_sparse_running_stats;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use matrix_util::sparse_stat::SparseRunningStatistics;
use matrix_util::traits::{IoOps, RunningStatOps};
use matrix_util::utils::generate_minibatch_intervals;
use rayon::prelude::*;

/// Fit the NB mean-variance trend on `data_vec` and return per-gene
/// Fisher-info weights in row order (same as `data_vec.row_names()`).
pub fn compute_nb_fisher_weights(
    data_vec: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();
    let stats = streaming_sparse_running_stats(data_vec, block_size, "NB-Fisher")?;

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

/// Same as [`compute_nb_fisher_weights`], but at the *coarse* feature
/// level after `FeatureCoarsening` aggregation. Streams cells, coarsens
/// each block via `aggregate_sparse_csc` (sum within group), then fits
/// a fresh NB dispersion trend at `D_coarse` and computes per-meta-gene
/// Fisher weights.
///
/// Use this when the encoder/decoder operate at a coarsened feature
/// resolution: the data the model actually sees is summed-coarsened
/// counts, so the dispersion trend (and hence Fisher info) must be
/// fit at that scale — not aggregated from fine-level weights.
pub fn compute_nb_fisher_weights_coarsened(
    data_vec: &SparseIoVec,
    coarsening: &FeatureCoarsening,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let n_features_coarse = coarsening.num_coarse;
    let n_total = data_vec.num_columns();
    let jobs = generate_minibatch_intervals(n_total, n_features_coarse, block_size);

    let pb = ProgressBar::new(jobs.len() as u64).with_style(
        ProgressStyle::with_template("NB-Fisher (coarse) {bar:40} {pos}/{len} blocks ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );

    let stats: SparseRunningStatistics<f32> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .try_fold(
            || SparseRunningStatistics::<f32>::new(n_features_coarse),
            |mut acc, &(lb, ub)| -> anyhow::Result<SparseRunningStatistics<f32>> {
                let chunk = data_vec.read_columns_csc(lb..ub)?;
                let coarse = coarsening.aggregate_sparse_csc(&chunk);
                // Add each coarsened cell column as a sparse column
                // (most coarse groups will be non-empty, but skipping
                // zeros keeps `n_pos` accurate per coarse feature).
                let n_block = ub - lb;
                let mut row_idx: Vec<usize> = Vec::with_capacity(n_features_coarse);
                let mut row_val: Vec<f32> = Vec::with_capacity(n_features_coarse);
                for j in 0..n_block {
                    row_idx.clear();
                    row_val.clear();
                    for c in 0..n_features_coarse {
                        let v = coarse[(c, j)];
                        if v > 0.0 {
                            row_idx.push(c);
                            row_val.push(v);
                        }
                    }
                    acc.add_sparse_column(&row_idx, &row_val);
                }
                Ok(acc)
            },
        )
        .try_reduce(
            || SparseRunningStatistics::<f32>::new(n_features_coarse),
            |mut a, b| {
                a.merge(&b);
                Ok(a)
            },
        )?;
    pb.finish_and_clear();

    let trend = DispersionTrend::from_sparse_stats(&stats);
    let means = stats.mean();
    let sums = stats.sum();
    let total_mass: f64 = sums.iter().map(|&s| s as f64).sum();
    let avg_s = if n_total > 0 {
        (total_mass / n_total as f64) as f32
    } else {
        1.0
    };
    let inv_total = if total_mass > 0.0 {
        1.0 / total_mass as f32
    } else {
        0.0
    };

    Ok((0..n_features_coarse)
        .map(|c| trend.fisher_weight(sums[c] * inv_total, avg_s, means[c]))
        .collect())
}

/// Save a per-gene weight vector as a single-column parquet keyed on
/// gene name. The output path is fully specified — callers control the
/// filename. For the standard `{out}.fisher_weights.parquet` convention
/// used by senna and chickpea, see [`save_fisher_weights`].
pub fn save_per_gene_weights(
    weights: &[f32],
    gene_names: &[Box<str>],
    out_path: &str,
) -> anyhow::Result<()> {
    let mat = nalgebra::DMatrix::<f32>::from_column_slice(weights.len(), 1, weights);
    let weight_col = vec![Box::<str>::from("weight")];
    mat.to_parquet_with_names(
        out_path,
        (Some(gene_names), Some("gene")),
        Some(&weight_col),
    )?;
    Ok(())
}

/// Convenience wrapper: write per-gene weights to the conventional
/// `{out_prefix}.fisher_weights.parquet` path used by senna `topic` and
/// chickpea `fit-topic`. Downstream consumers (e.g. `senna annotate`)
/// look for this filename to reload weights instead of re-scanning all
/// cells.
pub fn save_fisher_weights(
    out_prefix: &str,
    weights: &[f32],
    gene_names: &[Box<str>],
) -> anyhow::Result<()> {
    save_per_gene_weights(
        weights,
        gene_names,
        &format!("{out_prefix}.fisher_weights.parquet"),
    )
}

/// `(gene_names, per-gene weights)` returned by the per-gene weight loaders.
pub type GeneWeights = (Vec<Box<str>>, Vec<f32>);

/// Load a per-gene weight vector previously written by
/// [`save_per_gene_weights`]. Returns `(gene_names, weights)`.
pub fn load_per_gene_weights(path: &str) -> anyhow::Result<GeneWeights> {
    let result = nalgebra::DMatrix::<f32>::from_parquet_with_row_names(path, Some(0))?;
    anyhow::ensure!(
        result.mat.ncols() >= 1,
        "per-gene weights parquet at {path} has no value column",
    );
    let weights: Vec<f32> = result.mat.column(0).iter().copied().collect();
    Ok((result.rows, weights))
}

/// Convenience wrapper: load `{prefix}.fisher_weights.parquet`. Returns
/// `Ok(None)` when the file doesn't exist so callers can fall back to
/// recomputing via [`compute_nb_fisher_weights`].
pub fn load_fisher_weights(prefix: &str) -> anyhow::Result<Option<GeneWeights>> {
    let path = format!("{prefix}.fisher_weights.parquet");
    if !std::path::Path::new(&path).exists() {
        return Ok(None);
    }
    Ok(Some(load_per_gene_weights(&path)?))
}

/// Scale each row of `sum_gk` by its corresponding entry of `weights`.
/// `weights.len()` must equal `sum_gk.nrows()`.
pub fn apply_gene_weights(sum_gk: &mut nalgebra::DMatrix<f32>, weights: &[f32]) {
    debug_assert_eq!(weights.len(), sum_gk.nrows());
    for (g, &w) in weights.iter().enumerate() {
        sum_gk.row_mut(g).scale_mut(w);
    }
}
