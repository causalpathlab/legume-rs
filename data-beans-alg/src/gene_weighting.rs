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
use indicatif::ParallelProgressIterator;
use matrix_util::sparse_stat::SparseRunningStatistics;
use matrix_util::traits::{IoOps, RunningStatOps};
use matrix_util::utils::generate_minibatch_intervals;
use rayon::prelude::*;

/// Fit a fresh NB dispersion trend on `stats` and return per-feature
/// Fisher-info weights `w_d ∈ (0, 1]`. `n_entities` is the count divisor
/// used to compute the mean entity size factor `s̄` (typically the
/// number of cells or pseudobulks the stats were accumulated over).
pub fn fisher_weights_from_stats(
    stats: &SparseRunningStatistics<f32>,
    n_entities: usize,
) -> Vec<f32> {
    let trend = DispersionTrend::from_sparse_stats(stats);
    let means = stats.mean();
    let sums = stats.sum();
    let total_mass: f64 = sums.iter().map(|&s| s as f64).sum();
    let avg_s = if n_entities > 0 {
        (total_mass / n_entities as f64) as f32
    } else {
        1.0
    };
    let inv_total = if total_mass > 0.0 {
        1.0 / total_mass as f32
    } else {
        0.0
    };
    (0..means.len())
        .map(|g| trend.fisher_weight(sums[g] * inv_total, avg_s, means[g]))
        .collect()
}

/// Fit the NB mean-variance trend on `data_vec` and return per-gene
/// Fisher-info weights in row order (same as `data_vec.row_names()`).
pub fn compute_nb_fisher_weights(
    data_vec: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let n_cells = data_vec.num_columns();
    let stats = streaming_sparse_running_stats(data_vec, block_size, "NB-Fisher")?;
    Ok(fisher_weights_from_stats(&stats, n_cells))
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

    let prog_bar = matrix_util::progress::new_progress_bar(jobs.len() as u64)
        .with_message("NB-Fisher (coarse) blocks");

    let stats: SparseRunningStatistics<f32> = jobs
        .par_iter()
        .progress_with(prog_bar.clone())
        .try_fold(
            || SparseRunningStatistics::<f32>::new(n_features_coarse),
            |mut acc, &(lb, ub)| -> anyhow::Result<SparseRunningStatistics<f32>> {
                let chunk = data_vec.read_columns_csc(lb..ub)?;
                let coarse = coarsening.aggregate_sparse_csc(&chunk);
                acc.add_dense_columns(&coarse);
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
    prog_bar.finish_and_clear();

    Ok(fisher_weights_from_stats(&stats, n_total))
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

/// Save coarse-resolution NB-Fisher weights computed at the encoder/
/// decoder's `D_coarse` (post-feature-coarsening). Stored at
/// `{prefix}.fisher_weights_coarse.parquet` keyed on synthetic
/// `coarse_<i>` axis ids since coarse meta-genes have no biological
/// names. Loaded by `senna predict` / `senna eval-topic` to re-attach
/// the same per-feature multinomial-loss weights at inference.
pub fn save_fisher_weights_coarse(out_prefix: &str, weights: &[f32]) -> anyhow::Result<()> {
    let axis_names: Vec<Box<str>> = (0..weights.len())
        .map(|i| format!("coarse_{i}").into_boxed_str())
        .collect();
    save_per_gene_weights(
        weights,
        &axis_names,
        &format!("{out_prefix}.fisher_weights_coarse.parquet"),
    )
}

/// Convenience wrapper: load `{prefix}.fisher_weights_coarse.parquet`.
/// Returns `Ok(None)` when the file doesn't exist (older models trained
/// before coarse weights were saved).
pub fn load_fisher_weights_coarse(prefix: &str) -> anyhow::Result<Option<GeneWeights>> {
    let path = format!("{prefix}.fisher_weights_coarse.parquet");
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
