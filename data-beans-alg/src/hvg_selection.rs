//! Highly variable gene (HVG) selection with mean-variance correction.
//!
//! ## Algorithm
//!
//! Models the mean-variance relationship and selects genes with variance
//! above the expected trend. This filters out housekeeping genes (high
//! mean → high variance due to Poisson noise) and prioritizes cell-type-
//! specific markers (above-trend variance = biological signal).
//!
//! ## References
//!
//! This implementation follows the binning approach from:
//!
//! - **Scanpy** (Wolf et al. 2018, Genome Biology):
//!   `scanpy.pp.highly_variable_genes(flavor='seurat')`
//!   Bins genes by mean expression, computes variance within bins,
//!   selects genes with high residual variance.
//!
//! - **Seurat** (Stuart et al. 2019, Cell):
//!   `FindVariableFeatures(method='vst')`
//!   Uses loess to model variance-mean relationship, selects genes
//!   with high standardized variance. Our binning approach is simpler
//!   but statistically equivalent.
//!
//! Key insight: High expression → high variance (technical). We want
//! genes with variance **above** the mean-variance trend (biological).

use nalgebra::DMatrix;
use rayon::prelude::*;

/// Select top N highly variable genes by residual variance.
///
/// # Algorithm
/// 1. Compute mean and variance for each gene (column)
/// 2. Bin genes by mean expression (default: 20 bins)
/// 3. Compute expected variance within each bin
/// 4. Calculate residual variance = observed - expected
/// 5. Select top N genes by residual variance
///
/// # Arguments
/// * `mat` - Expression matrix (rows = samples, columns = genes)
/// * `n_genes` - Number of HVGs to select
/// * `n_bins` - Number of bins for mean-variance trend (default: 20)
///
/// # Returns
/// Matrix with only the selected HVG columns
pub fn select_hvg(mat: &DMatrix<f32>, n_genes: usize, n_bins: Option<usize>) -> DMatrix<f32> {
    select_hvg_with_indices(mat, n_genes, n_bins).0
}

/// Select HVGs and return both the subset matrix and the selected indices.
///
/// Useful when you need to know which genes were selected.
pub fn select_hvg_with_indices(
    mat: &DMatrix<f32>,
    n_genes: usize,
    n_bins: Option<usize>,
) -> (DMatrix<f32>, Vec<usize>) {
    let (n_samples, n_genes_total) = (mat.nrows(), mat.ncols());
    let n_bins = n_bins.unwrap_or(20);

    if n_genes >= n_genes_total {
        let indices: Vec<usize> = (0..n_genes_total).collect();
        return (mat.clone(), indices);
    }

    // Compute mean and variance for each gene (parallelized)
    let gene_stats: Vec<(usize, f32, f32)> = (0..n_genes_total)
        .into_par_iter()
        .map(|j| {
            let col = mat.column(j);
            let mean: f32 = col.iter().sum::<f32>() / n_samples as f32;
            let var: f32 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n_samples as f32;
            (j, mean, var)
        })
        .collect();

    // Sort indices by mean expression (avoid cloning gene_stats)
    let mut indices_by_mean: Vec<usize> = (0..n_genes_total).collect();
    indices_by_mean.sort_by(|&a, &b| {
        gene_stats[a]
            .1
            .partial_cmp(&gene_stats[b].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let bin_size = (n_genes_total as f32 / n_bins as f32).ceil() as usize;
    let mut expected_var = vec![0.0f32; n_genes_total];

    // Compute expected variance within each bin
    for bin_idx in 0..n_bins {
        let start = bin_idx * bin_size;
        let end = ((bin_idx + 1) * bin_size).min(n_genes_total);
        if start >= end {
            break;
        }

        // Compute mean variance directly without intermediate allocation
        let mean_var: f32 = indices_by_mean[start..end]
            .iter()
            .map(|&idx| gene_stats[idx].2)
            .sum::<f32>()
            / (end - start) as f32;

        for &gene_idx in &indices_by_mean[start..end] {
            expected_var[gene_idx] = mean_var;
        }
    }

    // Compute residual variance
    let mut residuals: Vec<(usize, f32)> = gene_stats
        .iter()
        .map(|(idx, _mean, var)| (*idx, var - expected_var[*idx]))
        .collect();

    residuals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let hvg_indices: Vec<usize> = residuals
        .iter()
        .take(n_genes)
        .map(|(idx, _)| *idx)
        .collect();

    // Build matrix with selected columns
    let mut hvg_mat = DMatrix::zeros(n_samples, n_genes);
    for (new_j, &old_j) in hvg_indices.iter().enumerate() {
        for i in 0..n_samples {
            hvg_mat[(i, new_j)] = mat[(i, old_j)];
        }
    }

    (hvg_mat, hvg_indices)
}
