//! Highly variable gene (HVG) selection via NB dispersion trend.
//!
//! Fits a smooth Negative-Binomial dispersion trend `σ²(μ) = μ + φ(μ)·μ²`
//! over all features and ranks genes by `(σ² − μ)/μ² − φ(μ)` — excess
//! dispersion above the trend. Continuous, self-reference-free, and shares
//! the same `φ(μ)` fit that [`crate::nb_dispersion::DispersionTrend`]
//! powers for DC-Poisson clustering.

use crate::nb_dispersion::DispersionTrend;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Select top N highly variable genes by NB excess dispersion.
///
/// # Arguments
/// * `mat` - Expression matrix (rows = samples, columns = genes)
/// * `n_genes` - Number of HVGs to select
///
/// # Returns
/// Matrix with only the selected HVG columns.
pub fn select_hvg(mat: &DMatrix<f32>, n_genes: usize) -> DMatrix<f32> {
    select_hvg_with_indices(mat, n_genes).0
}

/// Select HVGs and return both the subset matrix and the selected indices.
pub fn select_hvg_with_indices(mat: &DMatrix<f32>, n_genes: usize) -> (DMatrix<f32>, Vec<usize>) {
    let (n_samples, n_genes_total) = (mat.nrows(), mat.ncols());

    if n_genes >= n_genes_total {
        let indices: Vec<usize> = (0..n_genes_total).collect();
        return (mat.clone(), indices);
    }

    let (means, vars): (Vec<f32>, Vec<f32>) = (0..n_genes_total)
        .into_par_iter()
        .map(|j| {
            let col = mat.column(j);
            let mean: f32 = col.iter().sum::<f32>() / n_samples as f32;
            let var: f32 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n_samples as f32;
            (mean, var)
        })
        .unzip();

    let hvg_indices = select_hvg_by_stats(&means, &vars, n_genes);

    let mut hvg_mat = DMatrix::zeros(n_samples, hvg_indices.len());
    for (new_j, &old_j) in hvg_indices.iter().enumerate() {
        for i in 0..n_samples {
            hvg_mat[(i, new_j)] = mat[(i, old_j)];
        }
    }

    (hvg_mat, hvg_indices)
}

/// Select top-N HVG indices from pre-computed per-gene means and variances.
/// Returns indices sorted ascending.
pub fn select_hvg_by_stats(means: &[f32], vars: &[f32], n_genes: usize) -> Vec<usize> {
    assert_eq!(means.len(), vars.len());
    let n_genes_total = means.len();
    if n_genes >= n_genes_total {
        return (0..n_genes_total).collect();
    }

    let trend = DispersionTrend::fit(means, vars);
    let mut ranked: Vec<(usize, f32)> = means
        .iter()
        .zip(vars.iter())
        .enumerate()
        .map(|(j, (&mu, &v))| (j, trend.excess(mu, v)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut hvg_indices: Vec<usize> = ranked.iter().take(n_genes).map(|(idx, _)| *idx).collect();
    hvg_indices.sort_unstable();
    hvg_indices
}
