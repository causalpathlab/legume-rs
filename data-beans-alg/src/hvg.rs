//! Highly-variable-gene (HVG) selection.
//!
//! Two surfaces:
//! - [`select_hvg`] / [`select_hvg_with_indices`] / [`select_hvg_by_stats`] —
//!   dense-matrix scoring via NB dispersion trend (`σ²(μ) = μ + φ(μ)·μ²`),
//!   ranking genes by excess dispersion above the trend.
//! - [`select_hvg_streaming`] — sparse streaming wrapper that computes
//!   per-gene `(mean, variance)` of raw counts directly from CSC chunks
//!   (lock-free, per-thread accumulators merged at the end), then
//!   delegates to [`select_hvg_by_stats`].
//!
//! Used by senna and chickpea to reweight the random-projection basis so
//! the sketch geometry reflects variable biology rather than housekeeping
//! signal.

use crate::nb_dispersion::DispersionTrend;
use crate::sparse_streaming::streaming_sparse_running_stats;
use clap::Args;
use data_beans::sparse_io_vector::SparseIoVec;
use log::info;
use matrix_util::common_io::read_lines;
use matrix_util::traits::RunningStatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;

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

/// Shared CLI args for HVG gating of the random projection.
#[derive(Args, Debug, Clone)]
pub struct HvgCliArgs {
    #[arg(
        long = "n-hvg",
        default_value_t = 5000,
        help = "Keep top N highly variable genes (0 disables HVG)",
        long_help = "Select top N genes via binned residual-variance\n\
                     (scanpy/Seurat-style) for the random projection.\n\
                     Collapsing and batch-effect estimation still see\n\
                     all genes. 0 disables HVG. Ignored when\n\
                     --warm-start is set."
    )]
    pub n_hvg: usize,

    #[arg(
        long,
        help = "Pre-computed HVG list (one gene name per line)",
        long_help = "Takes precedence over --n-hvg. Ignored when\n\
                     --warm-start is set."
    )]
    pub feature_list_file: Option<Box<str>>,
}

/// HVG selection result used by SVD / topic / indexed-topic / joint-*
/// pipelines to subset or weight the feature axis.
#[derive(Clone)]
pub struct HvgSelection {
    pub selected_indices: Vec<usize>,
    pub selected_names: Vec<Box<str>>,
    #[allow(dead_code)]
    pub index_map: FxHashMap<usize, usize>,
}

impl HvgSelection {
    /// Per-feature weight vector suitable for `project_columns_weighted`:
    /// 1.0 at selected indices, 0.0 elsewhere.
    #[must_use]
    pub fn row_weights(&self, n_total: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; n_total];
        for &i in &self.selected_indices {
            if i < n_total {
                w[i] = 1.0;
            }
        }
        w
    }
}

/// Stream cells through the sparse backend to compute per-gene mean and
/// variance of raw expression, then select the top `n_features` HVGs via
/// the shared NB-trend scoring routine.
///
/// If `feature_list_file` is supplied it takes precedence and the HVG
/// computation is skipped entirely.
pub fn select_hvg_streaming(
    data_vec: &SparseIoVec,
    max_features: Option<usize>,
    feature_list_file: Option<&str>,
    block_size: Option<usize>,
) -> anyhow::Result<HvgSelection> {
    let feature_names = data_vec.row_names()?;

    if let Some(path) = feature_list_file {
        return load_feature_list_from_file(path, &feature_names);
    }

    let n_features = max_features
        .ok_or_else(|| anyhow::anyhow!("max_features or feature_list_file must be provided"))?;
    if n_features == 0 {
        return Err(anyhow::anyhow!("max_features must be >= 1"));
    }

    info!(
        "starting HVG streaming pass: D={}, N={}, block_size={:?}",
        data_vec.num_rows(),
        data_vec.num_columns(),
        block_size,
    );
    let hvg_t0 = Instant::now();
    let stat = streaming_sparse_running_stats(data_vec, block_size, "HVG")?;
    info!(
        "finished HVG streaming pass in {:.2?} ({} cells)",
        hvg_t0.elapsed(),
        stat.ncols_processed()
    );

    let means = stat.mean();
    let vars = stat.variance();

    let mut selected_indices = select_hvg_by_stats(&means, &vars, n_features);
    selected_indices.sort_unstable();

    info!(
        "Selected {} / {} highly variable features (NB dispersion-trend excess)",
        selected_indices.len(),
        feature_names.len()
    );

    Ok(build_selection(selected_indices, &feature_names))
}

fn load_feature_list_from_file(
    file_path: &str,
    all_feature_names: &[Box<str>],
) -> anyhow::Result<HvgSelection> {
    let names_from_file = read_lines(file_path)?;
    if names_from_file.is_empty() {
        return Err(anyhow::anyhow!("Feature list file is empty: {file_path}"));
    }

    let name_to_idx: FxHashMap<&str, usize> = all_feature_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i))
        .collect();

    let mut selected_indices: Vec<usize> = Vec::new();
    let mut not_found = 0usize;
    for name in &names_from_file {
        if let Some(&idx) = name_to_idx.get(name.as_ref()) {
            selected_indices.push(idx);
        } else {
            not_found += 1;
        }
    }
    if selected_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "No features from file matched data. File: {file_path}"
        ));
    }
    if not_found > 0 {
        info!("Warning: {not_found} features from file not found in data");
    }
    selected_indices.sort_unstable();

    info!(
        "Loaded {} features from {}",
        selected_indices.len(),
        file_path
    );

    Ok(build_selection(selected_indices, all_feature_names))
}

fn build_selection(selected_indices: Vec<usize>, feature_names: &[Box<str>]) -> HvgSelection {
    let selected_names: Vec<Box<str>> = selected_indices
        .iter()
        .map(|&i| feature_names[i].clone())
        .collect();
    let index_map: FxHashMap<usize, usize> = selected_indices
        .iter()
        .enumerate()
        .map(|(new_i, &old_i)| (old_i, new_i))
        .collect();
    HvgSelection {
        selected_indices,
        selected_names,
        index_map,
    }
}
