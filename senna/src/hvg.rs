//! Highly-variable-gene (HVG) selection for sparse backends.
//!
//! Streaming wrapper around `data_beans_alg::hvg_selection::select_hvg_by_stats`:
//! computes per-gene `(mean, variance)` of raw counts by streaming columns
//! through a visitor, then delegates the binned residual-variance logic to
//! the shared dense routine. Used to weight the random-projection sketch
//! (and for viz-layer feature gating); does not restrict the feature axis
//! of the Nyström dictionary or the topic decoder.

use crate::embed_common::*;
use clap::Args;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans_alg::hvg_selection::select_hvg_by_stats;
use matrix_util::common_io::read_lines;
use matrix_util::ndarray_stat::RunningStatistics;
use ndarray::Ix1;
use rustc_hash::FxHashMap;

/// Shared CLI args for HVG gating of the random projection.
/// Used by `senna svd`, `topic`, `indexed-topic`.
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
/// pipelines to subset the feature axis.
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
/// the shared binned residual-variance routine.
///
/// If `feature_list_file` is supplied it takes precedence and the HVG
/// computation is skipped entirely.
pub fn select_hvg_streaming(
    data_vec: &SparseIoVec,
    max_features: Option<usize>,
    feature_list_file: Option<&str>,
    block_size: usize,
    n_bins: Option<usize>,
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

    // Streaming pass: accumulate per-gene running mean+variance of raw counts.
    let mut stat = RunningStatistics::new(Ix1(data_vec.num_rows()));
    data_vec.visit_columns_by_block(
        &raw_stat_visitor,
        &EmptyArg {},
        &mut stat,
        Some(block_size),
    )?;

    let means = stat.mean().to_vec();
    let vars = stat.variance().to_vec();

    let mut selected_indices = select_hvg_by_stats(&means, &vars, n_features, n_bins);
    selected_indices.sort_unstable();

    info!(
        "Selected {} / {} highly variable features (binned residual variance)",
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

/// Running-statistics visitor over raw (un-transformed) column values.
fn raw_stat_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    _: &EmptyArg,
    arc_stat: Arc<Mutex<&mut RunningStatistics<Ix1>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let xx = data.read_columns_ndarray(lb..ub)?;
    let mut stat = arc_stat.lock().unwrap();
    for col in xx.axis_iter(ndarray::Axis(1)) {
        stat.add(&col);
    }
    Ok(())
}
