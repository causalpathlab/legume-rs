use crate::embed_common::*;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use fnv::FnvHashMap;
use matrix_util::common_io::read_lines;
use matrix_util::ndarray_stat::RunningStatistics;
use ndarray::Ix1;

/// Feature selection result with selected indices, names, and selection matrix
pub struct FeatureSelection {
    pub selected_indices: Vec<usize>,      // Sorted indices of selected features
    pub selected_names: Vec<Box<str>>,     // Names of selected features
    #[allow(dead_code)]
    pub index_map: FnvHashMap<usize, usize>,  // old_idx -> new_idx mapping
    pub selection_matrix: CscMat,          // Sparse selection matrix: S[new_i, old_i] = 1.0
}

/// Select highly variable features based on log-variance or from a pre-selected list
///
/// # Arguments
/// * `data_vec` - sparse data vector
/// * `max_features` - maximum number of features to select by log-variance (if None, all features are used)
/// * `feature_list_file` - path to file with pre-selected feature names (one per line), takes precedence over max_features
/// * `save_variance` - whether to save computed log-variance statistics
/// * `out_prefix` - output prefix for variance file
/// * `block_size` - block size for parallel processing
///
/// # Returns
/// * `FeatureSelection` - selected features with indices, names, and selection matrix
pub fn select_highly_variable_features(
    data_vec: &SparseIoVec,
    max_features: Option<usize>,
    feature_list_file: Option<&str>,
    save_variance: bool,
    out_prefix: &str,
    block_size: usize,
) -> anyhow::Result<FeatureSelection> {
    let feature_names = data_vec.row_names()?;

    // Option 1: Load from file
    if let Some(file_path) = feature_list_file {
        return load_feature_list_from_file(file_path, &feature_names);
    }

    // Option 2: Compute HVF from log-variance
    if let Some(n_features) = max_features {
        if n_features == 0 {
            return Err(anyhow::anyhow!("max_features must be >= 1"));
        }

        info!("Computing log-variance for {} features...", feature_names.len());

        // Compute log-variance using custom visitor
        let mut log_stat = RunningStatistics::new(Ix1(data_vec.num_rows()));
        data_vec.visit_columns_by_block(
            &log_variance_visitor,
            &EmptyArg {},
            &mut log_stat,
            Some(block_size),
        )?;

        let variance = log_stat.variance();

        // Save variance if requested
        if save_variance {
            let var_file = format!("{}.feature_variance.parquet", out_prefix);
            log_stat.save(&var_file, &feature_names, "\t")?;
            info!("Saved feature variance to {}", var_file);
        }

        // Rank features by variance (descending)
        let mut indexed_variance: Vec<(usize, f32)> = variance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        indexed_variance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top N features
        let n_select = n_features.min(indexed_variance.len());
        let mut selected_indices: Vec<usize> = indexed_variance[..n_select]
            .iter()
            .map(|(i, _)| *i)
            .collect();

        // Sort indices for efficient .select() operations
        selected_indices.sort_unstable();

        let selected_names = selected_indices
            .iter()
            .map(|&i| feature_names[i].clone())
            .collect();

        let index_map = selected_indices
            .iter()
            .enumerate()
            .map(|(new_i, &old_i)| (old_i, new_i))
            .collect();

        // Create sparse selection matrix once
        let selection_matrix = create_selection_matrix(&selected_indices, feature_names.len());

        info!(
            "Selected {} / {} highly variable features",
            n_select,
            feature_names.len()
        );

        Ok(FeatureSelection {
            selected_indices,
            selected_names,
            index_map,
            selection_matrix,
        })
    } else {
        Err(anyhow::anyhow!("Either max_features or feature_list_file must be provided"))
    }
}

/// Load feature list from a text file (one feature name per line)
///
/// # Arguments
/// * `file_path` - path to file with feature names
/// * `all_feature_names` - all available feature names from the data
///
/// # Returns
/// * `FeatureSelection` - selected features that match the data
fn load_feature_list_from_file(
    file_path: &str,
    all_feature_names: &[Box<str>],
) -> anyhow::Result<FeatureSelection> {
    let feature_names_from_file = read_lines(file_path)?;

    if feature_names_from_file.is_empty() {
        return Err(anyhow::anyhow!("Feature list file is empty: {}", file_path));
    }

    // Create a map from feature name to index
    let name_to_idx: FnvHashMap<&str, usize> = all_feature_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_ref(), i))
        .collect();

    // Find matching features
    let mut selected_indices: Vec<usize> = Vec::new();
    let mut selected_names: Vec<Box<str>> = Vec::new();
    let mut not_found = 0;

    for name in &feature_names_from_file {
        if let Some(&idx) = name_to_idx.get(name.as_ref()) {
            selected_indices.push(idx);
            selected_names.push(name.clone());
        } else {
            not_found += 1;
        }
    }

    if selected_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "No features from file matched data. File: {}",
            file_path
        ));
    }

    let match_rate = selected_indices.len() as f32 / feature_names_from_file.len() as f32;
    if match_rate < 0.5 {
        info!(
            "Warning: Only {:.1}% of features from file matched data ({}/{})",
            match_rate * 100.0,
            selected_indices.len(),
            feature_names_from_file.len()
        );
    }

    if not_found > 0 {
        info!("Warning: {} features from file not found in data", not_found);
    }

    // Sort indices for efficient .select() operations
    selected_indices.sort_unstable();

    // Create index map
    let index_map = selected_indices
        .iter()
        .enumerate()
        .map(|(new_i, &old_i)| (old_i, new_i))
        .collect();

    // Create sparse selection matrix once
    let selection_matrix = create_selection_matrix(&selected_indices, all_feature_names.len());

    info!("Loaded {} features from {}", selected_indices.len(), file_path);

    Ok(FeatureSelection {
        selected_indices,
        selected_names,
        index_map,
        selection_matrix,
    })
}

/// Compute log-variance for feature selection
///
/// This visitor processes blocks of columns to compute running statistics
/// of log-transformed values for feature selection.
fn log_variance_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    _: &EmptyArg,
    arc_stat: Arc<Mutex<&mut RunningStatistics<Ix1>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let xx = data.read_columns_ndarray(lb..ub)?;
    let xx_log = xx.mapv(|x| (x + 1.0).ln());  // log1p transform

    let mut stat = arc_stat.lock().unwrap();
    // Process each column (feature across samples)
    for col in xx_log.axis_iter(ndarray::Axis(1)) {
        stat.add(&col);
    }
    Ok(())
}

/// Create sparse selection matrix from selected row indices
///
/// Creates a sparse matrix S where S[new_i, old_i] = 1.0 for selected indices.
/// This allows efficient filtering: filtered_data = S * original_data
///
/// # Arguments
/// * `selected_indices` - sorted indices of selected features
/// * `n_total_rows` - total number of features in original data
///
/// # Returns
/// * `CscMat` - sparse selection matrix (n_selected × n_total)
fn create_selection_matrix(selected_indices: &[usize], n_total_rows: usize) -> CscMat {
    let n_selected = selected_indices.len();
    let mut triplets = Vec::with_capacity(n_selected);

    for (new_i, &old_i) in selected_indices.iter().enumerate() {
        triplets.push((new_i as u64, old_i as u64, 1.0_f32));
    }

    CscMat::from_nonzero_triplets(n_selected, n_total_rows, &triplets).expect("Failed to create selection matrix")
}

/// Filter CSC matrix by selected row indices using pre-computed selection matrix
///
/// # Arguments
/// * `selection_matrix` - sparse selection matrix (n_selected × n_total)
/// * `x` - input matrix to filter
///
/// # Returns
/// * `CscMat` - filtered matrix with only selected rows
pub fn filter_csc_by_rows(selection_matrix: &CscMat, x: &CscMat) -> CscMat {
    // Filtered result = selection_matrix * x
    selection_matrix * x
}
