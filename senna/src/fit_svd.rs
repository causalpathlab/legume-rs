use crate::embed_common::*;
use crate::senna_input::*;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use matrix_util::common_io::read_lines;
use matrix_util::ndarray_stat::RunningStatistics;
use ndarray::Ix1;
use std::collections::HashMap;

#[derive(Args, Debug)]
pub struct SvdArgs {
    #[arg(
        required = true,
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header for results.\n\
		     Specify the output file or prefix for generated files:\n\
		     - {out}.delta.parquet\n\
		     - {out}.dictionary.parquet\n\
		     - {out}.latent.parquet\n"
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Random projection dimension to project the data.\n\
		     Controls the dimensionality of the random projection step."
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection",
        long_help = "Use top {d} components of projection.\n\
		     Number of samples will be less than `2^{d}+1`."
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.csv,batch2.csv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Ignore batch adjustment",
        long_help = "Ignore batch adjustment.\n\
		     Disables batch effect correction during processing."
    )]
    ignore_batch_effects: bool,

    #[arg(
        short = 'w',
        long = "warm-start",
        help = "Warm start projection file",
        long_help = "Warm start from the previous projection (cell x k).\n\
		     Provide a file to initialize the projection."
    )]
    warm_start_proj_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of k-nearest neighbour batches",
        long_help = "Number of k-nearest neighbour batches.\n\
		     Controls the number of batches considered \n\
		     for nearest neighbour search."
    )]
    knn_batches: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch",
        long_help = "Number of k-nearest neighbours within each batch.\n\
		     Controls the number of cells considered \n\
		     for nearest neighbour search within each batch."
    )]
    knn_cells: usize,

    #[arg(
        long,
        value_delimiter(','),
        help = "Reference batch names",
        long_help = "Reference batch names (comma-separated).\n\
		     Specify batches to be used as reference during adjustment."
    )]
    reference_batches: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 30,
        help = "Optimization iterations",
        long_help = "Number of optimization iterations.\n\
		     Controls the number of steps for model optimization."
    )]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing",
        long_help = "Block size (number of columns) for parallel processing.\n\
		     Controls the granularity of parallel computation."
    )]
    block_size: usize,

    #[arg(
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column sum normalization scale",
        long_help = "Column sum normalization scale.\n\
		     Adjusts normalization of columns during processing."
    )]
    column_sum_norm: f32,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics",
        long_help = "Number of latent topics.\n\
		     Controls the dimensionality of the latent topic space."
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all columns data",
        long_help = "Preload all the columns data into memory.\n\
		     Improves performance for large datasets."
    )]
    preload_data: bool,

    #[arg(
        long,
        help = "Save batch-adjusted data",
        long_help = "If set, save the batch-adjusted data creating a new backend."
    )]
    save_adjusted: bool,

    #[arg(
        long,
        short,
        help = "Verbosity",
        long_help = "Enable verbose output.\n\
		     Prints additional information during execution."
    )]
    verbose: bool,

    #[arg(
        long,
        help = "Maximum number of highly variable features",
        long_help = "Select top N features by log-variance.\n\
		     If not specified, all features are used.\n\
		     Skipped if --warm-start is provided."
    )]
    max_features: Option<usize>,

    #[arg(
        long,
        help = "Pre-computed feature selection file",
        long_help = "Path to file with pre-selected feature names (one per line).\n\
		     Takes precedence over --max-features.\n\
		     Skipped if --warm-start is provided."
    )]
    feature_list_file: Option<Box<str>>,

    #[arg(
        long,
        help = "Save feature variance statistics",
        long_help = "Save computed log-variance for all features to {out}.feature_variance.parquet"
    )]
    save_feature_variance: bool,
}

pub fn fit_svd(args: &SvdArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
        nbatch,
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;

    // Feature selection (if requested)
    let selected_features: Option<FeatureSelection> = if args.warm_start_proj_file.is_some() {
        if args.max_features.is_some() || args.feature_list_file.is_some() {
            info!("Warm-start provided: skipping feature selection (may be incompatible)");
        }
        None
    } else if args.max_features.is_some() || args.feature_list_file.is_some() {
        Some(select_highly_variable_features(
            &data_vec,
            args.max_features,
            args.feature_list_file.as_deref(),
            args.save_feature_variance,
            &args.out,
            args.block_size,
        )?)
    } else {
        None
    };

    // 2. Random projection
    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file.as_deref() {
        use matrix_util::common_io::*;
        let ext = file_ext(proj_file)?;

        let MatWithNames {
            rows: cell_names,
            cols: _,
            mat: proj_nk,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_row_names(&proj_file, Some(0))?,
            _ => Mat::read_data_with_names(&proj_file, &['\t', ',', ' '], Some(0), Some(0))?,
        };

        if data_vec.column_names()? != cell_names {
            return Err(anyhow::anyhow!(
                "warm start projection rows don't match with the data"
            ));
        }

        proj_nk.transpose()
    } else {
        let proj_dim = args.proj_dim.max(args.n_latent_topics);

        let proj_out = data_vec.project_columns_with_batch_correction(
            proj_dim,
            Some(args.block_size),
            Some(&batch_membership),
        )?;

        proj_out.proj
    };

    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    let nsamp = data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), None)?;

    if !args.ignore_batch_effects && nbatch > 1 {
        info!("Registering batch information");
        data_vec.build_hnsw_per_batch(&proj_kn, &batch_membership)?;
    }

    // 3. Batch-adjusted collapsing (pseudobulk)
    let reference = args.reference_batches.as_ref().map(|x| x.as_slice());

    info!("Collapsing columns into {} pseudobulk samples ...", nsamp);
    let collapse_out = data_vec.collapse_columns(
        Some(args.knn_batches),
        Some(args.knn_cells),
        reference,
        Some(args.iter_opt),
    )?;

    // 4. batch-adjusted data
    let batch_dp = collapse_out.mu_residual.as_ref();

    if let Some(delta_dp) = batch_dp.map(|x| x.posterior_mean()) {
        info!("{} x {}", delta_dp.nrows(), delta_dp.ncols());

        if args.save_adjusted {
            info!("Generating batch-adjusted data...");

            let triplets = triplets_adjusted_by_pseudobulk(&data_vec, delta_dp)?;

            let mtx_shape = (
                data_vec.num_rows(),
                data_vec.num_columns(),
                triplets.len(),
            );

            let backend_file = args.out.to_string() + ".adjusted.zarr";
            let backend = SparseIoBackend::Zarr;
            remove_file(&backend_file)?;

            let mut adjusted_data = create_sparse_from_triplets(
                &triplets,
                mtx_shape,
                Some(&backend_file),
                Some(&backend),
            )?;

            adjusted_data.register_row_names_vec(&data_vec.row_names()?);
            adjusted_data.register_column_names_vec(&data_vec.column_names()?);

            info!("Batch-adjusted backend: {}", backend_file);
        }
    }

    if let Some(batch_db) = collapse_out.delta {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet(Some(&gene_names), batch_names.as_deref(), &outfile)?;
    }

    // 5. Nystrom projection
    let x_dn = match collapse_out.mu_adjusted.as_ref() {
        Some(adj) => adj,
        None => &collapse_out.mu_observed,
    };

    let nystrom_out = do_nystrom_proj(
        x_dn.posterior_log_mean().clone(),
        batch_dp.map(|x| x.posterior_mean()),
        &data_vec,
        args.n_latent_topics,
        args.column_sum_norm,
        Some(args.block_size),
        selected_features.as_ref(),
    )?;

    let cell_names = data_vec.column_names()?;
    let gene_names = data_vec.row_names()?;

    // Use selected feature names for dictionary if feature selection was applied
    let output_gene_names = selected_features
        .as_ref()
        .map(|sel| sel.selected_names.clone())
        .unwrap_or_else(|| gene_names.clone());

    nystrom_out.latent_nk.to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".latent.parquet"),
    )?;

    nystrom_out.dictionary_dk.to_parquet(
        Some(&output_gene_names),
        None,
        &(args.out.to_string() + ".dictionary.parquet"),
    )?;

    // Save selected feature list if feature selection was applied
    if let Some(sel) = &selected_features {
        use matrix_util::common_io::write_lines;
        let feature_file = args.out.to_string() + ".selected_features.txt";
        write_lines(&sel.selected_names, &feature_file)?;
        info!("Saved {} selected features to {}", sel.selected_names.len(), feature_file);
    }

    Ok(())
}

struct FeatureSelection {
    selected_indices: Vec<usize>,      // Sorted indices of selected features
    selected_names: Vec<Box<str>>,     // Names of selected features
    #[allow(dead_code)]
    index_map: HashMap<usize, usize>,  // old_idx -> new_idx mapping
    selection_matrix: CscMat,          // Sparse selection matrix: S[new_i, old_i] = 1.0
}

#[allow(dead_code)]
struct NystromParam<'a> {
    dictionary_dk: &'a Mat,
    basis_dk: &'a Mat,
    delta_dp: Option<&'a Mat>,
    column_sum_norm: f32,
    feature_selection: Option<&'a FeatureSelection>,
}

struct NystromOut {
    pub dictionary_dk: Mat,
    pub latent_nk: Mat,
}

/// Nystrom projection for fast latent representation
///
/// # Arguments
/// * `xx_dn` - feature x sample matrix
/// * `delta_db` - feature x batch batch effect matrix
/// * `full_data_vec` - full sparse data vector
/// * `rank` - matrix factorization rank
/// * `column_sum_norm` - column sum normalization scale
/// * `block_size` - online learning block size
///
///
fn do_nystrom_proj(
    log_xx_dn: Mat,
    delta_dp: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    rank: usize,
    column_sum_norm: f32,
    block_size: Option<usize>,
    feature_selection: Option<&FeatureSelection>,
) -> anyhow::Result<NystromOut> {
    let mut log_xx_dn = log_xx_dn.clone();

    // Apply feature selection to collapsed data before RSVD
    if let Some(sel) = feature_selection {
        log_xx_dn = log_xx_dn.select_rows(&sel.selected_indices);
    }

    log_xx_dn.scale_columns_inplace();

    let (u_dk, s_k, _) = log_xx_dn.rsvd(rank)?;
    let eps = 1e-8;
    let sinv_k = DVec::from_iterator(s_k.len(), s_k.iter().map(|&s| 1.0 / (s + eps)));
    let basis_dk = &u_dk * Mat::from_diagonal(&sinv_k);

    info!(
        "Constructed {} x {} projection matrix",
        u_dk.nrows(),
        u_dk.ncols()
    );

    let ntot = full_data_vec.num_columns();
    let kk = rank;

    let nystrom_param = NystromParam {
        dictionary_dk: &u_dk,
        basis_dk: &basis_dk,
        delta_dp,
        column_sum_norm,
        feature_selection,
    };

    let mut proj_kn = Mat::zeros(kk, ntot);

    full_data_vec.visit_columns_by_block(
        &nystrom_proj_visitor,
        &nystrom_param,
        &mut proj_kn,
        block_size,
    )?;

    let z_nk = proj_kn.transpose();

    Ok(NystromOut {
        dictionary_dk: u_dk,
        latent_nk: z_nk,
    })
}

fn nystrom_proj_visitor(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    proj_basis: &NystromParam,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    // let proj_dk = proj_basis.dictionary_dk;
    let basis_dk = proj_basis.basis_dk;
    let delta_dp = proj_basis.delta_dp;
    let column_sum_norm = proj_basis.column_sum_norm;
    let feature_selection = proj_basis.feature_selection;

    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

    // Apply feature selection lazily
    if let Some(sel) = feature_selection {
        x_dn = filter_csc_by_rows(&sel.selection_matrix, &x_dn);
    }

    x_dn.normalize_columns_inplace();
    x_dn *= column_sum_norm;

    if let Some(delta_dp) = delta_dp {
        let pseudobulk = full_data_vec.get_group_membership(lb..ub)?;
        x_dn.adjust_by_division_of_selected_inplace(delta_dp, &pseudobulk);
    }

    x_dn.log1p_inplace();
    x_dn.scale_columns_inplace();

    let chunk = (x_dn.transpose() * basis_dk).transpose();

    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in nystrom");

    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

/// Adjust the original data by eliminating batch effects `delta_db`
/// (`d x b`) from each column. We will directly call
/// `get_batch_membership` in `data_vec`.
///
/// # Arguments
/// * `data_vec` - sparse data vector
/// * `delta_dp` - row/feature by pseudobulk average effect matrix
///
/// # Returns
/// * `triplets` - we can feed this vector to create a new backend
fn triplets_adjusted_by_pseudobulk(
    data_vec: &SparseIoVec,
    delta_dp: &Mat,
) -> anyhow::Result<Vec<(u64, u64, f32)>> {
    let mut triplets = vec![];
    data_vec.visit_columns_by_block(&adjust_triplets_visitor, delta_dp, &mut triplets, None)?;
    Ok(triplets)
}

fn adjust_triplets_visitor(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    delta_dp: &Mat,
    triplets: Arc<Mutex<&mut Vec<(u64, u64, f32)>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;

    let pbs = full_data_vec.get_group_membership(lb..ub)?;
    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

    x_dn.adjust_by_division_of_selected_inplace(delta_dp, &pbs);

    let new_triplets = x_dn
        .triplet_iter()
        .filter_map(|(i, j, &x_ij)| {
            let x_ij = x_ij.round();
            if x_ij < 1_f32 {
                None
            } else {
                Some((i as u64, (j + lb) as u64, x_ij))
            }
        })
        .collect::<Vec<_>>();

    let mut triplets = triplets.lock().expect("lock triplets");
    triplets.extend(new_triplets);
    Ok(())
}

/// Create sparse selection matrix from selected row indices
fn create_selection_matrix(selected_indices: &[usize], n_total_rows: usize) -> CscMat {
    let n_selected = selected_indices.len();
    let mut triplets = Vec::with_capacity(n_selected);

    for (new_i, &old_i) in selected_indices.iter().enumerate() {
        triplets.push((new_i as u64, old_i as u64, 1.0_f32));
    }

    CscMat::from_nonzero_triplets(n_selected, n_total_rows, &triplets).expect("Failed to create selection matrix")
}

/// Load feature list from a text file (one feature name per line)
fn load_feature_list_from_file(
    file_path: &str,
    all_feature_names: &[Box<str>],
) -> anyhow::Result<FeatureSelection> {
    let feature_names_from_file = read_lines(file_path)?;

    if feature_names_from_file.is_empty() {
        return Err(anyhow::anyhow!("Feature list file is empty: {}", file_path));
    }

    // Create a map from feature name to index
    let name_to_idx: HashMap<&str, usize> = all_feature_names
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

/// Select highly variable features based on log-variance
fn select_highly_variable_features(
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

/// Filter CSC matrix by selected row indices using pre-computed selection matrix
fn filter_csc_by_rows(selection_matrix: &CscMat, x: &CscMat) -> CscMat {
    // Filtered result = selection_matrix * x
    selection_matrix * x
}
