use crate::embed_common::*;
use crate::senna_input::*;
use data_beans::sparse_data_visitors::VisitColumnsOps;

#[derive(Args, Debug)]
pub struct SvdArgs {
    /// Data files
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = false)]
    ignore_batch_effects: bool,

    /// warm start from the previous projection (`cell x k`)
    #[arg(short = 'w', long = "warm-start")]
    warm_start_proj_file: Option<Box<str>>,

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 10)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// reference batch names
    #[arg(long, value_delimiter(','))]
    reference_batches: Option<Vec<Box<str>>>,

    // /// #downsampling columns per each collapsed sample. If None, no
    // /// downsampling.
    // #[arg(long, short = 's')]
    // down_sample: Option<usize>,
    /// optimization iterations
    #[arg(long, default_value_t = 30)]
    iter_opt: usize,

    /// block_size (# columns) for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of latent topics
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// preload all the columns data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// save batch-adjusted data
    #[arg(long)]
    save_adjusted: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
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
    } = read_sparse_data_with_membership(ReadArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;

    // 2. Random projection
    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file.as_deref() {
        use matrix_util::common_io::*;
        let ext = extension(proj_file)?;

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
                data_vec.num_rows()?,
                data_vec.num_columns()?,
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
        Some(args.block_size),
    )?;

    let cell_names = data_vec.column_names()?;
    let gene_names = data_vec.row_names()?;

    nystrom_out.latent_nk.to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".latent.parquet"),
    )?;

    nystrom_out.dictionary_dk.to_parquet(
        Some(&gene_names),
        None,
        &(args.out.to_string() + ".dictionary.parquet"),
    )?;

    Ok(())
}

#[allow(dead_code)]
struct NystromParam<'a> {
    dictionary_dk: &'a Mat,
    basis_dk: &'a Mat,
    delta_dp: Option<&'a Mat>,
}

pub struct NystromOut {
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
/// * `block_size` - online learning block size
///
///
fn do_nystrom_proj(
    log_xx_dn: Mat,
    delta_dp: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    rank: usize,
    block_size: Option<usize>,
) -> anyhow::Result<NystromOut> {
    let mut log_xx_dn = log_xx_dn.clone();

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

    let ntot = full_data_vec.num_columns()?;
    let kk = rank;

    let nystrom_param = NystromParam {
        dictionary_dk: &u_dk,
        basis_dk: &basis_dk,
        delta_dp,
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

    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

    x_dn.normalize_columns_inplace();

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
