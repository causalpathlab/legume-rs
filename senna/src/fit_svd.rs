use crate::embed_common::*;
use crate::routines_post_process::*;
use crate::routines_pre_process::*;
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

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// reference batch names
    #[arg(long, value_delimiter(','))]
    reference_batches: Option<Vec<Box<str>>>,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value_t = 15)]
    iter_opt: usize,

    /// block_size (# columns) for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of latent topics
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// save batch-adjusted data
    #[arg(long)]
    save_adjusted: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn fit_svd(args: &SvdArgs) -> anyhow::Result<()> {
    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
    } = read_sparse_data_with_membership(ReadArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
    })?;

    // 2. Random projection
    let proj_dim = args.proj_dim.max(args.n_latent_topics);

    let proj_out = data_vec.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(&batch_membership),
    )?;

    let proj_kn = proj_out.proj;
    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    let nsamp =
        data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    if !args.ignore_batch_effects {
        if !batch_membership.is_empty() {
            info!("Registering batch information");
            data_vec.build_hnsw_per_batch(&proj_kn, &batch_membership)?;
        }
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

    let batch_db = collapse_out.delta.as_ref();

    if let Some(delta_db) = batch_db.map(|x| x.posterior_mean()) {
        if args.save_adjusted {
            info!("Generating batch-adjusted data...");

            let triplets = triplets_adjusted_by_batch(&data_vec, delta_db)?;

            let mtx_shape = (
                data_vec.num_rows()?,
                data_vec.num_columns()?,
                triplets.len(),
            );

            let backend_file = args.out.to_string() + ".adjusted.zarr";
            let backend = SparseIoBackend::Zarr;
            remove_file(&backend_file)?;

            let mut adjusted_data = create_sparse_from_triplets(
                triplets,
                mtx_shape,
                Some(&backend_file),
                Some(&backend),
            )?;

            adjusted_data.register_row_names_vec(&data_vec.row_names()?);
            adjusted_data.register_column_names_vec(&data_vec.column_names()?);

            info!("Batch-adjusted backend: {}", backend_file);
        }
    }

    if let Some(batch_db) = batch_db {
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
        batch_db.map(|x| x.posterior_mean()),
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

fn nystrom_proj_visitor(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    proj_basis: &NystromParam,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let u_dk = proj_basis.dictionary_dk;
    let delta_db = proj_basis.batch_db;

    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

    x_dn.normalize_columns_inplace();

    if let Some(delta_db) = delta_db {
        let batches = full_data_vec.get_batch_membership(lb..ub);
        x_dn.adjust_by_division_of_selected_inplace(delta_db, &batches);
    }

    x_dn.values_mut().iter_mut().for_each(|x| {
        *x = (*x + 1.).ln();
    });

    x_dn.scale_columns_inplace();

    let chunk = (x_dn.transpose() * u_dk).transpose();

    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in nystrom");

    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

struct NystromParam<'a> {
    dictionary_dk: &'a Mat,
    batch_db: Option<&'a Mat>,
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
    delta_db: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    rank: usize,
    block_size: Option<usize>,
) -> anyhow::Result<NystromOut> {
    let mut log_xx_dn = log_xx_dn.clone();

    log_xx_dn.scale_columns_inplace();

    let (u_dk, _, _) = log_xx_dn.rsvd(rank)?;

    info!(
        "Constructed {} x {} projection matrix",
        u_dk.nrows(),
        u_dk.ncols()
    );

    let ntot = full_data_vec.num_columns()?;
    let kk = rank;

    let nystrom_param = NystromParam {
        dictionary_dk: &u_dk,
        batch_db: delta_db,
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
