use crate::embed_common::*;
use crate::senna_input::*;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use matrix_util::dmatrix_util::concatenate_vertical;

#[derive(Args, Debug)]
pub struct JointSvdArgs {
    /// Data files
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Data modalities
    #[arg(short = 'm', long, required = true)]
    num_modalities: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// column sum normalization scale (will only affect decoder)
    #[arg(short = 'c', long, default_value_t = 1e4)]
    column_sum_norm: f32,

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

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

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn fit_joint_svd(args: &JointSvdArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 1. Read the data with batch membership
    let SparseStackWithBatch {
        mut data_stack,
        batch_stack,
        nbatch_stack,
    } = read_data_on_shared_columns(ReadSharedColumnsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        num_types: args.num_modalities,
        preload: args.preload_data,
    })?;

    // 2. Concatenate projections
    let proj_dim = args.proj_dim.max(args.n_latent_topics);
    let proj_out = data_stack.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(batch_stack[0].as_ref()),
    )?;
    let proj_kn = proj_out.proj;

    // 3. Batch-adjusted collapsing (pseudobulk)
    // assign pseudobulk samples by proj_kn
    let nsamp = data_stack.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), None)?;

    for (d, data_vec) in data_stack.stack.iter_mut().enumerate() {
        let nbatch = nbatch_stack[d];
        let batch_membership = &batch_stack[d];

        if nbatch > 1 {
            info!("Registering batch information");
            data_vec.build_hnsw_per_batch(&proj_kn, batch_membership)?;
        }
    }

    info!("Collapsing columns into {} pseudobulk samples ...", nsamp);

    let collapsed_data_vec = data_stack
        .stack
        .iter()
        .map(|x| {
            x.collapse_columns(
                Some(args.knn_batches),
                Some(args.knn_cells),
                None,
                Some(args.iter_opt),
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // 4. output batch effect information
    for (d, collapsed) in collapsed_data_vec.iter().enumerate() {
        if let Some(batch_db) = &collapsed.delta {
            let outfile = format!("{}_{}.delta.parquet", args.out, d);
            let data_vec = &data_stack.stack[d];
            let batch_names = data_vec.batch_names();
            let gene_names = data_vec.row_names()?;
            batch_db.to_parquet(Some(&gene_names), batch_names.as_deref(), &outfile)?;
        }
    }

    // 5. Nystrom projection for each modality
    let n_topics = args.n_latent_topics;
    let log_xx_vec = collapsed_data_vec
        .iter()
        .map(|x| x.mu_observed.posterior_log_mean())
        .collect::<Vec<_>>();

    let delta_vec = collapsed_data_vec
        .iter()
        .map(|x| x.mu_residual.as_ref().map(|y| y.posterior_mean()))
        .collect::<Vec<_>>();

    let nystrom_out = do_nystrom_proj(
        log_xx_vec,
        delta_vec,
        &data_stack,
        n_topics,
        Some(args.block_size),
    )?;

    let cell_names = data_stack.column_names()?;
    let gene_names = data_stack.row_names()?;

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

fn do_nystrom_proj(
    log_xx_dn_vec: Vec<&Mat>,
    delta_dp_vec: Vec<Option<&Mat>>,
    full_data: &SparseIoStack,
    rank: usize,
    block_size: Option<usize>,
) -> anyhow::Result<NystromOut> {
    // 1. construct a tall xx matrix and perform one svd
    let xx_vec = log_xx_dn_vec
        .into_iter()
        .map(|x| x.scale_columns())
        .collect::<Vec<_>>();

    let (u_dk, _, vv) = concatenate_vertical(&xx_vec)?.rsvd(rank)?;

    info!("{} x {}", vv.nrows(), vv.ncols());

    // 2. calibrate basis_dk for each modality
    let basis_vec = xx_vec.iter().map(|x| x * &vv).collect::<Vec<_>>();

    // 3. revisit data with basis_dk
    let ntot = full_data.num_columns()?;
    let kk = rank;
    let mut proj_tot_kn = Mat::zeros(kk, ntot);

    for d in 0..full_data.stack.len() {
        let data_vec = &full_data.stack[d];
        let basis_dk = &basis_vec[d];
        let delta_dp = delta_dp_vec[d];

        let nystrom_param = NystromParam { basis_dk, delta_dp };

        let mut proj_kn = Mat::zeros(kk, ntot);
        data_vec.visit_columns_by_block(
            &nystrom_proj_visitor,
            &nystrom_param,
            &mut proj_kn,
            block_size,
        )?;

        proj_tot_kn += &proj_kn;
    }

    let latent_nk = proj_tot_kn.transpose();
    let dictionary_dk = u_dk;

    Ok(NystromOut {
        dictionary_dk,
        latent_nk,
    })
}

#[allow(dead_code)]
struct NystromParam<'a> {
    basis_dk: &'a Mat,
    delta_dp: Option<&'a Mat>,
}

struct NystromOut {
    pub dictionary_dk: Mat,
    pub latent_nk: Mat,
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
