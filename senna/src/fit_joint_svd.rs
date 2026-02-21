use crate::embed_common::*;
use crate::senna_input::*;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use matrix_util::dmatrix_util::concatenate_vertical;

#[derive(Args, Debug)]
pub struct JointSvdArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided (space or comma separated)."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        short = 'm',
        long = "modalities",
        help = "Data modalities",
        long_help = "We will treat the provided data files as\n\
		     a table of data sets in a row-major order.\n\
		     This number of modalities will determine \n\
		     how many different data types are assumed,\n\
		     or the number of rows in the data table.",
        required = true
    )]
    num_modalities: usize,

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
        help = "Random projection dimension.",
        long_help = "Random projection dimension to project the data.\n\
		     Controls the dimensionality of the random projection step."
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection.",
        long_help = "Use top {d} components of projection.\n\
		     Number of samples will be less than `2^{d}+1`."
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files.",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.csv,batch2.csv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column sum normalization scale.",
        long_help = "Column sum normalization scale (affects decoder only).\n\
		     Adjusts normalization of columns in the decoder."
    )]
    column_sum_norm: f32,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of k-nearest neighbour batches.",
        long_help = "Number of k-nearest neighbour batches.\n\
		     Controls the number of batches considered \n\
		     for nearest neighbour search."
    )]
    knn_batches: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch.",
        long_help = "Number of k-nearest neighbours within each batch.\n\
		     Controls the number of cells considered \n\
		     for nearest neighbour search within each batch."
    )]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Optimization iterations.",
        long_help = "Number of optimization iterations.\n\
		     Controls the number of steps for model optimization."
    )]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing.",
        long_help = "Block size (number of columns) for parallel processing.\n\
		     Controls the granularity of parallel computation."
    )]
    block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics.",
        long_help = "Number of latent topics.\n\
		     Controls the dimensionality of the latent topic space."
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Number of multi-level collapsing levels.",
        long_help = "Number of multi-level collapsing levels.\n\
		     More levels = coarser-to-finer batch correction.\n\
		     Set to 1 to disable multi-level."
    )]
    num_levels: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all columns data.",
        long_help = "Preload all the columns data into memory.\n\
		     Improves performance for large datasets."
    )]
    preload_data: bool,
}

pub fn fit_joint_svd(args: &JointSvdArgs) -> anyhow::Result<()> {
    // 1. Read the data with batch membership
    let SparseStackWithBatch {
        mut data_stack,
        batch_stack,
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

    // 3. Batch-adjusted multilevel collapsing (pseudobulk)
    info!(
        "Multi-level collapsing across {} modalities ...",
        data_stack.num_types()
    );

    let collapsed_data_vec: Vec<CollapsedOut> = data_stack.collapse_columns_multilevel(
        &proj_kn,
        batch_stack[0].as_ref(),
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
        },
    )?;

    // 4. output batch effect information
    for (d, collapsed) in collapsed_data_vec.iter().enumerate() {
        if let Some(batch_db) = &collapsed.delta {
            let outfile = format!("{}_{}.delta.parquet", args.out, d);
            let data_vec = &data_stack.stack[d];
            let batch_names = data_vec.batch_names();
            let gene_names = data_vec.row_names()?;
            batch_db.to_parquet_with_names(
                &outfile,
                (Some(&gene_names), Some("gene")),
                batch_names.as_deref(),
            )?;
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
        args.column_sum_norm,
        Some(args.block_size),
    )?;

    let cell_names = data_stack.column_names()?;
    let gene_names = data_stack.row_names()?;

    nystrom_out.latent_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    nystrom_out.dictionary_dk.to_parquet_with_names(
        &(args.out.to_string() + ".dictionary.parquet"),
        (Some(&gene_names), Some("gene")),
        None,
    )?;

    Ok(())
}

fn do_nystrom_proj(
    log_xx_dn_vec: Vec<&Mat>,
    delta_dp_vec: Vec<Option<&Mat>>,
    full_data: &SparseIoStack,
    rank: usize,
    column_sum_norm: f32,
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

        let nystrom_param = NystromParam {
            basis_dk,
            delta_dp,
            column_sum_norm,
        };

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
    column_sum_norm: f32,
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
    let column_sum_norm = proj_basis.column_sum_norm;

    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

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
