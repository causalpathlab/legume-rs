use crate::embed_common::*;
use crate::senna_input::{
    read_data_on_shared_columns, ReadSharedColumnsArgs, SparseStackWithBatch,
};
use data_beans::sparse_data_visitors::VisitColumnsOps;
use matrix_util::dmatrix_util::concatenate_vertical;

#[derive(Args, Debug)]
pub struct JointSvdArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5), row-major (modality × batch)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Files are arranged as a row-major (modality × batch) table;\n\
                     use -m to set the number of modality rows."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        short = 'm',
        long = "modalities",
        required = true,
        help = "Number of modalities (rows of the data-file table)",
        long_help = "The input files are interpreted row-major as modality × batch.\n\
                     This value sets the number of modality rows."
    )]
    num_modalities: usize,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet  gene × component loadings\n  \
                     {out}.latent.parquet      cell × component scores\n  \
                     {out}.cell_proj.parquet   cached random projection (consumed by `senna layout`)\n  \
                     {out}_{d}.delta.parquet   per-batch effects for modality d\n  \
                     {out}.senna.json          run manifest consumed by `senna viz --from` and `senna plot --from`"
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Target rank of the initial random sketch used to seed\n\
                     batch correction and multi-level pseudobulk collapsing."
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Partition depth: ≤ 2^d + 1 pseudobulk groups",
        long_help = "Binary-tree partitioning over the top d projection components.\n\
                     Produces at most 2^d + 1 pseudobulk leaves."
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell in the same order as its\n\
                     matching data file. Example: batch1.tsv,batch2.tsv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column-sum normalization scale",
        long_help = "Target library size after per-cell normalization."
    )]
    column_sum_norm: f32,

    #[arg(
        long,
        default_value_t = 10,
        help = "In-batch k-NN for super-cell merging",
        long_help = "Number of within-batch nearest neighbours used when\n\
                     aggregating cells into pseudobulk super-cells."
    )]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Batch-correction optimizer iterations",
        long_help = "Coordinate-descent steps when fitting the per-batch delta."
    )]
    iter_opt: usize,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    block_size: Option<usize>,

    #[arg(
        long = "weighting",
        value_enum,
        default_value_t = crate::refine_weighting::WeightingArg::NbFisherInfo,
        help = crate::refine_weighting::WEIGHTING_HELP,
    )]
    refine_weighting: crate::refine_weighting::WeightingArg,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent components (K)"
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Multi-level coarsening levels",
        long_help = "Hierarchical pseudobulk refinement passes. Level sort dims are\n\
                     linearly spaced from 4 to --sort-dim. Set to 1 to disable."
    )]
    num_levels: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training"
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
        args.block_size,
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
            oversample: false,
            refine: Some(data_beans_alg::refine_multilevel::RefineParams {
                gene_weighting: args.refine_weighting.into(),
                ..data_beans_alg::refine_multilevel::RefineParams::default()
            }),
        },
    )?;

    // 4. output batch effect information
    for (d, collapsed) in collapsed_data_vec.iter().enumerate() {
        if let Some(batch_db) = &collapsed.delta {
            let outfile = format!("{}_{}.delta.parquet", args.out, d);
            let data_vec = &data_stack.stack[d];
            let batch_names = data_vec.batch_names();
            let gene_names = data_vec.row_names()?;
            batch_db.to_melted_parquet(
                &outfile,
                (Some(&gene_names), Some("gene")),
                (batch_names.as_deref(), Some("batch")),
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
        .map(|x| {
            x.mu_residual
                .as_ref()
                .map(matrix_param::traits::Inference::posterior_mean)
        })
        .collect::<Vec<_>>();

    let nystrom_out = do_nystrom_proj(
        log_xx_vec,
        delta_vec,
        &data_stack,
        n_topics,
        args.column_sum_norm,
        args.block_size,
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

    // Modality-0 only — joint multi-modality annotation is a follow-up.
    {
        let x_dn = match collapsed_data_vec[0].mu_adjusted.as_ref() {
            Some(adj) => adj,
            None => &collapsed_data_vec[0].mu_observed,
        };
        let pb_gene_gp: Mat = x_dn.posterior_mean().clone();
        let n_pb = pb_gene_gp.ncols();
        let pb_names: Vec<Box<str>> = (0..n_pb)
            .map(|i| format!("PB_{i}").into_boxed_str())
            .collect();
        let gene_names_0: Vec<Box<str>> = data_stack.stack[0].row_names()?;
        pb_gene_gp.to_parquet_with_names(
            &format!("{}.pb_gene.parquet", args.out),
            (Some(&gene_names_0), Some("gene")),
            Some(&pb_names),
        )?;
    }

    crate::postprocess::viz_prep::write_cell_proj(&args.out, &proj_kn, &cell_names)?;

    let input: Vec<String> = args.data_files.iter().map(|s| s.to_string()).collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: "joint-svd",
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: true,
        pb_gene_suffix: Some("pb_gene.parquet"),
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        default_colour_by: "cluster",
    })?;

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
        .map(matrix_util::traits::MatOps::scale_columns)
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
