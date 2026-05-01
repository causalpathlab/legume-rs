use crate::embed_common::*;
use crate::hvg::{select_hvg_streaming, HvgCliArgs, HvgSelection};
use crate::senna_input::{read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch};
use data_beans::sparse_data_visitors::VisitColumnsOps;

#[derive(Args, Debug)]
pub struct SvdArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Multiple files may be passed (comma- or space-separated)\n\
                     and are concatenated column-wise on a shared feature set."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet         gene × component loadings\n  \
                     {out}.latent.parquet             cell × component scores\n  \
                     {out}.delta.parquet              per-batch effects (if --batch-files)\n  \
                     {out}.adjusted.zarr              batch-adjusted backend (if --save-adjusted)\n  \
                     {out}.selected_features.txt      selected HVG names (if HVG enabled)\n  \
                     {out}.cell_proj.parquet          cached random projection (consumed by `senna layout`)\n  \
                     {out}.senna.json                 run manifest consumed by `senna viz --from` and `senna plot --from`"
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
        long,
        help = "Skip per-batch correction; treat all cells as a single batch",
        long_help = "Collapses batch membership to a single label so the random\n\
                     projection, multilevel collapsing, and δ estimation are all\n\
                     run as if there were no batch structure. Useful when batches\n\
                     are believed homogeneous, or for reference-only baselines."
    )]
    ignore_batch: bool,

    #[arg(
        short = 'w',
        long = "warm-start",
        help = "Warm-start projection file (cell × k)",
        long_help = "Skip random projection and use this matrix instead.\n\
                     Rows must match the concatenated cell order of the inputs.\n\
                     Disables feature selection."
    )]
    warm_start_proj_file: Option<Box<str>>,

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
        default_value_t = 3,
        help = "Multi-level coarsening levels",
        long_help = "Hierarchical pseudobulk refinement passes. Level sort dims are\n\
                     linearly spaced from 4 to --sort-dim. Set to 1 to disable."
    )]
    num_levels: usize,

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
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column-sum normalization scale",
        long_help = "Target library size after per-cell normalization."
    )]
    column_sum_norm: f32,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent components (K)"
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training"
    )]
    preload_data: bool,

    #[arg(long, help = "Write the batch-adjusted data to a new zarr backend")]
    save_adjusted: bool,

    #[command(flatten)]
    hvg: HvgCliArgs,

    #[command(flatten)]
    cnv: CnvArgs,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of k-means cell clusters used as cell-type proxy for CNV."
    )]
    cnv_svd_clusters: usize,
}

pub fn fit_svd(args: &SvdArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: mut batch_membership,
        ..
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;
    if args.ignore_batch {
        info!("--ignore-batch: collapsing all cells to a single batch");
        crate::senna_input::collapse_to_single_batch(&mut batch_membership);
    }

    // HVG selection (skipped with --warm-start for compatibility)
    let hvg_enabled = args.hvg.n_hvg > 0 || args.hvg.feature_list_file.is_some();
    let selected_features: Option<HvgSelection> = if args.warm_start_proj_file.is_some() {
        if hvg_enabled {
            info!("Warm-start provided: skipping HVG selection (may be incompatible)");
        }
        None
    } else if hvg_enabled {
        Some(select_hvg_streaming(
            &data_vec,
            (args.hvg.n_hvg > 0).then_some(args.hvg.n_hvg),
            args.hvg.feature_list_file.as_deref(),
            args.block_size,
        )?)
    } else {
        None
    };

    // 2. Random projection
    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file.as_deref() {
        use matrix_util::common_io::file_ext;
        let ext = file_ext(proj_file)?;

        let MatWithNames {
            rows: cell_names,
            cols: _,
            mat: proj_nk,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_row_names(proj_file, Some(0))?,
            _ => Mat::read_data_with_names(proj_file, &['\t', ',', ' '], Some(0), Some(0))?,
        };

        if data_vec.column_names()? != cell_names {
            return Err(anyhow::anyhow!(
                "warm start projection rows don't match with the data"
            ));
        }

        proj_nk.transpose()
    } else {
        let proj_dim = args.proj_dim.max(args.n_latent_topics);
        let n_rows = data_vec.num_rows();

        let proj_out = if let Some(sel) = selected_features.as_ref() {
            let weights = sel.row_weights(n_rows);
            data_vec.project_columns_weighted(
                proj_dim,
                args.block_size,
                Some(&batch_membership),
                &weights,
            )?
        } else {
            data_vec.project_columns_with_batch_correction(
                proj_dim,
                args.block_size,
                Some(&batch_membership),
            )?
        };

        proj_out.proj
    };

    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    // 3. Batch-adjusted collapsing (pseudobulk)
    let collapse_out = data_vec.collapse_columns_multilevel(
        &proj_kn,
        &batch_membership,
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
            refine: Some(data_beans_alg::refine_multilevel::RefineParams {
                gene_weighting: args.refine_weighting.into(),
                ..data_beans_alg::refine_multilevel::RefineParams::default()
            }),
        },
    )?;

    // 4. batch-adjusted data
    let batch_dp = collapse_out.mu_residual.as_ref();

    if let Some(delta_dp) = batch_dp.map(matrix_param::traits::Inference::posterior_mean) {
        info!("{} x {}", delta_dp.nrows(), delta_dp.ncols());

        if args.save_adjusted {
            info!("Generating batch-adjusted data...");

            let triplets = triplets_adjusted_by_pseudobulk(&data_vec, delta_dp)?;

            let mtx_shape = (data_vec.num_rows(), data_vec.num_columns(), triplets.len());

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

            info!("Batch-adjusted backend: {backend_file}");
        }
    }

    if let Some(batch_db) = collapse_out.delta {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_melted_parquet(
            &outfile,
            (Some(&gene_names), Some("gene")),
            (batch_names.as_deref(), Some("batch")),
        )?;
    }

    // 4b. Load gene positions for CNV (if requested)
    let cnv_positions = {
        let gene_names = data_vec.row_names()?;
        crate::cnv_pseudobulk::load_gene_positions(&args.cnv, &gene_names)?
    };

    // 5. Nystrom projection
    let x_dn = match collapse_out.mu_adjusted.as_ref() {
        Some(adj) => adj,
        None => &collapse_out.mu_observed,
    };

    let nystrom_out = do_nystrom_proj(
        x_dn.posterior_log_mean().clone(),
        batch_dp.map(matrix_param::traits::Inference::posterior_mean),
        &data_vec,
        args.n_latent_topics,
        args.column_sum_norm,
        args.block_size,
    )?;

    let cell_names = data_vec.column_names()?;
    let gene_names = data_vec.row_names()?;
    let output_gene_names = gene_names.clone();

    // SVD reuses the topic models' `T{c}` convention so `senna plot
    // --colour-by topic` reads the latent.parquet identically regardless
    // of upstream (`senna topic` or `senna svd`).
    let component_col_names = axis_id_names("T", nystrom_out.latent_nk.ncols());
    nystrom_out.latent_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&component_col_names),
    )?;

    nystrom_out.dictionary_dk.to_parquet_with_names(
        &(args.out.to_string() + ".dictionary.parquet"),
        (Some(&output_gene_names), Some("gene")),
        Some(&component_col_names),
    )?;

    {
        let pb_gene_gp: Mat = x_dn.posterior_mean().clone();
        let n_pb = pb_gene_gp.ncols();
        let pb_names: Vec<Box<str>> = (0..n_pb)
            .map(|i| format!("PB_{i}").into_boxed_str())
            .collect();
        pb_gene_gp.to_parquet_with_names(
            &format!("{}.pb_gene.parquet", args.out),
            (Some(&output_gene_names), Some("gene")),
            Some(&pb_names),
        )?;
    }

    // Save selected feature list if feature selection was applied
    if let Some(sel) = &selected_features {
        use matrix_util::common_io::write_lines;
        let feature_file = args.out.to_string() + ".selected_features.txt";
        write_lines(&sel.selected_names, &feature_file)?;
        info!(
            "Saved {} selected features to {}",
            sel.selected_names.len(),
            feature_file
        );
    }

    // 6. Cluster-informed CNV detection (after SVD, using latent for clustering)
    if let Some(positions) = cnv_positions {
        let cnv_config = crate::cnv_pseudobulk::build_cnv_config(&args.cnv);

        let cnv_result = crate::cnv_pseudobulk::detect_cnv_cluster_informed(
            data_vec,
            &nystrom_out.latent_nk,
            &batch_membership,
            &positions,
            args.cnv_svd_clusters.max(2),
            &cnv_config,
        )?;

        crate::cnv_pseudobulk::write_cnv_results(&cnv_result, &args.out, &gene_names)?;
    }

    crate::postprocess::viz_prep::write_cell_proj(&args.out, &proj_kn, &cell_names)?;

    let input: Vec<String> = args.data_files.iter().map(|s| s.to_string()).collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::run_manifest::RunKind::Svd,
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
        // SVD produces no topic / cluster labels on its own; users
        // typically run `senna clustering` next, so the viz column
        // `cluster` is the natural default.
        default_colour_by: "cluster",
    })?;

    Ok(())
}

#[allow(dead_code)]
struct NystromParam<'a> {
    dictionary_dk: &'a Mat,
    basis_dk: &'a Mat,
    delta_dp: Option<&'a Mat>,
    column_sum_norm: f32,
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
) -> anyhow::Result<NystromOut> {
    let mut log_xx_dn = log_xx_dn.clone();

    log_xx_dn.scale_columns_inplace();

    let (u_dk, s_k, _) = log_xx_dn.rsvd(rank)?;
    let basis_dk = nystrom_basis(&u_dk, &s_k);

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

#[allow(clippy::type_complexity)]
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
