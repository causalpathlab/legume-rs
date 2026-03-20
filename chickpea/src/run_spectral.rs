use crate::common::*;
use data_beans::convert::try_open_or_convert;
use data_beans_alg::collapse_data::MultilevelParams;
use matrix_util::traits::IoOps;

#[derive(Args, Debug)]
pub struct SpectralArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Data files (sparse backends: zarr, h5)"
    )]
    data_files: Vec<Box<str>>,

    #[arg(long, short, required = true, help = "Output prefix")]
    out: Box<str>,

    #[arg(long, short = 'd', default_value_t = 20, help = "Embedding dimension")]
    embedding_dim: usize,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension"
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Sort dimension for cell collapsing"
    )]
    sort_dim: usize,

    #[arg(long, default_value_t = 10, help = "KNN for super-cell matching")]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Optimization iterations for collapsing"
    )]
    iter_opt: usize,

    #[arg(long, default_value_t = 100, help = "Block size for I/O")]
    block_size: usize,

    #[arg(long, help = "Max coarse features (default: no coarsening)")]
    max_coarse_features: Option<usize>,

    #[arg(
        long,
        default_value_t = 1e-1,
        help = "Regularization tau for Laplacian (fraction of mean degree)"
    )]
    tau_frac: f64,

    #[arg(long, default_value_t = false, help = "Preload data into memory")]
    preload: bool,
}

/// Bipartite spectral SVD on collapsed pseudobulk.
///
/// 1. Cell collapse: Sparse [D × N] → CollapsedOut [D × S]  (S << N)
/// 2. Optional feature coarsen: [D × S] → [D_l × S]
/// 3. Regularized Laplacian SVD → feature embeddings (U) + cell embeddings (V)
pub fn run_spectral(args: &SpectralArgs) -> anyhow::Result<()> {
    // 1. Load data
    let mut data_vec = SparseIoVec::new();
    for data_file in args.data_files.iter() {
        info!("Loading: {}", data_file);
        let mut data = try_open_or_convert(data_file)?;
        if args.preload {
            data.preload_columns()?;
        }
        data_vec.push(Arc::from(data), None)?;
    }

    let d_full = data_vec.num_rows();
    let n_full = data_vec.num_columns();
    info!("Data: {} features × {} cells", d_full, n_full);

    // 2. Random projection for cell collapsing
    let proj_dim = args.proj_dim.min(d_full);
    let proj_out = data_vec.project_columns_with_batch_correction::<Box<str>>(
        proj_dim,
        Some(args.block_size),
        None,
    )?;
    let proj_kn = proj_out.proj;
    info!("Projection: {} × {}", proj_kn.nrows(), proj_kn.ncols());

    // 3. Cell collapse → pseudobulk [D × S]
    let batch_membership: Vec<Box<str>> = vec!["batch0".into(); n_full];
    let collapsed = data_vec.collapse_columns_multilevel(
        &proj_kn,
        &batch_membership,
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: 1,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
        },
    )?;

    let pseudobulk = collapsed.mu_observed.posterior_mean(); // [D × S]
    let s = pseudobulk.ncols();
    info!(
        "Collapsed pseudobulk: {} features × {} super-cells",
        d_full, s
    );

    // 4. Optional feature coarsening
    let (work_matrix, feat_coarsening) =
        if let Some(max_feat) = args.max_coarse_features.filter(|&m| m < d_full) {
            let fc = compute_feature_coarsening(pseudobulk, max_feat)?;
            let coarsened = fc.aggregate_rows_ds(pseudobulk); // [D_l × S]
            info!(
                "Feature coarsening: {} → {} features",
                d_full, fc.num_coarse
            );
            (coarsened, Some(fc))
        } else {
            (pseudobulk.clone(), None)
        };

    let d_l = work_matrix.nrows();
    let n_l = work_matrix.ncols(); // = S (super-cells)
    info!("SVD input: {} features × {} super-cells", d_l, n_l);

    // 5. Regularized bipartite Laplacian: L_tau = D_row^(-1/2) A D_col^(-1/2)
    let mut row_deg = DVec::zeros(d_l);
    for i in 0..d_l {
        row_deg[i] = work_matrix.row(i).iter().sum();
    }
    let mean_row_deg = row_deg.mean();
    let tau_row = (args.tau_frac * mean_row_deg as f64) as f32;

    let mut col_deg = DVec::zeros(n_l);
    for j in 0..n_l {
        col_deg[j] = work_matrix.column(j).iter().sum();
    }
    let mean_col_deg = col_deg.mean();
    let tau_col = (args.tau_frac * mean_col_deg as f64) as f32;

    info!(
        "Mean degrees: row={:.1}, col={:.1}, tau_row={:.2}, tau_col={:.2}",
        mean_row_deg, mean_col_deg, tau_row, tau_col
    );

    let row_scale: Vec<f32> = (0..d_l)
        .map(|i| 1.0 / (row_deg[i] + tau_row).sqrt())
        .collect();
    let col_scale: Vec<f32> = (0..n_l)
        .map(|j| 1.0 / (col_deg[j] + tau_col).sqrt())
        .collect();

    let mut laplacian = work_matrix.clone();
    for i in 0..d_l {
        for j in 0..n_l {
            laplacian[(i, j)] *= row_scale[i] * col_scale[j];
        }
    }

    // 6. SVD
    let rank = args.embedding_dim.min(d_l).min(n_l);
    info!("Computing SVD with rank {} ...", rank);

    let (u_dl_r, s_r, v_nl_r) = laplacian.rsvd(rank)?;
    let r = s_r.len().min(u_dl_r.ncols()).min(v_nl_r.ncols());
    info!(
        "SVD shapes: U [{} × {}], S [{}], V [{} × {}], rank {}",
        u_dl_r.nrows(),
        u_dl_r.ncols(),
        s_r.len(),
        v_nl_r.nrows(),
        v_nl_r.ncols(),
        r
    );

    // Feature embeddings: U * S^(1/2)  [D_l × r]
    let mut feat_embed = Mat::zeros(d_l, r);
    for i in 0..d_l {
        for k in 0..r {
            feat_embed[(i, k)] = u_dl_r[(i, k)] * s_r[k].sqrt();
        }
    }

    // Super-cell embeddings: V * S^(1/2)  [S × r]
    let mut cell_embed = Mat::zeros(n_l, r);
    for j in 0..n_l {
        for k in 0..r {
            cell_embed[(j, k)] = v_nl_r[(j, k)] * s_r[k].sqrt();
        }
    }

    info!(
        "Embeddings: features [{} × {}], super-cells [{} × {}]",
        feat_embed.nrows(),
        feat_embed.ncols(),
        cell_embed.nrows(),
        cell_embed.ncols()
    );

    // 7. Save outputs
    let gene_names = data_vec.row_names()?;
    let feat_names: Vec<Box<str>> = if let Some(fc) = &feat_coarsening {
        (0..d_l)
            .map(|c| gene_names[fc.coarse_to_fine[c][0]].clone())
            .collect()
    } else {
        gene_names.clone()
    };

    let sc_names: Vec<Box<str>> = (0..n_l).map(|c| format!("sc_{}", c).into()).collect();

    feat_embed.to_parquet_with_names(
        &(args.out.to_string() + ".feature_embedding.parquet"),
        (Some(&feat_names), Some("feature")),
        None,
    )?;

    cell_embed.to_parquet_with_names(
        &(args.out.to_string() + ".cell_embedding.parquet"),
        (Some(&sc_names), Some("cell")),
        None,
    )?;

    let sv_mat = Mat::from_column_slice(r, 1, &s_r.as_slice()[..r]);
    sv_mat.to_parquet_with_names(
        &(args.out.to_string() + ".singular_values.parquet"),
        (None, None),
        None,
    )?;

    info!(
        "Done. Outputs: {}.{{feature,cell}}_embedding.parquet",
        args.out
    );

    Ok(())
}
