use crate::srt_cell_pairs::*;
use crate::srt_collapse_pairs::*;
use crate::srt_common::*;
use crate::srt_input::*;
use crate::srt_random_projection::*;
use crate::srt_vertex_propensity::SrtVertPropOps;
use clap::Parser;
use matrix_param::traits::*;

#[derive(Parser, Debug, Clone)]
///
/// PINTO by Singular Value Decomposition
///
pub struct SrtSvdArgs {
    /// Data files of either `.zarr` or `.h5` format. All the formats
    /// in the given list should be identical. We can convert `.mtx`
    /// to `.zarr` or `.h5` using `data-beans from-mtx` command.
    #[arg(required = true, value_delimiter(','))]
    data_files: Vec<Box<str>>,

    /// An auxiliary cell coordinate file. Each coordinate file should
    /// correspond to each data file. Each line contains barcode, x, y, ...
    /// coordinates. We could include more columns.
    #[arg(long = "coord", short = 'c', required = true, value_delimiter(','))]
    coord_files: Vec<Box<str>>,

    /// Indicate the cell coordinate columns in the `coord` files (comma separated)
    #[arg(long = "coord_column_indices", value_delimiter(','))]
    coord_columns: Option<Vec<usize>>,

    /// The columns names in the `coord` files (comma separated)
    #[arg(
        long = "coord_column_names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres"
    )]
    coord_column_names: Vec<Box<str>>,

    /// Coordinate embedding dimension
    #[arg(long, default_value_t = 256)]
    coord_emb: usize,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short = 'b', value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// #k-nearest neighbours for spectral embedding for spatial coordinates
    #[arg(short = 'k', long, default_value_t = 10)]
    knn_spatial: usize,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of latent topics
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// number of (edge) clusters
    #[arg(long)]
    n_edge_clusters: Option<usize>,

    /// number of (edge) clusters
    #[arg(long, default_value_t = 100)]
    maxiter_clustering: usize,

    /// preload all the columns data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

/// Fits SVD and write down the dictionary matrix and pair-level
/// latent states.
pub fn fit_srt_svd(args: &SrtSvdArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }

    info!("Reading data files...");

    let SRTData {
        data,
        coordinates,
        coordinate_names,
        batches,
    } = read_data_vec(SRTReadArgs {
        data_files: args.data_files.clone(),
        coord_files: args.coord_files.clone(),
        preload_data: args.preload_data,
        coord_columns: args.coord_columns.clone().unwrap_or_default(),
        coord_column_names: args.coord_column_names.clone(),
        batch_files: args.batch_files.clone(),
    })?;

    let gene_names = data.row_names()?;
    let cell_names = data.column_names()?;

    info!("Constructing spatial nearest neighbourhood graphs");
    let mut srt_cell_pairs = SrtCellPairs::new(
        &data,
        &coordinates,
        SrtCellPairsArgs {
            knn: args.knn_spatial,
            coordinate_emb_dim: args.coord_emb,
            block_size: args.block_size,
        },
    )?;

    let proj_out =
        srt_cell_pairs.random_projection(args.proj_dim, args.block_size, Some(&batches))?;

    srt_cell_pairs.assign_pairs_to_samples(
        &proj_out,
        Some(args.sort_dim),
        args.down_sample.clone(),
    )?;

    info!("Collecting summary statistics across cell pairs...");
    let collapsed = srt_cell_pairs.collapse_pairs()?;
    let params = collapsed.optimize(None)?;

    info!("Randomized SVD on the Δ matrix");
    let training_dm = concatenate_horizontal(&[
        params.left_delta.posterior_log_mean().clone(),
        params.right_delta.posterior_log_mean().clone(),
    ])?
    .scale_columns();

    let (u_dk, s_k, _) = training_dm.rsvd(args.n_latent_topics)?;
    let eps = 1e-8;
    let sinv_k = DVec::from_iterator(s_k.len(), s_k.iter().map(|&s| 1.0 / (s + eps)));

    info!("Nystrom projection...");
    let mut proj_kn = Mat::zeros(args.n_latent_topics, srt_cell_pairs.num_pairs());

    let basis_dk = &u_dk * Mat::from_diagonal(&sinv_k);

    srt_cell_pairs.visit_pairs_by_block(
        &nystrom_proj_visitor,
        &basis_dk,
        &mut proj_kn,
        args.block_size,
    )?;

    proj_kn
        .transpose()
        .to_parquet(None, None, &(args.out.to_string() + ".latent.parquet"))?;

    u_dk.to_parquet(
        Some(&gene_names),
        None,
        &(args.out.to_string() + ".dictionary.parquet"),
    )?;

    srt_cell_pairs.to_parquet(
        &(args.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    collapsed.to_parquet(
        &(args.out.to_string() + ".collapsed_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    info!("clustering edges");
    let num_clusters = args.n_edge_clusters.unwrap_or(args.n_latent_topics);

    let edge_membership = proj_kn.kmeans_columns(KmeansArgs {
        num_clusters,
        max_iter: args.maxiter_clustering,
    });

    info!("calibrating propensity");
    let prop_kn = srt_cell_pairs.vertex_propensity(&edge_membership, args.block_size)?;

    prop_kn.transpose().to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".propensity.parquet"),
    )?;

    info!("Done");
    Ok(())
}

fn nystrom_proj_visitor(
    bound: (usize, usize),
    data: &SrtCellPairs,
    basis_dk: &Mat,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let left = pairs.into_iter().map(|pp| pp.left);
    let right = pairs.into_iter().map(|pp| pp.right);

    let y_left_nk = data
        .data
        .read_columns_csc(left)?
        .log1p()
        .scale_columns()
        .transpose()
        * basis_dk;
    let y_right_nk = data
        .data
        .read_columns_csc(right)?
        .log1p()
        .scale_columns()
        .transpose()
        * basis_dk;

    let chunk_kn = (y_left_nk.scale(0.5) + y_right_nk.scale(0.5)).transpose();

    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in nystrom");

    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk_kn);

    Ok(())
}
