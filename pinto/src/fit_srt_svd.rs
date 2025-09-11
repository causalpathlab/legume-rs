use crate::srt_cell_pairs::SrtCellPairs;
use crate::srt_collapse_pairs::*;
use crate::srt_common::*;
use crate::srt_random_projection::*;
use crate::srt_routines_latent_representation::*;
use crate::srt_routines_post_process::*;
use crate::srt_routines_pre_process::*;
use candle_util::candle_matched_data_loader::DataLoaderArgs;
use clap::{Parser, ValueEnum};
use data_beans_alg::random_projection::*;
use matrix_param::{
    io::ParamIo,
    traits::{Inference, TwoStatParam},
};

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

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    // /// optimization iterations
    // #[arg(long, default_value_t = 15)]
    // iter_opt: usize,
    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Block_size for parallel processing
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

    info!("Constructing spatial nearest neighbourhood graphs");
    let mut srt_cell_pairs =
        SrtCellPairs::new(&data, &coordinates, args.knn_spatial, Some(args.block_size))?;

    let proj_out =
        srt_cell_pairs.random_projection(args.proj_dim, args.block_size, Some(&batches))?;

    srt_cell_pairs.assign_pairs_to_samples(
        &proj_out,
        Some(args.sort_dim),
        args.down_sample.clone(),
    )?;

    info!("Collecting cell pairs...");
    let collapsed = srt_cell_pairs.collapse_pairs()?;
    let params = collapsed.optimize(None)?;

    info!("Randomized SVD on the Δ matrix");

    let log_x_delta_d2m = concatenate_horizontal(&[
        params.left_delta.posterior_log_mean().clone(),
        params.right_delta.posterior_log_mean().clone(),
    ])?
    .scale_columns();

    // todo: delta is the most informative...

    let log_x_marginal_d2m = concatenate_horizontal(&[
        params.left.posterior_log_mean().clone(),
        params.right.posterior_log_mean().clone(),
    ])?
    .scale_columns();

    let (dictionary_marginal_dk, _, _) = log_x_marginal_d2m.rsvd(args.n_latent_topics)?;

    let (dictionary_delta_dk, _, _) = log_x_delta_d2m.rsvd(args.n_latent_topics)?;

    info!("Nystrom projection...");
    let nystrom_param = NystromParam {
        dictionary_marginal_dk: &dictionary_delta_dk,
        dictionary_delta_dk: &dictionary_marginal_dk,
    };

    let mut nystrom_proj = NystromProj {
        marginal_kn: Mat::zeros(args.n_latent_topics, srt_cell_pairs.num_pairs()),
        delta_kn: Mat::zeros(args.n_latent_topics, srt_cell_pairs.num_pairs()),
    };

    srt_cell_pairs.visit_pairs_by_block(
        &nystrom_proj_visitor,
        &nystrom_param,
        &mut nystrom_proj,
        args.block_size,
    )?;

    let latent_marginal_nk = nystrom_proj.marginal_kn.transpose();
    let latent_delta_nk = nystrom_proj.delta_kn.transpose();

    latent_delta_nk.to_parquet(
        None,
        None,
        &(args.out.to_string() + ".latent_delta.parquet"),
    )?;

    latent_marginal_nk.to_parquet(
        None,
        None,
        &(args.out.to_string() + ".latent_marginal.parquet"),
    )?;

    dictionary_marginal_dk.to_parquet(
        Some(&gene_names),
        None,
        &(args.out.to_string() + ".dictionary_marginal.parquet"),
    )?;

    dictionary_delta_dk.to_parquet(
        Some(&gene_names),
        None,
        &(args.out.to_string() + ".dictionary_delta.parquet"),
    )?;

    srt_cell_pairs.to_parquet(
        &(args.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names),
    )?;

    info!("Done");
    Ok(())
}

struct NystromParam<'a> {
    dictionary_marginal_dk: &'a Mat,
    dictionary_delta_dk: &'a Mat,
}

struct NystromProj {
    marginal_kn: Mat,
    delta_kn: Mat,
}

fn nystrom_proj_visitor(
    bound: (usize, usize),
    data: &SrtCellPairs,
    proj_basis: &NystromParam,
    arc_proj: Arc<Mutex<&mut NystromProj>>,
) -> anyhow::Result<()> {
    let marginal_dk = proj_basis.dictionary_marginal_dk;
    let delta_dk = proj_basis.dictionary_delta_dk;

    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let left = pairs.into_iter().map(|pp| pp.left);
    let right = pairs.into_iter().map(|pp| pp.right);

    // todo: we can feed delta here?
    let y_left_nk = data
        .data
        .read_columns_csc(left)?
        .log1p()
        .scale_columns()
        .transpose()
        * delta_dk;
    let y_right_nk = data
        .data
        .read_columns_csc(right)?
        .log1p()
        .scale_columns()
        .transpose()
        * delta_dk;

    let marginal_kn = (y_left_nk.scale(0.5) + y_right_nk.scale(0.5)).transpose();

    ////////////////////////////////////////////////////
    // imputation by neighbours and update statistics //
    ////////////////////////////////////////////////////

    let pairs_neighbours = &data.pairs_neighbours[lb..ub];

    let y_delta_left = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, n)| -> anyhow::Result<Mat> {
            let left = pairs[j].left;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(left))?;
            let y_neigh_dm = data.data.read_columns_csc(n.right_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);
            // todo: we are applying marginal dk
            Ok(y_d1.log1p().scale_columns().transpose() * delta_dk)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let y_delta_right = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, n)| -> anyhow::Result<Mat> {
            let right = pairs[j].right;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(right))?;
            let y_neigh_dm = data.data.read_columns_csc(n.left_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);
            // todo: we are applying marginal dk?
            Ok(y_d1.log1p().scale_columns().transpose() * delta_dk)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let y_delta_left_nk = concatenate_vertical(&y_delta_left)?;

    let y_delta_right_nk = concatenate_vertical(&y_delta_right)?;

    let delta_kn = (y_delta_left_nk.scale(0.5) + y_delta_right_nk.scale(0.5)).transpose();

    let mut proj = arc_proj.lock().expect("lock proj in nystrom");

    proj.marginal_kn
        .columns_range_mut(lb..ub)
        .copy_from(&marginal_kn);
    proj.delta_kn.columns_range_mut(lb..ub).copy_from(&delta_kn);

    Ok(())
}
