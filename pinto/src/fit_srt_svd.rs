use crate::srt_cell_pairs::*;
use crate::srt_collapse_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::estimate_batch;
use crate::srt_estimate_batch_effects::EstimateBatchArgs;
use crate::srt_input::*;
use crate::srt_random_projection::*;

use clap::Parser;
use matrix_param::io::ParamIo;
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
    #[arg(long = "coord-column-indices", value_delimiter(','))]
    coord_columns: Option<Vec<usize>>,

    /// The columns names in the `coord` files (comma separated)
    #[arg(
        long = "coord-column-names",
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

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 10)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

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
        data: mut data_vec,
        coordinates,
        coordinate_names,
        batches: batch_membership,
    } = read_data_with_coordinates(SRTReadArgs {
        data_files: args.data_files.clone(),
        coord_files: args.coord_files.clone(),
        preload_data: args.preload_data,
        coord_columns: args.coord_columns.clone().unwrap_or_default(),
        coord_column_names: args.coord_column_names.clone(),
        batch_files: args.batch_files.clone(),
    })?;

    let gene_names = data_vec.row_names()?;

    // 0. identify gene-level batch effects
    info!("checking potential batch effects...");

    let batch_effects = estimate_batch(
        &mut data_vec,
        batch_membership.as_ref(),
        EstimateBatchArgs {
            proj_dim: args.proj_dim,
            sort_dim: args.sort_dim,
            block_size: args.block_size,
            knn_batches: args.knn_batches,
            knn_cells: args.knn_cells,
            down_sample: args.down_sample,
        },
    )?;

    if let Some(batch_db) = batch_effects.as_ref() {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet(Some(&gene_names), batch_names.as_deref(), &outfile)?;
    }

    info!("Constructing spatial nearest neighbourhood graphs");
    let mut srt_cell_pairs = SrtCellPairs::new(
        &data_vec,
        &coordinates,
        SrtCellPairsArgs {
            knn: args.knn_spatial,
            coordinate_emb_dim: args.coord_emb,
            block_size: args.block_size,
        },
    )?;

    srt_cell_pairs.to_parquet(
        &(args.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    let proj_out = srt_cell_pairs.random_projection(
        args.proj_dim,
        args.block_size,
        Some(&batch_membership),
    )?;

    srt_cell_pairs.assign_pairs_to_samples(
        &proj_out,
        Some(args.sort_dim),
        args.down_sample.clone(),
    )?;

    info!("Collecting summary statistics across cell pairs...");
    let batch_db = batch_effects.map(|x| x.posterior_mean().clone());
    let collapsed_data = srt_cell_pairs.collapse_pairs(batch_db.as_ref())?;
    let collapsed_params = collapsed_data.optimize(None)?;

    collapsed_data.to_parquet(
        &(args.out.to_string() + ".collapsed_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    info!("Randomized SVD ...");
    let training_dm = concatenate_horizontal(&[
        collapsed_params.left_delta.posterior_log_mean().clone(),
        collapsed_params.right_delta.posterior_log_mean().clone(),
        collapsed_params.left_resid.posterior_log_mean().clone(),
        collapsed_params.right_resid.posterior_log_mean().clone(),
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
        &BasisWithBatch {
            basis_dk: &basis_dk,
            batch_effect: batch_db.as_ref(),
        },
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

    info!("Done");
    Ok(())
}

struct BasisWithBatch<'a> {
    basis_dk: &'a Mat,
    batch_effect: Option<&'a Mat>,
}

fn nystrom_proj_visitor(
    bound: (usize, usize),
    data: &SrtCellPairs,
    basis_with_batch: &BasisWithBatch,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let basis_dk = basis_with_batch.basis_dk;
    let batch_effect = basis_with_batch.batch_effect;

    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let left = pairs.iter().map(|pp| pp.left);
    let right = pairs.iter().map(|pp| pp.right);

    let mut y_left_dn = data.data.read_columns_csc(left)?;
    let mut y_right_dn = data.data.read_columns_csc(right)?;

    // batch adjustment if needed
    if let Some(delta_db) = batch_effect {
        let left = pairs.iter().map(|x| x.left);
        let right = pairs.iter().map(|x| x.right);
        let left_batches = data.data.get_batch_membership(left);
        y_left_dn.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right_dn.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    let y_left_nk = y_left_dn.log1p().scale_columns().transpose() * basis_dk;
    let y_right_nk = y_right_dn.log1p().scale_columns().transpose() * basis_dk;

    let chunk_kn = (y_left_nk.scale(0.5) + y_right_nk.scale(0.5)).transpose();

    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in nystrom");

    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk_kn);

    Ok(())
}
