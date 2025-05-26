mod simulate;
mod srt_common;
mod srt_routines_latent_representation;
mod srt_routines_post_process;

use log::info;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use srt_common::*;

use srt_routines_latent_representation::*;
use srt_routines_post_process::*;

use asap_data::sparse_io_vector::*;
use matrix_param::traits::{Inference, ParamIo, TwoStatParam};
use matrix_util::common_io::{extension, read_lines, write_lines, write_types};
use matrix_util::dmatrix_util::*;
use matrix_util::traits::*;
use matrix_util::utils::partition_by_membership;

use asap_alg::collapse_data::CollapsingOps;
use asap_alg::random_projection::*;
use asap_data::sparse_io::*;

use candle_util::candle_decoder_topic::*;
use candle_util::candle_inference::TrainConfig;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_traits::DecoderModule;
use candle_util::candle_spatial_encoder_softmax::*;

use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "senna", version, about, long_about, term_width = 80)]
///
/// Embedding spatially resolved transcriptomic (SRT) data.
///
struct SRTArgs {
    /// Data files of either `.zarr` or `.h5` format. All the formats
    /// in the given list should be identical. We can convert `.mtx`
    /// to `.zarr` or `.h5` using `asap-data build` command.
    #[arg(required = true, value_delimiter(','))]
    data_files: Vec<Box<str>>,

    /// An auxiliary cell coordinate file. Each coordinate file should
    /// correspond to each data file. Each line contains x, y, ...
    /// coordinates. We could include more columns.
    #[arg(long = "coord", short = 'c', required = true, value_delimiter(','))]
    coord_files: Vec<Box<str>>,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short = 'b', value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value_t = 15)]
    iter_opt: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// Block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of latent topics
    #[arg(short = 'k', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// targeted number of row feature modules
    #[arg(short = 'r', long, default_value_t = 1000)]
    n_row_modules: usize,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    /// intensity levels for frequency embedding
    #[arg(long, default_value_t = 100)]
    vocab_size: usize,

    /// intensity embedding dimension
    #[arg(long, default_value_t = 3)]
    vocab_emb: usize,

    /// # training epochs
    #[arg(long, short = 'i', default_value_t = 1000)]
    epochs: usize,

    /// Minibatch size
    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

    /// exclude spatial information
    #[arg(long, default_value_t = false)]
    exclude_spatial_info: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args: SRTArgs = SRTArgs::parse();

    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    info!("Reading data files...");

    let (mut data_vec, coord_map, batch_membership) = read_data_vec(args.clone())?;

    //////////////////////////////////////////
    // 1. Randomly project the data columns //
    //////////////////////////////////////////

    let proj_dim = args.proj_dim.max(args.n_latent_topics);

    info!("Random projection of data onto {} dims", proj_dim);
    let proj_out = data_vec.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size.clone()),
        Some(&batch_membership),
    )?;

    let proj_kn = if args.exclude_spatial_info {
        info!("excluding spatial information in the initial random projection");
        proj_out.proj
    } else {
        info!(
            "Random projection of spatial coordinates onto {} dims",
            proj_dim
        );
        let coord_kn = positional_projection(&coord_map, &batch_membership, proj_dim)?;
        proj_out.proj + coord_kn
    };

    info!("Assigning {} columns to samples...", proj_kn.ncols());
    let nsamp =
        data_vec.assign_columns_to_samples(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    //////////////////////////////////
    // 2. Register batch membership //
    //////////////////////////////////

    if args.batch_files.is_some() && batch_membership.len() > 0 {
        info!("Registering batch information");
        data_vec.register_batches(&proj_kn, &batch_membership)?;
    }

    ///////////////////////////
    // 3. Collapsing columns //
    ///////////////////////////

    info!("Collapsing data columns... into {} samples", nsamp);
    let collapse_out = data_vec.collapse_columns(
        Some(args.knn_batches),
        Some(args.knn_cells),
        Some(args.iter_opt),
    )?;

    let group_to_cols = data_vec.take_grouped_columns().ok_or(anyhow::anyhow!(
        "The columns were not assigned before. Call `assign_columns_to_groups`"
    ))?;

    info!("Collapsing coordinates into {} samples", nsamp);

    let mut collapsed_coords = Mat::zeros(group_to_cols.len(), coord_map.ncols());

    group_to_cols
        .into_iter()
        .enumerate()
        .for_each(|(s, cells)| {
            collapsed_coords
                .row_mut(s)
                .copy_from(&coord_map.select_rows(cells).row_mean())
        });

    collapsed_coords.to_tsv(&(args.out.to_string() + ".collapsed.coords.gz"))?;

    let batch_db = collapse_out.delta.as_ref();

    if let Some(batch_db) = batch_db {
        batch_db.to_tsv(&(args.out.to_string() + ".delta"))?;
    }

    let row_names = data_vec.row_names()?;
    let col_names = data_vec.column_names()?;
    write_lines(&row_names, &(args.out.to_string() + ".rows.gz"))?;
    write_lines(&col_names, &(args.out.to_string() + ".cols.gz"))?;

    /////////////////////////////////////////////////////////
    // 4. Train embedded topic model on the collapsed data //
    /////////////////////////////////////////////////////////

    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;

    let aggregate_rows = if collapse_out.mu_observed.nrows() > args.n_row_modules {
        let log_x_nd = match collapse_out.mu_adjusted.as_ref() {
            Some(x) => x.posterior_log_mean().transpose().clone(),
            _ => collapse_out
                .mu_observed
                .posterior_log_mean()
                .transpose()
                .clone(),
        };
        let kk = (args.n_row_modules as f32).log2().ceil() as usize + 1;
        info!(
            "reduce data features: {} -> {}",
            log_x_nd.ncols(),
            args.n_row_modules
        );
        row_membership_matrix(binary_sort_columns(&log_x_nd, kk)?)?
    } else {
        let d = collapse_out.mu_observed.nrows();
        Mat::identity(d, d)
    };

    let mixed_dn = &collapse_out.mu_observed;
    let clean_dn = collapse_out.mu_adjusted.as_ref();
    let batch_dn = collapse_out.mu_residual.as_ref();

    // encoder input can be modularized
    let input_nm = mixed_dn.posterior_mean().transpose().clone() * &aggregate_rows;

    let batch_nm = batch_dn.map(|x| x.posterior_mean().transpose().clone() * &aggregate_rows);

    // additional spatial coordinates
    let spatial_nc = collapsed_coords;

    // output decoder should maintain the original dimension
    let output_nd = match clean_dn {
        Some(x) => x.posterior_mean().transpose().clone(),
        _ => mixed_dn.posterior_mean().transpose().clone(),
    };

    let parameters = candle_nn::VarMap::new();

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let train_config = TrainConfig {
        learning_rate: args.learning_rate,
        batch_size: args.minibatch_size,
        num_epochs: args.epochs,
        num_pretrain_epochs: 0,
        device: dev.clone(),
        verbose: args.verbose,
    };

    let dev = &train_config.device;

    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);

    ///////////////////////////////////////////////////
    // training variational autoencoder architecture //
    ///////////////////////////////////////////////////

    let n_features_encoder = input_nm.ncols();
    let n_coords = spatial_nc.ncols();
    let max_coord = spatial_nc.max();
    let n_features_decoder = output_nd.ncols();

    let encoder = SpatialLogSoftmaxEncoder::new(
        n_features_encoder,
        n_topics,
        n_coords,
        n_vocab,
        max_coord.into(),
        d_vocab_emb,
        &args.encoder_layers,
        param_builder.clone(),
    )?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features_encoder, n_features_decoder
    );

    let decoder = TopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

    let log_likelihood = train_encoder_decoder(
        &input_nm,
        &output_nd,
        &spatial_nc,
        batch_nm.as_ref(),
        &encoder,
        &decoder,
        &parameters,
        &loss_func::topic_likelihood,
        &train_config,
    )?;

    write_types::<f32>(&log_likelihood, &(args.out.to_string() + ".llik.gz"))?;

    decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?
        .to_tsv(&(args.out.to_string() + ".dictionary.gz"))?;

    let delta_db = batch_db.map(|x| x.posterior_mean());

    let z_nk = evaluate_latent_by_encoder(
        &data_vec,
        &coord_map,
        &encoder,
        &aggregate_rows,
        &train_config,
        delta_db,
    )?;
    z_nk.to_tsv(&(args.out.to_string() + ".latent.gz"))?;
    if let Some(batch_db) = batch_db {
        batch_db.to_tsv(&(args.out.to_string() + ".delta"))?;
    }

    info!("done");
    Ok(())
}

fn append_batch_coordinate<T>(coords: &Mat, batch_membership: &Vec<T>) -> anyhow::Result<Mat>
where
    T: Sync + Send + Clone + Eq + std::hash::Hash,
{
    if coords.nrows() != batch_membership.len() {
        return Err(anyhow::anyhow!("incompatible batch membership"));
    }

    let minval = coords.min();
    let maxval = coords.max();
    let width = (maxval - minval).max(1.);

    let uniq_batches = batch_membership.iter().collect::<HashSet<_>>();

    let batch_index = uniq_batches
        .into_iter()
        .enumerate()
        .map(|(k, v)| (v, k))
        .collect::<HashMap<_, _>>();

    let batch_coord = batch_membership
        .iter()
        .map(|k| {
            let b = batch_index[k];
            width * (b as f32)
        })
        .collect::<Vec<_>>();

    let bb = Mat::from_vec(coords.nrows(), 1, batch_coord);

    Ok(concatenate_horizontal(vec![coords.clone(), bb])?)
}

///
/// each position vector (1 x d) `*` random_proj (d x r)
///
fn positional_projection<T>(
    coords: &Mat,
    batch_membership: &Vec<T>,
    d: usize,
) -> anyhow::Result<Mat>
where
    T: Sync + Send + Clone + Eq + std::hash::Hash,
{
    if coords.nrows() != batch_membership.len() {
        return Err(anyhow::anyhow!("incompatible batch membership"));
    }

    let maxval = coords.max();
    let minval = coords.min();
    let coords = coords
        .map(|x| (x - minval) / (maxval - minval + 1.))
        .transpose();

    let batches = partition_by_membership(batch_membership, None);

    let mut ret = Mat::zeros(d, coords.ncols());

    for (_b, points) in batches.into_iter() {
        let rand_proj = Mat::rnorm(d, coords.nrows());

        points.into_iter().for_each(|p| {
            ret.column_mut(p)
                .copy_from(&(&rand_proj * coords.column(p)));
        });
    }

    let (lb, ub) = (-4., 4.);
    ret.scale_columns_inplace();

    if ret.max() > ub || ret.min() < lb {
        info!("Clamping values [{}, {}] after standardization", lb, ub);
        ret.iter_mut().for_each(|x| {
            *x = x.clamp(lb, ub);
        });
        ret.scale_columns_inplace();
    }
    Ok(ret)
}

fn read_data_vec(args: SRTArgs) -> anyhow::Result<(SparseIoVec, Mat, Vec<Box<str>>)> {
    // push data files and collect batch membership
    let file = args.data_files[0].as_ref();
    let backend = match extension(file)?.to_string().as_str() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => SparseIoBackend::Zarr,
    };

    if args.coord_files.len() != args.data_files.len() {
        return Err(anyhow::anyhow!("# coordinate files != # of data files"));
    }

    let mut data_vec = SparseIoVec::new();

    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        match extension(&data_file)?.as_ref() {
            "zarr" => {
                assert_eq!(backend, SparseIoBackend::Zarr);
            }
            "h5" => {
                assert_eq!(backend, SparseIoBackend::HDF5);
            }
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", data_file)),
        };

        let data = open_sparse_matrix(&data_file, &backend)?;
        data_vec.push(Arc::from(data))?;
    }

    // check if row names are the same
    let row_names = data_vec[0].row_names()?;

    for j in 1..data_vec.len() {
        let row_names_j = data_vec[j].row_names()?;
        if row_names != row_names_j {
            return Err(anyhow::anyhow!("Row names are not the same"));
        }
    }

    let mut coord_vec = Vec::with_capacity(args.coord_files.len());

    for coord_file in args.coord_files.iter() {
        info!("Reading coordinate file: {}", coord_file);
        let coord = Mat::read_file_delim(coord_file, vec!['\t', ',', ' '], None)?;
        coord_vec.push(coord);
    }

    let coord_nk = concatenate_vertical(coord_vec)?;

    // check batch membership
    let mut batch_membership = Vec::with_capacity(data_vec.len());

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!("# batch files != # of data files"));
        }

        for batch_file in batch_files.iter() {
            info!("Reading batch file: {}", batch_file);
            for s in read_lines(&batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
        }
    } else {
        for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
            batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
        }
    }

    if batch_membership.len() != data_vec.num_columns()? {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()?
        ));
    }

    // use batch index as another coordinate
    let uniq_batches = batch_membership.par_iter().collect::<HashSet<_>>();
    let n_batches = uniq_batches.len();
    let coord_nk = if n_batches > 1 {
        info!("attaching {} batch index coordinate(s)", n_batches);
        append_batch_coordinate(&coord_nk, &batch_membership)?
    } else {
        coord_nk
    };

    Ok((data_vec, coord_nk, batch_membership))
}
