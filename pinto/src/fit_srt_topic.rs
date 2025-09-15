use crate::srt_cell_pairs::*;
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

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Parser, Debug, Clone)]
///
/// PINTO by topic modelling
///
pub struct SrtTopicArgs {
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

    /// targeted number of row feature modules (to speed up)
    #[arg(short = 'r', long, default_value_t = 512)]
    n_row_modules: usize,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,128,128])]
    encoder_layers: Vec<usize>,

    /// intensity levels for frequency embedding
    #[arg(long, default_value_t = 20)]
    vocab_size: usize,

    /// intensity embedding dimension
    #[arg(long, default_value_t = 10)]
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

    /// preload all the columns data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn fit_srt_topic(args: &SrtTopicArgs) -> anyhow::Result<()> {
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

    //////////////////////////////////////////////////
    // 1. Take pairs of spatially interacting cells //
    //////////////////////////////////////////////////

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

    ////////////////////////////////////////////
    // 2. Randomly project the pairs of cells //
    ////////////////////////////////////////////

    let proj_out =
        srt_cell_pairs.random_projection(args.proj_dim, args.block_size, Some(&batches))?;

    srt_cell_pairs.assign_pairs_to_samples(
        &proj_out,
        Some(args.sort_dim),
        args.down_sample.clone(),
    )?;

    ///////////////////////////////////////////////
    // 3. Collapse these cell pairs into samples //
    ///////////////////////////////////////////////

    let collapsed = srt_cell_pairs.collapse_pairs()?;
    let params = collapsed.optimize(None)?;

    /////////////////////////////////////////////////////////
    // 5. Train embedded topic model on the collapsed data //
    /////////////////////////////////////////////////////////

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
        show_progress: true,
        verbose: args.verbose,
    };

    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;

    let marg_left_nd = params.left.posterior_mean().transpose();
    let marg_right_nd = params.right.posterior_mean().transpose();

    let delta_left_nd = params.left_delta.posterior_mean().transpose();
    let delta_right_nd = params.right_delta.posterior_mean().transpose();

    let train_data = DataLoaderArgs {
        input_marginal_left: &marg_left_nd,
        input_marginal_right: &marg_right_nd,
        input_delta_left: Some(&delta_left_nd),
        input_delta_right: Some(&delta_right_nd),
        output_marginal_left: Some(&marg_left_nd),
        output_marginal_right: Some(&marg_right_nd),
        output_delta_left: Some(&delta_left_nd),
        output_delta_right: Some(&delta_right_nd),
    };

    let parameters = candle_nn::VarMap::new();
    let dev = &train_config.device;
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);

    //////////////
    // training //
    //////////////

    let n_features_encoder = marg_left_nd.ncols();

    let encoder = MatchedLogSoftmaxEncoder::new(
        n_features_encoder,
        n_topics,
        n_vocab,
        d_vocab_emb,
        &args.encoder_layers,
        param_builder.clone(),
    )?;

    let n_features_decoder = marg_left_nd.ncols();

    let decoder = MatchedTopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features_encoder, n_features_decoder
    );

    let (log_likelihood, latent) = train_left_right_vae(
        train_data,
        &encoder,
        &decoder,
        &parameters,
        &loss_func::topic_likelihood,
        &train_config,
    )?;

    write_types::<f32>(&log_likelihood, &(args.out.to_string() + ".llik.gz"))?;

    tensor_parquet_out(&latent.marginal, &args.out, "collapsed_latent_marginal")?;
    tensor_parquet_out(&latent.border, &args.out, "collapsed_latent_border")?;

    named_tensor_parquet_out(
        &decoder.get_dictionary()?,
        Some(&gene_names),
        None,
        &args.out,
        "dictionary",
    )?;

    // let latent = srt_cell_pairs.evaluate_latent_states(
    //     &encoder,
    //     &train_config,
    //     args.block_size,
    // )?;

    // tensor_parquet_out(&latent.marginal, &args.out, "latent_marginal")?;
    // tensor_parquet_out(&latent.border, &args.out, "latent_border")?;
    // srt_cell_pairs.to_parquet(
    //     &(args.out.to_string() + ".coord_pairs.parquet"),
    //     Some(coordinate_names),
    // )?;

    // info!("done");
    Ok(())
}
