mod srt_cell_pairs;
mod srt_collapse_pairs;
mod srt_common;
mod srt_random_projection;
mod srt_routines_latent_representation;
mod srt_routines_post_process;
mod srt_routines_pre_process;

use candle_util::candle_data_loader::DataLoaderArgs;
use srt_routines_latent_representation::*;
use srt_routines_post_process::SrtLatentStatePairsOps;
use srt_routines_pre_process::*;
// use srt_routines_post_process::*;

use asap_alg::random_projection::binary_sort_columns;

use matrix_param::traits::{Inference, TwoStatParam};
use srt_cell_pairs::SrtCellPairs;
use srt_collapse_pairs::SrtCollapsePairsOps;
use srt_common::*;

use clap::{Parser, ValueEnum};
use srt_random_projection::SrtRandProjOps;

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
    /// correspond to each data file. Each line contains barcode, x, y, ...
    /// coordinates. We could include more columns.
    #[arg(long = "coord", short = 'c', required = true, value_delimiter(','))]
    coord_files: Vec<Box<str>>,

    /// which columns of cell coordinate files (0-based indexing)?
    #[arg(long = "coord_columns", value_delimiter(','), default_value = "3,4")]
    coord_columns: Vec<usize>,

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

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// #k-nearest neighbours for spectral embedding for spatial coordinates
    #[arg(long, default_value_t = 10)]
    knn_spatial: usize,

    /// maximum rank for spectral embedding for spatial coordinates
    #[arg(long, default_value_t = 10)]
    rank_spatial: usize,

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

    let (data, coordinates, _) = read_data_vec(args.clone())?;

    let gene_names = data.row_names()?;

    //////////////////////////////////////////////////
    // 1. Take pairs of spatially interacting cells //
    //////////////////////////////////////////////////

    let mut srt_cell_pairs =
        SrtCellPairs::new(&data, &coordinates, args.knn_spatial, Some(args.block_size))?;

    ////////////////////////////////////////////
    // 2. Randomly project the pairs of cells //
    ////////////////////////////////////////////

    let proj_out = srt_cell_pairs.random_projection(args.proj_dim, args.block_size)?;

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

    let coordinate_column_names: Vec<Box<str>> = (1..=collapsed.left_coordinates.nrows())
        .map(|x| format!("left_{}", x).into_boxed_str())
        .chain(
            (1..=collapsed.right_coordinates.nrows())
                .map(|x| format!("right_{}", x).into_boxed_str()),
        )
        .collect();

    concatenate_vertical(&[collapsed.left_coordinates, collapsed.right_coordinates])?
        .transpose()
        .to_parquet(
            None,
            Some(&coordinate_column_names),
            &(args.out.to_string() + ".collapsed_coord_pairs.parquet"),
        )?;

    /////////////////////////////////////////////////
    // 4. Collapse rows/genes/features to speed up //
    /////////////////////////////////////////////////

    let aggregate_rows = if params.left.nrows() > args.n_row_modules {
        let log_x_md = concatenate_vertical(&[
            params.left.posterior_log_mean().transpose().clone(),
            params.right.posterior_log_mean().transpose().clone(),
        ])?;
        let kk = (args.n_row_modules as f32).log2().ceil() as usize + 1;
        info!(
            "approximately reduce data features: {} -> {}",
            log_x_md.ncols(),
            args.n_row_modules
        );
        row_membership_matrix(binary_sort_columns(&log_x_md, kk)?)?
    } else {
        let d = params.left.nrows();
        Mat::identity(d, d)
    };

    /////////////////////////////////////////////////////////
    // 5. Train embedded topic model on the collapsed data //
    /////////////////////////////////////////////////////////

    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;

    // encoder input can be modularized
    let input_left_nm = params.left.posterior_mean().transpose() * &aggregate_rows;
    let input_right_nm = params.right.posterior_mean().transpose() * &aggregate_rows;

    // output decoder should maintain the original dimension
    let output_left_nd = params.left.posterior_mean().transpose();
    let output_right_nd = params.right.posterior_mean().transpose();

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

    let parameters = candle_nn::VarMap::new();
    let dev = &train_config.device;
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);

    //////////////
    // training //
    //////////////

    let n_features_encoder = input_left_nm.ncols();

    let encoder = MatchedLogSoftmaxEncoder::new(
        n_features_encoder,
        n_topics,
        n_vocab,
        d_vocab_emb,
        &args.encoder_layers,
        param_builder.clone(),
    )?;

    let n_features_decoder = output_left_nd.ncols();

    let decoder = MatchedTopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features_encoder, n_features_decoder
    );

    let train_data = DataLoaderArgs {
        input: &input_left_nm,
        input_null: None,
        input_matched: Some(&input_right_nm),
        output: Some(&output_left_nd),
        output_null: None,
        output_matched: Some(&output_right_nd),
    };

    let (log_likelihood, latent) = train_left_right_vae(
        train_data,
        &encoder,
        &decoder,
        &parameters,
        &loss_func::topic_likelihood,
        &train_config,
    )?;

    write_types::<f32>(&log_likelihood, &(args.out.to_string() + ".llik.gz"))?;

    tensor_parquet_out(&latent.average, &args.out, "collapsed_latent")?;
    tensor_parquet_out(&latent.left, &args.out, "collapsed_latent_left")?;
    tensor_parquet_out(&latent.right, &args.out, "collapsed_latent_right")?;

    named_tensor_parquet_out(
        &decoder.get_dictionary()?,
        Some(&gene_names),
        None,
        &args.out,
        "dictionary",
    )?;

    let latent = srt_cell_pairs.evaluate_latent_states(
        &encoder,
        &aggregate_rows,
        &train_config,
        args.block_size,
    )?;

    tensor_parquet_out(&latent.average, &args.out, "latent")?;
    tensor_parquet_out(&latent.left, &args.out, "latent_left")?;
    tensor_parquet_out(&latent.right, &args.out, "latent_right")?;

    let (_left, _right) = srt_cell_pairs.all_pairs_positions()?;

    concatenate_horizontal(&[_left, _right])?.to_parquet(
        None,
        Some(&coordinate_column_names),
        &(args.out.to_string() + ".coord_pairs.parquet"),
    )?;

    info!("done");
    Ok(())
}
