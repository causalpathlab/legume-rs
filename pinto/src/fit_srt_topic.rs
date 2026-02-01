use crate::srt_cell_pairs::*;
use crate::srt_collapse_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::estimate_batch;
use crate::srt_estimate_batch_effects::EstimateBatchArgs;
use crate::srt_input::*;
use crate::srt_random_projection::*;

use clap::{Parser, ValueEnum};
use matrix_param::io::ParamIo;
use matrix_param::traits::*;

use candle_util::candle_data_loader::*;
use candle_util::candle_decoder_topic::*;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_inference::TrainConfig;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_traits::*;
use candle_util::candle_vae_inference::*;
use indicatif::{ProgressBar, ProgressDrawTarget};

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
    #[arg(long = "coord-column-indices", value_delimiter(','))]
    coord_columns: Option<Vec<usize>>,

    /// The columns names in the `coord` files (comma separated)
    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres"
    )]
    coord_column_names: Vec<Box<str>>,

    /// header row in the coordinate information (feed 0 if the first line writes column names)
    #[arg(long)]
    coord_header_row: Option<usize>,

    /// Coordinate embedding dimension
    #[arg(long, default_value_t = 256)]
    coord_emb: usize,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short = 'b', value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// ignore batch adjustment
    #[arg(long, default_value_t = false)]
    ignore_batch_effects: bool,

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

    /// number of modules of the features in the encoder model.
    /// If not specified, `encoder_layers[0]` will be used.
    #[arg(short = 'm', long)]
    feature_modules: Option<usize>,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    /// intensity levels for frequency embedding
    #[arg(long, default_value_t = 10)]
    vocab_size: usize,

    /// intensity embedding dimension
    #[arg(long, default_value_t = 10)]
    vocab_emb: usize,

    /// # training epochs
    #[arg(long, short = 'i', default_value_t = 1000)]
    epochs: usize,

    /// data jitter interval
    #[arg(long, short = 'j', default_value_t = 5)]
    jitter_interval: usize,

    /// Minibatch size
    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

    /// column sum normalization scale (will only affect decoder)
    #[arg(short = 'c', long, default_value_t = 1e4)]
    column_sum_norm: f32,

    /// preload all the columns data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

/// Fits SVD and write down the dictionary matrix and pair-level
/// latent states.
pub fn fit_srt_topic(args: &SrtTopicArgs) -> anyhow::Result<()> {
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
        header_in_coord: args.coord_header_row,
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

    info!("Writing down collapsed pairs");
    collapsed_data.to_parquet(
        &(args.out.to_string() + ".collapsed_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    info!("Setting up training data...");
    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

    let n_features_decoder = data_vec.num_rows();
    let n_features_encoder = data_vec.num_rows();

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_features_encoder,
            n_topics,
            n_modules,
            n_vocab,
            d_vocab_emb,
            layers: &args.encoder_layers,
            use_sparsemax: false,
            temperature: 1.0,
        },
        param_builder.clone(),
    )?;

    let decoder = TopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features_encoder, n_features_decoder
    );

    let mut train_config = TrainConfig {
        learning_rate: args.learning_rate,
        batch_size: args.minibatch_size,
        num_epochs: args.epochs,
        num_pretrain_epochs: 0,
        device: dev.clone(),
        verbose: args.verbose,
        show_progress: true,
    };

    let pb = ProgressBar::new(train_config.num_epochs as u64);

    if !train_config.show_progress || train_config.verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut vae = Vae::build(&encoder, &decoder, &parameters);
    let mut log_likelihoods = Vec::with_capacity(train_config.num_epochs);

    for epoch in (0..args.epochs).step_by(args.jitter_interval) {
        // just train on the delta data
        let x_nd = concatenate_horizontal(&[
            collapsed_params.left_delta.posterior_sample()?,
            collapsed_params.right_delta.posterior_sample()?,
            (collapsed_params.left_delta.posterior_sample()?
                + collapsed_params.right_delta.posterior_sample()?),
        ])?
        .sum_to_one_columns()
        .scale(args.column_sum_norm)
        .transpose();

        let mut data_loader = InMemoryData::from(InMemoryArgs {
            input: &x_nd,
            input_null: None,
            output: Some(&x_nd),
            output_null: None,
        })?;

        train_config.verbose = false;
        train_config.show_progress = args.verbose;
        train_config.num_epochs = args.jitter_interval;

        let llik = vae.train_encoder_decoder(
            &mut data_loader,
            &loss_func::topic_likelihood,
            &train_config,
        )?;

        log_likelihoods.extend(llik);
        pb.inc(args.jitter_interval as u64);

        if args.verbose {
            info!(
                "[{}] log-likelihood: {}",
                epoch + args.jitter_interval,
                log_likelihoods.last().ok_or(anyhow::anyhow!("llik"))?
            );
        }
    }

    pb.finish_and_clear();

    if train_config.verbose {
        info!("Finished {} epochs", train_config.num_epochs);
    }

    info!("Writing down the model parameters");

    matrix_util::common_io::write_types::<f32>(
        &log_likelihoods,
        &(args.out.to_string() + ".log_likelihood.gz"),
    )?;

    named_tensor_parquet_out(
        &decoder.dictionary().weight_dk()?,
        Some(&gene_names),
        None,
        &args.out,
        "dictionary",
    )?;

    info!("Evaluating the latent states...");

    let latent = srt_cell_pairs.evaluate_latent(
        &encoder,
        batch_db.as_ref(),
        &train_config,
        args.block_size,
    )?;

    latent.to_parquet(None, None, &(args.out.to_string() + ".latent.parquet"))?;

    info!("done");
    Ok(())
}

trait SrtLatentTopicOps {
    fn evaluate_latent<'a, Enc>(
        &self,
        encoder: &Enc,
        batch_db: Option<&'a Mat>,
        train_config: &TrainConfig,
        block_size: usize,
    ) -> anyhow::Result<Mat>
    where
        Enc: EncoderModuleT + Send + Sync + 'static;
}

impl SrtLatentTopicOps for SrtCellPairs<'_> {
    fn evaluate_latent<'a, Enc>(
        &self,
        encoder: &Enc,
        batch_db: Option<&'a Mat>,
        train_config: &TrainConfig,
        block_size: usize,
    ) -> anyhow::Result<Mat>
    where
        Enc: EncoderModuleT + Send + Sync + 'static,
    {
        let njobs = self.num_pairs().div_ceil(block_size);
        let mut latent_vec = Vec::with_capacity(njobs);
        self.visit_pairs_by_block(
            &evaluate_latent_visitor,
            &EncoderBatchConfig {
                encoder: encoder,
                batch_db: batch_db,
                train_config: train_config,
            },
            &mut latent_vec,
            block_size,
        )?;

        latent_vec.sort_by_key(|&(lb, _)| lb);
        concatenate_vertical(&latent_vec.into_iter().map(|(_, x)| x).collect::<Vec<_>>())
    }
}

struct EncoderBatchConfig<'a, Enc> {
    encoder: &'a Enc,
    batch_db: Option<&'a Mat>,
    train_config: &'a TrainConfig,
}

/// Evaluate latent representation with the trained encoder network
///
/// #Arguments
/// * `data_vec` - full data vector
/// * `encoder` - encoder network
/// * `train_config` - training configuration
/// * `delta_db` - batch effect matrix (feature x batch)
fn evaluate_latent_visitor<'a, Enc>(
    bound: (usize, usize),
    data: &SrtCellPairs,
    encoder_batch_config: &EncoderBatchConfig<'a, Enc>,
    latent_vec: Arc<Mutex<&mut Vec<(usize, Mat)>>>,
) -> anyhow::Result<()>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
{
    let encoder = encoder_batch_config.encoder;
    let delta_db = encoder_batch_config.batch_db;
    let config = encoder_batch_config.train_config;
    let dev = &config.device;
    let (lb, ub) = bound;

    let pairs = &data.pairs[lb..ub];

    let left = pairs.iter().map(|x| x.left);
    let right = pairs.iter().map(|x| x.right);

    let mut y_left = data.data.read_columns_csc(left)?;
    let mut y_right = data.data.read_columns_csc(right)?;

    if let Some(delta_db) = delta_db {
        let left = pairs.iter().map(|x| x.left);
        let right = pairs.iter().map(|x| x.right);
        let left_batches = data.data.get_batch_membership(left);
        y_left.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    ////////////////////////////////////////////////////
    // imputation by neighbours and update statistics //
    ////////////////////////////////////////////////////

    let pairs_neighbours = data.pairs_neighbours[lb..ub]
        .iter()
        .map(|x| x)
        .collect::<Vec<_>>();

    // adjust the left by the neighbours of the right
    let mut y_delta_left = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<CscMat> {
            let left = pairs[j].left;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(left))?;
            let y_right_neigh_dm = data.data.read_columns_csc(n.right_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_right_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);

            Ok(y_d1)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // adjust the right by the neighbours of the left
    let mut y_delta_right = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<CscMat> {
            let right = pairs[j].right;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(right))?;
            let y_left_neigh_dm = data.data.read_columns_csc(n.left_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_left_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);

            Ok(y_d1)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // batch adjustment if needed
    if let Some(delta_db) = delta_db {
        let left = pairs.iter().map(|x| x.left);
        let right = pairs.iter().map(|x| x.right);
        let left_batches = data.data.get_batch_membership(left);
        let right_batches = data.data.get_batch_membership(right);

        y_delta_left
            .iter_mut()
            .zip(left_batches)
            .for_each(|(y_d, b)| {
                y_d.adjust_by_division_of_selected_inplace(delta_db, &[b]);
            });

        y_delta_right
            .iter_mut()
            .zip(right_batches)
            .for_each(|(y_d, b)| {
                y_d.adjust_by_division_of_selected_inplace(delta_db, &[b]);
            });
    }

    let y_delta_left_nd = Tensor::cat(
        &y_delta_left
            .into_iter()
            .map(|x| -> anyhow::Result<Tensor> { Ok(x.to_tensor(dev)?.transpose(0, 1)?) })
            .collect::<anyhow::Result<Vec<_>>>()?,
        0,
    )?;

    let y_delta_right_nd = Tensor::cat(
        &y_delta_right
            .into_iter()
            .map(|x| -> anyhow::Result<Tensor> { Ok(x.to_tensor(dev)?.transpose(0, 1)?) })
            .collect::<anyhow::Result<Vec<_>>>()?,
        0,
    )?;

    // combine two delta signals for this pair
    let (logits_theta, _) =
        encoder.forward_t(&y_delta_left_nd.add(&y_delta_right_nd)?, None, false)?;

    latent_vec
        .lock()
        .expect("lock")
        .push((lb, Mat::from_tensor(&logits_theta)?));

    Ok(())
}
