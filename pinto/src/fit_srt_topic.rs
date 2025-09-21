use crate::srt_cell_pairs::*;
use crate::srt_collapse_pairs::*;
use crate::srt_common::*;
use crate::srt_input::*;
use crate::srt_random_projection::*;
use crate::srt_vertex_propensity::*;

use candle_util::candle_inference::*;
use candle_util::candle_matched_data_loader::*;
use candle_util::candle_matched_vae_inference::*;
use clap::{Parser, ValueEnum};

use indicatif::{ProgressBar, ProgressDrawTarget};

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
    #[arg(long, default_value_t = 64)]
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

    /// targeted number of row feature modules (to speed up)
    #[arg(short = 'r', long, default_value_t = 128)]
    n_row_modules: usize,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,128,128])]
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
    let cell_names = data.column_names()?;

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

    info!("Collecting summary statistics across cell pairs...");
    let collapsed = srt_cell_pairs.collapse_pairs()?;

    /////////////////////////////////////////////////////////
    // 4. Train embedded topic model on the collapsed data //
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

    let parameters = candle_nn::VarMap::new();
    let dev = &train_config.device;
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);

    //////////////
    // training //
    //////////////

    let encoder = MatchedEncoder::new(
        MatchedEncoderArg {
            dim_feature: collapsed.nrows(),
            dim_latent: args.n_latent_topics,
            dim_coord: collapsed.num_coordinate_embedding(),
            n_vocab_emb: args.vocab_size,
            dim_emb: args.coord_emb,
            num_feature_modules: args.n_row_modules,
            layers: &args.encoder_layers,
        },
        param_builder.clone(),
    )?;

    let decoder = MatchedTopicDecoder::new(
        collapsed.nrows(),
        args.n_latent_topics,
        param_builder.clone(),
    )?;

    let log_likelihood = train_encoder_decoder_stochastic(
        &collapsed,
        &encoder,
        &decoder,
        &parameters,
        &train_config,
        args.jitter_interval,
    )?;

    write_types::<f32>(&log_likelihood, &(args.out.to_string() + ".llik.gz"))?;

    // tensor_parquet_out(&latent.marginal, &args.out, "collapsed_latent_marginal")?;
    // tensor_parquet_out(&latent.border, &args.out, "collapsed_latent_border")?;

    named_tensor_parquet_out(
        &decoder.dictionary().weight()?,
        Some(&gene_names),
        None,
        &args.out,
        "dictionary",
    )?;

    let latent = srt_cell_pairs.evaluate_latent_states(&encoder, &train_config, args.block_size)?;

    tensor_parquet_out(&latent.logits_theta, &args.out, "latent")?;
    srt_cell_pairs.to_parquet(
        &(args.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names),
    )?;

    let proj_kn = Mat::from_tensor(&latent.logits_theta)?.transpose();

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

    info!("done");
    Ok(())
}

fn train_encoder_decoder_stochastic<Enc, Dec>(
    collapsed: &SrtCollapsedStat,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    train_config: &TrainConfig,
    jitter: usize,
) -> anyhow::Result<Vec<f32>>
where
    Enc: MatchedEncoderModuleT + Send + Sync + 'static,
    Dec: MatchedDecoderModuleT + Send + Sync + 'static,
{
    let full_data = collapsed.optimize(None)?;

    let mut vae = MatchedVae::build(encoder, decoder, parameters);

    if train_config.verbose {
        info!("Built the VAE model");
    }

    for var in parameters.all_vars() {
        var.to_device(&train_config.device)?;
    }

    let mut llik_trace = vec![];

    let sub_train_config = TrainConfig {
        learning_rate: train_config.learning_rate, // override
        batch_size: train_config.batch_size,       // train
        num_epochs: jitter,                        // config
        num_pretrain_epochs: 0,                    //
        device: train_config.device.clone(),       //
        verbose: false,
        show_progress: false,
    };

    let pb = ProgressBar::new(train_config.num_epochs as u64);

    if !train_config.show_progress || train_config.verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let aux_left_nc = collapsed.left_coord_emb.transpose();
    let aux_right_nc = &collapsed.right_coord_emb.transpose();

    for epoch in (0..train_config.num_epochs).step_by(jitter) {
        let input_left_nd = full_data.left.posterior_sample()?.transpose();
        let input_right_nd = full_data.right.posterior_sample()?.transpose();

        let output_left_nd = full_data.left_delta.posterior_sample()?.transpose();
        let output_right_nd = full_data.right_delta.posterior_sample()?.transpose();

        let mut data_loader = InMemoryData::from(DataLoaderArgs {
            input_left: &input_left_nd,
            input_right: &input_right_nd,
            input_aux_left: Some(&aux_left_nc),
            input_aux_right: Some(&aux_right_nc),
            output_left: Some(&output_left_nd),
            output_right: Some(&output_right_nd),
        })?;

        let llik = vae.train_encoder_decoder(
            &mut data_loader,
            &loss_func::topic_likelihood,
            &sub_train_config,
        )?;
        llik_trace.extend(llik);
        pb.inc(jitter as u64);

        if train_config.verbose {
            info!(
                "[{}] log-likelihood: {}",
                epoch + 1,
                llik_trace.last().ok_or(anyhow::anyhow!("llik"))?
            );
        }
    }
    pb.finish_and_clear();

    if train_config.verbose {
        info!("Finished {} epochs", train_config.num_epochs);
    }

    Ok(llik_trace)
}

pub trait SrtLatentTopicOps {
    fn evaluate_latent_states<Enc>(
        &self,
        encoder: &Enc,
        train_config: &TrainConfig,
        block_size: usize,
    ) -> anyhow::Result<MatchedEncoderLatent>
    where
        Enc: MatchedEncoderModuleT + Send + Sync + 'static;
}

impl<'a> SrtLatentTopicOps for SrtCellPairs<'a> {
    fn evaluate_latent_states<Enc>(
        &self,
        encoder: &Enc,
        train_config: &TrainConfig,
        block_size: usize,
    ) -> anyhow::Result<MatchedEncoderLatent>
    where
        Enc: MatchedEncoderModuleT + Send + Sync + 'static,
    {
        let njobs = self.num_pairs().div_ceil(block_size);
        let mut latent_vec = Vec::with_capacity(njobs);
        self.visit_pairs_by_block(
            &evaluate_latent_state_visitor,
            &(encoder, train_config),
            &mut latent_vec,
            block_size,
        )?;

        latent_vec.sort_by_key(|&(lb, _)| lb);
        latent_vec
            .into_iter()
            .map(|(_, x)| x)
            .collect::<Vec<_>>()
            .concatenate()
    }
}

fn evaluate_latent_state_visitor<Enc>(
    bound: (usize, usize),
    data: &SrtCellPairs,
    encoder_config: &(&Enc, &TrainConfig),
    latent_vec: Arc<Mutex<&mut Vec<(usize, MatchedEncoderLatent)>>>,
) -> anyhow::Result<()>
where
    Enc: MatchedEncoderModuleT + Send + Sync + 'static,
{
    let (encoder, config) = *encoder_config;
    let dev = &config.device;

    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let left = pairs.into_iter().map(|pp| pp.left);
    let right = pairs.into_iter().map(|pp| pp.right);

    let y_left = data
        .data
        .read_columns_dmatrix(left)?
        .transpose()
        .to_tensor(dev)?;

    let y_right = data
        .data
        .read_columns_dmatrix(right)?
        .transpose()
        .to_tensor(dev)?;

    let aux_left = concatenate_vertical(
        &pairs
            .into_iter()
            .map(|j| data.coordinate_embedding.row(j.left))
            .collect::<Vec<_>>(),
    )?
    .to_tensor(dev)?;

    let aux_right = concatenate_vertical(
        &pairs
            .into_iter()
            .map(|j| data.coordinate_embedding.row(j.right))
            .collect::<Vec<_>>(),
    )?
    .to_tensor(dev)?;

    let latent = encoder.forward_t(
        MatchedEncoderData {
            left: &y_left,
            right: &y_right,
            aux_left: Some(&aux_left),
            aux_right: Some(&aux_right),
        },
        false,
    )?;

    latent_vec
        .lock()
        .expect("latent vec lock")
        .push((lb, latent));

    Ok(())
}

pub trait MatchedEncoderLatentVecOps {
    fn concatenate(&self) -> anyhow::Result<MatchedEncoderLatent>;
}

impl MatchedEncoderLatentVecOps for Vec<MatchedEncoderLatent> {
    fn concatenate(&self) -> anyhow::Result<MatchedEncoderLatent> {
        // Collect references to tensors for each field
        let logits_theta: Vec<&Tensor> = self.iter().map(|latent| &latent.logits_theta).collect();

        let kl_divs: Vec<&Tensor> = self.iter().map(|latent| &latent.kl_div).collect();

        // Concatenate tensors along dimension 0
        let logits_theta = Tensor::cat(&logits_theta, 0)?;
        let kl_div = Tensor::cat(&kl_divs, 0)?;

        // Return the concatenated MatchedEncoderLatent
        Ok(MatchedEncoderLatent {
            logits_theta,
            kl_div,
        })
    }
}

// pub trait MatchedEncoderEvaluateOps {
//     fn evaluate<DataL: DataLoader>(
//         &self,
//         data: &DataL,
//         train_config: &TrainConfig,
//     ) -> anyhow::Result<MatchedEncoderLatent>;
// }

// impl MatchedEncoderEvaluateOps for MatchedLogSoftmaxEncoder {
//     fn evaluate<DataL: DataLoader>(
//         &self,
//         data: &DataL,
//         train_config: &TrainConfig,
//     ) -> anyhow::Result<MatchedEncoderLatent> {
//         let device = &train_config.device;
//         let ntot = data.num_data();
//         let batch_size = train_config.batch_size;
//         let jobs = generate_minibatch_intervals(ntot, batch_size);
//         let num_jobs = jobs.len();

//         let mut ret = Vec::with_capacity(num_jobs);

//         for (lb, ub) in jobs {
//             let mb = data.minibatch_ordered(lb, ub, device)?;

//             let latent = self.forward_t(
//                 MatchedEncoderData {
//                     left: mb.input_left.as_ref(),
//                     right: mb.input_right.as_ref(),
//                     aux_left: mb.input_aux_left.as_ref(),
//                     aux_right: mb.input_aux_right.as_ref(),
//                 },
//                 false,
//             )?;
//             ret.push(latent);
//         }
//         ret.concatenate()
//     }
// }
