use crate::srt_cell_pairs::*;
use crate::srt_collapse_pairs::*;
use crate::srt_common::*;
use crate::srt_input::*;
use crate::srt_random_projection::*;
use crate::srt_vertex_propensity::*;

use candle_util::candle_matched_data_loader::*;
use candle_util::candle_matched_decoder_topic::*;
use candle_util::candle_matched_encoder::*;
use candle_util::candle_matched_vae_inference::*;

use clap::{Parser, ValueEnum};

use candle_util::candle_inference::TrainConfig;
use candle_util::candle_loss_functions as loss_func;

use matrix_param::traits::Inference;
use matrix_util::utils::*;

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

    /// number of modules of the features in the encoder model.
    /// If not specified, `encoder_layers[0]` will be used.
    #[arg(short = 'm', long)]
    feature_modules: Option<usize>,

    /// number of (edge) clusters
    #[arg(long)]
    n_edge_clusters: Option<usize>,

    /// number of (edge) clusters
    #[arg(long, default_value_t = 100)]
    maxiter_clustering: usize,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,128,128])]
    encoder_layers: Vec<usize>,

    /// intensity levels for frequency embedding
    #[arg(long, default_value_t = 100)]
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
    let training_data = collapsed.optimize(None)?;

    collapsed.to_parquet(
        &(args.out.to_string() + ".collapsed_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    // let training_dm = concatenate_horizontal(&[
    //     params.left_delta.posterior_log_mean().clone(),
    //     params.right_delta.posterior_log_mean().clone(),
    // ])?
    // .scale_columns();

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

    let encoder = MatchedEncoder::new(
        MatchedEncoderArg {
            dim_feature: collapsed.nrows(),
            dim_latent: args.n_latent_topics,
            dim_coord: collapsed.num_coordinate_embedding(),
            num_feature_modules: args.feature_modules.unwrap_or(args.encoder_layers[0]),
            layers: &args.encoder_layers,
        },
        param_builder.clone(),
    )?;

    let decoder = MatchedTopicDecoder::new(
        collapsed.nrows(),
        args.n_latent_topics,
        param_builder.clone(),
    )?;

    //////////////
    // training //
    //////////////

    let mut vae = MatchedVae::build(&encoder, &decoder, &parameters);

    // let aux_left_nc = collapsed.left_coord_emb.transpose();
    // let aux_right_nc = &collapsed.right_coord_emb.transpose();

    let mixed_left_nd = training_data.left.posterior_mean().transpose();
    let mixed_right_nd = training_data.right.posterior_mean().transpose();

    let delta_left_nd = training_data.left_delta.posterior_mean().transpose();
    let delta_right_nd = training_data.right_delta.posterior_mean().transpose();

    let mut data_loader = InMemoryData::from(DataLoaderArgs {
        input_left: &mixed_left_nd,
        input_right: &mixed_right_nd,
        input_aux_left: None,  // Some(&aux_left_nc),
        input_aux_right: None, // Some(&aux_right_nc),
        output_left: Some(&delta_left_nd),
        output_right: Some(&delta_right_nd),
        output_delta_left: None,  // Some(&delta_left_nd),
        output_delta_right: None, // Some(&delta_right_nd),
    })?;

    info!("Set up training data");

    let log_likelihoods = vae.train_encoder_decoder(
        &mut data_loader,
        &loss_func::topic_likelihood,
        &train_config,
    )?;

    if train_config.verbose {
        info!("Finished {} epochs", train_config.num_epochs);
    }

    info!("Writing down the model parameters");

    write_types::<f32>(
        &log_likelihoods,
        &(args.out.to_string() + ".log_likelihood.gz"),
    )?;

    let latent = encoder.evaluate(&data_loader, &train_config)?;
    tensor_parquet_out(
        &latent.logits_theta_left,
        &args.out,
        "collapsed_latent_left",
    )?;
    tensor_parquet_out(
        &latent.logits_theta_right,
        &args.out,
        "collapsed_latent_right",
    )?;

    named_tensor_parquet_out(
        &decoder.dictionary().weight()?,
        Some(&gene_names),
        None,
        &args.out,
        "dictionary",
    )?;

    named_tensor_parquet_out(
        &decoder.dictionary_delta().weight()?,
        Some(&gene_names),
        None,
        &args.out,
        "dictionary_delta",
    )?;

    let latent = srt_cell_pairs.evaluate_latent_states(&encoder, &train_config, args.block_size)?;
    latent.to_parquet_simple(&(args.out.to_string() + ".latent.parquet"))?;

    srt_cell_pairs.to_parquet(
        &(args.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names),
    )?;

    info!("clustering edges");
    let proj_kn = latent.map(|x| x.exp()).transpose();
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

pub trait SrtLatentTopicOps {
    fn evaluate_latent_states<Enc>(
        &self,
        encoder: &Enc,
        train_config: &TrainConfig,
        block_size: usize,
    ) -> anyhow::Result<Mat>
    where
        Enc: MatchedEncoderModuleT + Send + Sync + 'static;
}

impl<'a> SrtLatentTopicOps for SrtCellPairs<'a> {
    fn evaluate_latent_states<Enc>(
        &self,
        encoder: &Enc,
        train_config: &TrainConfig,
        block_size: usize,
    ) -> anyhow::Result<Mat>
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
        concatenate_vertical(&latent_vec.into_iter().map(|(_, x)| x).collect::<Vec<_>>())
    }
}

fn evaluate_latent_state_visitor<Enc>(
    bound: (usize, usize),
    data: &SrtCellPairs,
    encoder_config: &(&Enc, &TrainConfig),
    latent_vec: Arc<Mutex<&mut Vec<(usize, Mat)>>>,
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

    ////////////////////////////////////////////////////
    // imputation by neighbours and update statistics //
    ////////////////////////////////////////////////////

    let pairs_neighbours = (lb..ub)
        .map(|j| data.pairs_neighbours.get(j).unwrap())
        .collect::<Vec<_>>();

    // adjust the left by the neighbours of the right
    let y_delta_left = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<Tensor> {
            let left = pairs[j].left;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(left))?;
            let y_right_neigh_dm = data.data.read_columns_csc(n.right_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_right_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);
            y_d1.transpose().to_tensor(dev)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let y_delta_left = Tensor::cat(&y_delta_left, 0)?;

    // adjust the right by the neighbours of the left
    let y_delta_right = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<Tensor> {
            let right = pairs[j].right;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(right))?;
            let y_left_neigh_dm = data.data.read_columns_csc(n.left_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_left_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);
            y_d1.transpose().to_tensor(dev)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let y_delta_right = Tensor::cat(&y_delta_right, 0)?;

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
            aux_left: None,
            aux_right: None,
            // aux_left: Some(&aux_left),
            // aux_right: Some(&aux_right),
        },
        false,
    )?;

    // let logits_theta = candle_nn::ops::log_softmax(
    //     &latent
    //         .logits_theta_left
    //         .exp()?
    //         .add(&latent.logits_theta_right.exp()?)?
    //         .log()?,
    //     latent.logits_theta_left.rank() - 1,
    // )?;

    let logits_theta = candle_nn::ops::log_softmax(
        &latent.logits_theta_left.add(&latent.logits_theta_right)?,
        latent.logits_theta_left.rank() - 1,
    )?;

    latent_vec
        .lock()
        .expect("lock")
        .push((lb, Mat::from_tensor(&logits_theta)?));

    Ok(())
}

pub trait MatchedEncoderLatentVecOps {
    fn concatenate(&self) -> anyhow::Result<MatchedEncoderLatent>;
}

impl MatchedEncoderLatentVecOps for Vec<MatchedEncoderLatent> {
    fn concatenate(&self) -> anyhow::Result<MatchedEncoderLatent> {
        // Collect references to tensors for each field
        let logits_theta_left: Vec<&Tensor> = self
            .iter()
            .map(|latent| &latent.logits_theta_left)
            .collect();

        let logits_theta_right: Vec<&Tensor> = self
            .iter()
            .map(|latent| &latent.logits_theta_right)
            .collect();

        let kl_divs: Vec<&Tensor> = self.iter().map(|latent| &latent.kl_div).collect();

        // Concatenate tensors along dimension 0
        let logits_theta_left = Tensor::cat(&logits_theta_left, 0)?;
        let logits_theta_right = Tensor::cat(&logits_theta_right, 0)?;

        let kl_div = Tensor::cat(&kl_divs, 0)?;

        // Return the concatenated MatchedEncoderLatent
        Ok(MatchedEncoderLatent {
            logits_theta_left,
            logits_theta_right,
            kl_div,
        })
    }
}

pub trait MatchedEncoderEvaluateOps {
    fn evaluate<DataL: DataLoader>(
        &self,
        data: &DataL,
        train_config: &TrainConfig,
    ) -> anyhow::Result<MatchedEncoderLatent>;
}

impl MatchedEncoderEvaluateOps for MatchedEncoder {
    fn evaluate<DataL: DataLoader>(
        &self,
        data_loader: &DataL,
        train_config: &TrainConfig,
    ) -> anyhow::Result<MatchedEncoderLatent> {
        let device = &train_config.device;
        let ntot = data_loader.num_data();
        let batch_size = train_config.batch_size;
        let jobs = generate_minibatch_intervals(ntot, batch_size);
        let num_jobs = jobs.len();

        let mut ret = Vec::with_capacity(num_jobs);

        for (lb, ub) in jobs {
            let mb = data_loader.minibatch_ordered(lb, ub, device)?;

            let latent = self.forward_t(
                MatchedEncoderData {
                    left: mb.input_left.as_ref(),
                    right: mb.input_right.as_ref(),
                    aux_left: None,  // mb.input_aux_left.as_ref(),
                    aux_right: None, // mb.input_aux_right.as_ref(),
                },
                false,
            )?;
            ret.push(latent);
        }
        ret.concatenate()
    }
}
