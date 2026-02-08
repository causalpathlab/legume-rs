use crate::fit_srt_delta_svd::*;
use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::estimate_batch;
use crate::srt_estimate_batch_effects::EstimateBatchArgs;
use crate::srt_input::*;
use crate::srt_random_projection::*;

use clap::Parser;
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

#[derive(Parser, Debug, Clone)]
pub struct SrtTopicArgs {
    #[arg(required = true, value_delimiter(','),
          help = "Data files (.zarr or .h5 format, comma separated)")]
    data_files: Vec<Box<str>>,

    #[arg(long = "coord", short = 'c', required = true, value_delimiter(','),
          help = "Spatial coordinate files, one per data file",
          long_help = "Spatial coordinate files, one per data file (comma separated).\n\
                       Each file: barcode, x, y, ... per line.")]
    coord_files: Vec<Box<str>>,

    #[arg(long = "coord-column-indices", value_delimiter(','),
          help = "Column indices for coordinates in coord files",
          long_help = "Column indices for coordinates in coord files (comma separated).\n\
                       Use when coord files have extra columns beyond barcode,x,y.")]
    coord_columns: Option<Vec<usize>>,

    #[arg(long = "coord-column-names", value_delimiter(','),
          default_value = "pxl_row_in_fullres,pxl_col_in_fullres",
          help = "Column names to look up in coord files")]
    coord_column_names: Vec<Box<str>>,

    #[arg(long,
          help = "Header row index in coord files (0 = first line is column names)")]
    coord_header_row: Option<usize>,

    #[arg(long, default_value_t = 256,
          help = "Dimension for spectral embedding of spatial coordinates")]
    coord_emb: usize,

    #[arg(long, short = 'b', value_delimiter(','),
          help = "Batch membership files, one per data file",
          long_help = "Batch membership files, one per data file (comma separated).\n\
                       Each file maps cells to batch labels for batch effect correction.")]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = false,
          help = "Skip batch effect estimation and correction")]
    ignore_batch_effects: bool,

    #[arg(long, short = 'p', default_value_t = 50,
          help = "Random projection dimension for pseudobulk sample construction")]
    proj_dim: usize,

    #[arg(long, short = 'd', default_value_t = 10,
          help = "Number of top projection components for binary sort",
          long_help = "Number of top projection components for binary sort.\n\
                       Produces up to 2^S pseudobulk samples.")]
    sort_dim: usize,

    #[arg(short = 'k', long, default_value_t = 10,
          help = "Number of nearest neighbours for spatial cell-pair graph")]
    knn_spatial: usize,

    #[arg(long, default_value_t = 10,
          help = "Number of nearest-neighbour batches for batch effect estimation")]
    knn_batches: usize,

    #[arg(long, default_value_t = 10,
          help = "Number of nearest neighbours within each batch for batch estimation")]
    knn_cells: usize,

    #[arg(long, short = 's',
          help = "Maximum cells per pseudobulk sample (downsampling)")]
    down_sample: Option<usize>,

    #[arg(long, short, required = true,
          help = "Output file prefix",
          long_help = "Output file prefix.\n\
                       Generates: {out}.delta.parquet (when multiple batches), {out}.coord_pairs.parquet,\n\
                       {out}.dictionary.parquet, {out}.latent.parquet,\n\
                       {out}.log_likelihood.gz")]
    out: Box<str>,

    #[arg(long, default_value_t = 100,
          help = "Block size for parallel processing of cell pairs")]
    block_size: usize,

    #[arg(short = 't', long, default_value_t = 10,
          help = "Number of latent topics")]
    n_latent_topics: usize,

    #[arg(short = 'm', long,
          help = "Number of feature modules in the encoder",
          long_help = "Number of feature modules in the encoder (smaller = faster).\n\
                       Defaults to encoder_layers[0] if not specified.")]
    feature_modules: Option<usize>,

    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128],
          help = "Encoder hidden layer sizes (comma separated)")]
    encoder_layers: Vec<usize>,

    #[arg(long, default_value_t = 10,
          help = "Number of intensity levels for frequency embedding")]
    vocab_size: usize,

    #[arg(long, default_value_t = 10,
          help = "Dimension of intensity embedding vectors")]
    vocab_emb: usize,

    #[arg(long, short = 'i', default_value_t = 1000,
          help = "Total number of training epochs")]
    epochs: usize,

    #[arg(long, short = 'j', default_value_t = 5,
          help = "Posterior resampling interval (epochs between data jittering)")]
    jitter_interval: usize,

    #[arg(long, default_value_t = 100,
          help = "Minibatch size for SGD training")]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3,
          help = "Learning rate for Adam optimizer")]
    learning_rate: f32,

    #[arg(long, value_enum, default_value = "cpu",
          help = "Compute device for neural network training (cpu, cuda, metal)")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 1e4,
          help = "Column sum normalization scale for decoder reconstruction")]
    column_sum_norm: f32,

    #[arg(long, default_value_t = false,
          help = "Preload all sparse column data into memory for faster access")]
    preload_data: bool,

    #[arg(long, short,
          help = "Enable verbose logging (sets RUST_LOG=info)")]
    verbose: bool,
}

/// Fits SVD and write down the dictionary matrix and pair-level
/// latent states.
pub fn fit_srt_delta_topic(args: &SrtTopicArgs) -> anyhow::Result<()> {
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
        args.down_sample,
    )?;

    info!("Collecting shared/difference statistics across cell pairs...");
    let batch_db = batch_effects.map(|x| x.posterior_mean().clone());
    let n_genes = data_vec.num_rows();
    let batch_ref = batch_db.as_ref();
    let mut collapsed_stat =
        PairDeltaCollapsedStat::new(n_genes, srt_cell_pairs.num_samples()?);
    srt_cell_pairs.visit_pairs_by_sample(
        &collect_pair_delta_visitor,
        &batch_ref,
        &mut collapsed_stat,
    )?;
    let collapsed_params = collapsed_stat.optimize(None)?;

    info!("Setting up training data...");
    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

    let n_features_decoder = 2 * n_genes;
    let n_features_encoder = 2 * n_genes;

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
        let half_norm = args.column_sum_norm / 2.0;
        let x_nd = concatenate_vertical(&[
            collapsed_params.shared.posterior_sample()?.sum_to_one_columns().scale(half_norm),
            collapsed_params.diff.posterior_sample()?.sum_to_one_columns().scale(half_norm),
        ])?
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

    let dict_row_names: Vec<Box<str>> = gene_names
        .iter()
        .map(|g| format!("{}@shared", g).into_boxed_str())
        .chain(
            gene_names
                .iter()
                .map(|g| format!("{}@diff", g).into_boxed_str()),
        )
        .collect();

    named_tensor_parquet_out(
        &decoder.dictionary().weight_dk()?,
        Some(&dict_row_names),
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
        n_genes,
        args.column_sum_norm,
    )?;

    latent.to_parquet(None, None, &(args.out.to_string() + ".latent.parquet"))?;

    info!("done");
    Ok(())
}

trait SrtLatentTopicOps {
    fn evaluate_latent<Enc>(
        &self,
        encoder: &Enc,
        batch_db: Option<&Mat>,
        train_config: &TrainConfig,
        block_size: usize,
        n_genes: usize,
        column_sum_norm: f32,
    ) -> anyhow::Result<Mat>
    where
        Enc: EncoderModuleT + Send + Sync + 'static;
}

impl SrtLatentTopicOps for SrtCellPairs<'_> {
    fn evaluate_latent<Enc>(
        &self,
        encoder: &Enc,
        batch_db: Option<&Mat>,
        train_config: &TrainConfig,
        block_size: usize,
        n_genes: usize,
        column_sum_norm: f32,
    ) -> anyhow::Result<Mat>
    where
        Enc: EncoderModuleT + Send + Sync + 'static,
    {
        let njobs = self.num_pairs().div_ceil(block_size);
        let mut latent_vec = Vec::with_capacity(njobs);
        self.visit_pairs_by_block(
            &evaluate_latent_visitor,
            &EncoderBatchConfig {
                encoder,
                batch_db,
                train_config,
                n_genes,
                column_sum_norm,
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
    n_genes: usize,
    column_sum_norm: f32,
}

/// Evaluate latent representation with shared/diff features
///
/// For each pair, reads left/right sparse columns, computes per-gene
/// shared = log1p(left) + log1p(right) and diff = |log1p(left) - log1p(right)|,
/// builds a dense (2*n_genes × n_pairs_block) feature matrix, normalizes,
/// and encodes through the trained network.
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
    let n_genes = encoder_batch_config.n_genes;
    let column_sum_norm = encoder_batch_config.column_sum_norm;
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

    let n_pairs_block = ub - lb;
    let mut features = Mat::zeros(2 * n_genes, n_pairs_block);

    for (pair_idx, (left_col, right_col)) in
        y_left.col_iter().zip(y_right.col_iter()).enumerate()
    {
        let right_log: HashMap<usize, f32> = right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
            .map(|(&g, &v)| (g, v.ln_1p()))
            .collect();

        let mut left_visited = HashSet::new();

        for (&gene, &val) in left_col
            .row_indices()
            .iter()
            .zip(left_col.values().iter())
        {
            let log_left = val.ln_1p();
            let log_right = right_log.get(&gene).copied().unwrap_or(0.0);
            features[(gene, pair_idx)] = log_left + log_right;
            features[(n_genes + gene, pair_idx)] = (log_left - log_right).abs();
            left_visited.insert(gene);
        }

        for (&gene, _) in right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
        {
            if !left_visited.contains(&gene) {
                let log_right = right_log[&gene];
                features[(gene, pair_idx)] = log_right;
                features[(n_genes + gene, pair_idx)] = log_right;
            }
        }
    }

    let half_norm = column_sum_norm / 2.0;
    let mut shared_part = features.rows(0, n_genes).clone_owned();
    let mut diff_part = features.rows(n_genes, n_genes).clone_owned();
    shared_part.sum_to_one_columns_inplace();
    shared_part *= half_norm;
    diff_part.sum_to_one_columns_inplace();
    diff_part *= half_norm;

    let x_nd = concatenate_vertical(&[shared_part, diff_part])?.transpose();

    let x_tensor = x_nd.to_tensor(dev)?;
    let (logits_theta, _) = encoder.forward_t(&x_tensor, None, false)?;

    latent_vec
        .lock()
        .expect("lock")
        .push((lb, Mat::from_tensor(&logits_theta)?));

    Ok(())
}
