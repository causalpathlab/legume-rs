use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_gene_graph::*;
use crate::srt_gene_pairs::*;
use crate::srt_input::*;

use clap::Parser;
use data_beans_alg::random_projection::*;
use matrix_param::dmatrix_gamma::GammaMatrix;
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
///
/// PINTO gene-gene interaction analysis by topic modelling
///
pub struct SrtGenePairTopicArgs {
    /// Data files of either `.zarr` or `.h5` format.
    #[arg(required = true, value_delimiter(','))]
    data_files: Vec<Box<str>>,

    /// Auxiliary cell coordinate files (comma separated).
    #[arg(long = "coord", short = 'c', required = true, value_delimiter(','))]
    coord_files: Vec<Box<str>>,

    /// Cell coordinate column indices in the `coord` files (comma separated)
    #[arg(long = "coord-column-indices", value_delimiter(','))]
    coord_columns: Option<Vec<usize>>,

    /// Column names in the `coord` files (comma separated)
    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres"
    )]
    coord_column_names: Vec<Box<str>>,

    /// Header row in coordinate file (0 if first line is column names)
    #[arg(long)]
    coord_header_row: Option<usize>,

    /// Coordinate embedding dimension
    #[arg(long, default_value_t = 256)]
    coord_emb: usize,

    /// Batch membership files (comma separated).
    #[arg(long, short = 'b', value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// Random projection dimension
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// Use top S components for sample assignment. #samples < 2^S+1.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// k-nearest neighbours for gene-gene graph
    #[arg(long, default_value_t = 20)]
    knn_gene: usize,

    /// k-nearest neighbours for spatial cell pairs
    #[arg(short = 'k', long, default_value_t = 10)]
    knn_spatial: usize,

    /// Downsampling columns per collapsed sample
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Block size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// Number of latent topics
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// Number of feature modules in the encoder.
    /// If not specified, `encoder_layers[0]` will be used.
    #[arg(short = 'm', long)]
    feature_modules: Option<usize>,

    /// Encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128, 1024, 128])]
    encoder_layers: Vec<usize>,

    /// Intensity levels for frequency embedding
    #[arg(long, default_value_t = 10)]
    vocab_size: usize,

    /// Intensity embedding dimension
    #[arg(long, default_value_t = 10)]
    vocab_emb: usize,

    /// Training epochs
    #[arg(long, short = 'i', default_value_t = 1000)]
    epochs: usize,

    /// Data jitter interval
    #[arg(long, short = 'j', default_value_t = 5)]
    jitter_interval: usize,

    /// Minibatch size
    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// Candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

    /// Column sum normalization scale
    #[arg(long, default_value_t = 1e4)]
    column_sum_norm: f32,

    /// Preload all column data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// Verbosity
    #[arg(long, short)]
    verbose: bool,
}

/// Gene-gene interaction pipeline with topic modelling:
///
/// 1. Load data + coordinates
/// 2. Build spatial cell-cell KNN graph
/// 3. Assign cell pairs to samples (random projection + binary sort)
/// 4. Preliminary collapse → gene × sample matrix
/// 5. Build gene-gene KNN graph from posterior means
/// 6. Compute gene log means (μ̃_g)
/// 7. Compute gene-pair deltas (δ⁺/δ⁻) by visiting cells
/// 8. Fit Poisson-Gamma on gene-pair stats
/// 9. Train encoder-decoder topic model on gene-pair posterior samples
/// 10. Encoder projection → per-cell → per-pair latent codes
/// 11. Export
pub fn fit_srt_gene_pair_topic(args: &SrtGenePairTopicArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }

    // 1. Load data
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
    let n_genes = data_vec.num_rows();

    // 2. Build spatial cell-cell KNN graph and extract pair info
    info!("Constructing spatial nearest neighbourhood graphs");
    let cell_pairs: Vec<(usize, usize)>;
    {
        let srt_cell_pairs = SrtCellPairs::new(
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

        cell_pairs = srt_cell_pairs
            .pairs
            .iter()
            .map(|p| (p.left, p.right))
            .collect();
    }

    // 3. Assign individual cells to samples
    info!("Projecting cells for sample assignment...");

    let cell_proj_out = data_vec.project_columns_with_batch_correction(
        args.proj_dim,
        Some(args.block_size),
        Some(&batch_membership),
    )?;

    let n_samples = data_vec.partition_columns_to_groups(
        &cell_proj_out.proj,
        Some(args.sort_dim),
        args.down_sample,
    )?;

    info!("Assigned cells to {} samples", n_samples);

    // 4. Preliminary collapse: gene × sample sums
    let (gene_sum_ds, size_s) = preliminary_collapse(&data_vec, n_genes, n_samples)?;

    // Compute posterior means via Poisson-Gamma
    let (a0, b0) = (1_f32, 1_f32);
    let mut mu_param = GammaMatrix::new((n_genes, n_samples), a0, b0);
    let denom_ds = DVec::from_element(n_genes, 1_f32) * size_s.transpose();
    mu_param.update_stat(&gene_sum_ds, &denom_ds);
    mu_param.calibrate();

    // 5. Build gene-gene KNN graph
    info!("Building gene-gene KNN graph...");

    let gene_graph = GenePairGraph::from_posterior_means(
        mu_param.posterior_mean(),
        gene_names.clone(),
        GenePairGraphArgs {
            knn: args.knn_gene,
            block_size: args.block_size,
        },
    )?;

    gene_graph.to_parquet(&(args.out.to_string() + ".gene_graph.parquet"))?;

    // 6. Compute gene log means
    let gene_log_means = compute_gene_log_means(&data_vec, args.block_size)?;

    // 7. Compute gene-pair deltas
    info!("Calibrating gene-gene interaction statistics...");

    let gene_pair_stat = compute_gene_interaction_deltas(
        &data_vec,
        &gene_graph,
        &gene_log_means,
        n_samples,
    )?;

    gene_pair_stat.to_parquet(&(args.out.to_string() + ".gene_pairs.parquet"))?;

    // 8. Fit Poisson-Gamma
    info!("Fitting Poisson-Gamma on gene-pair statistics...");
    let gene_pair_params = gene_pair_stat.optimize(None)?;

    // 9. Train encoder-decoder topic model
    info!("Setting up training data...");

    let n_edges = gene_graph.num_edges();
    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);
    let n_features = 2 * n_edges;

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
            n_features,
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

    let decoder = TopicDecoder::new(n_features, n_topics, param_builder.clone())?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features, n_features
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
        let x_nd = concatenate_vertical(&[
            gene_pair_params.delta_pos.posterior_sample()?,
            gene_pair_params.delta_neg.posterior_sample()?,
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

    info!("Writing down the model parameters");

    matrix_util::common_io::write_types::<f32>(
        &log_likelihoods,
        &(args.out.to_string() + ".log_likelihood.gz"),
    )?;

    let dict_row_names = gene_graph.edge_names_with_channels();

    named_tensor_parquet_out(
        &decoder.dictionary().weight_dk()?,
        Some(&dict_row_names),
        None,
        &args.out,
        "dictionary",
    )?;

    // 10. Encoder projection: per-cell first, then convert to per-pair
    info!("Encoding gene-pair projection...");

    let cell_latent_nk = encode_gene_pair_projection(
        &data_vec,
        &gene_graph,
        &gene_log_means,
        &encoder,
        &dev,
        args.column_sum_norm,
        args.block_size,
    )?;

    // Convert cell-level latents to pair-level:
    // pair_latent = 0.5 * (cell_latent[left] + cell_latent[right])
    info!("Converting cell latents to pair latents...");
    let n_pairs = cell_pairs.len();
    let mut pair_latent = Mat::zeros(n_pairs, n_topics);

    for (pair_idx, &(left, right)) in cell_pairs.iter().enumerate() {
        let left_row = cell_latent_nk.row(left);
        let right_row = cell_latent_nk.row(right);
        let avg = (&left_row + &right_row) * 0.5;
        pair_latent.row_mut(pair_idx).copy_from(&avg);
    }

    pair_latent.to_parquet(None, None, &(args.out.to_string() + ".latent.parquet"))?;

    info!("Done");
    Ok(())
}

/// Encoder-based projection: project individual cells onto the gene-pair
/// dictionary using a trained encoder network.
///
/// For each cell, builds a dense feature vector of gene-pair deltas
/// (δ⁺ in top half, |δ⁻| in bottom half), normalizes, and passes
/// through the encoder to obtain per-cell latent codes.
///
/// Returns latent matrix of shape (n_cells × n_topics).
fn encode_gene_pair_projection<Enc>(
    data_vec: &SparseIoVec,
    gene_graph: &GenePairGraph,
    gene_log_means: &DVec,
    encoder: &Enc,
    device: &candle_core::Device,
    column_sum_norm: f32,
    block_size: usize,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync,
{
    let n_cells = data_vec.num_columns();
    let n_edges = gene_graph.num_edges();

    info!(
        "Encoder gene-pair projection: {} cells, {} edges",
        n_cells, n_edges,
    );

    let gene_adj = gene_graph.build_directed_adjacency();

    let shared_in = EncoderSharedInput {
        gene_log_means: gene_log_means.clone(),
        gene_adj,
        n_edges,
        encoder,
        device,
        column_sum_norm,
    };

    let mut latent_vec = Vec::<(usize, Mat)>::new();

    data_vec.visit_columns_by_block(
        &encode_gene_pair_visitor,
        &shared_in,
        &mut latent_vec,
        Some(block_size),
    )?;

    latent_vec.sort_by_key(|&(lb, _)| lb);
    let sorted: Vec<Mat> = latent_vec.into_iter().map(|(_, m)| m).collect();
    concatenate_vertical(&sorted)
}

struct EncoderSharedInput<'a, Enc> {
    gene_log_means: DVec,
    gene_adj: Vec<Vec<(usize, usize)>>,
    n_edges: usize,
    encoder: &'a Enc,
    device: &'a candle_core::Device,
    column_sum_norm: f32,
}

fn encode_gene_pair_visitor<Enc>(
    bound: (usize, usize),
    data_vec: &SparseIoVec,
    shared_in: &EncoderSharedInput<Enc>,
    arc_out: Arc<Mutex<&mut Vec<(usize, Mat)>>>,
) -> anyhow::Result<()>
where
    Enc: EncoderModuleT + Send + Sync,
{
    let (lb, ub) = bound;
    let gene_log_means = &shared_in.gene_log_means;
    let gene_adj = &shared_in.gene_adj;
    let n_edges = shared_in.n_edges;
    let dev = shared_in.device;

    let yy = data_vec.read_columns_csc(lb..ub)?;
    let n_cells_block = ub - lb;

    // Build dense delta feature matrix: [δ⁺; |δ⁻|] (2*n_edges × n_cells_block)
    let mut features = Mat::zeros(2 * n_edges, n_cells_block);

    for (cell_idx, y_j) in yy.col_iter().enumerate() {
        let rows = y_j.row_indices();
        let vals = y_j.values();

        visit_gene_pair_deltas(rows, vals, gene_adj, gene_log_means, |edge_idx, delta| {
            if delta > 0.0 {
                features[(edge_idx, cell_idx)] = delta;
            } else if delta < 0.0 {
                features[(n_edges + edge_idx, cell_idx)] = -delta;
            }
        });
    }

    // Normalize and encode: (n_cells_block × 2*n_edges)
    let x_nd = features
        .sum_to_one_columns()
        .scale(shared_in.column_sum_norm)
        .transpose();

    let x_tensor = x_nd.to_tensor(dev)?;
    let (logits_theta, _) = shared_in.encoder.forward_t(&x_tensor, None, false)?;
    let latent_nk = Mat::from_tensor(&logits_theta)?;

    arc_out
        .lock()
        .expect("lock encode proj")
        .push((lb, latent_nk));

    Ok(())
}
