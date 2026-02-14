use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::estimate_batch;
use crate::srt_estimate_batch_effects::EstimateBatchArgs;
use crate::srt_input::*;
use crate::srt_random_projection::*;

use candle_util::candle_core::{self, Device, Tensor};
use candle_util::candle_decoder_topic::TopicDecoder;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_loss_functions::topic_likelihood;
use candle_util::candle_model_traits::{DecoderModuleT, EncoderModuleT};
use candle_util::candle_data_loader::*;
use candle_util::candle_nn::{self, AdamW, Optimizer};

use std::collections::{HashMap, HashSet};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressDrawTarget};
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::io::ParamIo;
use matrix_param::traits::*;
use matrix_util::utils::generate_minibatch_intervals;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Parser, Debug, Clone)]
pub struct SrtLinkCommunityArgs {
    #[arg(
        required = true,
        value_delimiter(','),
        help = "Data files (.zarr or .h5 format, comma separated)"
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long = "coord",
        short = 'c',
        required = true,
        value_delimiter(','),
        help = "Spatial coordinate files, one per data file"
    )]
    coord_files: Vec<Box<str>>,

    #[arg(
        long = "coord-column-indices",
        value_delimiter(','),
        help = "Column indices for coordinates in coord files"
    )]
    coord_columns: Option<Vec<usize>>,

    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres",
        help = "Column names to look up in coord files"
    )]
    coord_column_names: Vec<Box<str>>,

    #[arg(
        long,
        help = "Header row index in coord files (0 = first line is column names)"
    )]
    coord_header_row: Option<usize>,

    #[arg(
        long,
        default_value_t = 256,
        help = "Dimension for spectral embedding of spatial coordinates"
    )]
    coord_emb: usize,

    #[arg(
        long,
        short = 'b',
        value_delimiter(','),
        help = "Batch membership files, one per data file"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension for pseudobulk sample construction"
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Number of top projection components for binary sort"
    )]
    sort_dim: usize,

    #[arg(
        short = 'k',
        long,
        default_value_t = 10,
        help = "Number of nearest neighbours for spatial cell-pair graph"
    )]
    knn_spatial: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of nearest neighbours within each batch for batch estimation"
    )]
    knn_cells: usize,

    #[arg(long, short = 's', help = "Maximum cells per pseudobulk sample")]
    down_sample: Option<usize>,

    #[arg(long, short, required = true, help = "Output file prefix")]
    out: Box<str>,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing of cell pairs"
    )]
    block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent community topics (K)"
    )]
    n_topics: usize,

    #[arg(
        short = 'f',
        long,
        help = "Number of feature modules in encoder (default: encoder_layers[0])"
    )]
    feature_modules: Option<usize>,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder layers (comma-separated)"
    )]
    encoder_layers: Vec<usize>,

    #[arg(
        long,
        short = 'i',
        default_value_t = 1000,
        help = "Number of training epochs"
    )]
    epochs: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Jitter resampling interval (epochs between posterior resampling)"
    )]
    jitter_interval: usize,

    #[arg(long, default_value_t = 100, help = "Minibatch size for training")]
    batch_size: usize,

    #[arg(long, default_value_t = 1e-3, help = "Learning rate")]
    learning_rate: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "KL warmup: kl_weight = 1 - exp(-epoch / warmup). 0 = no annealing."
    )]
    kl_warmup_epochs: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Topic smoothing: z = (1-α)z + α/K. 0 = disabled."
    )]
    topic_smoothing: f64,

    #[arg(
        long,
        short = 'n',
        default_value_t = 1e4,
        help = "Column sum normalization scale for encoder input"
    )]
    column_sum_norm: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Compute device (cpu, cuda, metal)"
    )]
    device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "Device ordinal for cuda/metal"
    )]
    device_no: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse column data into memory"
    )]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = 2,
        help = "Number of multi-level collapsing levels (coarse→fine)"
    )]
    num_levels: usize,

    #[arg(long, short, help = "Enable verbose logging (sets RUST_LOG=info)")]
    verbose: bool,
}

/////////////////////////////////////////
// Pair + Null collapsed statistics    //
/////////////////////////////////////////

pub(crate) struct PairNullCollapsedStat {
    pair_ds: Mat,
    null_ds: Mat,
    pair_size_s: DVec,
    null_size_s: DVec,
    n_genes: usize,
    n_samples: usize,
}

impl PairNullCollapsedStat {
    pub(crate) fn new(n_genes: usize, n_samples: usize) -> Self {
        Self {
            pair_ds: Mat::zeros(n_genes, n_samples),
            null_ds: Mat::zeros(n_genes, n_samples),
            pair_size_s: DVec::zeros(n_samples),
            null_size_s: DVec::zeros(n_samples),
            n_genes,
            n_samples,
        }
    }

    pub(crate) fn optimize(
        &self,
        hyper_param: Option<(f32, f32)>,
    ) -> anyhow::Result<PairNullParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));
        let shape = (self.n_genes, self.n_samples);

        let mut pair = GammaMatrix::new(shape, a0, b0);
        let mut null = GammaMatrix::new(shape, a0, b0);

        let pair_size = &self.pair_size_s.transpose();
        let pair_size_ds = Mat::from_rows(&vec![pair_size.clone(); shape.0]);

        let null_size = &self.null_size_s.transpose();
        let null_size_ds = Mat::from_rows(&vec![null_size.clone(); shape.0]);

        info!("Calibrating null statistics");

        null.update_stat(&self.null_ds, &null_size_ds);
        null.calibrate();

        // Use null posterior mean as denominator for pair Gamma,
        // so pair posterior models fold-change over null baseline.
        info!("Calibrating pair statistics (null-adjusted denominator)");

        let null_mean = null.posterior_mean();
        let pair_denom = null_mean.component_mul(&pair_size_ds);
        pair.update_stat(&self.pair_ds, &pair_denom);
        pair.calibrate();

        info!("Resolved pair/null collapsed statistics");

        Ok(PairNullParameters { pair, null })
    }
}

pub(crate) struct PairNullParameters {
    pub(crate) pair: GammaMatrix,
    pub(crate) null: GammaMatrix,
}

/// Collapse visitor: accumulate pair (sum of both cells) and null
/// (sum of random non-neighbor cell pairs) expression per gene per sample.
pub(crate) fn collect_pair_null_visitor(
    indices: &[usize],
    data: &SrtCellPairs,
    sample: usize,
    batch_effect: &Option<&Mat>,
    arc_stat: Arc<Mutex<&mut PairNullCollapsedStat>>,
) -> anyhow::Result<()> {
    let pairs: Vec<&Pair> = indices.iter().filter_map(|&j| data.pairs.get(j)).collect();
    let n_pairs = pairs.len();
    if n_pairs == 0 {
        return Ok(());
    }

    // 1. Collect unique cell indices from all pairs in this sample
    let mut unique_set = HashSet::new();
    for p in &pairs {
        unique_set.insert(p.left);
        unique_set.insert(p.right);
    }
    let unique_cells: Vec<usize> = {
        let mut v: Vec<usize> = unique_set.into_iter().collect();
        v.sort_unstable();
        v
    };
    let n_unique = unique_cells.len();

    // 2. Generate random non-neighbor pairs via rejection sampling
    //    Cap attempts to avoid infinite loops when too few unique cells
    let mut rng = SmallRng::seed_from_u64(sample as u64);
    let mut random_pairs: Vec<(usize, usize)> = Vec::with_capacity(n_pairs);
    let max_attempts = n_pairs * 20;
    let mut attempts = 0;

    while random_pairs.len() < n_pairs && attempts < max_attempts {
        attempts += 1;
        let a = unique_cells[rng.random_range(0..n_unique)];
        let b = unique_cells[rng.random_range(0..n_unique)];
        if a == b {
            continue;
        }
        let key = (a.min(b), a.max(b));
        if data.graph.edges.binary_search(&key).is_ok() {
            continue; // reject real spatial neighbours
        }
        random_pairs.push((a, b));
    }

    // 3. Read expression for all unique cells once
    let mut all_columns = data.data.read_columns_csc(unique_cells.iter().copied())?;

    // 4. Batch adjustment
    if let Some(delta_db) = *batch_effect {
        let batches = data.data.get_batch_membership(unique_cells.iter().copied());
        all_columns.adjust_by_division_of_selected_inplace(delta_db, &batches);
    }

    // 5. Build cell → column-index mapping
    let cell_to_col: HashMap<usize, usize> = unique_cells
        .iter()
        .enumerate()
        .map(|(col_idx, &cell)| (cell, col_idx))
        .collect();

    let n_genes = all_columns.nrows();
    let mut local_pair = DVec::zeros(n_genes);
    let mut local_null = DVec::zeros(n_genes);

    // Helper: accumulate a sparse column into a dense vector
    let accumulate = |target: &mut DVec, col_idx: usize| {
        let col = all_columns.col(col_idx);
        for (&gene, &val) in col.row_indices().iter().zip(col.values().iter()) {
            target[gene] += val;
        }
    };

    // 6. Pair accumulation: sum both cells per real neighbour pair
    for p in &pairs {
        accumulate(&mut local_pair, cell_to_col[&p.left]);
        accumulate(&mut local_pair, cell_to_col[&p.right]);
    }

    // 7. Null accumulation: sum both cells per random non-neighbour pair
    for &(a, b) in &random_pairs {
        accumulate(&mut local_null, cell_to_col[&a]);
        accumulate(&mut local_null, cell_to_col[&b]);
    }

    // 8. Lock and update shared statistics
    let mut stat = arc_stat.lock().expect("lock pair null stat");
    let mut col_pair = stat.pair_ds.column_mut(sample);
    col_pair += &local_pair;
    let mut col_null = stat.null_ds.column_mut(sample);
    col_null += &local_null;
    stat.pair_size_s[sample] += n_pairs as f32;
    stat.null_size_s[sample] += random_pairs.len() as f32;

    Ok(())
}

/////////////////////////////////////////
// Training loop                       //
/////////////////////////////////////////

struct TrainScores {
    llik: Vec<f32>,
    kl: Vec<f32>,
}

impl TrainScores {
    fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        let mat = Mat::from_columns(&[
            DVec::from_vec(self.llik.clone()),
            DVec::from_vec(self.kl.clone()),
        ]);

        let score_types = vec![
            "log_likelihood".to_string().into_boxed_str(),
            "kl_divergence".to_string().into_boxed_str(),
        ];

        let epochs: Vec<Box<str>> = (0..mat.nrows())
            .map(|x| (x + 1).to_string().into_boxed_str())
            .collect();

        mat.to_parquet_with_names(
            file_path,
            (Some(&epochs), Some("epoch")),
            Some(&score_types),
        )
    }
}

fn train_link_community_progressive(
    all_params: &[PairNullParameters],
    encoder: &mut LogSoftmaxEncoder,
    decoder: &TopicDecoder,
    parameters: &candle_nn::VarMap,
    args: &SrtLinkCommunityArgs,
) -> anyhow::Result<TrainScores> {
    let dev = match args.device {
        ComputeDevice::Metal => Device::new_metal(args.device_no)?,
        ComputeDevice::Cuda => Device::new_cuda(args.device_no)?,
        _ => Device::Cpu,
    };

    let mut adam = AdamW::new_lr(parameters.all_vars(), args.learning_rate as f64)?;

    let num_levels = all_params.len();
    let total_weight: usize = (1..=num_levels).sum();
    let mut llik_trace = Vec::with_capacity(args.epochs);
    let mut kl_trace = Vec::with_capacity(args.epochs);

    let pb = ProgressBar::new(args.epochs as u64);
    if args.verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    info!("Start progressive training ({} levels, {} total epochs)...", num_levels, args.epochs);

    let mut epoch_counter = 0usize;

    for (level, params) in all_params.iter().enumerate() {
        let level_epochs = (args.epochs * (num_levels - level)).div_ceil(total_weight);
        info!(
            "Progressive level {}/{}: {} epochs",
            level + 1,
            num_levels,
            level_epochs
        );

        for epoch in (0..level_epochs).step_by(args.jitter_interval) {
            let kl_weight = if args.kl_warmup_epochs > 0.0 {
                1.0 - (-(epoch_counter as f64) / args.kl_warmup_epochs).exp()
            } else {
                1.0
            };

            for jitter in 0..args.jitter_interval {
                if epoch + jitter >= level_epochs {
                    break;
                }

                let pair_sg = params
                    .pair
                    .posterior_sample()?
                    .sum_to_one_columns()
                    .scale(args.column_sum_norm)
                    .transpose();

                let null_sg = params
                    .null
                    .posterior_sample()?
                    .sum_to_one_columns()
                    .scale(args.column_sum_norm)
                    .transpose();

                let mut data_loader = InMemoryData::from(InMemoryArgs {
                    input: &pair_sg,
                    input_null: Some(&null_sg),
                    output: None,
                    output_null: None,
                })?;

                data_loader.shuffle_minibatch(args.batch_size)?;

                let mut llik_tot = 0f32;
                let mut kl_tot = 0f32;

                for b in 0..data_loader.num_minibatch() {
                    let mb = data_loader.minibatch_shuffled(b, &dev)?;
                    let x_pair = &mb.input;
                    let x_null = mb.input_null.as_ref();
                    let y_nd = x_pair;

                    let (theta, kl) = encoder.forward_t(x_pair, x_null, true)?;

                    let theta = if args.topic_smoothing > 0.0 {
                        let alpha = args.topic_smoothing;
                        let kk = theta.dim(1)? as f64;
                        ((&theta * (1.0 - alpha))? + alpha / kk)?
                    } else {
                        theta
                    };

                    let (_, llik) = decoder.forward_with_llik(&theta, y_nd, &topic_likelihood)?;
                    let loss = ((&kl * kl_weight)? - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;

                    llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                }

                let n_mb = data_loader.num_minibatch().max(1) as f32;
                kl_trace.push(kl_tot / n_mb);
                llik_trace.push(llik_tot / n_mb);

                pb.inc(1);
                epoch_counter += 1;

                if args.verbose {
                    info!(
                        "[L{}][{}][{}] llik={:.2} kl={:.2}",
                        level, epoch, jitter, llik_tot, kl_tot
                    );
                }
            }
        }
    }

    pb.finish_and_clear();
    info!("Done progressive model training");

    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

fn train_link_community(
    params: &PairNullParameters,
    encoder: &mut LogSoftmaxEncoder,
    decoder: &TopicDecoder,
    parameters: &candle_nn::VarMap,
    args: &SrtLinkCommunityArgs,
) -> anyhow::Result<TrainScores> {
    let dev = match args.device {
        ComputeDevice::Metal => Device::new_metal(args.device_no)?,
        ComputeDevice::Cuda => Device::new_cuda(args.device_no)?,
        _ => Device::Cpu,
    };

    let mut adam = AdamW::new_lr(parameters.all_vars(), args.learning_rate as f64)?;

    let pb = ProgressBar::new(args.epochs as u64);
    if args.verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut llik_trace = Vec::with_capacity(args.epochs);
    let mut kl_trace = Vec::with_capacity(args.epochs);

    info!("Start training topic link community VAE...");

    for epoch in (0..args.epochs).step_by(args.jitter_interval) {
        // KL annealing (changes slowly, per jitter block)
        let kl_weight = if args.kl_warmup_epochs > 0.0 {
            1.0 - (-(epoch as f64) / args.kl_warmup_epochs).exp()
        } else {
            1.0
        };

        for jitter in 0..args.jitter_interval {
            // Fresh posterior sample each sub-epoch for data variety
            let pair_sg = params
                .pair
                .posterior_sample()?
                .sum_to_one_columns()
                .scale(args.column_sum_norm)
                .transpose();

            let null_sg = params
                .null
                .posterior_sample()?
                .sum_to_one_columns()
                .scale(args.column_sum_norm)
                .transpose();

            let mut data_loader = InMemoryData::from(InMemoryArgs {
                input: &pair_sg,
                input_null: Some(&null_sg),
                output: None,
                output_null: None,
            })?;

            data_loader.shuffle_minibatch(args.batch_size)?;

            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;

            for b in 0..data_loader.num_minibatch() {
                let mb = data_loader.minibatch_shuffled(b, &dev)?;

                let x_pair = &mb.input;
                let x_null = mb.input_null.as_ref();
                let y_nd = x_pair; // reconstruct pair (mixture) data

                // Encoder: h(pair) - h(null) in feature space
                let (theta, kl) = encoder.forward_t(x_pair, x_null, true)?;

                // Topic smoothing
                let theta = if args.topic_smoothing > 0.0 {
                    let alpha = args.topic_smoothing;
                    let kk = theta.dim(1)? as f64;
                    ((&theta * (1.0 - alpha))? + alpha / kk)?
                } else {
                    theta
                };

                let (_, llik) = decoder.forward_with_llik(&theta, y_nd, &topic_likelihood)?;

                let loss = ((&kl * kl_weight)? - &llik)?.mean_all()?;
                adam.backward_step(&loss)?;

                let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                let kl_val = kl.sum_all()?.to_scalar::<f32>()?;
                llik_tot += llik_val;
                kl_tot += kl_val;
            }

            let n_mb = data_loader.num_minibatch().max(1) as f32;
            kl_trace.push(kl_tot / n_mb);
            llik_trace.push(llik_tot / n_mb);

            pb.inc(1);

            if args.verbose {
                info!("[{}][{}] llik={:.2} kl={:.2}", epoch, jitter, llik_tot, kl_tot);
            }
        }
    }
    pb.finish_and_clear();

    info!("Done model training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

/////////////////////////////////////////
// Global null + cell propensities     //
/////////////////////////////////////////

/// Compute global null from the random-pair posterior mean.
/// Average across S samples → [G] vector, normalize to sum-to-one, scale.
fn compute_global_null(params: &PairNullParameters, column_sum_norm: f32) -> DVec {
    let null_mean = params.null.posterior_mean();
    let g = null_mean.nrows();
    let s = null_mean.ncols();

    let mut g_vec = DVec::zeros(g);
    for j in 0..s {
        g_vec += &null_mean.column(j);
    }
    g_vec /= s as f32;

    let total = g_vec.sum();
    if total > 0.0 {
        g_vec *= column_sum_norm / total;
    }
    g_vec
}

fn evaluate_cell_propensities(
    data_vec: &SparseIoVec,
    encoder: &LogSoftmaxEncoder,
    global_null: &Tensor,
    args: &SrtLinkCommunityArgs,
) -> anyhow::Result<Mat> {
    let dev = match args.device {
        ComputeDevice::Metal => Device::new_metal(args.device_no)?,
        ComputeDevice::Cuda => Device::new_cuda(args.device_no)?,
        _ => Device::Cpu,
    };

    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();
    let block_size = args.batch_size;

    let jobs = generate_minibatch_intervals(ntot, block_size);
    let njobs = jobs.len() as u64;
    let arc_enc = Arc::new(encoder);

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&(lb, ub)| {
            let x_dn = data_vec.read_columns_csc(lb..ub)?;
            let x_nd = x_dn.to_tensor(&dev)?.transpose(0, 1)?;
            let n = x_nd.dim(0)?;
            let null_nd = global_null.broadcast_as((n, global_null.dim(1)?))?.contiguous()?;
            let (theta_nk, _) = arc_enc.forward_t(&x_nd, Some(&null_nd), false)?;
            let theta_nk = theta_nk.to_device(&Device::Cpu)?;
            Ok((lb, Mat::from_tensor(&theta_nk)?))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    chunks.par_sort_by_key(|&(lb, _)| lb);

    let mut ret = Mat::zeros(ntot, kk);
    let mut lb = 0;
    for (_, z) in chunks {
        let ub = lb + z.nrows();
        ret.rows_range_mut(lb..ub).copy_from(&z);
        lb = ub;
    }
    Ok(ret)
}

/////////////////////////////////////////
// Main pipeline                       //
/////////////////////////////////////////

/// Linked Community Model pipeline (pair + random non-pair null):
///
/// 1.  Load data + coordinates
/// 2.  Estimate batch effects
/// 3.  Build spatial cell-cell KNN graph
/// 4.  Random projection → assign pairs to samples
/// 5.  Collapse: accumulate pair/null expression per gene per sample
/// 6.  Fit Poisson-Gamma on pair and null channels
/// 7.  Initialize encoder + topic decoder
/// 8.  Train with jitter resampling (encoder: pair − null)
/// 9.  Write dictionary
/// 10. Compute global null for cell-level inference
/// 11. Inference: encoder on individual cells → propensities
pub fn fit_srt_link_community(args: &SrtLinkCommunityArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 1. Load data
    info!("[1/11] Loading data files...");

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
    let n_cells = data_vec.num_columns();

    anyhow::ensure!(args.proj_dim > 0, "proj_dim must be > 0");
    anyhow::ensure!(args.sort_dim > 0, "sort_dim must be > 0");
    anyhow::ensure!(
        args.sort_dim <= args.proj_dim,
        "sort_dim ({}) must be <= proj_dim ({})",
        args.sort_dim,
        args.proj_dim
    );
    anyhow::ensure!(args.knn_spatial > 0, "knn_spatial must be > 0");
    anyhow::ensure!(args.n_topics > 0, "n_topics must be > 0");
    anyhow::ensure!(
        args.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        args.knn_spatial,
        n_cells
    );

    // 2. Estimate batch effects
    info!("[2/11] Estimating batch effects...");

    let batch_effects = estimate_batch(
        &mut data_vec,
        &batch_membership,
        EstimateBatchArgs {
            proj_dim: args.proj_dim,
            sort_dim: args.sort_dim,
            block_size: args.block_size,
            knn_cells: args.knn_cells,
        },
    )?;

    if let Some(batch_db) = batch_effects.as_ref() {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet_with_names(
            &outfile,
            (Some(&gene_names), Some("gene")),
            batch_names.as_deref(),
        )?;
    }

    // 3. Build spatial KNN graph
    info!(
        "[3/11] Building spatial KNN graph (k={})...",
        args.knn_spatial
    );

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

    // 4. Random projection + sample assignment
    info!("[4/11] Random projection and sample assignment...");
    let proj_out = srt_cell_pairs.random_projection(
        args.proj_dim,
        args.block_size,
        Some(&batch_membership),
    )?;

    // 5. Multi-level collapse: accumulate pair/null expression statistics
    info!(
        "[5/11] Multi-level collapsing ({} levels, sort_dim={})...",
        args.num_levels, args.sort_dim
    );

    let batch_db = batch_effects.map(|x| x.posterior_mean().clone());
    let batch_ref = batch_db.as_ref();

    let collapse_args = CollapsePairsArgs {
        proj_out: &proj_out,
        finest_sort_dim: args.sort_dim,
        num_levels: args.num_levels,
        down_sample: args.down_sample,
    };

    let collapsed_stats = srt_cell_pairs.collapse_pairs_multilevel(
        &collapse_args,
        &collect_pair_null_visitor,
        &batch_ref,
        |n_samples| PairNullCollapsedStat::new(n_genes, n_samples),
    )?;

    // 6. Fit Poisson-Gamma on each level
    info!("[6/11] Fitting Poisson-Gamma model ({} levels)...", collapsed_stats.len());
    let all_params: Vec<PairNullParameters> = collapsed_stats
        .iter()
        .map(|stat| stat.optimize(None))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let params = all_params.last().ok_or(anyhow::anyhow!("no levels"))?;

    // 7. Initialize encoder + topic decoder
    info!(
        "[7/11] Initializing encoder ({} topics, {} genes)...",
        args.n_topics, n_genes
    );

    let n_topics = args.n_topics;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

    let dev = match args.device {
        ComputeDevice::Metal => Device::new_metal(args.device_no)?,
        ComputeDevice::Cuda => Device::new_cuda(args.device_no)?,
        _ => Device::Cpu,
    };

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &dev);

    let mut encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_genes,
            n_topics,
            n_modules,
            layers: &args.encoder_layers,
            use_sparsemax: false,
        },
        vb.clone(),
    )?;

    let decoder = TopicDecoder::new(n_genes, n_topics, vb.clone())?;

    info!(
        "Encoder: {} features -> {} modules -> {:?} -> {} topics",
        n_genes, n_modules, args.encoder_layers, n_topics
    );

    // 8. Train with jitter resampling
    info!("[8/11] Training ({} epochs, jitter every {})...", args.epochs, args.jitter_interval);

    let scores = if all_params.len() > 1 {
        train_link_community_progressive(&all_params, &mut encoder, &decoder, &var_map, args)?
    } else {
        train_link_community(params, &mut encoder, &decoder, &var_map, args)?
    };

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    // 9. Write dictionary
    info!("[9/11] Writing dictionary...");

    let dict_gk = decoder.get_dictionary()?;
    dict_gk
        .to_device(&Device::Cpu)?
        .to_parquet_with_names(
            &(args.out.to_string() + ".dictionary.parquet"),
            (Some(&gene_names), Some("gene")),
            None,
        )?;

    // 10. Compute global null for cell-level inference
    info!("[10/11] Computing global null...");

    let global_null = compute_global_null(params, args.column_sum_norm);
    let global_null_data: Vec<f32> = global_null.as_slice().to_vec();
    let global_null_tensor = Tensor::from_vec(global_null_data, (1, n_genes), &dev)?;

    // 11. Inference: encoder on individual cells
    info!("[11/11] Inferring cell propensities...");

    let propensity_nk =
        evaluate_cell_propensities(&data_vec, &encoder, &global_null_tensor, args)?;

    let cell_names = data_vec.column_names()?;

    propensity_nk.to_parquet_with_names(
        &(args.out.to_string() + ".propensity.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    info!("Done");
    Ok(())
}
