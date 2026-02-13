use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::estimate_batch;
use crate::srt_estimate_batch_effects::EstimateBatchArgs;
use crate::srt_input::*;
use crate::srt_random_projection::*;

use candle_util::candle_core::{self, Device, Tensor};
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_aux_linear::non_neg_linear;
use candle_util::candle_loss_functions::{poisson_likelihood, topic_log_likelihood};
use candle_util::candle_model_traits::EncoderModuleT;
use candle_util::candle_data_loader::*;
use candle_util::candle_nn::{self, AdamW, Module, Optimizer};

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

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum Likelihood {
    Poisson,
    Multinomial,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum Interaction {
    /// θ_left ⊙ θ_right (Hadamard product)
    Multiply,
    /// θ_left + θ_right (additive)
    Add,
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
        help = "Number of nearest-neighbour batches for batch effect estimation"
    )]
    knn_batches: usize,

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
        default_value_t = false,
        help = "Normalize interaction to simplex (divide by row sum)"
    )]
    normalize_interaction: bool,

    #[arg(
        long,
        value_enum,
        default_value = "multiply",
        help = "Interaction type: multiply (θ_left ⊙ θ_right) or add (θ_left + θ_right)"
    )]
    interaction: Interaction,

    #[arg(
        long,
        value_enum,
        default_value = "poisson",
        help = "Likelihood model: poisson (rate-based) or multinomial (composition-based)"
    )]
    likelihood: Likelihood,

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

    #[arg(long, short, help = "Enable verbose logging (sets RUST_LOG=info)")]
    verbose: bool,
}

/////////////////////////////////////////
// Left/Right collapsed statistics     //
/////////////////////////////////////////

pub(crate) struct PairLeftRightCollapsedStat {
    left_ds: Mat,
    right_ds: Mat,
    size_s: DVec,
    n_genes: usize,
    n_samples: usize,
}

impl PairLeftRightCollapsedStat {
    pub(crate) fn new(n_genes: usize, n_samples: usize) -> Self {
        Self {
            left_ds: Mat::zeros(n_genes, n_samples),
            right_ds: Mat::zeros(n_genes, n_samples),
            size_s: DVec::zeros(n_samples),
            n_genes,
            n_samples,
        }
    }

    pub(crate) fn optimize(
        &self,
        hyper_param: Option<(f32, f32)>,
    ) -> anyhow::Result<PairLeftRightParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));
        let shape = (self.n_genes, self.n_samples);

        let mut left = GammaMatrix::new(shape, a0, b0);
        let mut right = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating left/right pair statistics");

        left.update_stat(&self.left_ds, &sample_size_ds);
        left.calibrate();
        right.update_stat(&self.right_ds, &sample_size_ds);
        right.calibrate();

        info!("Resolved pair left/right collapsed statistics");

        Ok(PairLeftRightParameters { left, right })
    }
}

pub(crate) struct PairLeftRightParameters {
    pub(crate) left: GammaMatrix,
    pub(crate) right: GammaMatrix,
}

/// Collapse visitor: accumulate left/right expression per gene per sample.
pub(crate) fn collect_pair_left_right_visitor(
    indices: &[usize],
    data: &SrtCellPairs,
    sample: usize,
    batch_effect: &Option<&Mat>,
    arc_stat: Arc<Mutex<&mut PairLeftRightCollapsedStat>>,
) -> anyhow::Result<()> {
    let pairs: Vec<&Pair> = indices.iter().filter_map(|&j| data.pairs.get(j)).collect();

    let left = pairs.iter().map(|&x| x.left);
    let right = pairs.iter().map(|&x| x.right);

    let mut y_left = data.data.read_columns_csc(left)?;
    let mut y_right = data.data.read_columns_csc(right)?;

    // batch adjustment: divide raw counts by batch effect
    if let Some(delta_db) = *batch_effect {
        let left = pairs.iter().map(|&x| x.left);
        let right = pairs.iter().map(|&x| x.right);
        let left_batches = data.data.get_batch_membership(left);
        y_left.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    let n_genes = y_left.nrows();
    let mut local_left = DVec::zeros(n_genes);
    let mut local_right = DVec::zeros(n_genes);
    let mut local_size = 0_f32;

    for (left_col, right_col) in y_left.col_iter().zip(y_right.col_iter()) {
        for (&gene, &val) in left_col.row_indices().iter().zip(left_col.values().iter()) {
            local_left[gene] += val;
        }
        for (&gene, &val) in right_col.row_indices().iter().zip(right_col.values().iter()) {
            local_right[gene] += val;
        }
        local_size += 1.0;
    }

    let mut stat = arc_stat.lock().expect("lock pair left/right stat");
    let mut col_left = stat.left_ds.column_mut(sample);
    col_left += &local_left;
    let mut col_right = stat.right_ds.column_mut(sample);
    col_right += &local_right;
    stat.size_s[sample] += local_size;

    Ok(())
}

/////////////////////////////////////////
// Bilinear Interaction Decoder        //
/////////////////////////////////////////

struct BilinearInteractionDecoder {
    n_topics: usize,
    normalize_interaction: bool,
    interaction: Interaction,
    dictionary: candle_util::candle_aux_linear::NonNegLinear,
}

impl BilinearInteractionDecoder {
    fn new(
        n_genes: usize,
        n_topics: usize,
        normalize_interaction: bool,
        interaction: Interaction,
        vb: candle_nn::VarBuilder,
    ) -> candle_core::Result<Self> {
        let dictionary = non_neg_linear(n_topics, n_genes, vb.pp("dictionary"))?;
        Ok(Self {
            n_topics,
            normalize_interaction,
            interaction,
            dictionary,
        })
    }

    /// Forward: combine θ_left and θ_right, then decode through dictionary [N, G]
    fn forward(
        &self,
        theta_left: &Tensor,
        theta_right: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let interaction = match self.interaction {
            Interaction::Multiply => (theta_left * theta_right)?,
            Interaction::Add => (theta_left + theta_right)?,
        };
        let interaction = if self.normalize_interaction {
            let row_sum = interaction.sum_keepdim(1)?;
            interaction.broadcast_div(&row_sum)?
        } else {
            interaction
        };
        self.dictionary.forward(&interaction)
    }

    /// Forward with Poisson log-likelihood
    fn forward_with_llik(
        &self,
        theta_left: &Tensor,
        theta_right: &Tensor,
        y_nd: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let rate_nd = self.forward(theta_left, theta_right)?;
        let llik = poisson_likelihood(y_nd, &rate_nd)?;
        Ok((rate_nd, llik))
    }

    /// Forward with multinomial log-likelihood:
    /// log_p = log_softmax(rate, dim=1) over genes, then Σ_g y_g log_p_g
    fn forward_with_multinomial_llik(
        &self,
        theta_left: &Tensor,
        theta_right: &Tensor,
        y_nd: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let rate_nd = self.forward(theta_left, theta_right)?;
        let log_p_nd = candle_nn::ops::log_softmax(&rate_nd, 1)?;
        let llik = topic_log_likelihood(y_nd, &log_p_nd)?;
        Ok((rate_nd, llik))
    }

    /// Get dictionary [G, K] as non-negative weights
    fn get_dictionary(&self, dev: &Device) -> candle_core::Result<Tensor> {
        let eye = Tensor::eye(self.n_topics, candle_core::DType::F32, dev)?;
        self.dictionary.forward(&eye)?.t()
    }

    /// Get dictionary [G, K] as log-probabilities (log_softmax over genes per topic)
    fn get_log_dictionary(&self, dev: &Device) -> candle_core::Result<Tensor> {
        let eye = Tensor::eye(self.n_topics, candle_core::DType::F32, dev)?;
        let dict_kg = self.dictionary.forward(&eye)?; // [K, G]
        candle_nn::ops::log_softmax(&dict_kg, 1)?.t() // [G, K]
    }
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

fn train_link_community(
    params: &PairLeftRightParameters,
    encoder: &mut LogSoftmaxEncoder,
    decoder: &BilinearInteractionDecoder,
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

    info!("Start training bilinear link community VAE...");

    for epoch in (0..args.epochs).step_by(args.jitter_interval) {
        // Resample from Poisson-Gamma posterior (stochastic data augmentation)
        let left_sg = params
            .left
            .posterior_sample()?
            .sum_to_one_columns()
            .scale(args.column_sum_norm)
            .transpose();

        let right_sg = params
            .right
            .posterior_sample()?
            .sum_to_one_columns()
            .scale(args.column_sum_norm)
            .transpose();

        // Create data loader with paired left/right data
        let mut data_loader = InMemoryData::from(InMemoryArgs {
            input: &left_sg,
            input_null: None,
            output: Some(&right_sg),
            output_null: None,
        })?;

        data_loader.shuffle_minibatch(args.batch_size)?;

        // KL annealing
        let kl_weight = if args.kl_warmup_epochs > 0.0 {
            1.0 - (-(epoch as f64) / args.kl_warmup_epochs).exp()
        } else {
            1.0
        };

        for jitter in 0..args.jitter_interval {
            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;

            for b in 0..data_loader.num_minibatch() {
                let mb = data_loader.minibatch_shuffled(b, &dev)?;

                let x_left = &mb.input;
                let x_right = mb.output.as_ref().unwrap_or(&mb.input);
                let y_nd = (x_left + x_right)?;

                let (theta_left, kl_l) = encoder.forward_t(x_left, None, true)?;
                let (theta_right, kl_r) = encoder.forward_t(x_right, None, true)?;

                // Topic smoothing
                let theta_left = if args.topic_smoothing > 0.0 {
                    let alpha = args.topic_smoothing;
                    let kk = theta_left.dim(1)? as f64;
                    ((&theta_left * (1.0 - alpha))? + alpha / kk)?
                } else {
                    theta_left
                };

                let theta_right = if args.topic_smoothing > 0.0 {
                    let alpha = args.topic_smoothing;
                    let kk = theta_right.dim(1)? as f64;
                    ((&theta_right * (1.0 - alpha))? + alpha / kk)?
                } else {
                    theta_right
                };

                let (_, llik) = match args.likelihood {
                    Likelihood::Poisson => {
                        decoder.forward_with_llik(&theta_left, &theta_right, &y_nd)?
                    }
                    Likelihood::Multinomial => {
                        decoder.forward_with_multinomial_llik(&theta_left, &theta_right, &y_nd)?
                    }
                };

                let kl = (&kl_l + &kl_r)?;
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
// Inference: cell propensities        //
/////////////////////////////////////////

fn evaluate_cell_propensities(
    data_vec: &SparseIoVec,
    encoder: &LogSoftmaxEncoder,
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
            let (theta_nk, _) = arc_enc.forward_t(&x_nd, None, false)?;
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

/// Bipartite Linked Community Model pipeline:
///
/// 1. Load data + coordinates
/// 2. Estimate batch effects
/// 3. Build spatial cell-cell KNN graph
/// 4. Random projection → assign pairs to samples
/// 5. Collapse: accumulate left/right expression per gene per sample
/// 6. Fit Poisson-Gamma on each channel
/// 7. Initialize encoder + bilinear decoder
/// 8. Train with jitter resampling
/// 9. Inference: encoder on individual cells → propensities
/// 10. Save outputs
pub fn fit_srt_link_community(args: &SrtLinkCommunityArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }

    // 1. Load data
    info!("[1/10] Loading data files...");

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
    info!("[2/10] Estimating batch effects...");

    let batch_effects = estimate_batch(
        &mut data_vec,
        &batch_membership,
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
        batch_db.to_parquet_with_names(
            &outfile,
            (Some(&gene_names), Some("gene")),
            batch_names.as_deref(),
        )?;
    }

    // 3. Build spatial KNN graph
    info!(
        "[3/10] Building spatial KNN graph (k={})...",
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
    info!("[4/10] Random projection and sample assignment...");
    let proj_out = srt_cell_pairs.random_projection(
        args.proj_dim,
        args.block_size,
        Some(&batch_membership),
    )?;

    srt_cell_pairs.assign_pairs_to_samples(&proj_out, Some(args.sort_dim), args.down_sample)?;

    // 5. Collapse: accumulate left/right expression per gene per sample
    info!("[5/10] Collapsing left/right expression statistics...");

    let batch_db = batch_effects.map(|x| x.posterior_mean().clone());
    let batch_ref = batch_db.as_ref();

    let mut collapsed_stat =
        PairLeftRightCollapsedStat::new(n_genes, srt_cell_pairs.num_samples()?);

    srt_cell_pairs.visit_pairs_by_sample(
        &collect_pair_left_right_visitor,
        &batch_ref,
        &mut collapsed_stat,
    )?;

    // 6. Fit Poisson-Gamma on each channel
    info!("[6/10] Fitting Poisson-Gamma model...");
    let params = collapsed_stat.optimize(None)?;

    // 7. Initialize encoder + bilinear decoder
    info!(
        "[7/10] Initializing encoder ({} topics, {} genes)...",
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

    let decoder =
        BilinearInteractionDecoder::new(n_genes, n_topics, args.normalize_interaction, args.interaction.clone(), vb.clone())?;

    info!(
        "Encoder: {} features -> {} modules -> {:?} -> {} topics",
        n_genes, n_modules, args.encoder_layers, n_topics
    );

    // 8. Train with jitter resampling
    info!("[8/10] Training ({} epochs, jitter every {})...", args.epochs, args.jitter_interval);

    let scores = train_link_community(&params, &mut encoder, &decoder, &var_map, args)?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    // Write dictionary
    info!("[9/10] Writing dictionary...");

    let dict_gk = match args.likelihood {
        Likelihood::Multinomial => decoder.get_log_dictionary(&dev)?,
        Likelihood::Poisson => decoder.get_dictionary(&dev)?,
    };
    dict_gk
        .to_device(&Device::Cpu)?
        .to_parquet_with_names(
            &(args.out.to_string() + ".dictionary.parquet"),
            (Some(&gene_names), Some("gene")),
            None,
        )?;

    // 9. Inference: encoder on individual cells
    info!("[10/10] Inferring cell propensities...");

    let propensity_nk = evaluate_cell_propensities(&data_vec, &encoder, args)?;

    let cell_names = data_vec.column_names()?;

    propensity_nk.to_parquet_with_names(
        &(args.out.to_string() + ".propensity.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    info!("Done");
    Ok(())
}
