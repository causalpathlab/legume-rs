use crate::embed_common::*;
use crate::feature_selection::*;
use crate::fit_topic::{AdjMethod, ComputeDevice, TrainScores};
use crate::senna_input::*;

use candle_core::{Device, Tensor};
use candle_nn::AdamW;
use candle_nn::Optimizer;
use candle_util::candle_decoder_indexed_topic::*;
use candle_util::candle_encoder_indexed::*;
use candle_util::candle_indexed_data_loader::*;
use candle_util::candle_indexed_model_traits::*;
use indicatif::ParallelProgressIterator;
use indicatif::ProgressBar;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct IndexedTopicArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided (space or comma separated)."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header for results.\n\
		     Specify the output file or prefix for generated files:\n\
		     - {out}.delta.parquet\n\
		     - {out}.dictionary.parquet\n\
		     - {out}.latent.parquet (log-softmax topic proportions)\n"
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension"
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection"
    )]
    sort_dim: usize,

    #[arg(long, short, value_delimiter(','), help = "Batch membership files")]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = false, help = "Ignore batch adjustment")]
    ignore_batch_effects: bool,

    #[arg(short = 'w', long = "warm-start", help = "Warm start projection file")]
    warm_start_proj_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of k-nearest neighbour batches"
    )]
    knn_batches: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch"
    )]
    knn_cells: usize,

    #[arg(long, value_delimiter(','), help = "Reference batch names")]
    reference_batches: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of multi-level coarsening levels"
    )]
    num_levels: usize,

    #[arg(long, default_value_t = false, help = "Disable super-cell matching")]
    no_supercell: bool,

    #[arg(long, default_value_t = 30, help = "Optimization iterations")]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing"
    )]
    block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics"
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder layers (comma-separated)"
    )]
    encoder_layers: Vec<usize>,

    #[arg(long, default_value_t = 0.0, help = "KL annealing warmup epochs")]
    kl_warmup_epochs: f64,

    #[arg(
        long,
        short = 'i',
        default_value_t = 1000,
        help = "Number of training epochs"
    )]
    epochs: usize,

    #[arg(long, short = 'j', default_value_t = 5, help = "Data jitter interval")]
    jitter_interval: usize,

    #[arg(long, default_value_t = 100, help = "Minibatch size")]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3, help = "Learning rate")]
    learning_rate: f32,

    #[arg(long, value_enum, default_value = "cpu", help = "Candle device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "A device for cuda")]
    device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Adjustment method"
    )]
    adj_method: AdjMethod,

    #[arg(long, default_value_t = false, help = "Preload all columns data")]
    preload_data: bool,

    #[arg(long, help = "Maximum number of highly variable features")]
    max_features: Option<usize>,

    #[arg(long, help = "Pre-computed feature selection file")]
    feature_list_file: Option<Box<str>>,

    #[arg(
        long,
        alias = "high-sd",
        help = "Exclude highly expressed features (SD threshold)"
    )]
    exclude_high_expression_sd: Option<f32>,

    #[arg(long, help = "Save feature variance statistics")]
    save_feature_variance: bool,

    #[arg(long, default_value_t = 0.01, help = "Topic smoothing during training")]
    topic_smoothing: f64,

    // Indexed-specific args
    #[arg(
        long,
        default_value_t = 512,
        help = "Top-K features per sample (context window size)",
        long_help = "Number of top features to keep per sample by value.\n\
                     Each sample selects its top-K features; minibatches use\n\
                     the union of selected indices. Smaller K = faster decoder."
    )]
    context_size: usize,

    #[arg(
        long,
        default_value_t = 128,
        help = "Feature embedding dimension",
        long_help = "Dimensionality of per-feature embeddings.\n\
                     Features are aggregated via [N, S] × [S, H] matmul\n\
                     instead of dense [N, D] × [D, M] in the standard encoder."
    )]
    embedding_dim: usize,
}

pub fn fit_indexed_topic_model(args: &IndexedTopicArgs) -> anyhow::Result<()> {
    let reference = args.reference_batches.as_deref();

    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
        nbatch,
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;

    // Feature selection (if requested)
    let selected_features: Option<FeatureSelection> = if args.warm_start_proj_file.is_some() {
        if args.max_features.is_some() || args.feature_list_file.is_some() {
            info!("Warm-start provided: skipping feature selection (may be incompatible)");
        }
        None
    } else if args.max_features.is_some() || args.feature_list_file.is_some() {
        Some(select_highly_variable_features(
            &data_vec,
            args.max_features,
            args.feature_list_file.as_deref(),
            args.save_feature_variance,
            &args.out,
            args.block_size,
            args.exclude_high_expression_sd,
        )?)
    } else {
        None
    };

    // 2. Projection
    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file.as_deref() {
        use matrix_util::common_io::*;
        let ext = file_ext(proj_file)?;

        let MatWithNames {
            rows: cell_names,
            cols: _,
            mat: proj_nk,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_row_names(proj_file, Some(0))?,
            _ => Mat::read_data_with_names(proj_file, &['\t', ',', ' '], Some(0), Some(0))?,
        };

        if data_vec.column_names()? != cell_names {
            return Err(anyhow::anyhow!(
                "warm start projection rows don't match with the data"
            ));
        }

        proj_nk.transpose()
    } else {
        let proj_dim = args.proj_dim.max(args.n_latent_topics);

        let proj_out = data_vec.project_columns_with_batch_correction(
            proj_dim,
            Some(args.block_size),
            Some(&batch_membership),
        )?;

        proj_out.proj
    };

    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    // 3. Multi-level collapsing (pseudobulk)
    let use_multilevel = args.num_levels > 1 && !args.no_supercell;

    let collapsed_levels: Vec<CollapsedOut> = if use_multilevel {
        info!("Multi-level collapsing with super-cells ...");
        let mut levels = data_vec.collapse_columns_multilevel_vec(
            &proj_kn,
            &batch_membership,
            &MultilevelParams {
                knn_super_cells: args.knn_cells,
                num_levels: args.num_levels,
                sort_dim: args.sort_dim,
                num_opt_iter: args.iter_opt,
            },
        )?;
        // Reverse so training goes coarse→fine: coarsest (fewest samples)
        // gets the most epochs for a warm start, finest gets brief refinement.
        levels.reverse();
        levels
    } else {
        let nsamp = data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), None)?;

        if !args.ignore_batch_effects && nbatch > 1 {
            info!("Registering batch information");
            data_vec.build_hnsw_per_batch(&proj_kn, &batch_membership)?;
        }

        info!("Collapsing columns into {} pseudobulk samples ...", nsamp);
        vec![data_vec.collapse_columns(
            Some(args.knn_batches),
            Some(args.knn_cells),
            reference,
            Some(args.iter_opt),
        )?]
    };

    // After reversing, the finest level (most groups) is the last element.
    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();

    let batch_db = finest_collapsed.delta.as_ref();

    if let Some(batch_db) = batch_db {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet_with_names(
            &outfile,
            (Some(&gene_names), Some("gene")),
            batch_names.as_deref(),
        )?;
    }

    // 4. Train indexed topic model on collapsed data
    let n_topics = args.n_latent_topics;

    let n_features_decoder = selected_features
        .as_ref()
        .map(|sel| sel.selected_indices.len())
        .unwrap_or_else(|| data_vec.num_rows());

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(args.device_no)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(args.device_no)?,
        _ => candle_core::Device::Cpu,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_decoder,
            n_topics,
            embedding_dim: args.embedding_dim,
            layers: &args.encoder_layers,
        },
        param_builder.pp("enc"),
    )?;

    let decoder = IndexedTopicDecoder::new(n_features_decoder, n_topics, param_builder.pp("dec"))?;

    info!(
        "input: {} -> indexed encoder (emb={}, ctx={}) -> indexed decoder -> output: {}",
        n_features_decoder, args.embedding_dim, args.context_size, n_features_decoder
    );

    let gene_names = data_vec.row_names()?;

    let output_gene_names = selected_features
        .as_ref()
        .map(|sel| sel.selected_names.clone())
        .unwrap_or_else(|| gene_names.clone());

    // Set up graceful stop flag
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            info!("Interrupt received — stopping training early and saving results...");
            stop.store(true, Ordering::SeqCst);
        })
        .expect("failed to set signal handler");
    }

    let train_config = ProgressiveTrainConfig {
        parameters: &parameters,
        dev: &dev,
        epochs: args.epochs,
        jitter_interval: args.jitter_interval,
        minibatch_size: args.minibatch_size,
        learning_rate: args.learning_rate,
        kl_warmup_epochs: args.kl_warmup_epochs,
        topic_smoothing: args.topic_smoothing,
        context_size: args.context_size,
        feature_selection: selected_features.as_ref(),
        stop: &stop,
    };

    let scores = train_indexed_progressive(&collapsed_levels, &encoder, &decoder, &train_config)?;

    info!("Writing down the model parameters");

    decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?
        .to_parquet_with_names(
            &(args.out.to_string() + ".dictionary.parquet"),
            (Some(&output_gene_names), Some("gene")),
            None,
        )?;

    info!("Writing down the latent states");
    let eval_config = EvaluateLatentConfig {
        dev: &dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size,
        context_size: args.context_size,
        feature_selection: selected_features.as_ref(),
    };
    let z_nk =
        evaluate_latent_by_indexed_encoder(&data_vec, &encoder, finest_collapsed, &eval_config)?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    if let Some(sel) = &selected_features {
        use matrix_util::common_io::write_lines;
        let feature_file = args.out.to_string() + ".selected_features.txt";
        write_lines(&sel.selected_names, &feature_file)?;
        info!(
            "Saved {} selected features to {}",
            sel.selected_names.len(),
            feature_file
        );
    }

    info!("Done");
    Ok(())
}

///////////////////////
// training routines //
///////////////////////

struct ProgressiveTrainConfig<'a> {
    parameters: &'a candle_nn::VarMap,
    dev: &'a Device,
    epochs: usize,
    jitter_interval: usize,
    minibatch_size: usize,
    learning_rate: f32,
    kl_warmup_epochs: f64,
    topic_smoothing: f64,
    context_size: usize,
    feature_selection: Option<&'a FeatureSelection>,
    stop: &'a AtomicBool,
}

/// Compute per-level epoch allocation for progressive training.
fn compute_level_epochs(num_levels: usize, total_epochs: usize) -> Vec<usize> {
    let total_weight: usize = (1..=num_levels).sum();
    (0..num_levels)
        .map(|i| {
            let w = num_levels - i;
            (total_epochs * w / total_weight).max(1)
        })
        .collect()
}

/// Sample collapsed data for one jitter interval.
fn sample_collapsed_data(
    collapsed: &CollapsedOut,
    feature_selection: Option<&FeatureSelection>,
) -> anyhow::Result<(Mat, Option<Mat>, Option<Mat>)> {
    let mut mixed_nd = collapsed.mu_observed.posterior_sample()?.transpose();

    if let Some(sel) = feature_selection {
        mixed_nd = mixed_nd.select_columns(&sel.selected_indices);
    }

    let clean_nd = collapsed.mu_adjusted.as_ref().map(|x| {
        let mut ret: Mat = x.posterior_sample().unwrap();
        ret = ret.transpose();
        if let Some(sel) = feature_selection {
            ret = ret.select_columns(&sel.selected_indices);
        }
        ret
    });

    let batch_nd = collapsed.mu_residual.as_ref().map(|x| {
        let mut ret: Mat = x.posterior_sample().unwrap();
        ret = ret.transpose();
        if let Some(sel) = feature_selection {
            ret = ret.select_columns(&sel.selected_indices);
        }
        ret
    });

    if let Some(ref batch) = batch_nd {
        if batch.ncols() != mixed_nd.ncols() {
            return Err(anyhow::anyhow!(
                "mixed_nd and batch_nd have different feature dimensions: {} vs {}",
                mixed_nd.ncols(),
                batch.ncols()
            ));
        }
    }

    Ok((mixed_nd, clean_nd, batch_nd))
}

/// Apply topic smoothing in log-space.
fn smooth_topics(log_z_nk: Tensor, alpha: f64) -> candle_core::Result<Tensor> {
    if alpha > 0.0 {
        let kk = log_z_nk.dim(1)? as f64;
        ((log_z_nk.exp()? * (1.0 - alpha))? + alpha / kk)?.log()
    } else {
        Ok(log_z_nk)
    }
}

fn train_indexed_progressive(
    collapsed_levels: &[CollapsedOut],
    encoder: &IndexedEmbeddingEncoder,
    decoder: &IndexedTopicDecoder,
    config: &ProgressiveTrainConfig,
) -> anyhow::Result<TrainScores> {
    let num_levels = collapsed_levels.len();
    let level_epochs = compute_level_epochs(num_levels, config.epochs);

    info!(
        "Progressive training: {} levels, epoch allocation: {:?} (total {})",
        num_levels,
        level_epochs,
        level_epochs.iter().sum::<usize>()
    );

    let mut adam = AdamW::new_lr(config.parameters.all_vars(), config.learning_rate as f64)?;
    let total_actual_epochs: usize = level_epochs.iter().sum();
    let pb = ProgressBar::new(total_actual_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_actual_epochs);
    let mut kl_trace = Vec::with_capacity(total_actual_epochs);
    let mut global_epoch: usize = 0;

    for (level, (collapsed, &level_ep)) in
        collapsed_levels.iter().zip(level_epochs.iter()).enumerate()
    {
        let label = if level == 0 {
            "coarsest"
        } else if level + 1 == num_levels {
            "finest"
        } else {
            ""
        };
        info!(
            "Level {}/{}: {} epochs, {} samples {}",
            level + 1,
            num_levels,
            level_ep,
            collapsed.mu_observed.ncols(),
            label,
        );

        for epoch in (0..level_ep).step_by(config.jitter_interval) {
            let (mixed_nd, clean_nd, batch_nd) =
                sample_collapsed_data(collapsed, config.feature_selection)?;

            let mut data_loader = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                input: &mixed_nd,
                input_null: batch_nd.as_ref(),
                output: clean_nd.as_ref(),
                output_null: None,
                context_size: config.context_size,
            })?;

            data_loader.shuffle_minibatch(config.minibatch_size);

            let kl_weight = if config.kl_warmup_epochs > 0.0 {
                1.0 - (-(global_epoch as f64) / config.kl_warmup_epochs).exp()
            } else {
                1.0
            };

            let jitter_end = config.jitter_interval.min(level_ep - epoch);
            for jitter in 0..jitter_end {
                let mut llik_tot = 0f32;
                let mut kl_tot = 0f32;
                let mut count_tot = 0f32;

                for b in 0..data_loader.num_minibatch() {
                    let mb = data_loader.minibatch_shuffled(b, config.dev)?;
                    let (log_z_nk, kl) = encoder.forward_indexed_t(
                        &mb.union_indices,
                        &mb.indexed_x,
                        mb.indexed_x_null.as_ref(),
                        true,
                    )?;

                    let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;

                    let target_x = mb.indexed_y.as_ref().unwrap_or(&mb.indexed_x);
                    let (_, llik) =
                        decoder.forward_indexed(&log_z_nk, &mb.union_indices, target_x)?;

                    let loss = ((&kl * kl_weight)? - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;

                    llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    count_tot += target_x.sum_all()?.to_scalar::<f32>()?;
                }

                let n = data_loader.num_data() as f32;
                llik_trace.push(llik_tot / count_tot);
                kl_trace.push(kl_tot / n);

                pb.inc(1);
                global_epoch += 1;

                info!(
                    "[level {}/{}][{}][{}] {} {}",
                    level + 1,
                    num_levels,
                    epoch,
                    jitter,
                    llik_trace.last().unwrap(),
                    kl_trace.last().unwrap()
                );

                if config.stop.load(Ordering::SeqCst) {
                    pb.finish_and_clear();
                    info!(
                        "Stopping early at level {}/{}, epoch {}",
                        level + 1,
                        num_levels,
                        epoch
                    );
                    return Ok(TrainScores {
                        llik: llik_trace,
                        kl: kl_trace,
                    });
                }
            }
        }
    }

    pb.finish_and_clear();
    info!("done progressive model training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

/// Convert a dense [N, D] tensor to indexed form: union_indices [S] and indexed_x [N, S].
fn dense_to_indexed(
    x_nd: &Tensor,
    context_size: usize,
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let rows: Vec<Vec<f32>> = x_nd.to_vec2()?;
    let n_batch = rows.len();

    let mut union_set = BTreeSet::new();
    let mut all_top_k: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(n_batch);

    for row in &rows {
        let (indices, values) = top_k_indices(row, context_size);
        for &idx in &indices {
            union_set.insert(idx);
        }
        all_top_k.push((indices, values));
    }

    let union_vec: Vec<u32> = union_set.into_iter().collect();
    let s = union_vec.len();

    let pos_map: HashMap<u32, usize> = union_vec
        .iter()
        .enumerate()
        .map(|(pos, &idx)| (idx, pos))
        .collect();

    let mut x_data = vec![0.0f32; n_batch * s];
    for (row, (indices, values)) in all_top_k.iter().enumerate() {
        for (k, &feat_idx) in indices.iter().enumerate() {
            let col = pos_map[&feat_idx];
            x_data[row * s + col] = values[k];
        }
    }

    let union_indices =
        Tensor::from_vec(union_vec, (s,), dev)?.to_dtype(candle_core::DType::U32)?;
    let indexed_x = Tensor::from_vec(x_data, (n_batch, s), dev)?;

    Ok((union_indices, indexed_x))
}

/////////////////////////
// evaluation routines //
/////////////////////////

struct EvaluateLatentConfig<'a> {
    dev: &'a Device,
    adj_method: &'a AdjMethod,
    minibatch_size: usize,
    context_size: usize,
    feature_selection: Option<&'a FeatureSelection>,
}

fn evaluate_latent_by_indexed_encoder(
    data_vec: &SparseIoVec,
    encoder: &IndexedEmbeddingEncoder,
    collapsed: &CollapsedOut,
    config: &EvaluateLatentConfig,
) -> anyhow::Result<Mat> {
    let ntot = data_vec.num_columns();
    let kk = encoder.n_topics();

    let block_size = config.minibatch_size;

    let jobs = create_jobs(ntot, Some(block_size));
    let njobs = jobs.len() as u64;

    let delta = match config.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| {
        let mut delta_db = x.posterior_mean().clone();
        if let Some(sel) = config.feature_selection {
            delta_db = delta_db.select_rows(&sel.selected_indices);
        }
        delta_db
    })
    .map(|delta_db| {
        delta_db
            .to_tensor(config.dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
    });

    let block_config = EvaluateBlockConfig {
        dev: config.dev,
        adj_method: config.adj_method,
        delta: delta.as_ref(),
        context_size: config.context_size,
        feature_selection: config.feature_selection,
    };

    let eval_block = |block| -> anyhow::Result<(usize, Mat)> {
        evaluate_indexed_block(block, data_vec, encoder, &block_config)
    };

    // GPU forward passes are not thread-safe — run sequentially on Metal/CUDA
    let mut chunks: Vec<(usize, Mat)> = if config.dev.is_cpu() {
        jobs.par_iter()
            .progress_count(njobs)
            .map(|&block| eval_block(block))
            .collect::<anyhow::Result<Vec<_>>>()?
    } else {
        let pb = ProgressBar::new(njobs);
        let result = jobs
            .iter()
            .map(|&block| {
                let r = eval_block(block);
                pb.inc(1);
                r
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        pb.finish_and_clear();
        result
    };

    chunks.sort_by_key(|&(lb, _)| lb);

    let mut ret = Mat::zeros(ntot, kk);
    let mut lb = 0;
    for (_, z) in chunks {
        let ub = lb + z.nrows();
        ret.rows_range_mut(lb..ub).copy_from(&z);
        lb = ub;
    }
    Ok(ret)
}

struct EvaluateBlockConfig<'a> {
    dev: &'a Device,
    adj_method: &'a AdjMethod,
    delta: Option<&'a Tensor>,
    context_size: usize,
    feature_selection: Option<&'a FeatureSelection>,
}

fn evaluate_indexed_block(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &IndexedEmbeddingEncoder,
    config: &EvaluateBlockConfig,
) -> anyhow::Result<(usize, Mat)> {
    let (lb, ub) = block;

    // Read sparse -> dense [D, N] -> transpose -> [N, D]
    let mut x_dn = data_vec.read_columns_csc(lb..ub)?;

    if let Some(sel) = config.feature_selection {
        x_dn = filter_csc_by_rows(&sel.selection_matrix, &x_dn);
    }

    let x_nd = x_dn.to_tensor(config.dev)?.transpose(0, 1)?;

    // Get batch/residual correction if available
    let x0_nd = config.delta.map(|delta_bm| match config.adj_method {
        AdjMethod::Batch => {
            let batches = data_vec
                .get_batch_membership(lb..ub)
                .into_iter()
                .map(|x| x as u32);
            let batches = Tensor::from_iter(batches, config.dev).unwrap();
            delta_bm.index_select(&batches, 0).expect("expand delta")
        }
        AdjMethod::Residual => {
            let groups = data_vec
                .get_group_membership(lb..ub)
                .expect("failed to get group membership")
                .into_iter()
                .map(|x| x as u32);
            let groups = Tensor::from_iter(groups, config.dev).unwrap();
            delta_bm.index_select(&groups, 0).expect("expand delta")
        }
    });

    // Convert dense to indexed format
    let (union_indices, indexed_x) = dense_to_indexed(&x_nd, config.context_size, config.dev)?;

    // Scatter batch correction at union positions if available
    let indexed_x_null = if let Some(x0) = &x0_nd {
        let union_vec: Vec<u32> = union_indices.to_vec1()?;
        let s = union_vec.len();
        let n_batch = ub - lb;
        let pos_map: HashMap<u32, usize> = union_vec
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();
        let x0_vec: Vec<Vec<f32>> = x0.to_vec2()?;
        let mut x0_data = vec![0.0f32; n_batch * s];
        for (row, x0_row) in x0_vec.iter().enumerate() {
            for &feat_idx in &union_vec {
                let col = pos_map[&feat_idx];
                x0_data[row * s + col] = x0_row[feat_idx as usize];
            }
        }
        Some(Tensor::from_vec(x0_data, (n_batch, s), config.dev)?)
    } else {
        None
    };

    let (log_z_nk, _) =
        encoder.forward_indexed_t(&union_indices, &indexed_x, indexed_x_null.as_ref(), false)?;

    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}
