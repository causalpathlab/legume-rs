use crate::common::*;
use candle_util::candle_bipartite_decoder::BipartiteDecoder;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_encoder_softmax::{LogSoftmaxEncoder, LogSoftmaxEncoderArgs};
use candle_util::candle_model_traits::EncoderModuleT;
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans::convert::try_open_or_convert;
use data_beans_alg::collapse_data::{CollapsedOut, MultilevelParams};
use matrix_util::traits::IoOps;
use rayon::prelude::*;

#[derive(clap::ValueEnum, Clone, Debug)]
#[clap(rename_all = "lowercase")]
pub enum LevelSchedule {
    /// All levels trained simultaneously each epoch
    Mixed,
    /// Coarse→fine, more epochs for coarser levels
    Progressive,
}

#[derive(Args, Debug)]
pub struct EmbedArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Data files (sparse backends: zarr, h5)"
    )]
    data_files: Vec<Box<str>>,

    #[arg(long, short, required = true, help = "Output prefix")]
    out: Box<str>,

    #[arg(long, short = 't', default_value_t = 20, help = "Number of topics")]
    n_topics: usize,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension"
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Sort dimension for cell collapsing"
    )]
    sort_dim: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of multi-level collapsing levels"
    )]
    num_levels: usize,

    #[arg(long, default_value_t = 10, help = "KNN for super-cell matching")]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Optimization iterations for collapsing"
    )]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 5000,
        help = "Max coarse features for encoder/decoder"
    )]
    max_coarse_features: usize,

    #[arg(long, default_value_t = 100, help = "Block size for I/O")]
    block_size: usize,

    #[arg(long, default_value_t = 500, help = "Total training epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 1e-3, help = "Learning rate")]
    learning_rate: f64,

    #[arg(long, default_value_t = 50.0, help = "KL warmup epochs")]
    kl_warmup: f64,

    #[arg(
        long,
        default_value_t = 5,
        help = "Jitter interval (resample collapsed data)"
    )]
    jitter_interval: usize,

    #[arg(
        long,
        value_enum,
        default_value = "progressive",
        help = "Multi-level training schedule"
    )]
    level_schedule: LevelSchedule,

    #[arg(
        long,
        default_value_t = false,
        help = "Use full relation matrix R [K,K] instead of diagonal"
    )]
    full_relation: bool,

    #[arg(
        short = 'b',
        long,
        value_delimiter(','),
        help = "Batch membership files (comma-separated, one per data file)"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = false, help = "Preload data into memory")]
    preload: bool,
}

/// Bipartite VAE: jointly embed features and cells.
///
/// Data flow (no full dense matrix ever materialized):
/// 1. Cell collapse: Sparse [D × N] → CollapsedOut [D × S]  (S << N)
/// 2. Feature coarsen: [D × S] → [D_l × S] per decoder level
/// 3. Cell encoder (dense): D_encoder features → z_c [S, K]
/// 4. Feature encoder (dense): S super-cells → z_f [D_encoder, K]
/// 5. Per-level decoders at [D_l × S], all trained simultaneously
pub fn run_embed(args: &EmbedArgs) -> anyhow::Result<()> {
    let dev = Device::Cpu;

    // 1. Load data with batch membership
    let mut data_vec = SparseIoVec::new();
    let attach_data_name = args.data_files.len() > 1;
    for data_file in args.data_files.iter() {
        info!("Loading: {}", data_file);
        let mut data = try_open_or_convert(data_file)?;
        if args.preload {
            data.preload_columns()?;
        }
        let data_name = attach_data_name
            .then(|| matrix_util::common_io::basename(data_file))
            .transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }

    let d_full = data_vec.num_rows();
    let n_full = data_vec.num_columns();
    info!("Data: {} features × {} cells", d_full, n_full);

    // Batch membership
    let batch_membership: Vec<Box<str>> = if let Some(batch_files) = &args.batch_files {
        let mut membership = Vec::new();
        for batch_file in batch_files {
            info!("Reading batch file: {}", batch_file);
            for s in matrix_util::common_io::read_lines(batch_file)? {
                membership.push(s.to_string().into_boxed_str());
            }
        }
        if membership.len() != n_full {
            return Err(anyhow::anyhow!(
                "batch membership length {} != number of cells {}",
                membership.len(),
                n_full
            ));
        }
        membership
    } else {
        vec!["batch0".into(); n_full]
    };

    // 2. Random projection + cell collapse → pseudobulk [D × S]
    let proj_dim = args.proj_dim.min(d_full);
    let proj_out = data_vec.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(&batch_membership),
    )?;
    let proj_kn = proj_out.proj;

    let mut collapsed_levels = data_vec.collapse_columns_multilevel_vec(
        &proj_kn,
        &batch_membership,
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
        },
    )?;
    collapsed_levels.reverse(); // coarsest first

    let num_levels = collapsed_levels.len();
    for (i, c) in collapsed_levels.iter().enumerate() {
        info!(
            "Collapsed level {}: {} features × {} super-cells",
            i,
            c.mu_observed.nrows(),
            c.mu_observed.ncols()
        );
    }

    // Save batch effects (delta) if present
    let finest_collapsed = collapsed_levels.last().unwrap();
    if let Some(batch_db) = &finest_collapsed.delta {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_melted_parquet(
            &outfile,
            (Some(&gene_names), Some("gene")),
            (batch_names.as_deref(), Some("batch")),
        )?;
        info!("Saved batch effects to {}", outfile);
    }

    // 3. Per-level feature coarsenings
    let n_features_full = data_vec.num_rows();
    let sketch_ds = finest_collapsed.mu_observed.posterior_mean().clone();

    let level_coarsenings: Vec<Option<FeatureCoarsening>> = if args.max_coarse_features > 0
        && n_features_full > args.max_coarse_features
    {
        let finest_target = args.max_coarse_features;
        let min_target = (finest_target / num_levels).max(50);
        (0..num_levels)
            .map(|i| {
                let frac = if num_levels > 1 {
                    i as f64 / (num_levels - 1) as f64
                } else {
                    1.0
                };
                let log_min = (min_target as f64).ln();
                let log_max = (finest_target as f64).ln();
                let target = (log_min + frac * (log_max - log_min)).exp().round() as usize;
                let target = target.clamp(min_target, finest_target);
                Some(compute_feature_coarsening(&sketch_ds, target).expect("feature coarsening"))
            })
            .collect()
    } else {
        vec![None; num_levels]
    };

    let finest_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());
    let d_encoder = finest_coarsening
        .map(|c| c.num_coarse)
        .unwrap_or(n_features_full);
    let n_supercells = finest_collapsed.mu_observed.ncols();

    let level_decoder_dims: Vec<usize> = level_coarsenings
        .iter()
        .map(|fc| fc.as_ref().map(|c| c.num_coarse).unwrap_or(n_features_full))
        .collect();

    info!(
        "Encoder: D_enc={}, S={}, decoder dims: {:?}",
        d_encoder, n_supercells, level_decoder_dims
    );

    // 4. Build model
    let k = args.n_topics;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    // Cell encoder: D_encoder features → z_c [K]  (encodes super-cells)
    let cell_encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: d_encoder,
            n_topics: k,
            layers: &[128, 64],
        },
        vb.pp("cell_enc"),
    )?;

    // Feature encoder: S super-cells → z_f [K]  (encodes features)
    let feat_encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_supercells,
            n_topics: k,
            layers: &[128, 64],
        },
        vb.pp("feat_enc"),
    )?;

    // Per-level bipartite decoders
    let decoders: Vec<BipartiteDecoder> = collapsed_levels
        .iter()
        .zip(level_decoder_dims.iter())
        .enumerate()
        .map(|(i, (collapsed, &d_l))| {
            let s_l = collapsed.mu_observed.ncols();
            if args.full_relation {
                BipartiteDecoder::new_full(k, d_l, s_l, vb.pp(format!("dec_{i}")))
            } else {
                BipartiteDecoder::new_diagonal(k, d_l, s_l, vb.pp(format!("dec_{i}")))
            }
            .expect("decoder creation")
        })
        .collect();

    // 5. Optimizer
    let mut adam = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.learning_rate,
            weight_decay: 1e-4,
            ..Default::default()
        },
    )?;

    // 6. Training
    let total_epochs = args.epochs;

    match args.level_schedule {
        LevelSchedule::Mixed => train_mixed(
            &collapsed_levels,
            &level_coarsenings,
            finest_coarsening,
            &cell_encoder,
            &feat_encoder,
            &decoders,
            d_encoder,
            n_supercells,
            &mut adam,
            total_epochs,
            args.jitter_interval,
            args.kl_warmup,
            &dev,
        )?,
        LevelSchedule::Progressive => train_progressive(
            &collapsed_levels,
            &level_coarsenings,
            finest_coarsening,
            &cell_encoder,
            &feat_encoder,
            &decoders,
            d_encoder,
            n_supercells,
            &mut adam,
            total_epochs,
            args.jitter_interval,
            args.kl_warmup,
            &dev,
        )?,
    };

    // 7. Per-cell embeddings: project all N original cells through trained encoder
    info!("Evaluating per-cell embeddings ({} cells)...", n_full);

    // Precompute batch delta for encoder null subtraction
    let delta_tensor = finest_collapsed
        .mu_residual
        .as_ref()
        .map(|x| {
            let mut delta_ds = x.posterior_mean().clone();
            if let Some(fc) = finest_coarsening {
                delta_ds = fc.aggregate_rows_ds(&delta_ds);
            }
            delta_ds.to_tensor(&dev).and_then(|t| {
                t.transpose(0, 1)
                    .map_err(|e| anyhow::anyhow!("transpose: {}", e))
            })
        })
        .transpose()?;

    let block_size = 256usize;
    let jobs: Vec<(usize, usize)> = (0..n_full)
        .step_by(block_size)
        .map(|lb| (lb, (lb + block_size).min(n_full)))
        .collect();

    let cell_chunks: Vec<(usize, Mat)> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Mat)> {
            // Batch delta lookup for this block
            let x0_nd = delta_tensor
                .as_ref()
                .map(|delta_bm| -> anyhow::Result<Tensor> {
                    let membership: Vec<u32> = data_vec
                        .get_group_membership(lb..ub)?
                        .into_iter()
                        .map(|x| x as u32)
                        .collect();
                    let indices = Tensor::from_iter(membership.into_iter(), &dev)?;
                    Ok(delta_bm.index_select(&indices, 0)?)
                })
                .transpose()?;

            // Read sparse columns and coarsen features
            let x_dn = data_vec.read_columns_csc(lb..ub)?;
            let x_enc_nd = if let Some(fc) = finest_coarsening {
                fc.aggregate_sparse_csc(&x_dn)
                    .to_tensor(&dev)?
                    .transpose(0, 1)?
            } else {
                x_dn.to_tensor(&dev)?.transpose(0, 1)?
            };

            let (log_z_nk, _) = cell_encoder.forward_t(&x_enc_nd, x0_nd.as_ref(), false)?;
            let z_nk = log_z_nk.to_device(&dev)?;
            Ok((lb, Mat::from_tensor(&z_nk)?))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut cell_embed = Mat::zeros(n_full, k);
    for (lb, z) in &cell_chunks {
        let ub = lb + z.nrows();
        cell_embed.rows_range_mut(*lb..ub).copy_from(z);
    }

    // 8. Per-feature embeddings: expand D_enc → D_full
    info!("Evaluating per-feature embeddings...");
    let final_sample = finest_collapsed.mu_observed.posterior_mean();
    let feat_input = maybe_coarsen_rows(final_sample, finest_coarsening);
    let feat_t = Tensor::from_slice(
        feat_input.as_slice(),
        (feat_input.nrows(), feat_input.ncols()),
        &dev,
    )?;
    let (log_z_f, _) = feat_encoder.forward_t(&feat_t, None, false)?;
    let z_f_mat = Mat::from_tensor(&log_z_f.to_device(&dev)?)?;

    // Expand from D_enc to D_full
    let feat_embed = if let Some(fc) = finest_coarsening {
        // z_f_mat is [D_enc, K] in log-softmax space → expand to [D_full, K]
        fc.expand_log_dict_dk(&z_f_mat, d_full)
    } else {
        z_f_mat
    };

    // 9. Save
    let gene_names = data_vec.row_names()?;
    let cell_names = data_vec.column_names()?;

    feat_embed.to_parquet_with_names(
        &(args.out.to_string() + ".feature_embedding.parquet"),
        (Some(&gene_names), Some("feature")),
        None,
    )?;
    cell_embed.to_parquet_with_names(
        &(args.out.to_string() + ".cell_embedding.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    info!(
        "Done. Feature [{} × {}], Cell [{} × {}]",
        feat_embed.nrows(),
        feat_embed.ncols(),
        cell_embed.nrows(),
        cell_embed.ncols()
    );

    Ok(())
}

/// Apply feature coarsening to rows if coarsening exists, else return clone.
fn maybe_coarsen_rows(mat: &Mat, fc: Option<&FeatureCoarsening>) -> Mat {
    if let Some(fc) = fc {
        fc.aggregate_rows_ds(mat)
    } else {
        mat.clone()
    }
}

/// Coarsen embeddings by averaging within groups.
fn coarsen_embeddings(
    log_z: &Tensor,
    fine_to_coarse: &[usize],
    n_groups: usize,
    dev: &Device,
) -> candle_util::candle_core::Result<Tensor> {
    let (n, k) = log_z.dims2()?;
    let z_vec: Vec<f32> = log_z.to_vec2::<f32>()?.into_iter().flatten().collect();

    let mut sums = vec![0f32; n_groups * k];
    let mut counts = vec![0f32; n_groups];

    for i in 0..n {
        let g = fine_to_coarse[i];
        counts[g] += 1.0;
        for j in 0..k {
            sums[g * k + j] += z_vec[i * k + j];
        }
    }

    for g in 0..n_groups {
        if counts[g] > 0.0 {
            for j in 0..k {
                sums[g * k + j] /= counts[g];
            }
        }
    }

    let coarse = Tensor::from_vec(sums, (n_groups, k), dev)?;
    candle_util::candle_nn::ops::log_softmax(&coarse, 1)
}

/// Sample encoder inputs from a collapsed level.
/// Returns (cell_input [S × D_enc], cell_null, feat_input [D_enc × S], feat_null).
fn sample_encoder_inputs(
    collapsed: &CollapsedOut,
    finest_coarsening: Option<&FeatureCoarsening>,
) -> anyhow::Result<(Mat, Option<Mat>, Mat, Option<Mat>)> {
    let sample = collapsed.mu_observed.posterior_sample()?;
    let batch = collapsed
        .mu_residual
        .as_ref()
        .map(|x| x.posterior_sample())
        .transpose()?;

    let coarsened = maybe_coarsen_rows(&sample, finest_coarsening);
    let coarsened_batch = batch
        .as_ref()
        .map(|b| maybe_coarsen_rows(b, finest_coarsening));

    let cell_input = coarsened.transpose();
    let cell_null = coarsened_batch.as_ref().map(|b| b.transpose());
    let feat_input = coarsened;
    let feat_null = coarsened_batch;

    Ok((cell_input, cell_null, feat_input, feat_null))
}

/// Sample decoder target from a collapsed level.
fn sample_decoder_target(
    collapsed: &CollapsedOut,
    fc: Option<&FeatureCoarsening>,
) -> anyhow::Result<Mat> {
    let target = if let Some(adj) = &collapsed.mu_adjusted {
        adj.posterior_sample()?
    } else {
        collapsed.mu_observed.posterior_sample()?
    };
    Ok(if let Some(fc) = fc {
        fc.aggregate_rows_ds(&target)
    } else {
        target
    })
}

/// Run one training step: encode → decode across given decoders → backward.
/// Returns (llik_total, n_entries_total).
#[allow(clippy::too_many_arguments)]
fn train_step(
    cell_encoder: &LogSoftmaxEncoder,
    feat_encoder: &LogSoftmaxEncoder,
    decoders: &[(usize, &BipartiteDecoder, &Mat)], // (level_idx, decoder, target)
    level_coarsenings: &[Option<FeatureCoarsening>],
    d_encoder: usize,
    n_supercells: usize,
    cell_input_t: &Tensor,
    cell_null_t: Option<&Tensor>,
    feat_input_t: &Tensor,
    feat_null_t: Option<&Tensor>,
    kl_weight: f32,
    adam: &mut AdamW,
    dev: &Device,
) -> anyhow::Result<(f32, f32)> {
    let (log_z_c, kl_c) = cell_encoder.forward_t(cell_input_t, cell_null_t, true)?;
    let (log_z_f, kl_f) = feat_encoder.forward_t(feat_input_t, feat_null_t, true)?;

    let mut llik_total = 0f32;
    let mut n_entries_total = 0f32;
    let mut llik_sum: Option<Tensor> = None;

    for &(level, decoder, dec_target) in decoders {
        let d_l = dec_target.nrows();
        let s_l = dec_target.ncols();

        let log_z_f_l = if let Some(fc) = &level_coarsenings[level] {
            if fc.num_coarse < d_encoder {
                coarsen_embeddings(&log_z_f, &fc.fine_to_coarse, d_l, dev)?
            } else {
                log_z_f.clone()
            }
        } else {
            log_z_f.clone()
        };

        let log_z_c_l = if s_l < n_supercells {
            log_z_c.narrow(0, 0, s_l)?
        } else {
            log_z_c.clone()
        };

        let a_tensor = Tensor::from_slice(dec_target.as_slice(), (d_l, s_l), dev)?;
        let (llik_col, llik_row) =
            decoder.forward_symmetric_llik(&log_z_f_l, &log_z_c_l, &a_tensor)?;

        let llik = (&llik_col + &llik_row)?;
        llik_total += llik.to_scalar::<f32>()?;
        n_entries_total += (d_l * s_l) as f32;
        llik_sum = Some(match llik_sum {
            Some(prev) => (prev + llik)?,
            None => llik,
        });
    }

    let kl = (kl_c.sum_all()? + kl_f.sum_all()?)?;
    let loss = ((kl * kl_weight as f64)? - llik_sum.unwrap())?;
    adam.backward_step(&loss)?;

    Ok((llik_total, n_entries_total))
}

/// Mixed training: all levels simultaneously each epoch.
#[allow(clippy::too_many_arguments)]
fn train_mixed(
    collapsed_levels: &[CollapsedOut],
    level_coarsenings: &[Option<FeatureCoarsening>],
    finest_coarsening: Option<&FeatureCoarsening>,
    cell_encoder: &LogSoftmaxEncoder,
    feat_encoder: &LogSoftmaxEncoder,
    decoders: &[BipartiteDecoder],
    d_encoder: usize,
    n_supercells: usize,
    adam: &mut AdamW,
    total_epochs: usize,
    jitter_interval: usize,
    kl_warmup: f64,
    dev: &Device,
) -> anyhow::Result<()> {
    let num_levels = collapsed_levels.len();
    info!(
        "Mixed training: {} levels, {} epochs",
        num_levels, total_epochs
    );

    let finest_collapsed = collapsed_levels.last().unwrap();

    for epoch in (0..total_epochs).step_by(jitter_interval) {
        let (cell_input, cell_null, feat_input, feat_null) =
            sample_encoder_inputs(finest_collapsed, finest_coarsening)?;

        let level_targets: Vec<Mat> = collapsed_levels
            .iter()
            .zip(level_coarsenings.iter())
            .map(|(c, fc)| sample_decoder_target(c, fc.as_ref()).unwrap())
            .collect();

        let cell_input_t = Tensor::from_slice(
            cell_input.as_slice(),
            (cell_input.nrows(), cell_input.ncols()),
            dev,
        )?;
        let cell_null_t = cell_null
            .as_ref()
            .map(|b| Tensor::from_slice(b.as_slice(), (b.nrows(), b.ncols()), dev))
            .transpose()?;
        let feat_input_t = Tensor::from_slice(
            feat_input.as_slice(),
            (feat_input.nrows(), feat_input.ncols()),
            dev,
        )?;
        let feat_null_t = feat_null
            .as_ref()
            .map(|b| Tensor::from_slice(b.as_slice(), (b.nrows(), b.ncols()), dev))
            .transpose()?;

        let decoder_refs: Vec<(usize, &BipartiteDecoder, &Mat)> = decoders
            .iter()
            .zip(level_targets.iter())
            .enumerate()
            .map(|(i, (d, t))| (i, d, t))
            .collect();

        let jitter_end = jitter_interval.min(total_epochs - epoch);
        for jitter in 0..jitter_end {
            let ep = epoch + jitter;
            let kl_weight = if kl_warmup > 0.0 {
                (1.0 - (-(ep as f64) / kl_warmup).exp()) as f32
            } else {
                1.0
            };

            let (llik, n_entries) = train_step(
                cell_encoder,
                feat_encoder,
                &decoder_refs,
                level_coarsenings,
                d_encoder,
                n_supercells,
                &cell_input_t,
                cell_null_t.as_ref(),
                &feat_input_t,
                feat_null_t.as_ref(),
                kl_weight,
                adam,
                dev,
            )?;

            if ep % 50 == 0 || ep == total_epochs - 1 {
                info!(
                    "  epoch {}/{}: llik/entry={:.4}, kl_w={:.3}",
                    ep + 1,
                    total_epochs,
                    llik / n_entries,
                    kl_weight,
                );
            }
        }
    }
    Ok(())
}

/// Progressive training: coarse→fine, more epochs for coarser levels.
#[allow(clippy::too_many_arguments)]
fn train_progressive(
    collapsed_levels: &[CollapsedOut],
    level_coarsenings: &[Option<FeatureCoarsening>],
    finest_coarsening: Option<&FeatureCoarsening>,
    cell_encoder: &LogSoftmaxEncoder,
    feat_encoder: &LogSoftmaxEncoder,
    decoders: &[BipartiteDecoder],
    d_encoder: usize,
    n_supercells: usize,
    adam: &mut AdamW,
    total_epochs: usize,
    jitter_interval: usize,
    kl_warmup: f64,
    dev: &Device,
) -> anyhow::Result<()> {
    let num_levels = collapsed_levels.len();

    let total_weight: usize = (1..=num_levels).sum();
    let level_epochs: Vec<usize> = (0..num_levels)
        .map(|i| {
            let w = num_levels - i;
            (total_epochs * w / total_weight).max(1)
        })
        .collect();

    info!(
        "Progressive training: {} levels, epoch allocation: {:?}",
        num_levels, level_epochs
    );

    let mut global_epoch = 0usize;

    for (level, &level_ep) in level_epochs.iter().enumerate() {
        let collapsed = &collapsed_levels[level];
        let decoder = &decoders[level];

        info!(
            "Level {}/{}: {} epochs, {} super-cells",
            level + 1,
            num_levels,
            level_ep,
            collapsed.mu_observed.ncols()
        );

        for epoch in (0..level_ep).step_by(jitter_interval) {
            // Encoder always sees finest collapsed data
            let (cell_input, cell_null, feat_input, feat_null) =
                sample_encoder_inputs(collapsed_levels.last().unwrap(), finest_coarsening)?;

            let dec_target = sample_decoder_target(collapsed, level_coarsenings[level].as_ref())?;

            let cell_input_t = Tensor::from_slice(
                cell_input.as_slice(),
                (cell_input.nrows(), cell_input.ncols()),
                dev,
            )?;
            let cell_null_t = cell_null
                .as_ref()
                .map(|b| Tensor::from_slice(b.as_slice(), (b.nrows(), b.ncols()), dev))
                .transpose()?;
            let feat_input_t = Tensor::from_slice(
                feat_input.as_slice(),
                (feat_input.nrows(), feat_input.ncols()),
                dev,
            )?;
            let feat_null_t = feat_null
                .as_ref()
                .map(|b| Tensor::from_slice(b.as_slice(), (b.nrows(), b.ncols()), dev))
                .transpose()?;

            let decoder_refs = vec![(level, decoder, &dec_target)];

            let jitter_end = jitter_interval.min(level_ep - epoch);
            for _jitter in 0..jitter_end {
                let kl_weight = if kl_warmup > 0.0 {
                    (1.0 - (-(global_epoch as f64) / kl_warmup).exp()) as f32
                } else {
                    1.0
                };

                let (llik, n_entries) = train_step(
                    cell_encoder,
                    feat_encoder,
                    &decoder_refs,
                    level_coarsenings,
                    d_encoder,
                    n_supercells,
                    &cell_input_t,
                    cell_null_t.as_ref(),
                    &feat_input_t,
                    feat_null_t.as_ref(),
                    kl_weight,
                    adam,
                    dev,
                )?;

                if global_epoch % 50 == 0 {
                    info!(
                        "  [level {}/{}] epoch {}: llik/entry={:.4}, kl_w={:.3}",
                        level + 1,
                        num_levels,
                        global_epoch + 1,
                        llik / n_entries,
                        kl_weight,
                    );
                }
                global_epoch += 1;
            }
        }
    }

    info!("Progressive training done ({} total epochs)", global_epoch);
    Ok(())
}
