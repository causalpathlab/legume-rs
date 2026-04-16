use super::common::{
    apply_column_delta, compute_level_epochs, resample_levels, sample_collapsed_data,
};
use super::eval_indexed::{dense_to_indexed, refine_indexed_topic_proportions};
use crate::embed_common::*;
use crate::logging::new_progress_bar;
use data_beans_alg::collapse_data::resample_and_optimize;

use candle_core::{Device, Tensor};
use candle_nn::{ops, AdamW, Optimizer};
use candle_util::candle_encoder_indexed::*;
use candle_util::candle_ess::batched_ess_steps;
use candle_util::candle_indexed_data_loader::*;
use candle_util::candle_indexed_model_traits::*;
use candle_util::candle_loss_functions::gaussian_neg_log_prob;
use candle_util::candle_topic_refinement::TopicRefinementConfig;
use matrix_param::dmatrix_gamma::GammaMatrix;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct IndexedTrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub epochs: usize,
    pub jitter_interval: usize,
    pub minibatch_size: usize,
    pub learning_rate: f32,
    pub topic_smoothing: f64,
    pub enc_context_size: usize,
    pub dec_context_size: usize,
    pub sort_dim_budget: usize,
    pub vcd_epochs: usize,
    pub vcd_ess_steps: usize,
    pub ess_max_shrink: usize,
    pub stop: &'a AtomicBool,
    /// Per-level `[K, D_l]` anchor β prior tensors (pre-transposed, on device).
    pub anchor_prior_per_level: Option<&'a [Tensor]>,
    /// Cross-entropy penalty strength λ applied per minibatch.
    pub anchor_penalty: f32,
}

impl<'a> IndexedTrainConfig<'a> {
    #[inline]
    fn add_anchor_penalty(&self, loss: Tensor, level: usize) -> anyhow::Result<Tensor> {
        crate::topic::anchor_prior::anchor_penalty_at_level(
            loss,
            self.parameters,
            self.anchor_prior_per_level,
            self.anchor_penalty,
            level,
        )
    }
}

/// Estimate bulk-vs-SC bias as a GammaMatrix [D_sc, 1].
pub(crate) fn estimate_bulk_delta(
    bulk_dm: &Mat,
    collapsed: &CollapsedOut,
) -> anyhow::Result<GammaMatrix> {
    let mu_adj = collapsed
        .mu_adjusted
        .as_ref()
        .unwrap_or(&collapsed.mu_observed);
    let mu_adj_mean = mu_adj.posterior_mean();

    let n_sc = mu_adj_mean.ncols() as f32;
    let mu_gene_mean: Mat = Mat::from_fn(mu_adj_mean.nrows(), 1, |i, _| {
        mu_adj_mean.row(i).iter().sum::<f32>() / n_sc
    });

    let m = bulk_dm.ncols() as f32;
    let bulk_sum: Mat = Mat::from_fn(bulk_dm.nrows(), 1, |i, _| {
        bulk_dm.row(i).iter().sum::<f32>()
    });

    let expected: Mat = &mu_gene_mean * m;

    let (a0, b0) = (1.0f32, 1.0f32);
    let mut bulk_delta = GammaMatrix::new((bulk_dm.nrows(), 1), a0, b0);
    bulk_delta.update_stat(&bulk_sum, &expected);
    bulk_delta.calibrate();

    Ok(bulk_delta)
}

pub(crate) fn train_mixed<Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[Dec],
    config: &IndexedTrainConfig,
    bulk_with_deltas: Option<(&Mat, &[GammaMatrix])>,
) -> anyhow::Result<TrainScores>
where
    Dec: IndexedDecoderT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.epochs;

    for (level, (collapsed, decoder)) in collapsed_levels.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} samples, decoder dim {}",
            level + 1,
            num_levels,
            collapsed.mu_observed.ncols(),
            decoder.dim_obs(),
        );
    }

    let vcd_epochs = config.vcd_epochs;
    let vcd_ess_steps = config.vcd_ess_steps;
    let ess_max_shrink = config.ess_max_shrink;

    if vcd_epochs > 0 {
        info!(
            "Mixed multi-level training: {} levels, {} epochs ({} VCD + {} SGVB), {} ESS steps/batch",
            num_levels, total_epochs, vcd_epochs, total_epochs.saturating_sub(vcd_epochs), vcd_ess_steps
        );
    } else {
        info!(
            "Mixed multi-level training: {} levels, {} epochs",
            num_levels, total_epochs
        );
    }

    let mut adam = AdamW::new_lr(config.parameters.all_vars(), config.learning_rate as f64)?;
    let pb = new_progress_bar(total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    let target_total = config.sort_dim_budget;
    let level_sizes: Vec<usize> = collapsed_levels
        .iter()
        .map(|c| c.mu_observed.ncols())
        .collect();
    let level_budgets = compute_level_budgets(&level_sizes, target_total);
    info!(
        "Sample budget per epoch: {} total, per level: {:?} (from {:?})",
        target_total, level_budgets, level_sizes
    );

    let mut rng = rand::rng();

    for epoch in (0..total_epochs).step_by(config.jitter_interval) {
        let resampled = resample_levels(collapsed_levels, &mut rng);
        let effective_levels = resampled.as_deref().unwrap_or(collapsed_levels);

        let level_data: Vec<(Mat, Option<Mat>, Mat)> = effective_levels
            .iter()
            .zip(level_budgets.iter())
            .map(|(collapsed, &budget)| {
                let data = sample_collapsed_data(collapsed)?;
                Ok(subsample_rows(data, budget, &mut rng))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let data_loaders: Vec<IndexedInMemoryData> = level_data
            .iter()
            .map(|(mixed, batch, target)| {
                let mut loader = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                    input: mixed,
                    input_null: batch.as_ref(),
                    output: target,
                    input_context_size: config.enc_context_size,
                    output_context_size: config.dec_context_size,
                })
                .expect("data loader creation");
                loader.shuffle_minibatch(config.minibatch_size);
                loader
                    .precompute_all_minibatches(config.dev)
                    .expect("precompute");
                loader
            })
            .collect();

        let jitter_end = config.jitter_interval.min(total_epochs - epoch);
        for jitter in 0..jitter_end {
            let cur_epoch = epoch + jitter;
            let use_vcd = vcd_epochs > 0 && cur_epoch < vcd_epochs;

            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;
            let mut count_tot = 0f32;
            let mut n_tot = 0usize;

            for (level, loader) in data_loaders.iter().enumerate() {
                let decoder = &decoders[level];
                n_tot += loader.num_data();

                for b in 0..loader.num_minibatch() {
                    let mb = loader.minibatch_cached(b);

                    if use_vcd {
                        let (z_mean, z_lnvar) = encoder.latent_gaussian_params_indexed(
                            &mb.input_union_indices,
                            &mb.input_indexed_x,
                            mb.input_indexed_x_null.as_ref(),
                            true,
                        )?;
                        let z_init = encoder.reparameterize(&z_mean, &z_lnvar, true)?;

                        let (log_beta_ks, beta_ks) = decoder.prepare_dictionary_slice(
                            &mb.output_union_indices,
                            &mb.output_log_q_s,
                        )?;

                        let ess_llik = Dec::build_ess_llik_from_beta(
                            beta_ks,
                            &mb.output_indexed_x,
                            config.topic_smoothing,
                            decoder.dim_latent(),
                        )?;

                        let (z_refined, _) = batched_ess_steps(
                            &z_init.detach(),
                            &|z: &Tensor| ess_llik(z),
                            vcd_ess_steps,
                            ess_max_shrink,
                        )?;
                        let z_refined = z_refined.detach();

                        let log_z_refined = ops::log_softmax(&z_refined, 1)?;
                        let log_z_refined = smooth_topics(log_z_refined, config.topic_smoothing)?;
                        let (_, llik) = decoder.forward_indexed_with_log_beta(
                            &log_z_refined,
                            &log_beta_ks,
                            &mb.output_indexed_x,
                        )?;
                        let dec_loss = llik.neg()?.mean_all()?;

                        let enc_nll_n = gaussian_neg_log_prob(&z_refined, &z_mean, &z_lnvar)?;
                        let enc_loss = enc_nll_n.mean_all()?;

                        let loss = (dec_loss + enc_loss)?;
                        let loss = config.add_anchor_penalty(loss, level)?;
                        adam.backward_step(&loss)?;

                        llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                        kl_tot += enc_nll_n.sum_all()?.to_scalar::<f32>()?;
                    } else {
                        let (log_z_nk, kl) = encoder.forward_indexed_t(
                            &mb.input_union_indices,
                            &mb.input_indexed_x,
                            mb.input_indexed_x_null.as_ref(),
                            true,
                        )?;
                        let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;

                        let (_, llik) = decoder.forward_indexed(
                            &log_z_nk,
                            &mb.output_union_indices,
                            &mb.output_indexed_x,
                            &mb.output_log_q_s,
                        )?;
                        let loss = (&kl - &llik)?.mean_all()?;
                        let loss = config.add_anchor_penalty(loss, level)?;

                        adam.backward_step(&loss)?;

                        llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                        kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    }
                    count_tot += mb.output_indexed_x.sum_all()?.to_scalar::<f32>()?;

                    if config.stop.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }

            // Bulk training step (if present) — use finest decoder, SGVB only
            if let Some((bulk_full, bulk_deltas)) = &bulk_with_deltas {
                let finest_idx = num_levels - 1;
                let finest_decoder = &decoders[finest_idx];
                let bulk_delta = &bulk_deltas[finest_idx];

                let delta_sample = bulk_delta.posterior_sample()?.transpose();
                let corrected = apply_column_delta(bulk_full, &delta_sample, 1e-8);

                let mut bulk_loader = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                    input: *bulk_full,
                    input_null: None,
                    output: &corrected,
                    input_context_size: config.enc_context_size,
                    output_context_size: config.dec_context_size,
                })?;
                bulk_loader.shuffle_minibatch(config.minibatch_size);

                for b in 0..bulk_loader.num_minibatch() {
                    let mb = bulk_loader.minibatch_shuffled(b, config.dev)?;
                    let (log_z_nk, kl) = encoder.forward_indexed_t(
                        &mb.input_union_indices,
                        &mb.input_indexed_x,
                        None,
                        true,
                    )?;
                    let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                    let (_, llik) = finest_decoder.forward_indexed(
                        &log_z_nk,
                        &mb.output_union_indices,
                        &mb.output_indexed_x,
                        &mb.output_log_q_s,
                    )?;
                    let loss = (&kl - &llik)?.mean_all()?;
                    let loss = config.add_anchor_penalty(loss, finest_idx)?;
                    adam.backward_step(&loss)?;

                    llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    count_tot += mb.output_indexed_x.sum_all()?.to_scalar::<f32>()?;

                    if config.stop.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }

            llik_trace.push(llik_tot / count_tot);
            kl_trace.push(kl_tot / n_tot as f32);

            pb.inc(1);

            info!(
                "[epoch {}] llik={} kl={}",
                cur_epoch,
                llik_trace.last().unwrap(),
                kl_trace.last().unwrap()
            );

            if config.stop.load(Ordering::SeqCst) {
                pb.finish_and_clear();
                info!("Stopping early at epoch {}", epoch);
                return Ok(TrainScores {
                    llik: llik_trace,
                    kl: kl_trace,
                });
            }
        }
    }

    pb.finish_and_clear();
    info!("done mixed multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

pub(crate) fn train_progressive<Enc, Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &Enc,
    decoders: &[Dec],
    config: &IndexedTrainConfig,
    bulk_with_deltas: Option<(&Mat, &[GammaMatrix])>,
) -> anyhow::Result<TrainScores>
where
    Enc: IndexedEncoderT,
    Dec: IndexedDecoderT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.epochs;

    let level_epochs = compute_level_epochs(total_epochs, num_levels);

    for (level, (collapsed, decoder)) in collapsed_levels.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} epochs, {} samples, decoder dim {}",
            level + 1,
            num_levels,
            level_epochs[level],
            collapsed.mu_observed.ncols(),
            decoder.dim_obs(),
        );
    }

    info!(
        "Progressive training: {} levels, epoch allocation: {:?} (total {})",
        num_levels,
        level_epochs,
        level_epochs.iter().sum::<usize>()
    );

    let mut adam = AdamW::new_lr(config.parameters.all_vars(), config.learning_rate as f64)?;
    let total_actual_epochs: usize = level_epochs.iter().sum();
    let pb = new_progress_bar(total_actual_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_actual_epochs);
    let mut kl_trace = Vec::with_capacity(total_actual_epochs);
    let mut rng = rand::rng();

    for (level, (collapsed, &level_ep)) in
        collapsed_levels.iter().zip(level_epochs.iter()).enumerate()
    {
        let decoder = &decoders[level];

        for epoch in (0..level_ep).step_by(config.jitter_interval) {
            let resampled_one = collapsed
                .overresolved_stat
                .as_ref()
                .map(|s| resample_and_optimize(s, &mut rng, 20).expect("resample"));
            let effective = resampled_one.as_ref().unwrap_or(collapsed);
            let (mixed_nd, batch_nd, target_nd) = sample_collapsed_data(effective)?;

            let mut data_loader = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                input: &mixed_nd,
                input_null: batch_nd.as_ref(),
                output: &target_nd,
                input_context_size: config.enc_context_size,
                output_context_size: config.dec_context_size,
            })?;

            data_loader.shuffle_minibatch(config.minibatch_size);
            data_loader.precompute_all_minibatches(config.dev)?;

            let jitter_end = config.jitter_interval.min(level_ep - epoch);
            for jitter in 0..jitter_end {
                let mut llik_tot = 0f32;
                let mut kl_tot = 0f32;
                let mut count_tot = 0f32;

                for b in 0..data_loader.num_minibatch() {
                    let mb = data_loader.minibatch_cached(b);
                    let (log_z_nk, kl) = encoder.forward_indexed_t(
                        &mb.input_union_indices,
                        &mb.input_indexed_x,
                        mb.input_indexed_x_null.as_ref(),
                        true,
                    )?;

                    let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;

                    let (_, llik) = decoder.forward_indexed(
                        &log_z_nk,
                        &mb.output_union_indices,
                        &mb.output_indexed_x,
                        &mb.output_log_q_s,
                    )?;

                    let loss = (&kl - &llik)?.mean_all()?;
                    let loss = config.add_anchor_penalty(loss, level)?;
                    adam.backward_step(&loss)?;

                    llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    count_tot += mb.output_indexed_x.sum_all()?.to_scalar::<f32>()?;

                    if config.stop.load(Ordering::Relaxed) {
                        break;
                    }
                }

                // Bulk training step — use finest decoder
                if level + 1 == num_levels {
                    if let Some((bulk_full, bulk_deltas)) = &bulk_with_deltas {
                        let bulk_delta = &bulk_deltas[level];
                        let delta_sample = bulk_delta.posterior_sample()?.transpose();
                        let corrected = apply_column_delta(bulk_full, &delta_sample, 1e-8);

                        let mut bulk_loader =
                            IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                                input: *bulk_full,
                                input_null: None,
                                output: &corrected,
                                input_context_size: config.enc_context_size,
                                output_context_size: config.dec_context_size,
                            })?;
                        bulk_loader.shuffle_minibatch(config.minibatch_size);
                        bulk_loader.precompute_all_minibatches(config.dev)?;

                        for b in 0..bulk_loader.num_minibatch() {
                            let mb = bulk_loader.minibatch_cached(b);
                            let (log_z_nk, kl) = encoder.forward_indexed_t(
                                &mb.input_union_indices,
                                &mb.input_indexed_x,
                                None,
                                true,
                            )?;
                            let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                            let (_, llik) = decoder.forward_indexed(
                                &log_z_nk,
                                &mb.output_union_indices,
                                &mb.output_indexed_x,
                                &mb.output_log_q_s,
                            )?;
                            let loss = (&kl - &llik)?.mean_all()?;
                            let loss = config.add_anchor_penalty(loss, level)?;
                            adam.backward_step(&loss)?;

                            llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                            kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                            count_tot += mb.output_indexed_x.sum_all()?.to_scalar::<f32>()?;

                            if config.stop.load(Ordering::Relaxed) {
                                break;
                            }
                        }
                    }
                }

                let n = data_loader.num_data() as f32;
                llik_trace.push(llik_tot / count_tot);
                kl_trace.push(kl_tot / n);

                pb.inc(1);

                info!(
                    "[level {}/{}][epoch {}] llik={} kl={}",
                    level + 1,
                    num_levels,
                    epoch + jitter,
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
    info!("done progressive multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

pub(crate) struct BulkEvalConfig<'a, Dec> {
    pub dev: &'a Device,
    pub enc_context_size: usize,
    pub dec_context_size: usize,
    pub refine_config: Option<&'a TopicRefinementConfig>,
    pub decoder: &'a Dec,
    pub gene_names: &'a [Box<str>],
    pub out_prefix: &'a str,
}

/// Evaluate bulk samples using the given encoder/decoder and write results.
pub(crate) fn evaluate_bulk_samples<Enc, Dec>(
    bulk: &BulkDataOut,
    bulk_deltas: &[GammaMatrix],
    encoder: &Enc,
    config: &BulkEvalConfig<Dec>,
) -> anyhow::Result<()>
where
    Enc: IndexedEncoderT,
    Dec: IndexedDecoderT,
{
    info!("Evaluating bulk samples ...");
    let finest_delta = bulk_deltas.last().unwrap();
    let delta_mean = finest_delta.posterior_mean().clone();

    let bulk_nd = bulk.data.transpose();
    let delta_row = delta_mean.transpose();
    let bulk_corrected = apply_column_delta(&bulk_nd, &delta_row, 1e-8);

    let bulk_tensor = bulk_nd
        .to_tensor(config.dev)?
        .to_dtype(candle_core::DType::F32)?;
    let (enc_union, enc_indexed_x) =
        dense_to_indexed(&bulk_tensor, config.enc_context_size, config.dev)?;
    let (log_z_nk, _) = encoder.forward_indexed_t(&enc_union, &enc_indexed_x, None, false)?;

    let log_z_nk = if let Some(cfg) = config.refine_config {
        let corrected_tensor = bulk_corrected
            .to_tensor(config.dev)?
            .to_dtype(candle_core::DType::F32)?;
        let (dec_union, dec_indexed_x) =
            dense_to_indexed(&corrected_tensor, config.dec_context_size, config.dev)?;
        let s = dec_union.dim(0)?;
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, config.dev)?;
        refine_indexed_topic_proportions(
            &log_z_nk,
            &dec_union,
            &dec_indexed_x,
            &log_q_s,
            config.decoder,
            cfg,
        )?
    } else {
        log_z_nk
    };
    let z_nk_bulk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    let z_nk_bulk = Mat::from_tensor(&z_nk_bulk)?;

    z_nk_bulk.to_parquet_with_names(
        &(config.out_prefix.to_string() + ".deconv.parquet"),
        (Some(&bulk.samples), Some("sample")),
        None,
    )?;

    delta_mean.to_parquet_with_names(
        &(config.out_prefix.to_string() + ".bulk_delta.parquet"),
        (Some(config.gene_names), Some("gene")),
        None,
    )?;
    info!("Wrote bulk deconvolution results");
    Ok(())
}

/// Write dictionary at full resolution (no coarsening for indexed model).
pub(crate) fn write_indexed_dictionary<Dec: IndexedDecoderT>(
    decoder: &Dec,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let dict_tensor = decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?;

    dict_tensor.to_parquet_with_names(
        &(out_prefix.to_string() + ".dictionary.parquet"),
        (Some(gene_names), Some("gene")),
        None,
    )?;
    Ok(())
}
