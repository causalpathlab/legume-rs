use crate::embed_common::*;
use crate::fit_topic::TopicArgs;

use candle_core::Device;
use candle_nn::AdamW;
use candle_nn::Optimizer;
use candle_util::candle_data_loader::*;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_ess::batched_ess_steps;
use candle_util::candle_loss_functions::{gaussian_neg_log_prob, topic_likelihood};
use candle_util::candle_model_traits::*;
use indicatif::ProgressBar;
use std::sync::atomic::{AtomicBool, Ordering};

use super::common::{compute_level_epochs, sample_collapsed_data};

/// Configuration for training
pub(crate) struct TrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub args: &'a TopicArgs,
    pub stop: &'a AtomicBool,
}

/// Mixed multi-level VAE training.
///
/// Encoder operates at D_coarse (finest level's feature coarsening).
/// Per-level decoders operate at D_l. All levels are trained simultaneously
/// each epoch — the shared encoder sees data from all levels, while each
/// decoder handles its own feature resolution.
pub(crate) fn train_mixed<Enc, Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &mut Enc,
    decoders: &[Dec],
    level_coarsenings: &[Option<FeatureCoarsening>],
    config: &TrainConfig,
) -> anyhow::Result<TrainScores>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.args.epochs;

    // Encoder coarsening = finest level's coarsening
    let enc_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());

    for (level, (collapsed, decoder)) in collapsed_levels.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} samples, decoder dim {}",
            level + 1,
            num_levels,
            collapsed.mu_observed.ncols(),
            decoder.dim_obs(),
        );
    }

    info!(
        "Mixed multi-level training: {} levels, {} epochs",
        num_levels, total_epochs
    );

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        config.args.learning_rate as f64,
    )?;

    let pb = ProgressBar::new(total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    // Budget: 2^sort_dim total samples per epoch, weighted inversely by level size.
    let target_total = 1usize << config.args.sort_dim;
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

    for epoch in (0..total_epochs).step_by(config.args.jitter_interval) {
        let level_data: Vec<(Mat, Option<Mat>, Mat)> = collapsed_levels
            .iter()
            .zip(level_coarsenings.iter())
            .zip(level_budgets.iter())
            .map(|((collapsed, dec_fc), &budget)| {
                let (full_mixed, full_batch, target_full) =
                    sample_collapsed_data(collapsed).unwrap();

                let (sub_mixed, sub_batch, sub_target) =
                    subsample_rows((full_mixed, full_batch, target_full), budget, &mut rng);

                let enc_nd = if let Some(fc) = enc_coarsening {
                    fc.aggregate_columns_nd(&sub_mixed)
                } else {
                    sub_mixed
                };

                let batch_nd = sub_batch.map(|b| {
                    if let Some(fc) = enc_coarsening {
                        fc.aggregate_columns_nd(&b)
                    } else {
                        b
                    }
                });

                let dec_target = if let Some(fc) = dec_fc.as_ref() {
                    fc.aggregate_columns_nd(&sub_target)
                } else {
                    sub_target
                };

                (enc_nd, batch_nd, dec_target)
            })
            .collect();

        let data_loaders: Vec<InMemoryData> = level_data
            .iter()
            .map(|(enc, batch, target)| {
                let mut loader = InMemoryData::from(InMemoryArgs {
                    input: enc,
                    input_null: batch.as_ref(),
                    output: Some(target),
                    output_null: None,
                })
                .expect("data loader creation");
                loader
                    .shuffle_minibatch(config.args.minibatch_size)
                    .expect("shuffle");
                loader
            })
            .collect();

        let jitter_end = config.args.jitter_interval.min(total_epochs - epoch);
        for jitter in 0..jitter_end {
            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;
            let mut count_tot = 0f32;
            let mut n_tot = 0usize;

            for (level, loader) in data_loaders.iter().enumerate() {
                let decoder = &decoders[level];
                n_tot += loader.num_data();

                for b in 0..loader.num_minibatch() {
                    let mb = loader.minibatch_shuffled(b, config.dev)?;
                    let (log_z_nk, kl) =
                        encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;

                    let log_z_nk = smooth_topics(log_z_nk, config.args.topic_smoothing)?;

                    let y_nd = mb.output.unwrap_or(mb.input);
                    let (_, llik) =
                        decoder.forward_with_llik(&log_z_nk, &y_nd, &topic_likelihood)?;

                    let loss = (&kl - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;

                    llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    count_tot += y_nd.sum_all()?.to_scalar::<f32>()?;
                }
            }

            let llik_avg = llik_tot / count_tot;
            let kl_avg = kl_tot / n_tot as f32;
            llik_trace.push(llik_avg);
            kl_trace.push(kl_avg);

            pb.inc(1);

            info!("[epoch {}] llik={} kl={}", epoch + jitter, llik_avg, kl_avg);

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

/// Mixed multi-level VCD training.
///
/// Like `train_mixed`, but uses elliptical slice sampling (ESS) to refine
/// latent z toward the true posterior, then trains the encoder to match
/// ESS-refined samples (inclusive KL / wake-phase update).
pub(crate) fn train_mixed_vcd<Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &mut LogSoftmaxEncoder,
    decoders: &[Dec],
    level_coarsenings: &[Option<FeatureCoarsening>],
    config: &TrainConfig,
) -> anyhow::Result<TrainScores>
where
    Dec: DecoderModuleT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.args.epochs;
    let vcd_epochs = config.args.vcd_epochs;
    let vcd_ess_steps = config.args.vcd_ess_steps;
    let ess_max_shrink = config.args.ess_max_shrink;

    let enc_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());

    for (level, (collapsed, decoder)) in collapsed_levels.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} samples, decoder dim {}",
            level + 1,
            num_levels,
            collapsed.mu_observed.ncols(),
            decoder.dim_obs(),
        );
    }

    info!(
        "Mixed training: {} levels, {} epochs ({} VCD + {} SGVB), {} ESS steps/batch",
        num_levels,
        total_epochs,
        vcd_epochs,
        total_epochs.saturating_sub(vcd_epochs),
        vcd_ess_steps
    );

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        config.args.learning_rate as f64,
    )?;

    let pb = ProgressBar::new(total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    let target_total = 1usize << config.args.sort_dim;
    let level_sizes: Vec<usize> = collapsed_levels
        .iter()
        .map(|c| c.mu_observed.ncols())
        .collect();
    let level_budgets = compute_level_budgets(&level_sizes, target_total);

    let mut rng = rand::rng();

    for epoch in (0..total_epochs).step_by(config.args.jitter_interval) {
        let level_data: Vec<(Mat, Option<Mat>, Mat)> = collapsed_levels
            .iter()
            .zip(level_coarsenings.iter())
            .zip(level_budgets.iter())
            .map(|((collapsed, dec_fc), &budget)| {
                let (full_mixed, full_batch, target_full) =
                    sample_collapsed_data(collapsed).unwrap();

                let (sub_mixed, sub_batch, sub_target) =
                    subsample_rows((full_mixed, full_batch, target_full), budget, &mut rng);

                let enc_nd = if let Some(fc) = enc_coarsening {
                    fc.aggregate_columns_nd(&sub_mixed)
                } else {
                    sub_mixed
                };

                let batch_nd = sub_batch.map(|b| {
                    if let Some(fc) = enc_coarsening {
                        fc.aggregate_columns_nd(&b)
                    } else {
                        b
                    }
                });

                let dec_target = if let Some(fc) = dec_fc.as_ref() {
                    fc.aggregate_columns_nd(&sub_target)
                } else {
                    sub_target
                };

                (enc_nd, batch_nd, dec_target)
            })
            .collect();

        let data_loaders: Vec<InMemoryData> = level_data
            .iter()
            .map(|(enc, batch, target)| {
                let mut loader = InMemoryData::from(InMemoryArgs {
                    input: enc,
                    input_null: batch.as_ref(),
                    output: Some(target),
                    output_null: None,
                })
                .expect("data loader creation");
                loader
                    .shuffle_minibatch(config.args.minibatch_size)
                    .expect("shuffle");
                loader
            })
            .collect();

        let jitter_end = config.args.jitter_interval.min(total_epochs - epoch);
        for jitter in 0..jitter_end {
            let cur_epoch = epoch + jitter;
            let use_vcd = cur_epoch < vcd_epochs;

            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;
            let mut n_tot = 0usize;

            for (level, loader) in data_loaders.iter().enumerate() {
                let decoder = &decoders[level];
                n_tot += loader.num_data();

                for b in 0..loader.num_minibatch() {
                    let mb = loader.minibatch_shuffled(b, config.dev)?;

                    if use_vcd {
                        let (z_mean, z_lnvar) = encoder.latent_gaussian_params(
                            &mb.input,
                            mb.input_null.as_ref(),
                            true,
                        )?;
                        let z_init = encoder.reparameterize(&z_mean, &z_lnvar, true)?;

                        let y_nd = mb.output.unwrap_or(mb.input);
                        let ess_llik =
                            decoder.build_ess_llik(&y_nd, config.args.topic_smoothing)?;
                        let (z_refined, _) = batched_ess_steps(
                            &z_init.detach(),
                            &|z: &Tensor| ess_llik(z),
                            vcd_ess_steps,
                            ess_max_shrink,
                        )?;
                        let z_refined = z_refined.detach();

                        let log_z_refined = candle_nn::ops::log_softmax(&z_refined, 1)?;
                        let log_z_refined =
                            smooth_topics(log_z_refined, config.args.topic_smoothing)?;
                        let (_, llik) =
                            decoder.forward_with_llik(&log_z_refined, &y_nd, &topic_likelihood)?;
                        let dec_loss = llik.neg()?.mean_all()?;

                        let enc_nll_n = gaussian_neg_log_prob(&z_refined, &z_mean, &z_lnvar)?;
                        let enc_loss = enc_nll_n.mean_all()?;

                        let loss = (dec_loss + enc_loss)?;
                        adam.backward_step(&loss)?;

                        llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                        kl_tot += enc_nll_n.sum_all()?.to_scalar::<f32>()?;
                    } else {
                        let (log_z_nk, kl) =
                            encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;
                        let y_nd = mb.output.unwrap_or(mb.input);
                        let log_z_nk = smooth_topics(log_z_nk, config.args.topic_smoothing)?;

                        let (_, llik) =
                            decoder.forward_with_llik(&log_z_nk, &y_nd, &topic_likelihood)?;

                        let loss = (&kl - &llik)?.mean_all()?;
                        adam.backward_step(&loss)?;

                        llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                        kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    }
                }
            }

            let llik_avg = llik_tot / n_tot as f32;
            let kl_avg = kl_tot / n_tot as f32;
            llik_trace.push(llik_avg);
            kl_trace.push(kl_avg);

            pb.inc(1);

            info!("[epoch {}] llik={} kl={}", cur_epoch, llik_avg, kl_avg);

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
    info!("done mixed multi-level training (VCD + SGVB)");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

/// Progressive training: coarse→fine, more epochs for coarser levels.
pub(crate) fn train_progressive<Enc, Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &mut Enc,
    decoders: &[Dec],
    level_coarsenings: &[Option<FeatureCoarsening>],
    config: &TrainConfig,
) -> anyhow::Result<TrainScores>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.args.epochs;

    let enc_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());

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

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        config.args.learning_rate as f64,
    )?;

    let total_actual_epochs: usize = level_epochs.iter().sum();
    let pb = ProgressBar::new(total_actual_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_actual_epochs);
    let mut kl_trace = Vec::with_capacity(total_actual_epochs);

    for (level, (collapsed, &level_ep)) in
        collapsed_levels.iter().zip(level_epochs.iter()).enumerate()
    {
        let decoder = &decoders[level];
        let dec_coarsening = level_coarsenings[level].as_ref();

        for epoch in (0..level_ep).step_by(config.args.jitter_interval) {
            let (full_mixed, full_batch, target_full) = sample_collapsed_data(collapsed)?;

            let enc_nd = if let Some(fc) = enc_coarsening {
                fc.aggregate_columns_nd(&full_mixed)
            } else {
                full_mixed
            };

            let batch_nd = full_batch.map(|b| {
                if let Some(fc) = enc_coarsening {
                    fc.aggregate_columns_nd(&b)
                } else {
                    b
                }
            });

            let dec_target = if let Some(fc) = dec_coarsening {
                fc.aggregate_columns_nd(&target_full)
            } else {
                target_full
            };

            let mut data_loader = InMemoryData::from(InMemoryArgs {
                input: &enc_nd,
                input_null: batch_nd.as_ref(),
                output: Some(&dec_target),
                output_null: None,
            })?;

            data_loader.shuffle_minibatch(config.args.minibatch_size)?;

            let jitter_end = config.args.jitter_interval.min(level_ep - epoch);
            for _jitter in 0..jitter_end {
                let mut llik_tot = 0f32;
                let mut kl_tot = 0f32;
                let mut count_tot = 0f32;

                for b in 0..data_loader.num_minibatch() {
                    let mb = data_loader.minibatch_shuffled(b, config.dev)?;
                    let (log_z_nk, kl) =
                        encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;

                    let log_z_nk = smooth_topics(log_z_nk, config.args.topic_smoothing)?;

                    let y_nd = mb.output.unwrap_or(mb.input);
                    let (_, llik) =
                        decoder.forward_with_llik(&log_z_nk, &y_nd, &topic_likelihood)?;

                    let loss = (&kl - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;

                    llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    count_tot += y_nd.sum_all()?.to_scalar::<f32>()?;
                }

                let n = data_loader.num_data() as f32;
                llik_trace.push(llik_tot / count_tot);
                kl_trace.push(kl_tot / n);

                pb.inc(1);

                info!(
                    "[level {}/{}][epoch {}] llik={} kl={}",
                    level + 1,
                    num_levels,
                    epoch,
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
