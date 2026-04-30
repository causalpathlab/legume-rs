use crate::embed_common::*;
use crate::fit_topic::TopicArgs;
use crate::logging::new_progress_bar;

use candle_core::{Device, Tensor};
use candle_nn::AdamW;
use candle_nn::Optimizer;
use candle_util::candle_data_loader::*;
use candle_util::candle_loss_functions::topic_likelihood;
use candle_util::candle_model_traits::*;
use std::sync::atomic::{AtomicBool, Ordering};

use super::anchor_prior::anchor_penalty_at_level;
use super::common::sample_collapsed_data;

/// Configuration for training
pub(crate) struct TrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub args: &'a TopicArgs,
    pub stop: &'a AtomicBool,
    /// Per-level `[K, D_l]` anchor β prior tensors (pre-transposed and on
    /// device). `None` means no anchor prior is attached — training runs
    /// unchanged.
    pub anchor_prior_per_level: Option<&'a [Tensor]>,
    /// Cross-entropy penalty strength λ applied per minibatch.
    pub anchor_penalty: f32,
}

impl TrainConfig<'_> {
    #[inline]
    fn add_anchor_penalty(&self, loss: Tensor, level: usize) -> anyhow::Result<Tensor> {
        anchor_penalty_at_level(
            loss,
            self.parameters,
            self.anchor_prior_per_level,
            self.anchor_penalty,
            level,
        )
    }
}

/// Materialize `(encoder-input, batch, decoder-target)` `Mat` triples
/// once per training run, applying the encoder's and per-level decoder
/// coarsenings.
fn build_level_data(
    collapsed_levels: &[CollapsedOut],
    level_coarsenings: &[Option<FeatureCoarsening>],
    enc_coarsening: Option<&FeatureCoarsening>,
) -> anyhow::Result<Vec<(Mat, Option<Mat>, Mat)>> {
    collapsed_levels
        .iter()
        .zip(level_coarsenings.iter())
        .map(|(collapsed, dec_fc)| {
            let (mixed_nd, batch_nd, target_nd) = sample_collapsed_data(collapsed)?;

            let enc_nd = if let Some(fc) = enc_coarsening {
                fc.aggregate_columns_nd(&mixed_nd)
            } else {
                mixed_nd
            };

            let batch_nd = batch_nd.map(|b| {
                if let Some(fc) = enc_coarsening {
                    fc.aggregate_columns_nd(&b)
                } else {
                    b
                }
            });

            let dec_target = if let Some(fc) = dec_fc.as_ref() {
                fc.aggregate_columns_nd(&target_nd)
            } else {
                target_nd
            };

            Ok((enc_nd, batch_nd, dec_target))
        })
        .collect()
}

fn build_device_loaders(
    level_data: &[(Mat, Option<Mat>, Mat)],
    dev: &Device,
) -> anyhow::Result<Vec<InMemoryData>> {
    level_data
        .iter()
        .map(|(enc, batch, target)| {
            InMemoryData::from_device(
                InMemoryArgs {
                    input: enc,
                    input_null: batch.as_ref(),
                    output: Some(target),
                    output_null: None,
                },
                dev,
            )
        })
        .collect()
}

/// Mixed multi-level VAE training.
///
/// Encoder operates at `D_coarse` (finest level's feature coarsening).
/// Per-level decoders operate at `D_l`. All levels are trained simultaneously
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

    info!("Mixed multi-level training: {num_levels} levels, {total_epochs} epochs");

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        f64::from(config.args.learning_rate),
    )?;

    let pb = new_progress_bar(total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    let level_data = build_level_data(collapsed_levels, level_coarsenings, enc_coarsening)?;
    let mut data_loaders = build_device_loaders(&level_data, config.dev)?;

    for epoch in 0..total_epochs {
        for loader in data_loaders.iter_mut() {
            loader.shuffle_minibatch_on_device(config.args.minibatch_size)?;
        }

        let mut llik_tot = 0f32;
        let mut kl_tot = 0f32;
        let mut count_tot = 0f32;
        let mut n_tot = 0usize;

        for (level, loader) in data_loaders.iter().enumerate() {
            let decoder = &decoders[level];
            n_tot += loader.num_data();

            for b in 0..loader.num_minibatch() {
                let mb = loader.minibatch_cached(b);
                let (log_z_nk, kl) = encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;

                let log_z_nk = smooth_topics(log_z_nk, config.args.topic_smoothing)?;

                let y_nd = mb.output.as_ref().unwrap_or(&mb.input);
                let (_, llik) = decoder.forward_with_llik(&log_z_nk, y_nd, &topic_likelihood)?;

                let loss = (&kl - &llik)?.mean_all()?;
                let loss = config.add_anchor_penalty(loss, level)?;
                adam.backward_step(&loss)?;

                llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                count_tot += y_nd.sum_all()?.to_scalar::<f32>()?;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        let llik_avg = llik_tot / count_tot;
        let kl_avg = kl_tot / n_tot as f32;
        llik_trace.push(llik_avg);
        kl_trace.push(kl_avg);

        pb.inc(1);

        info!("[epoch {}] llik={} kl={}", epoch, llik_avg, kl_avg);

        if config.stop.load(Ordering::SeqCst) {
            pb.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            return Ok(TrainScores {
                llik: llik_trace,
                kl: kl_trace,
            });
        }
    }

    pb.finish_and_clear();
    info!("done mixed multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

/// Mixed multi-level training with multiple simultaneous decoders.
///
/// Each level has a `Vec<Box<dyn DynDecoderModuleT>>` of decoders.
/// The shared encoder produces z, and each decoder computes its own
/// likelihood. The total likelihood is a weighted sum across decoders.
pub(crate) fn train_mixed_multi_decoder<Enc: EncoderModuleT>(
    collapsed_levels: &[CollapsedOut],
    encoder: &mut Enc,
    decoders_per_level: &[Vec<Box<dyn candle_util::candle_dyn_decoder::DynDecoderModuleT>>],
    level_coarsenings: &[Option<FeatureCoarsening>],
    decoder_weights: &[f64],
    config: &TrainConfig,
) -> anyhow::Result<TrainScores> {
    use candle_core::Tensor;

    let num_levels = collapsed_levels.len();
    let total_epochs = config.args.epochs;

    let enc_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());

    for (level, (collapsed, decoders)) in collapsed_levels
        .iter()
        .zip(decoders_per_level.iter())
        .enumerate()
    {
        let names: Vec<&str> = decoders.iter().map(|d| d.decoder_name()).collect();
        info!(
            "Level {}/{}: {} samples, {} decoders {:?}",
            level + 1,
            num_levels,
            collapsed.mu_observed.ncols(),
            decoders.len(),
            names,
        );
    }

    info!(
        "Mixed multi-decoder training: {} levels, {} decoders, {} epochs",
        num_levels,
        decoders_per_level[0].len(),
        total_epochs,
    );

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        f64::from(config.args.learning_rate),
    )?;

    let pb = new_progress_bar(total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    let level_data = build_level_data(collapsed_levels, level_coarsenings, enc_coarsening)?;
    let mut data_loaders = build_device_loaders(&level_data, config.dev)?;

    for epoch in 0..total_epochs {
        for loader in data_loaders.iter_mut() {
            loader.shuffle_minibatch_on_device(config.args.minibatch_size)?;
        }

        let mut llik_tot = 0f32;
        let mut kl_tot = 0f32;
        let mut count_tot = 0f32;
        let mut n_tot = 0usize;

        for (level, loader) in data_loaders.iter().enumerate() {
            let decoders = &decoders_per_level[level];
            n_tot += loader.num_data();

            for b in 0..loader.num_minibatch() {
                let mb = loader.minibatch_cached(b);
                let (log_z_nk, kl) = encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;

                let log_z_nk = smooth_topics(log_z_nk, config.args.topic_smoothing)?;

                let y_nd = mb.output.as_ref().unwrap_or(&mb.input);

                // Weighted sum of likelihoods across decoders
                let mut weighted_llik = Tensor::zeros_like(&kl)?;
                for (dec, &w) in decoders.iter().zip(decoder_weights) {
                    let (_, llik) = dec.forward_llik(&log_z_nk, y_nd)?;
                    weighted_llik = (weighted_llik + llik * w)?;
                }

                let loss = (&kl - &weighted_llik)?.mean_all()?;
                adam.backward_step(&loss)?;

                llik_tot += weighted_llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                count_tot += y_nd.sum_all()?.to_scalar::<f32>()?;
            }
        }

        let llik_avg = llik_tot / count_tot;
        let kl_avg = kl_tot / n_tot as f32;
        llik_trace.push(llik_avg);
        kl_trace.push(kl_avg);

        pb.inc(1);

        info!("[epoch {}] llik={} kl={}", epoch, llik_avg, kl_avg);

        if config.stop.load(Ordering::SeqCst) {
            pb.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            return Ok(TrainScores {
                llik: llik_trace,
                kl: kl_trace,
            });
        }
    }

    pb.finish_and_clear();
    info!("done mixed multi-decoder multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}
