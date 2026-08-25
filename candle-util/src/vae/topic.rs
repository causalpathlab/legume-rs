//! Dense VAE trainer for topic models (`EncoderModuleT` + `DecoderModuleT`).
//!
//! Two entry points:
//! - [`train_mixed`]: shared encoder + one decoder per cascade level.
//! - [`train_mixed_multi_decoder`]: shared encoder + multiple weighted
//!   decoders per level (via [`DynDecoderModuleT`]).
//!
//! Callers pre-build per-level `(input, batch, target)` `Mat` triples;
//! all three are borrowed so a single matrix can back both input and
//! target without cloning.

use super::{clip_grads_and_step, smooth_topics, LevelLossHook, TrainScores};
use crate::data::indexed::labeled_bar;
use crate::data::loader::{DataLoader, InMemoryArgs, InMemoryData};
use crate::decoder::dyn_decoder::DynDecoderModuleT;
use crate::loss::topic_likelihood;
use crate::traits::model::{DecoderModuleT, EncoderModuleT};
use candle_core::{Device, Tensor};
use candle_nn::AdamW;
use log::info;
use nalgebra::DMatrix;
use std::sync::atomic::{AtomicBool, Ordering};

type Mat = DMatrix<f32>;

/// Hyperparameter bundle passed by reference to [`train_mixed`] and
/// [`train_mixed_multi_decoder`].
pub struct TrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub epochs: usize,
    /// `Some(frac)`: on CUDA, probe one forward per candidate size and
    /// SHRINK `minibatch_size` when free device memory says so (never
    /// grow past it). `None`, CPU, or an unavailable query keep the
    /// configured size exactly.
    pub gpu_mem_fraction: Option<f32>,
    pub minibatch_size: usize,
    pub learning_rate: f32,
    pub topic_smoothing: f64,
    pub grad_clip: f32,
    pub stop: &'a AtomicBool,
    /// Optional per-level hook applied to the ELBO loss before backward.
    /// Senna passes a closure that calls its anchor-prior CE penalty;
    /// other callers pass `None`.
    pub loss_hook: Option<&'a LevelLossHook<'a>>,
}

/// Per-level training triple: `(encoder input, optional batch null, decoder target)`.
pub type LevelData<'a> = (&'a Mat, Option<&'a Mat>, &'a Mat);

fn build_device_loaders(
    level_data: &[LevelData],
    dev: &Device,
) -> anyhow::Result<Vec<InMemoryData>> {
    level_data
        .iter()
        .map(|&(enc, batch, target)| {
            InMemoryData::from_device(
                InMemoryArgs {
                    input: enc,
                    input_null: batch,
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
/// Encoder is shared across levels; each level has its own decoder at its
/// own feature resolution. All levels train simultaneously each epoch.
pub fn train_mixed<Enc, Dec>(
    level_data: &[LevelData],
    encoder: &mut Enc,
    decoders: &[Dec],
    config: &TrainConfig,
) -> anyhow::Result<TrainScores>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let num_levels = level_data.len();
    let total_epochs = config.epochs;

    for (level, (&(mixed, _, _), decoder)) in level_data.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} samples, decoder dim {}",
            level + 1,
            num_levels,
            mixed.ncols(),
            decoder.dim_obs(),
        );
    }

    info!("Mixed multi-level training: {num_levels} levels, {total_epochs} epochs");

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        f64::from(config.learning_rate),
    )?;
    let prog_bar = labeled_bar("Epochs", total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    let mut data_loaders = build_device_loaders(level_data, config.dev)?;

    // On CUDA, optionally shrink the minibatch size to fit free device
    // memory. The probe mirrors the loop's forward with the loss held
    // un-backwarded; `auto_chunk_size` reserves half the measured budget
    // for backward's gradient copies. It probes level 0 only: callers
    // enabling this must keep every level's per-row footprint at or
    // below level 0's (true for the full-D-decoder vae path; senna
    // topic's coarse-to-fine decoders pin this to `None` instead).
    let minibatch_size = match (config.gpu_mem_fraction, data_loaders.first_mut()) {
        (Some(frac), Some(loader)) => {
            let cap = config.minibatch_size;
            crate::device::auto_chunk_size(config.dev, cap, 16.min(cap), frac, |n| {
                loader
                    .shuffle_minibatch_on_device(n)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                if loader.num_minibatch() == 0 {
                    return Err(candle_core::Error::Msg("probe loader is empty".into()));
                }
                let mb = loader.minibatch_cached(0);
                let (log_z_nk, kl) = encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                let y_nd = mb.output.as_ref().unwrap_or(&mb.input);
                let (_, llik) =
                    decoders[0].forward_with_llik(&log_z_nk, y_nd, &topic_likelihood)?;
                (&kl - &llik)?.mean_all()
            })
            .unwrap_or(cap)
        }
        _ => config.minibatch_size,
    };

    for epoch in 0..total_epochs {
        for loader in data_loaders.iter_mut() {
            loader.shuffle_minibatch_on_device(minibatch_size)?;
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

                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;

                let y_nd = mb.output.as_ref().unwrap_or(&mb.input);
                let (_, llik) = decoder.forward_with_llik(&log_z_nk, y_nd, &topic_likelihood)?;

                let loss = (&kl - &llik)?.mean_all()?;
                let loss = match config.loss_hook {
                    Some(hook) => hook(loss, level)?,
                    None => loss,
                };
                clip_grads_and_step(&mut adam, &loss, f64::from(config.grad_clip))?;

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

        prog_bar.set_message(format!("llik={llik_avg:.3}"));
        prog_bar.inc(1);

        info!("[epoch {}] llik={} kl={}", epoch, llik_avg, kl_avg);

        if config.stop.load(Ordering::SeqCst) {
            prog_bar.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            return Ok(TrainScores {
                llik: llik_trace,
                kl: kl_trace,
            });
        }
    }

    prog_bar.finish_and_clear();
    info!("done mixed multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

/// Multi-decoder variant: each level has a `Vec<Box<dyn DynDecoderModuleT>>`.
///
/// The shared encoder produces `z`, every decoder computes its own
/// likelihood, and the total likelihood is a `decoder_weights`-weighted
/// sum across decoders. Useful when one cell is observed under multiple
/// modalities (e.g. RNA + ATAC) sharing a single topic posterior.
pub fn train_mixed_multi_decoder<Enc: EncoderModuleT>(
    level_data: &[LevelData],
    encoder: &mut Enc,
    decoders_per_level: &[Vec<Box<dyn DynDecoderModuleT>>],
    decoder_weights: &[f64],
    config: &TrainConfig,
) -> anyhow::Result<TrainScores> {
    let num_levels = level_data.len();
    let total_epochs = config.epochs;

    for (level, (&(mixed, _, _), decoders)) in
        level_data.iter().zip(decoders_per_level.iter()).enumerate()
    {
        let names: Vec<&str> = decoders.iter().map(|d| d.decoder_name()).collect();
        info!(
            "Level {}/{}: {} samples, {} decoders {:?}",
            level + 1,
            num_levels,
            mixed.ncols(),
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
        f64::from(config.learning_rate),
    )?;
    let prog_bar = labeled_bar("Epochs", total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    let mut data_loaders = build_device_loaders(level_data, config.dev)?;

    // Same CUDA sizing probe as `train_mixed`, over the first level's
    // full decoder set.
    let minibatch_size = match (config.gpu_mem_fraction, data_loaders.first_mut()) {
        (Some(frac), Some(loader)) => {
            let cap = config.minibatch_size;
            crate::device::auto_chunk_size(config.dev, cap, 16.min(cap), frac, |n| {
                loader
                    .shuffle_minibatch_on_device(n)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                if loader.num_minibatch() == 0 {
                    return Err(candle_core::Error::Msg("probe loader is empty".into()));
                }
                let mb = loader.minibatch_cached(0);
                let (log_z_nk, kl) = encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                let y_nd = mb.output.as_ref().unwrap_or(&mb.input);
                let mut weighted_llik = Tensor::zeros_like(&kl)?;
                for (dec, &w) in decoders_per_level[0].iter().zip(decoder_weights) {
                    let (_, llik) = dec.forward_llik(&log_z_nk, y_nd)?;
                    weighted_llik = (weighted_llik + llik * w)?;
                }
                (&kl - &weighted_llik)?.mean_all()
            })
            .unwrap_or(cap)
        }
        _ => config.minibatch_size,
    };

    for epoch in 0..total_epochs {
        for loader in data_loaders.iter_mut() {
            loader.shuffle_minibatch_on_device(minibatch_size)?;
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

                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;

                let y_nd = mb.output.as_ref().unwrap_or(&mb.input);

                // Weighted sum of likelihoods across decoders.
                let mut weighted_llik = Tensor::zeros_like(&kl)?;
                for (dec, &w) in decoders.iter().zip(decoder_weights) {
                    let (_, llik) = dec.forward_llik(&log_z_nk, y_nd)?;
                    weighted_llik = (weighted_llik + llik * w)?;
                }

                let loss = (&kl - &weighted_llik)?.mean_all()?;
                let loss = match config.loss_hook {
                    Some(hook) => hook(loss, level)?,
                    None => loss,
                };
                clip_grads_and_step(&mut adam, &loss, f64::from(config.grad_clip))?;

                llik_tot += weighted_llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                count_tot += y_nd.sum_all()?.to_scalar::<f32>()?;
            }
        }

        let llik_avg = llik_tot / count_tot;
        let kl_avg = kl_tot / n_tot as f32;
        llik_trace.push(llik_avg);
        kl_trace.push(kl_avg);

        prog_bar.set_message(format!("llik={llik_avg:.3}"));
        prog_bar.inc(1);

        info!("[epoch {}] llik={} kl={}", epoch, llik_avg, kl_avg);

        if config.stop.load(Ordering::SeqCst) {
            prog_bar.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            return Ok(TrainScores {
                llik: llik_trace,
                kl: kl_trace,
            });
        }
    }

    prog_bar.finish_and_clear();
    info!("done mixed multi-decoder multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}
