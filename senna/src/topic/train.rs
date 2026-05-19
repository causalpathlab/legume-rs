//! Senna-side glue for the dense VAE topic trainer.
//!
//! The training hot loop lives in [`candle_util::vae::topic`]. This
//! module keeps senna's `TrainConfig` shape (keyed on `&TopicArgs`),
//! builds per-level data from `CollapsedOut` + `FeatureCoarsening`, and
//! wires senna's anchor-prior penalty into the trainer's `loss_hook`.

use crate::embed_common::*;
use crate::fit_topic::TopicArgs;

use candle_core::{Device, Tensor};
use candle_util::decoder::DynDecoderModuleT;
use candle_util::traits::*;
use std::sync::atomic::AtomicBool;

use super::anchor_prior::anchor_penalty_at_level;
use super::common::sample_collapsed_data;

/// Configuration for training (senna-side bundle).
pub(crate) struct TrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub args: &'a TopicArgs,
    pub stop: &'a AtomicBool,
    /// Per-level `[K, D_l]` anchor β prior tensors (pre-transposed and on
    /// device). `None` means no anchor prior is attached.
    pub anchor_prior_per_level: Option<&'a [Tensor]>,
    /// Cross-entropy penalty strength λ applied per minibatch.
    pub anchor_penalty: f32,
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

/// Build the candle-util-side `TrainConfig` from the senna-side bundle.
/// The anchor-prior hook is included only when priors are attached and
/// λ > 0; otherwise the hook is `None` and the trainer takes the bare
/// ELBO path.
fn make_candle_config<'a>(
    config: &'a TrainConfig<'a>,
    hook: Option<&'a candle_util::vae::LevelLossHook<'a>>,
) -> candle_util::vae::topic::TrainConfig<'a> {
    candle_util::vae::topic::TrainConfig {
        parameters: config.parameters,
        dev: config.dev,
        epochs: config.args.epochs,
        minibatch_size: config.args.minibatch_size,
        learning_rate: config.args.learning_rate,
        topic_smoothing: config.args.topic_smoothing,
        grad_clip: config.args.grad_clip,
        stop: config.stop,
        loss_hook: hook,
    }
}

/// Mixed multi-level VAE training.
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
    let enc_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());
    let level_data = build_level_data(collapsed_levels, level_coarsenings, enc_coarsening)?;
    let level_refs: Vec<candle_util::vae::topic::LevelData> = level_data
        .iter()
        .map(|(a, b, c)| (a, b.as_ref(), c))
        .collect();

    // Anchor-prior loss hook: senna injects the CE penalty per level
    // through the candle-util `loss_hook` slot.
    let priors = config.anchor_prior_per_level;
    let lambda = config.anchor_penalty;
    let parameters = config.parameters;
    let hook_owned = move |loss: Tensor, level: usize| {
        anchor_penalty_at_level(loss, parameters, priors, lambda, level)
    };
    let hook_ref: &candle_util::vae::LevelLossHook = &hook_owned;
    let candle_cfg = make_candle_config(config, Some(hook_ref));

    let scores = candle_util::vae::topic::train_mixed(&level_refs, encoder, decoders, &candle_cfg)?;
    Ok(TrainScores {
        llik: scores.llik,
        kl: scores.kl,
    })
}

/// Mixed multi-level training with multiple simultaneous decoders.
pub(crate) fn train_mixed_multi_decoder<Enc: EncoderModuleT>(
    collapsed_levels: &[CollapsedOut],
    encoder: &mut Enc,
    decoders_per_level: &[Vec<Box<dyn DynDecoderModuleT>>],
    level_coarsenings: &[Option<FeatureCoarsening>],
    decoder_weights: &[f64],
    config: &TrainConfig,
) -> anyhow::Result<TrainScores> {
    let enc_coarsening = level_coarsenings.last().and_then(|c| c.as_ref());
    let level_data = build_level_data(collapsed_levels, level_coarsenings, enc_coarsening)?;
    let level_refs: Vec<candle_util::vae::topic::LevelData> = level_data
        .iter()
        .map(|(a, b, c)| (a, b.as_ref(), c))
        .collect();

    // Multi-decoder path historically did not apply the anchor-prior
    // penalty (senna's `fit_topic` passes `anchor_prior_per_level: None`,
    // `anchor_penalty: 0.0` here). Keep that behaviour explicitly.
    let candle_cfg = make_candle_config(config, None);

    let scores = candle_util::vae::topic::train_mixed_multi_decoder(
        &level_refs,
        encoder,
        decoders_per_level,
        decoder_weights,
        &candle_cfg,
    )?;
    Ok(TrainScores {
        llik: scores.llik,
        kl: scores.kl,
    })
}
