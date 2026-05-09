//! Inner count-NCE training loop. Polls `stop` at minibatch boundaries
//! so SIGINT cleanly returns to the caller for output finalization.

use crate::coarsen::AxisCoarsenings;
use crate::data::UnifiedData;
use crate::feature_network::FeatureNetworkSmoother;
use crate::loss::{nce_loss, sample_edge_batch, EdgeBatchArgs, PerBatchSampler};
use crate::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::{AdamW, Optimizer};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct TrainingContext<'a> {
    pub unified: &'a UnifiedData,
    pub cell_axis: &'a AxisCoarsenings,
    pub feat_weights: &'a [f32],
    /// Active (non-empty) per-batch samplers — a minibatch picks one
    /// uniformly so every batch contributes equally regardless of size.
    pub batch_samplers: &'a [PerBatchSampler],
    pub dev: &'a Device,
    pub stop: &'a Arc<AtomicBool>,
}

pub struct TrainingParams {
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub num_negatives: usize,
    pub seed: u64,
}

pub fn train(
    model: &JointEmbedModel,
    opt: &mut AdamW,
    ctx: &TrainingContext,
    params: &TrainingParams,
    smoother: Option<&mut FeatureNetworkSmoother>,
) -> anyhow::Result<()> {
    let pb = ProgressBar::new(params.epochs as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:30} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut rng = StdRng::seed_from_u64(params.seed);
    let n_seeds = ctx.cell_axis.coarsenings.len();
    let n_batches = ctx.batch_samplers.len();
    assert!(n_batches > 0, "no non-empty batches");

    let refresh_every = smoother.as_ref().map(|s| s.refresh_epochs).unwrap_or(0);
    let mut smoother = smoother;

    for epoch in 0..params.epochs {
        if let Some(sm) = smoother.as_deref_mut() {
            if epoch % refresh_every == 0 {
                sm.refresh(&model.e_feat, ctx.dev)?;
            }
        }

        let mut loss_sum = 0f32;
        let mut n_steps = 0usize;

        for _ in 0..params.batches_per_epoch {
            let seed_k = rng.random_range(0..n_seeds);
            let cc = &ctx.cell_axis.coarsenings[seed_k];

            let batch_id = rng.random_range(0..n_batches);
            let bs = &ctx.batch_samplers[batch_id];

            let batch = sample_edge_batch(
                EdgeBatchArgs {
                    triplets: &ctx.unified.triplets,
                    batch_sampler: bs,
                    cell_coarsening: cc,
                    fine_feature_weights: Some(ctx.feat_weights),
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                &mut rng,
            );

            let loss = nce_loss(
                model,
                batch,
                &cc.coarse_to_fine,
                smoother.as_deref(),
                ctx.dev,
            )?;

            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()?;
            n_steps += 1;

            if ctx.stop.load(Ordering::Relaxed) {
                break;
            }
        }

        let avg = loss_sum / n_steps.max(1) as f32;
        pb.set_message(format!("loss={:.3}", avg));
        pb.inc(1);
        if epoch % 10 == 0 || epoch + 1 == params.epochs {
            info!("epoch {}/{}: loss={:.3}", epoch + 1, params.epochs, avg);
        }

        if ctx.stop.load(Ordering::SeqCst) {
            pb.finish_and_clear();
            info!(
                "Stopping early at epoch {}/{} — finalizing outputs",
                epoch + 1,
                params.epochs
            );
            return Ok(());
        }
    }
    pb.finish_and_clear();

    Ok(())
}
