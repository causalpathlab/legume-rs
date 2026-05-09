//! Inner count-NCE training loop for `chickpea embed-graph`.

use crate::embed_common::*;
use crate::gbe::coarsen::AxisCoarsenings;
use crate::gbe::data::UnifiedData;
use crate::gbe::loss::{nce_loss, sample_edge_batch, EdgeBatchArgs, PerFileSampler};
use crate::gbe::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::{AdamW, Optimizer};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{rngs::StdRng, RngExt, SeedableRng};

pub struct TrainingContext<'a> {
    pub unified: &'a UnifiedData,
    pub cell_axis: &'a AxisCoarsenings,
    pub feat_weights: &'a [f32],
    pub file_samplers: &'a [PerFileSampler],
    pub dev: &'a Device,
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
) -> anyhow::Result<()> {
    let pb = ProgressBar::new(params.epochs as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:30} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut rng = StdRng::seed_from_u64(params.seed);
    let n_seeds = ctx.cell_axis.coarsenings.len();
    let n_files = ctx.file_samplers.len();
    assert!(n_files > 0, "no input files");

    for epoch in 0..params.epochs {
        let mut loss_sum = 0f32;
        let mut n_steps = 0usize;

        for _ in 0..params.batches_per_epoch {
            let seed_k = rng.random_range(0..n_seeds);
            let cc = &ctx.cell_axis.coarsenings[seed_k];

            let file_id = rng.random_range(0..n_files);
            let fs = &ctx.file_samplers[file_id];

            let batch = sample_edge_batch(
                EdgeBatchArgs {
                    triplets: &ctx.unified.triplets,
                    file_sampler: fs,
                    cell_coarsening: cc,
                    fine_feature_weights: Some(ctx.feat_weights),
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                &mut rng,
            );

            let loss = nce_loss(model, batch, &cc.coarse_to_fine, ctx.dev)?;

            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()?;
            n_steps += 1;
        }

        let avg = loss_sum / n_steps.max(1) as f32;
        pb.set_message(format!("loss={:.3}", avg));
        pb.inc(1);
        if epoch % 10 == 0 || epoch + 1 == params.epochs {
            info!("epoch {}/{}: loss={:.3}", epoch + 1, params.epochs, avg);
        }
    }
    pb.finish_and_clear();

    Ok(())
}
