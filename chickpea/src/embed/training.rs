//! Inner count-NCE training loop for `chickpea embed-graph`.
//!
//! Per epoch: `batches_per_epoch` minibatches; per minibatch: pick a
//! coarsening seed, sample positive `(feature, cell)` edges from the
//! unified triplet stream (genes + peaks), compute count-NCE loss,
//! AdamW step.

use crate::common::*;
use crate::embed::coarsen::AxisCoarsenings;
use crate::embed::data::UnifiedData;
use crate::embed::loss::{nce_loss, sample_edge_batch, EdgeBatchArgs};
use crate::embed::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::{AdamW, Optimizer};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use rand_distr::weighted::WeightedIndex;

/// Borrowed inputs to the training loop.
pub struct TrainingContext<'a> {
    pub unified: &'a UnifiedData,
    pub cell_axis: &'a AxisCoarsenings,
    pub feat_axis: &'a AxisCoarsenings,
    /// Per-fine-feature loss weight (NB-Fisher for genes; 1.0 for peaks).
    pub feat_weights: &'a [f32],
    /// Count-weighted positive edge sampler **restricted to RNA** triplets
    /// (the underlying triplet stream is unified, but the WeightedIndex
    /// gives zero mass to ATAC entries).
    pub rna_sampler: &'a WeightedIndex<f32>,
    /// Same for ATAC.
    pub atac_sampler: &'a WeightedIndex<f32>,
    /// Per-seed marginal-weighted negative sampler over unified feature blocks.
    pub neg_samplers: &'a [WeightedIndex<f32>],
    pub dev: &'a Device,
}

/// Hyperparameters for the training loop.
pub struct TrainingParams {
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub num_negatives: usize,
    pub seed: u64,
}

/// Run the count-NCE training loop. Mutates `model` parameters via `opt`.
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

    for epoch in 0..params.epochs {
        let mut loss_sum = 0f32;
        let mut n_steps = 0usize;

        for _ in 0..params.batches_per_epoch {
            let seed_k = rng.random_range(0..n_seeds);
            let cc = &ctx.cell_axis.coarsenings[seed_k];
            let fc = &ctx.feat_axis.coarsenings[seed_k];

            // Modality-balanced positive sampling: half the batch from
            // RNA triplets, half from ATAC. The shared E_feat table is
            // updated by both — only the positive distribution is
            // balanced. Negatives still come from the unified marginal
            // sampler (so a gene positive can be contrasted against
            // peak negatives and vice versa, exposing cross-modal
            // discrimination).
            let half = params.batch_size / 2;
            let rna_batch = sample_edge_batch(
                EdgeBatchArgs {
                    triplets: &ctx.unified.triplets,
                    edge_weights: ctx.rna_sampler,
                    cell_coarsening: cc,
                    feat_coarsening: fc,
                    neg_sampler: &ctx.neg_samplers[seed_k],
                    fine_feature_weights: Some(ctx.feat_weights),
                    batch_size: half,
                    n_negatives: params.num_negatives,
                },
                &mut rng,
            );
            let atac_batch = sample_edge_batch(
                EdgeBatchArgs {
                    triplets: &ctx.unified.triplets,
                    edge_weights: ctx.atac_sampler,
                    cell_coarsening: cc,
                    feat_coarsening: fc,
                    neg_sampler: &ctx.neg_samplers[seed_k],
                    fine_feature_weights: Some(ctx.feat_weights),
                    batch_size: params.batch_size - half,
                    n_negatives: params.num_negatives,
                },
                &mut rng,
            );

            let l_rna = nce_loss(
                model,
                &rna_batch,
                &cc.coarse_to_fine,
                &fc.coarse_to_fine,
                ctx.dev,
            )?;
            let l_atac = nce_loss(
                model,
                &atac_batch,
                &cc.coarse_to_fine,
                &fc.coarse_to_fine,
                ctx.dev,
            )?;
            let loss = (&l_rna + &l_atac)?;

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
