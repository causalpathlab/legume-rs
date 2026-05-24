//! Composite multi-axis count-NCE training loop.
//!
//! Each minibatch step samples one positive batch from *every* axis
//! (the per-cell axis plus every pseudobulk-level axis), computes the
//! NCE loss on each, and sums them with per-axis weights `λ_k`. A
//! single AdamW step then updates the shared `E_feat` / `b_feat` Vars
//! (gradients accumulate across all axes' losses naturally — they
//! reference the same tensors) plus each axis's own cell-side Vars.
//!
//! Cell-cell NCE is an additional positive-pair term that attaches
//! only to the per-cell axis (it operates on real `E_cell`, not on
//! pseudobulk embeddings).
//!
//! Polls `stop` at minibatch boundaries so SIGINT cleanly returns to
//! the caller for output finalization.

use crate::coarsen::AxisCoarsenings;
use crate::data::UnifiedData;
use crate::feature_network::FeatureNetworkSmoother;
use crate::loss::{
    nce_loss, nce_loss_chain, nce_loss_identity, sample_chain_batch, sample_edge_batch,
    sample_per_batch_stratified_edge_batch, sample_stratified_edge_batch, ChainAxis,
    ChainBatchArgs, ChainSampler, EdgeBatch, EdgeBatchArgs, PerBatchCellSampler, PerBatchSampler,
    PerBatchStratifiedCellSampler, PerBatchStratifiedEdgeBatchArgs, StratifiedEdgeBatchArgs,
    StratifiedSampler,
};
use crate::model::JointEmbedModel;
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// How `train_composite` mixes per-axis NCE losses each step.
///
/// - [`Sum`]: every step computes one minibatch per axis and sums the
///   losses (with per-axis `λ` weights). Lower-variance gradient per
///   step but `O(n_axes)` work per step.
/// - [`Sample`]: every step picks a single axis with probability
///   `λ_k / Σλ`, computes its NCE on one minibatch, and scales by
///   `Σλ`. Same expected gradient as `Sum`, higher variance per step,
///   `O(1)` work per step.
/// - [`Chain`]: every step samples a coordinated bottom-up chain — a
///   real `(cell, feature)` triplet whose pb ancestors at each level
///   are derived via the cell→pb_per_level map. All axes share the
///   same positive feature and negatives per chain; cell-side indices
///   differ per axis. One feature-side gather + one backward per step,
///   with coherent across-level gradients on `E_feat`. Lowest variance
///   per step, comparable per-step compute to `Sum`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompositeMode {
    #[default]
    Sum,
    Sample,
    Chain,
}

/// One axis in the composite training objective. `model` shares its
/// `e_feat` / `b_feat` Tensors with every other axis (same Var under
/// the hood); `e_cell` / `b_cell` are unique per axis.
pub struct CompositeAxis<'a> {
    pub model: &'a JointEmbedModel,
    pub unified: &'a UnifiedData,
    pub cell_axis: &'a AxisCoarsenings,
    pub sampler: AxisSampler<'a>,
    /// Mixing weight in the summed objective. Defaults to 1.0; tune
    /// down for axes that should have less influence on `E_feat`.
    pub lambda: f32,
    /// Optional cell-cell NCE term — wired only on the per-cell axis
    /// (pseudobulk axes don't have meaningful pair edges). `None`
    /// disables.
    pub cell_cell: Option<CellCellTraining<'a>>,
    /// Short label for log lines (e.g. "cell", "pb_l0"). Cosmetic.
    pub label: &'a str,
}

/// Bipartite sampler attached to a composite axis. Three variants:
/// - `PerBatch`: flat per-batch positive draw weighted by `count·fisher`.
/// - `PerBatchStratified`: per-batch two-stage draw — cell by
///   `degree^α_cell`, feature within cell by `count·fisher`. Guarantees
///   per-cell coverage so rare/shallow cells aren't drowned by deeply
///   sequenced ones. Used by the cell axis by default.
/// - `Stratified`: single-sampler two-stage draw over pb's — pb by
///   `pb_size^α_pb`, feature within pb by `count·fisher`. Used by the
///   pb axes (one synthetic batch each).
pub enum AxisSampler<'a> {
    PerBatch(&'a [PerBatchSampler]),
    PerBatchStratified(&'a [PerBatchStratifiedCellSampler]),
    Stratified(&'a StratifiedSampler),
}

impl<'a> AxisSampler<'a> {
    fn is_empty(&self) -> bool {
        match self {
            Self::PerBatch(s) => s.is_empty(),
            Self::PerBatchStratified(s) => s.is_empty(),
            Self::Stratified(_) => false,
        }
    }
}

pub struct CellCellTraining<'a> {
    pub samplers: &'a [Option<PerBatchCellSampler>],
    pub edges: &'a [(u32, u32)],
    pub lambda: f32,
    pub n_negatives: usize,
    /// Multi-level chain over pseudobulk resolutions: each level scores
    /// the same `(left, right)` positive pair against per-level sibling
    /// negatives summed with `lambdas`.
    pub pb_chain: CellCellPbChainTraining<'a>,
}

/// Resolved per-step view of the cell-cell chain. Owned slices live in
/// [`crate::fit::CellCellPrepared`] so this struct only borrows.
pub struct CellCellPbChainTraining<'a> {
    /// Resolved level indices into `cell_to_pb_per_level` (coarsest-first).
    pub levels: &'a [usize],
    /// Same length as `levels`; per-level mixing weight inside the chain
    /// sum. Multiplied by the outer cell-cell `lambda` at loss time.
    pub lambdas: &'a [f32],
    /// `cell_to_pb_per_level[L][cell] = pb id at level L`, coarsest-first.
    pub cell_to_pb_per_level: &'a [Vec<usize>],
}

pub struct TrainingParams {
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub num_negatives: usize,
    pub seed: u64,
    pub composite_mode: CompositeMode,
    /// Explicit L2 penalty `λ · ‖E_feat‖_F²` on the shared feature
    /// embedding, added to the per-step composite loss before backward.
    /// `0.0` disables. Equivalent to a zero-mean Gaussian prior on
    /// `E_feat` with precision `2 · λ`.
    pub feature_embedding_l2: f32,
}

pub struct CompositeTrainContext<'a> {
    pub axes: &'a [CompositeAxis<'a>],
    pub feat_weights: &'a [f32],
    pub dev: &'a Device,
    pub stop: &'a Arc<AtomicBool>,
    /// Per-level cell→pb mappings (coarsest-first; length = number of
    /// pb axes = `axes.len() - 1`). Required for `CompositeMode::Chain`;
    /// ignored otherwise. Comes from
    /// `MultilevelCollapseOut::cell_to_pb_per_level`.
    pub cell_to_pb_per_level: Option<&'a [Vec<usize>]>,
}

pub fn train_composite(
    ctx: &CompositeTrainContext,
    opt: &mut AdamW,
    params: &TrainingParams,
    smoother: Option<&mut FeatureNetworkSmoother>,
) -> anyhow::Result<()> {
    assert!(!ctx.axes.is_empty(), "composite training needs >= 1 axis");

    let prog_bar = ProgressBar::new(params.epochs as u64);
    prog_bar.set_style(
        ProgressStyle::with_template("{bar:30} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut rng = StdRng::seed_from_u64(params.seed);
    let refresh_every = smoother.as_ref().map(|s| s.refresh_epochs).unwrap_or(0);
    let mut smoother = smoother;

    // Smoother refreshes against the *shared* E_feat — pull it from the
    // first axis (every axis points at the same tensor).
    let shared_e_feat = ctx.axes[0].model.e_feat.clone();

    // Pre-build the axis sampler for `Sample` mode. Reused every step;
    // weights = `λ_k`, so picking axis `k` happens with probability
    // `λ_k / Σλ`. The `Σλ` scale gets applied to the chosen axis's loss
    // so `E_k[L_step] = Σ_k λ_k · L_k` matches `Sum` mode in expectation.
    let lambda_sum: f32 = ctx.axes.iter().map(|a| a.lambda).sum();
    let axis_picker: Option<WeightedIndex<f32>> = if params.composite_mode == CompositeMode::Sample
    {
        let weights: Vec<f32> = ctx.axes.iter().map(|a| a.lambda.max(1e-8)).collect();
        Some(WeightedIndex::new(weights).expect("non-empty axis weights"))
    } else {
        None
    };

    for epoch in 0..params.epochs {
        if let Some(sm) = smoother.as_deref_mut() {
            if refresh_every > 0 && epoch % refresh_every == 0 {
                sm.refresh(&shared_e_feat, ctx.dev)?;
            }
        }

        let mut loss_sum = 0f32;
        let mut n_steps = 0usize;

        for _ in 0..params.batches_per_epoch {
            let loss = match params.composite_mode {
                CompositeMode::Sum => sum_step(ctx, &mut rng, params, smoother.as_deref())?,
                CompositeMode::Sample => sample_step(
                    ctx,
                    &mut rng,
                    params,
                    smoother.as_deref(),
                    axis_picker.as_ref().unwrap(),
                    lambda_sum,
                )?,
                CompositeMode::Chain => chain_step(ctx, &mut rng, params, smoother.as_deref())?,
            };
            let Some(mut loss) = loss else { continue };
            if params.feature_embedding_l2 > 0.0 {
                // `mean_all` keeps λ scale-invariant across `D · H`.
                let l2 = shared_e_feat
                    .sqr()?
                    .mean_all()?
                    .affine(f64::from(params.feature_embedding_l2), 0.0)?;
                loss = (loss + l2)?;
            }
            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()?;
            n_steps += 1;

            if ctx.stop.load(Ordering::Relaxed) {
                break;
            }
        }

        let avg = loss_sum / n_steps.max(1) as f32;
        prog_bar.set_message(format!("loss={:.3}", avg));
        prog_bar.inc(1);
        // Every-epoch info; senna/pinto's `--verbose` flag raises the
        // log level to `info`, so this is gated by the user's choice
        // there. Quiet runs (warn level) suppress it.
        info!(
            "epoch {}/{}: composite loss={:.3}",
            epoch + 1,
            params.epochs,
            avg
        );

        if ctx.stop.load(Ordering::SeqCst) {
            prog_bar.finish_and_clear();
            info!(
                "Stopping early at epoch {}/{} — finalizing outputs",
                epoch + 1,
                params.epochs
            );
            return Ok(());
        }
    }
    prog_bar.finish_and_clear();

    Ok(())
}

/// One step of `CompositeMode::Sum` — sample a minibatch from every
/// axis, compute each axis's NCE loss, return the λ-weighted sum.
fn sum_step(
    ctx: &CompositeTrainContext,
    rng: &mut StdRng,
    params: &TrainingParams,
    smoother: Option<&FeatureNetworkSmoother>,
) -> anyhow::Result<Option<Tensor>> {
    let mut total_loss: Option<Tensor> = None;
    for axis in ctx.axes {
        let Some(loss) = single_axis_step(axis, rng, params, ctx.feat_weights, smoother, ctx.dev)?
        else {
            continue;
        };
        let scaled = (loss * axis.lambda as f64)?;
        total_loss = Some(match total_loss {
            Some(prev) => (prev + scaled)?,
            None => scaled,
        });
    }
    Ok(total_loss)
}

/// One step of `CompositeMode::Sample` — pick a single axis weighted
/// by λ and run its NCE forward. Multiplied by `Σλ` so the per-step
/// gradient is unbiased for the same multi-task objective as `Sum`.
fn sample_step(
    ctx: &CompositeTrainContext,
    rng: &mut StdRng,
    params: &TrainingParams,
    smoother: Option<&FeatureNetworkSmoother>,
    axis_picker: &WeightedIndex<f32>,
    lambda_sum: f32,
) -> anyhow::Result<Option<Tensor>> {
    let axis_idx = axis_picker.sample(rng);
    let axis = &ctx.axes[axis_idx];
    let Some(loss) = single_axis_step(axis, rng, params, ctx.feat_weights, smoother, ctx.dev)?
    else {
        return Ok(None);
    };
    Ok(Some((loss * lambda_sum as f64)?))
}

/// One step of `CompositeMode::Chain` — sample a coordinated chain
/// batch (real cell-axis triplets, with pb ancestors derived from the
/// stored cell→pb maps), then score every axis (cell + each pb level)
/// with the same shared positive feature and shared negatives. One
/// `nce_loss_chain` call returns the λ-weighted sum across all axes.
fn chain_step(
    ctx: &CompositeTrainContext,
    rng: &mut StdRng,
    params: &TrainingParams,
    smoother: Option<&FeatureNetworkSmoother>,
) -> anyhow::Result<Option<Tensor>> {
    let cell_to_pb = ctx.cell_to_pb_per_level.ok_or_else(|| {
        anyhow::anyhow!(
            "CompositeMode::Chain requires CompositeTrainContext.cell_to_pb_per_level = Some(..)"
        )
    })?;
    let cell_axis = ctx
        .axes
        .first()
        .ok_or_else(|| anyhow::anyhow!("CompositeMode::Chain needs the cell axis as axes[0]"))?;
    let pb_axes = &ctx.axes[1..];
    anyhow::ensure!(
        pb_axes.len() == cell_to_pb.len(),
        "Chain mode: {} pb axes vs {} cell→pb levels — must match",
        pb_axes.len(),
        cell_to_pb.len()
    );

    // Chain mode accepts either flat per-batch or stratified per-batch
    // cell samplers — both produce real `(cell, feature)` positives that
    // can be walked up the pb tree. Stratified is the recommended default.
    let (chain, batch_id) = match cell_axis.sampler {
        AxisSampler::PerBatch(samplers) => {
            if samplers.is_empty() {
                return Ok(None);
            }
            anyhow::ensure!(
                !cell_axis.unified.triplets.is_empty(),
                "flat PerBatch cell sampler needs a materialized edge list, but \
                 unified.triplets is empty (the streaming PerBatchStratified path \
                 leaves it empty) — call materialize_cell_triplets() to revive the \
                 flat path"
            );
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            let chain_sampler = ChainSampler {
                batch_sampler: bs,
                cell_to_pb_per_level: cell_to_pb,
            };
            let chain = sample_chain_batch(
                ChainBatchArgs {
                    triplets: &cell_axis.unified.triplets,
                    sampler: &chain_sampler,
                    fine_feature_weights: ctx.feat_weights,
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                rng,
            );
            (chain, id)
        }
        AxisSampler::PerBatchStratified(samplers) => {
            if samplers.is_empty() {
                return Ok(None);
            }
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            let chain = sample_chain_batch_stratified(
                bs,
                ctx.feat_weights,
                params.batch_size,
                params.num_negatives,
                rng,
            );
            (chain, id)
        }
        AxisSampler::Stratified(_) => {
            anyhow::bail!(
                "Chain mode: cell axis must use PerBatch or PerBatchStratified \
                 (need real triplets). Got Stratified (pb-only)."
            );
        }
    };
    let crate::loss::ChainBatch { leaf_cells, feats } = chain;
    let b = leaf_cells.len();

    // Build all per-axis cell-side index tensors up front. The cell
    // axis uses leaf_cells directly; each pb axis derives its indices
    // by mapping leaf_cells through that level's cell→pb on host
    // *before* moving leaf_cells into a tensor (avoids a `to_vec1`
    // round-trip back from device). One Vec→Tensor allocation per
    // axis per step instead of two (the loss function used to clone
    // again to call `Tensor::from_vec` itself).
    let mut idx_tensors: Vec<Tensor> = Vec::with_capacity(ctx.axes.len());
    for c2p in cell_to_pb {
        let pb_ids: Vec<u32> = leaf_cells.iter().map(|&c| c2p[c as usize] as u32).collect();
        idx_tensors.push(Tensor::from_vec(pb_ids, b, ctx.dev)?);
    }
    let cell_idx_tensor = Tensor::from_vec(leaf_cells, b, ctx.dev)?;

    let mut chain_axes: Vec<ChainAxis> = Vec::with_capacity(ctx.axes.len());
    chain_axes.push(ChainAxis {
        e_cell: &cell_axis.model.e_cell,
        b_cell: &cell_axis.model.b_cell,
        indices: &cell_idx_tensor,
        lambda: cell_axis.lambda,
        label: cell_axis.label,
    });
    for (i, axis) in pb_axes.iter().enumerate() {
        chain_axes.push(ChainAxis {
            e_cell: &axis.model.e_cell,
            b_cell: &axis.model.b_cell,
            indices: &idx_tensors[i],
            lambda: axis.lambda,
            label: axis.label,
        });
    }

    // Cell-cell loss attaches to the cell axis only. Reuses the existing
    // PerBatchCellSampler logic — independent draw, summed in.
    let cc_loss: Option<Tensor> = match cell_axis.cell_cell.as_ref() {
        Some(cc_ctx) => match cc_ctx.samplers[batch_id].as_ref() {
            Some(cc_sampler) => Some(compute_cell_cell_loss(
                cell_axis.model,
                cc_ctx,
                cc_sampler,
                params,
                rng,
                ctx.dev,
            )?),
            None => None,
        },
        None => None,
    };

    let chain_loss = nce_loss_chain(
        &cell_axis.model.e_feat,
        &cell_axis.model.b_feat,
        feats,
        &chain_axes,
        smoother,
        ctx.dev,
    )?;
    let total = match cc_loss {
        Some(cc) => (chain_loss + cc)?,
        None => chain_loss,
    };
    Ok(Some(total))
}

/// Chain-batch sampler for the stratified per-batch cell sampler.
/// Mirrors `loss::sample_chain_batch` but draws each leaf
/// `(cell, feature)` via the two-stage `cell_picker` → `per_cell` path
/// instead of a flat triplet pick. Shared negatives come from the same
/// per-batch feature pool, so downstream `nce_loss_chain` consumes the
/// same `ChainBatch` shape regardless of which sampler produced it.
fn sample_chain_batch_stratified(
    bs: &PerBatchStratifiedCellSampler,
    fine_feature_weights: &[f32],
    batch_size: usize,
    n_negatives: usize,
    rng: &mut StdRng,
) -> crate::loss::ChainBatch {
    let mut leaf_cells = Vec::with_capacity(batch_size);
    let mut fine_feats = Vec::with_capacity(batch_size);
    let mut weights = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let lc = bs.cell_picker.sample(rng);
        let c = bs.active_cells[lc];
        let pf = &bs.per_cell[lc];
        let lf = pf.picker.sample(rng);
        let f = pf.features[lf];
        leaf_cells.push(c);
        fine_feats.push(f);
        weights.push(fine_feature_weights[f as usize]);
    }

    let mut neg_feats = Vec::with_capacity(batch_size * n_negatives);
    for _ in 0..(batch_size * n_negatives) {
        let local = bs.neg.sample(rng);
        neg_feats.push(bs.feature_pool[local]);
    }

    crate::loss::ChainBatch {
        leaf_cells,
        feats: crate::loss::ChainFeatureSide {
            fine_feats,
            neg_feats,
            edge_weights: weights,
            n_negatives,
        },
    }
}

/// Sample a minibatch from a single axis, compute its bipartite NCE
/// loss (taking the identity fast path when the axis has identity
/// coarsening), and add the optional cell-cell term. Returns `None`
/// when the axis has no positives to sample.
fn single_axis_step(
    axis: &CompositeAxis,
    rng: &mut StdRng,
    params: &TrainingParams,
    feat_weights: &[f32],
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> anyhow::Result<Option<Tensor>> {
    if axis.sampler.is_empty() {
        return Ok(None);
    }
    let n_seeds = axis.cell_axis.coarsenings.len();
    if n_seeds == 0 {
        return Ok(None);
    }
    let seed_k = if n_seeds == 1 {
        0
    } else {
        rng.random_range(0..n_seeds)
    };
    let cc = &axis.cell_axis.coarsenings[seed_k];

    let (batch, batch_id): (EdgeBatch, Option<usize>) = match axis.sampler {
        AxisSampler::PerBatch(samplers) => {
            anyhow::ensure!(
                !axis.unified.triplets.is_empty(),
                "flat PerBatch sampler needs a materialized edge list, but \
                 unified.triplets is empty (streaming PerBatchStratified leaves it \
                 empty) — call materialize_cell_triplets() to revive the flat path"
            );
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            let batch = sample_edge_batch(
                EdgeBatchArgs {
                    triplets: &axis.unified.triplets,
                    batch_sampler: bs,
                    cell_coarsening: cc,
                    fine_feature_weights: Some(feat_weights),
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                rng,
            );
            (batch, Some(id))
        }
        AxisSampler::PerBatchStratified(samplers) => {
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            let batch = sample_per_batch_stratified_edge_batch(
                PerBatchStratifiedEdgeBatchArgs {
                    sampler: bs,
                    cell_coarsening: cc,
                    fine_feature_weights: feat_weights,
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                rng,
            );
            (batch, Some(id))
        }
        AxisSampler::Stratified(s) => {
            let batch = sample_stratified_edge_batch(
                StratifiedEdgeBatchArgs {
                    sampler: s,
                    fine_feature_weights: feat_weights,
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                rng,
            );
            (batch, None)
        }
    };

    let bip_loss = if axis.cell_axis.is_identity {
        nce_loss_identity(axis.model, batch, smoother, dev)?
    } else {
        nce_loss(axis.model, batch, &cc.coarse_to_fine, smoother, dev)?
    };
    let mut axis_loss = bip_loss;

    if let (Some(cc_ctx), Some(batch_id)) = (axis.cell_cell.as_ref(), batch_id) {
        if let Some(cc_sampler) = cc_ctx.samplers[batch_id].as_ref() {
            let cc_loss = compute_cell_cell_loss(axis.model, cc_ctx, cc_sampler, params, rng, dev)?;
            axis_loss = (axis_loss + cc_loss)?;
        }
    }
    Ok(Some(axis_loss))
}

/// Cell-cell loss step. Runs the multi-level chain with per-level
/// sibling negatives and returns the loss already scaled by
/// `cc_ctx.lambda`.
fn compute_cell_cell_loss(
    model: &JointEmbedModel,
    cc_ctx: &CellCellTraining<'_>,
    cc_sampler: &PerBatchCellSampler,
    params: &TrainingParams,
    rng: &mut StdRng,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let chain = &cc_ctx.pb_chain;
    let pb_maps: Vec<&[usize]> = chain
        .levels
        .iter()
        .map(|&l| chain.cell_to_pb_per_level[l].as_slice())
        .collect();
    let (batch, _stats) = crate::loss::sample_cell_chain_batch(
        crate::loss::CellChainBatchArgs {
            edges: cc_ctx.edges,
            batch_sampler: cc_sampler,
            batch_size: params.batch_size,
            n_negatives: cc_ctx.n_negatives,
            pb_maps: &pb_maps,
        },
        rng,
    );
    let raw = crate::loss::cell_cell_nce_loss_chain(model, batch, chain.lambdas, dev)?;
    Ok((raw * cc_ctx.lambda as f64)?)
}
