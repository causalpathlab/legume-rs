//! Composite multi-axis count-NCE training loop.
//!
//! Each minibatch step samples one positive batch from *every* axis
//! (the per-cell axis plus every pseudobulk-level axis), computes the
//! NCE loss on each, and sums them with per-axis weights `λ_k`. A
//! single `AdamW` step then updates the shared `E_feat` / `b_feat` Vars
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
use crate::fit::lineage::PbLineageLevel;
use crate::loss::{
    nce_loss, nce_loss_identity, sample_per_batch_stratified_edge_batch,
    sample_stratified_edge_batch, EdgeBatch, PerBatchStratifiedCellSampler,
    PerBatchStratifiedEdgeBatchArgs, StratifiedEdgeBatchArgs, StratifiedSampler,
};
use crate::model::{JointEmbedModel, GATE_KL_REF_UNITS, GATE_KL_STEP_WEIGHT};
use crate::progress::new_progress_bar;
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::AdamW;
use log::info;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    /// Short label for log lines (e.g. "cell", "`pb_l0`"). Cosmetic.
    pub label: &'a str,
}

/// Bipartite sampler attached to a composite axis. Two variants:
/// - `PerBatchStratified`: per-batch two-stage draw — cell by
///   `degree^α_cell`, feature within cell by `count`. Guarantees
///   per-cell coverage so rare/shallow cells aren't drowned by deeply
///   sequenced ones. Used by the cell axis by default.
/// - `Stratified`: single-sampler two-stage draw over pb's — pb by
///   `pb_size^α_pb`, feature within pb by `count`. Used by the
///   pb axes (one synthetic batch each).
///
/// A third, `PerBatch` — a flat per-batch positive draw weighted by `count` — was
/// matched in two places and constructed in none. It read `UnifiedData::triplets`,
/// which the cell axis leaves empty because positives are drawn from the backend at
/// sample time.
pub enum AxisSampler<'a> {
    PerBatchStratified(&'a [PerBatchStratifiedCellSampler]),
    Stratified(&'a StratifiedSampler),
}

impl AxisSampler<'_> {
    fn is_empty(&self) -> bool {
        match self {
            Self::PerBatchStratified(s) => s.is_empty(),
            Self::Stratified(_) => false,
        }
    }

    /// Number of "draw units" on this axis — used by auto
    /// `--batches-per-epoch` (one weighted pass = `n_units / batch_size`).
    /// A cell axis exposes one unit per *cell* (summed across its per-batch
    /// samplers) so a "pass" sweeps every cell once; pb axes expose
    /// `active_pbs.len()`. (The cell axis previously reported the number of
    /// batches here, which starved per-cell training — the cell axis was
    /// invisible to the budget and got the same ~1 step/epoch as the pb
    /// axes despite having orders of magnitude more units.)
    #[must_use]
    pub fn n_units(&self) -> usize {
        match self {
            Self::PerBatchStratified(s) => s.iter().map(|x| x.active_cells.len()).sum(),
            Self::Stratified(s) => s.active_pbs.len(),
        }
    }
}

pub struct TrainingParams {
    pub epochs: usize,
    /// `None` = auto: one weighted pass over the largest axis
    /// (`ceil(max_axis_units / batch_size)`). `Some(n)` = fixed budget.
    pub batches_per_epoch: Option<usize>,
    pub batch_size: usize,
    pub num_negatives: usize,
    pub seed: u64,
    /// Which NCE objective the feature side trains with ([`crate::loss::NceObjective`]).
    /// Defaults to `Softmax` (InfoNCE). `senna gem`, `senna bge` and `pinto cage` all
    /// expose it as `--nce-objective` and all default to `Softmax`; `Logistic` is opt-in.
    pub objective: crate::loss::NceObjective,
    /// Explicit L2 penalty `λ · ‖E_feat‖_F²` on the shared feature
    /// embedding, added to the per-step composite loss before backward.
    /// `0.0` disables. Equivalent to a zero-mean Gaussian prior on
    /// `E_feat` with precision `2 · λ`.
    pub feature_embedding_l2: f32,
    /// Global-norm gradient clip per `AdamW` step (`0.0` = off). Bounds the
    /// update magnitude so embeddings don't inflate on NCE loss spikes.
    pub max_grad_norm: f32,
    /// L2 (ridge) penalty `λ · mean(δ_g²)` on the per-gene splice offset (factored
    /// β-sharing only). Shrinks `δ_g` toward 0 so the splice signal is explained
    /// on the cell axis unless a gene's nascent deviation genuinely lowers the
    /// loss — a dense prior that fits the (dense) per-gene γ structure and is
    /// well-behaved under AdamW. `0.0` disables (plain β-sharing, no `δ_g`).
    pub delta_l2: f32,
}

pub struct CompositeTrainContext<'a> {
    pub axes: &'a [CompositeAxis<'a>],
    pub dev: &'a Device,
    pub stop: &'a Arc<AtomicBool>,
    /// Per-level cell→pb mappings (coarsest-first; length = number of pb axes =
    /// `axes.len() - 1`), from `MultilevelCollapseOut::cell_to_pb_per_level`.
    ///
    /// Currently unread — its one consumer was the chain composite mode, which was
    /// deleted as unreachable. Kept because a nested chain sampler is the standing plan
    /// for this field and rebuilding the plumbing is the expensive part.
    pub cell_to_pb_per_level: Option<&'a [Vec<usize>]>,
    /// Optional per-axis velocity-drift SEM term (lineage-DAG refine pass, fixed velocity-KNN
    /// structure). Aligned 1:1 with `axes`: `None` for axes with no lineage structure
    /// (the cell axis, and any pb level with no oriented edges), `Some` otherwise.
    /// When set, each `Some` term's penalty is added to the per-step loss so its
    /// pb embedding is pulled toward the velocity-consistent geometry.
    pub lineage_sem: Option<&'a [Option<PbSemTerm>]>,
    /// Optional SECOND per-axis SEM term: the θ-pseudotime DAG (same `PbSemTerm` form,
    /// but the drift is the θ-manifold pseudotime gradient, not velocity). Added
    /// alongside `lineage_sem` so the embedding is shaped by both the
    /// velocity flow AND the dense identity manifold — robust where δ is sparse (a
    /// δ-less pb is dropped by the velocity graph but kept by this one). Aligned 1:1
    /// with `axes`; `None` (default) is a no-op.
    pub lineage_sem_theta: Option<&'a [Option<PbSemTerm>]>,
}

/// Device-side velocity-drift SEM term for one pb axis (lineage-DAG, fixed velocity-KNN
/// structure). Precomputes the per-edge source/target index tensors, weights, and
/// the constant drift `s·v̂_i`, so each training step is two `index_select`s plus
/// elementwise ops. Penalizes `Σ_{i→j} w_ij ‖e_j − e_i − s·v̂_i‖² · λ / Σw`, i.e.
/// a pb node should sit one velocity-step ahead of its parent along the flow.
pub struct PbSemTerm {
    /// `[E]` parent (source) node id per edge.
    src: Tensor,
    /// `[E]` child (target) node id per edge.
    dst: Tensor,
    /// `[E]` edge weight.
    w: Tensor,
    /// `[E, H]` constant drift `s·v̂_i` (unit velocity of the parent, scaled).
    drift: Tensor,
    /// `λ_sem / Σw` — the weight, folded with the `Σw` normalizer.
    scale: f64,
}

impl PbSemTerm {
    /// Build the device term for one level, or `None` when the level has no
    /// oriented edges (nothing to penalize). `step` is `s`, `weight` is `λ_sem`.
    pub fn new(
        level: &PbLineageLevel,
        h: usize,
        step: f32,
        weight: f32,
        dev: &Device,
    ) -> anyhow::Result<Option<Self>> {
        if level.edges.is_empty() {
            return Ok(None);
        }
        let e = level.edges.len();
        let mut src = Vec::with_capacity(e);
        let mut dst = Vec::with_capacity(e);
        let mut w = Vec::with_capacity(e);
        let mut drift = Vec::with_capacity(e * h);
        let mut wsum = 0f32;
        for &(i, j, wij) in &level.edges {
            src.push(i);
            dst.push(j);
            w.push(wij);
            wsum += wij;
            let vi = &level.velocity[i as usize * h..(i as usize + 1) * h];
            for &vk in vi {
                drift.push(step * vk);
            }
        }
        Ok(Some(Self {
            src: Tensor::from_vec(src, e, dev)?,
            dst: Tensor::from_vec(dst, e, dev)?,
            w: Tensor::from_vec(w, e, dev)?,
            drift: Tensor::from_vec(drift, (e, h), dev)?,
            scale: f64::from(weight) / f64::from(wsum.max(1e-8)),
        }))
    }
}

/// Velocity-drift SEM penalty on one axis's pb embedding `e_cell`, differentiable
/// in `e_cell`: `λ · (Σ_ij w_ij ‖e_j − e_i − s·v̂_i‖²) / Σw`.
fn sem_penalty(e_cell: &Tensor, term: &PbSemTerm) -> anyhow::Result<Tensor> {
    let e_src = e_cell.index_select(&term.src, 0)?;
    let e_dst = e_cell.index_select(&term.dst, 0)?;
    let resid = e_dst.sub(&e_src)?.sub(&term.drift)?; // [E, H]
    let sq = resid.sqr()?.sum(1)?; // [E]
    let weighted = sq.mul(&term.w)?.sum_all()?; // scalar
    Ok(weighted.affine(term.scale, 0.0)?)
}

/// Returns the final epoch's mean composite loss (an NCE ≈ neg-log-likelihood proxy),
/// used as a fit-hygiene signal by the lineage QC.
pub fn train_composite(
    ctx: &CompositeTrainContext,
    opt: &mut AdamW,
    params: &TrainingParams,
) -> anyhow::Result<f32> {
    assert!(!ctx.axes.is_empty(), "composite training needs >= 1 axis");

    // Shared style — consistent with every other faba/senna progress bar
    // (`[elapsed] bar pos/len (eta) msg`).
    let prog_bar = new_progress_bar(params.epochs as u64);

    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut last_avg = 0f32; // final-epoch mean loss, returned as the fit-hygiene signal

    // `--feature-embedding-l2` penalizes the *shared* E_feat — pull it from the
    // first axis (every axis points at the same tensor).
    let shared_e_feat = ctx.axes[0].model.e_feat.clone();

    // Shared per-gene splice offset δ_g (factored splice models), for the L2 (ridge)
    // penalty below. `None` for free / plain-β-sharing models.
    let shared_delta = ctx.axes[0]
        .model
        .factor
        .as_ref()
        .and_then(|f| f.splice_delta.as_ref().map(|(delta, _)| delta.clone()));
    // Resolve `batches_per_epoch`: explicit override, or auto = one
    // weighted pass over the largest axis. `n_units` is per-cell for the
    // cell axis and `active_pbs.len()` for the pb axes.
    let max_axis_units = ctx
        .axes
        .iter()
        .map(|a| a.sampler.n_units())
        .max()
        .unwrap_or(0);
    let batches_per_epoch = resolve_batches_per_epoch(params, max_axis_units);
    // The gate KL used to be `λ/batch_size`; it is now pinned to the reference
    // so a throughput knob stops retuning feature sparsity. At the default that
    // is the same number, but a non-default `--batch-size` DOES change results
    // versus older builds, so say so rather than let it look like seed noise.
    if params.batch_size != GATE_KL_REF_UNITS as usize
        && ctx.axes.iter().any(|a| a.model.gate.is_some())
    {
        info!(
            "gate KL is pinned at 1/{:.0} (was 1/batch_size); with --batch-size {} \
             the gate is {:.2}x the strength older builds applied",
            GATE_KL_REF_UNITS,
            params.batch_size,
            params.batch_size as f64 / GATE_KL_REF_UNITS
        );
    }
    log::info!(
        "train_composite: {} epochs × {} batches (auto={}, max_axis_units={})",
        params.epochs,
        batches_per_epoch,
        params.batches_per_epoch.is_none(),
        max_axis_units,
    );

    for epoch in 0..params.epochs {
        // One `z ~ Bern(pip)` draw for the whole epoch, shared by every axis.
        //
        // Per-EPOCH, not per-minibatch: `z` is a latent for the DATASET, so a per-batch
        // draw would model it as if each minibatch had its own inclusion state and would
        // add gradient variance for nothing. No-op unless a `gate_pip` is installed.
        ctx.axes[0].model.resample_gate_mask()?;
        // Loss kept **on-device** and synced to a scalar once per epoch (not
        // per minibatch) — `detach()` keeps the running sum off the autograd
        // graph so each step's forward graph is still freed immediately,
        // while avoiding a per-step GPU→CPU stall. Mirrors senna gem.
        let mut loss_acc: Option<Tensor> = None;
        let mut n_steps = 0usize;

        for _ in 0..batches_per_epoch {
            let loss = sum_step(ctx, &mut rng, params)?;
            let Some(mut loss) = loss else { continue };
            if params.feature_embedding_l2 > 0.0 {
                // `λ · mean_g ‖e_g‖²` — see `loss::embedding_ridge` for why the
                // reduction sums over the latent axis instead of averaging over it.
                let l2 = crate::loss::embedding_ridge(
                    &shared_e_feat,
                    f64::from(params.feature_embedding_l2),
                )?;
                loss = (loss + l2)?;
            }
            // SuSiE single-effect KL on the gate (identity + velocity): the fixed
            // `GATE_KL_WEIGHT · (categorical + Gaussian) KL`. `None` when the gate is
            // off. The shared gate lives on every axis identically; axes[0] is the
            // representative (counted once, not once per axis).
            //
            // Weighted by `gate_kl_step_weight`. `data_units = 1` is deliberate: it
            // reproduces the historical level `λ/1024` EXACTLY at the default
            // `--batch-size`, which is what makes this a correctness fix. The share
            // `w/u` therefore still carries a `1/axis_count` factor, so
            // `--phase1-cells-per-pb` and `--num-levels` continue to retune the prior.
            // Normalizing that away is a real behavioural change and is deferred.
            //
            // This replaced `GATE_KL_WEIGHT / batch_size`, which made that share
            // `∝ 1/B`: `--batch-size 4096` quartered the prior and `64` raised it 16×,
            // so a throughput flag retuned feature sparsity. At the default
            // `--batch-size 1024` — which both `senna bge` and `senna gem` carry — the
            // new weight is numerically identical to the old one, so default runs are
            // unchanged.
            if let Some(kl) = ctx.axes[0].model.gate_kl()? {
                let w = GATE_KL_STEP_WEIGHT;
                loss = (loss + kl.affine(w, 0.0)?)?;
            }
            // L2 (ridge) shrinkage on the per-gene splice offset δ_g (factored
            // models with a splice split). `mean(δ_g²)` keeps λ scale-invariant
            // across `G · H` (mirrors the feature-embedding L2 above).
            if let (Some(delta), l2) = (&shared_delta, params.delta_l2) {
                if l2 > 0.0 {
                    let pen = delta.sqr()?.mean_all()?.affine(f64::from(l2), 0.0)?;
                    loss = (loss + pen)?;
                }
            }
            // Velocity-drift SEM residual (lineage-DAG refine pass). One penalty per
            // pb axis with oriented edges, pulling its pb embedding toward the
            // velocity-consistent geometry; gradients reach `E_feat` through the NCE
            // coupling. `None` (default) is a no-op → byte-identical training.
            if let Some(terms) = ctx.lineage_sem {
                for (axis, term) in ctx.axes.iter().zip(terms) {
                    if let Some(t) = term {
                        loss = (loss + sem_penalty(&axis.model.e_cell, t)?)?;
                    }
                }
            }
            // θ-pseudotime DAG: the SECOND lineage term, same `sem_penalty` form but
            // drifting along the θ-manifold pseudotime instead of velocity. Added every
            // step alongside the velocity term (fixed structure, like `lineage_sem`).
            if let Some(terms) = ctx.lineage_sem_theta {
                for (axis, term) in ctx.axes.iter().zip(terms) {
                    if let Some(t) = term {
                        loss = (loss + sem_penalty(&axis.model.e_cell, t)?)?;
                    }
                }
            }
            // Backward + optional global-norm gradient clip + step.
            candle_util::grad_clip::clipped_backward_step(
                opt,
                &loss,
                f64::from(params.max_grad_norm),
            )?;
            let ld = loss.detach();
            loss_acc = Some(match loss_acc.take() {
                None => ld,
                Some(a) => (a + ld)?,
            });
            n_steps += 1;

            if ctx.stop.load(Ordering::Relaxed) {
                break;
            }
        }

        // Single GPU→CPU sync per epoch.
        let avg = match &loss_acc {
            Some(t) => t.to_scalar::<f32>()? / n_steps.max(1) as f32,
            None => 0f32,
        };
        last_avg = avg;
        prog_bar.set_message(format!("loss={avg:.3}"));
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
            return Ok(last_avg);
        }
    }
    prog_bar.finish_and_clear();

    Ok(last_avg)
}

/// Resolve the per-epoch step budget: an explicit override, or auto = one
/// weighted pass over the largest axis.
///
/// Lifted out of `train_composite` so the gate-KL invariance test can call it
/// without standing up a training context.
pub fn resolve_batches_per_epoch(params: &TrainingParams, max_axis_units: usize) -> usize {
    params.batches_per_epoch.unwrap_or_else(|| {
        let bs = params.batch_size.max(1);
        max_axis_units.div_ceil(bs).max(1)
    })
}

/// One training step — sample a minibatch from every
/// axis, compute each axis's NCE loss, return the λ-weighted sum.
fn sum_step(
    ctx: &CompositeTrainContext,
    rng: &mut StdRng,
    params: &TrainingParams,
) -> anyhow::Result<Option<Tensor>> {
    let mut total_loss: Option<Tensor> = None;
    for axis in ctx.axes {
        let Some(loss) = single_axis_step(axis, rng, params, ctx.dev)? else {
            continue;
        };
        let scaled = (loss * f64::from(axis.lambda))?;
        total_loss = Some(match total_loss {
            Some(prev) => (prev + scaled)?,
            None => scaled,
        });
    }
    Ok(total_loss)
}

/// Sample a minibatch from a single axis and compute its bipartite NCE
/// loss (taking the identity fast path when the axis has identity
/// coarsening). Returns `None` when the axis has no positives to sample.
fn single_axis_step(
    axis: &CompositeAxis,
    rng: &mut StdRng,
    params: &TrainingParams,
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

    let batch: EdgeBatch = match axis.sampler {
        AxisSampler::PerBatchStratified(samplers) => {
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            sample_per_batch_stratified_edge_batch(
                PerBatchStratifiedEdgeBatchArgs {
                    sampler: bs,
                    cell_coarsening: cc,
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                rng,
            )
        }
        AxisSampler::Stratified(s) => sample_stratified_edge_batch(
            StratifiedEdgeBatchArgs {
                sampler: s,
                batch_size: params.batch_size,
                n_negatives: params.num_negatives,
            },
            rng,
        ),
    };

    let bip_loss = if axis.cell_axis.is_identity {
        nce_loss_identity(axis.model, batch, params.objective, dev)?
    } else {
        nce_loss(axis.model, batch, &cc.coarse_to_fine, params.objective, dev)?
    };
    Ok(Some(bip_loss))
}

#[cfg(test)]
mod tests;
