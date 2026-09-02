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
    dense_count_block, draw_gene_keep_mask, log_membership_diagnostics, masked_membership,
    membership_rows_host, module_priors, module_step_loss, nce_loss, nce_loss_identity,
    sample_per_batch_stratified_edge_batch, sample_stratified_edge_batch, EdgeBatch, ModulePools,
    PerBatchStratifiedCellSampler, PerBatchStratifiedEdgeBatchArgs, StratifiedEdgeBatchArgs,
    StratifiedSampler,
};
use crate::model::{FeatModules, JointEmbedModel, GATE_KL_REF_UNITS, GATE_KL_STEP_WEIGHT};
use crate::progress::new_progress_bar;
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::AdamW;
use log::info;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use rand_distr::Distribution;
use rayon::prelude::*;
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

    /// One expressed-feature pool per sampler on this axis, in sampler order — the
    /// index the within-module negative pools are keyed by.
    fn feature_pools(&self) -> Vec<&[u32]> {
        match self {
            Self::PerBatchStratified(s) => s.iter().map(|x| x.feature_pool.as_slice()).collect(),
            Self::Stratified(s) => vec![s.feature_pool.as_slice()],
        }
    }
}

/// Training-side knobs of the learned gene modules (see
/// [`crate::fit::GeneModuleConfig`] for the caller-facing form and the meaning of
/// each field). `Some` only for a module-parameterized model.
#[derive(Clone, Debug)]
pub struct ModuleTrainParams {
    pub warmup_epochs: usize,
    pub gene_dropout: f32,
    pub units_per_step: usize,
    pub lambda_module: f32,
    pub lambda_balance: f32,
    pub residual_l2: f32,
}

/// Per-run state of the module layer that lives with the training loop rather
/// than the model: the host-side negative pools per axis and sampler, refreshed
/// from the membership once per epoch.
struct ModuleState<'a> {
    modules: &'a FeatModules,
    params: &'a ModuleTrainParams,
    /// `pools[axis][sampler]`.
    pools: Vec<Vec<ModulePools>>,
    /// The host copy of `π` the pools were built from; also feeds the diagnostics.
    pi_host: Vec<f32>,
}

impl<'a> ModuleState<'a> {
    fn new(
        modules: &'a FeatModules,
        params: &'a ModuleTrainParams,
        ctx: &CompositeTrainContext,
    ) -> anyhow::Result<Self> {
        let mut st = Self {
            modules,
            params,
            pools: Vec::new(),
            pi_host: Vec::new(),
        };
        st.refresh(ctx)?;
        Ok(st)
    }

    /// Rebuild the pools from the current membership (one device→host copy).
    fn refresh(&mut self, ctx: &CompositeTrainContext) -> anyhow::Result<()> {
        let pi = candle_util::nn::sparsemax(&self.modules.logits.detach())?;
        self.pi_host = pi.flatten_all()?.to_vec1::<f32>()?;
        let n_features = self.modules.logits.dim(0)?;
        let rows = membership_rows_host(&self.pi_host, n_features, self.modules.n_modules);
        self.pools = ctx
            .axes
            .par_iter()
            .map(|axis| {
                axis.sampler
                    .feature_pools()
                    .into_iter()
                    .map(|pool| ModulePools::build(rows.clone(), self.modules.n_modules, pool))
                    .collect()
            })
            .collect();
        Ok(())
    }

    fn log_diagnostics(&self, epoch: usize, epochs: usize) -> anyhow::Result<()> {
        let fallbacks: usize = self
            .pools
            .iter()
            .flatten()
            .map(ModulePools::take_fallbacks)
            .sum();
        log_membership_diagnostics(self.modules, &self.pi_host, epoch, epochs, fallbacks)?;
        Ok(())
    }
}

#[derive(Clone)]
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
    /// `Some(frac)`: on a CUDA device, probe one forward per candidate
    /// size through [`candle_util::device::auto_chunk_size`] and SHRINK
    /// `batch_size` from its configured value when free device memory
    /// says so (never grow past it: batch size is not fit-neutral).
    /// `None`, a CPU device, or an unavailable memory query all keep
    /// `batch_size` exactly as configured.
    pub gpu_mem_fraction: Option<f32>,
    /// L2 (ridge) penalty `λ · mean(δ_g²)` on the per-gene splice offset (factored
    /// β-sharing only). Shrinks `δ_g` toward 0 so the splice signal is explained
    /// on the cell axis unless a gene's nascent deviation genuinely lowers the
    /// loss — a dense prior that fits the (dense) per-gene γ structure and is
    /// well-behaved under AdamW. `0.0` disables (plain β-sharing, no `δ_g`).
    pub delta_l2: f32,
    /// Learned-module training knobs; `Some` iff the model is module-parameterized.
    pub module: Option<ModuleTrainParams>,
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

    // The module layer, when the model has one: hold the warm-start membership for
    // the warm-up, build the within-module negative pools, and pick the ridge that
    // applies (the residual is the module model's only per-row table).
    let mut module_state = match (ctx.axes[0].model.modules.as_ref(), params.module.as_ref()) {
        (Some(m), Some(p)) => {
            m.set_frozen(p.warmup_epochs > 0);
            info!(
                "gene modules: membership held for {} of {} epochs, then trained; {} units per \
                 step per axis pooled for the exact module term, gene dropout {}",
                p.warmup_epochs, params.epochs, p.units_per_step, p.gene_dropout
            );
            Some(ModuleState::new(m, p, ctx)?)
        }
        (Some(_), None) => {
            anyhow::bail!("a module-parameterized model needs `TrainingParams::module`; got None")
        }
        _ => None,
    };
    let ridge_lambda = match &module_state {
        Some(st) => st.params.residual_l2,
        None => params.feature_embedding_l2,
    };

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
    // Memory-aware batch size, resolved BEFORE the per-epoch budget so
    // the step count matches the size actually trained with. The probe
    // runs the same `sum_step` forward the loop uses, never backward.
    let params = &{
        let mut resolved = params.clone();
        if let Some(frac) = params.gpu_mem_fraction {
            let mut probe_rng = StdRng::seed_from_u64(params.seed ^ 0x9e37_79b9);
            if let Some(n) = candle_util::device::auto_chunk_size(
                ctx.dev,
                params.batch_size,
                16.min(params.batch_size),
                frac,
                |n| {
                    let mut probe_params = params.clone();
                    probe_params.batch_size = n;
                    match sum_step(ctx, &mut probe_rng, &probe_params, None) {
                        Ok(Some(loss)) => Ok(loss),
                        Ok(None) => Err(candle_util::candle_core::Error::Msg(
                            "probe sampled nothing".into(),
                        )),
                        Err(e) => Err(candle_util::candle_core::Error::Msg(e.to_string())),
                    }
                },
            ) {
                resolved.batch_size = n;
            }
        }
        resolved
    };

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
        // Release the membership once the warm-up is over. Pools are refreshed at the
        // END of each epoch below, so the first trained epoch still contrasts within
        // the warm-start modules — one epoch of lag, the same as the gate mask.
        if let Some(st) = &module_state {
            if epoch == st.params.warmup_epochs && st.modules.is_frozen() {
                st.modules.set_frozen(false);
                info!(
                    "epoch {}/{}: module membership released — π now trains with the exact \
                     module term and the balance prior",
                    epoch + 1,
                    params.epochs
                );
            }
        }
        // Loss kept **on-device** and synced to a scalar once per epoch (not
        // per minibatch) — `detach()` keeps the running sum off the autograd
        // graph so each step's forward graph is still freed immediately,
        // while avoiding a per-step GPU→CPU stall. Mirrors senna gem.
        let mut loss_acc: Option<Tensor> = None;
        let mut n_steps = 0usize;

        for _ in 0..batches_per_epoch {
            let loss = sum_step(ctx, &mut rng, params, module_state.as_ref())?;
            let Some(mut loss) = loss else { continue };
            if ridge_lambda > 0.0 {
                // `λ · mean_g ‖e_g‖²` on whichever per-row table this parameterization
                // trains — see `loss::embedding_ridge` for the reduction, and
                // `JointEmbedModel::feature_ridge` for why the model picks the table:
                // ridging `e_feat` directly is silently inert on a model where that
                // field is a detached snapshot.
                if let Some(l2) = ctx.axes[0].model.feature_ridge(f64::from(ridge_lambda))? {
                    loss = (loss + l2)?;
                }
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
        if let Some(st) = &mut module_state {
            // The membership moves only once released; the pools and the host copy
            // stay valid until then.
            if !st.modules.is_frozen() {
                st.refresh(ctx)?;
            }
            st.log_diagnostics(epoch, params.epochs)?;
        }
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
    module_state: Option<&ModuleState>,
) -> anyhow::Result<Option<Tensor>> {
    // The module layer's per-step tensors, once for every axis: the live
    // membership (detached while frozen), and — when dropout is on — one keep mask
    // and its renormalized membership, shared so every axis pools under the same
    // hidden genes this step.
    let per_step: Option<(Tensor, Tensor)> = match module_state {
        Some(st) => {
            let pi = st.modules.membership()?;
            let pi_masked = if st.params.gene_dropout > 0.0 {
                let n_features = pi.dim(0)?;
                let keep = draw_gene_keep_mask(n_features, st.params.gene_dropout, rng, ctx.dev)?;
                masked_membership(&pi, &keep)?
            } else {
                pi.detach()
            };
            Some((pi, pi_masked))
        }
        None => None,
    };
    let mut total_loss: Option<Tensor> = None;
    for (axis_idx, axis) in ctx.axes.iter().enumerate() {
        let step_modules = match (module_state, &per_step) {
            (Some(st), Some((_, pi_masked))) => Some(AxisModuleStep {
                modules: st.modules,
                params: st.params,
                pools: &st.pools[axis_idx],
                pi_masked,
            }),
            _ => None,
        };
        let Some(loss) = single_axis_step(axis, rng, params, ctx.dev, step_modules.as_ref())?
        else {
            continue;
        };
        let scaled = (loss * f64::from(axis.lambda))?;
        total_loss = Some(match total_loss {
            Some(prev) => (prev + scaled)?,
            None => scaled,
        });
    }
    // Membership priors, once per step (not per axis); `None` while frozen.
    if let (Some(st), Some((pi, _)), Some(loss)) = (module_state, &per_step, total_loss.as_mut()) {
        if let Some(prior) = module_priors(st.modules, pi, st.params.lambda_balance)? {
            *loss = (&*loss + prior)?;
        }
    }
    Ok(total_loss)
}

/// What one axis step needs from the module layer.
struct AxisModuleStep<'a> {
    modules: &'a FeatModules,
    params: &'a ModuleTrainParams,
    /// This axis's pools, one per sampler.
    pools: &'a [ModulePools],
    /// This step's (dropout-masked, detached) membership `[D, M]`.
    pi_masked: &'a Tensor,
}

/// The exact module term for one axis step: draw `units_per_step` units from the
/// axis's own picker (the same `degree^α` weighting the positives use, so the two
/// levels weight units alike), pool their count rows through the masked
/// membership, and score every module with a full softmax. The pooled target is
/// detached inside the loss, so this term trains `μ`, the module bias and the
/// cell side; the membership trains through the within-module NCE and the priors.
fn module_term(
    axis: &CompositeAxis,
    cc: &data_beans_alg::feature_coarsening::FeatureCoarsening,
    rng: &mut StdRng,
    step: &AxisModuleStep,
    dev: &Device,
) -> anyhow::Result<Option<Tensor>> {
    let u = step.params.units_per_step;
    if u == 0 || step.params.lambda_module <= 0.0 {
        return Ok(None);
    }
    let n_features = axis.unified.n_features();
    let mut rows: Vec<(&[u32], &[f32])> = Vec::with_capacity(u);
    let mut coarse: Vec<u32> = Vec::with_capacity(u);
    match axis.sampler {
        AxisSampler::PerBatchStratified(samplers) => {
            let s = &samplers[rng.random_range(0..samplers.len())];
            for _ in 0..u {
                let lc = s.cell_picker.sample(rng);
                let pf = &s.per_cell[lc];
                rows.push((&pf.features, &pf.counts));
                coarse.push(cc.fine_to_coarse[s.active_cells[lc] as usize] as u32);
            }
        }
        AxisSampler::Stratified(s) => {
            for _ in 0..u {
                let lp = s.pb_picker.sample(rng);
                let pf = &s.per_pb[lp];
                rows.push((&pf.features, &pf.counts));
                coarse.push(s.active_pbs[lp]);
            }
        }
    }
    let x = dense_count_block(&rows, n_features, dev)?;
    let e_units = if axis.cell_axis.is_identity {
        let idx = Tensor::from_vec(coarse, u, dev)?;
        axis.model.e_cell.index_select(&idx, 0)?
    } else {
        axis.model.pool_cells(&coarse, &cc.coarse_to_fine, dev)?.0
    };
    Ok(Some(module_step_loss(
        step.modules,
        step.pi_masked,
        &x,
        &e_units,
        step.params.lambda_module,
    )?))
}

/// Sample a minibatch from a single axis and compute its bipartite NCE
/// loss (taking the identity fast path when the axis has identity
/// coarsening), plus the exact module term when the model has modules.
/// Returns `None` when the axis has no positives to sample.
fn single_axis_step(
    axis: &CompositeAxis,
    rng: &mut StdRng,
    params: &TrainingParams,
    dev: &Device,
    modules: Option<&AxisModuleStep>,
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
                    module_pools: modules.map(|m| &m.pools[id]),
                },
                rng,
            )
        }
        AxisSampler::Stratified(s) => sample_stratified_edge_batch(
            StratifiedEdgeBatchArgs {
                sampler: s,
                batch_size: params.batch_size,
                n_negatives: params.num_negatives,
                module_pools: modules.map(|m| &m.pools[0]),
            },
            rng,
        ),
    };

    let mut loss = if axis.cell_axis.is_identity {
        nce_loss_identity(axis.model, batch, params.objective, dev)?
    } else {
        nce_loss(axis.model, batch, &cc.coarse_to_fine, params.objective, dev)?
    };
    if let Some(step) = modules {
        if let Some(term) = module_term(axis, cc, rng, step, dev)? {
            loss = (loss + term)?;
        }
    }
    Ok(Some(loss))
}

#[cfg(test)]
mod tests;
