//! One optimizer step of the exact two-level softmax, over one or more TRACKS
//! of the same genes, as a candle forward pass and one `backward()`.
//!
//! ```text
//! S_t = { g : gene g has a row on track t }      M_t = { m : m ∩ S_t ≠ ∅ }
//! p^t_um   = softmax_{m ∈ M_t}     ( ⟨e_u, μ^t_m⟩ + b^t_m )
//! p^t_ug|m = softmax_{g ∈ m ∩ S_t} ( ⟨e_u, r^t_g⟩ + b^t_g )
//! L₁(u,t)   = − Σ_{m ∈ M_t}     q^t_um   · log p^t_um    q^t_um   = n^t_um / N^t_u
//! L₂(u,t,m) = − Σ_{g ∈ m ∩ S_t} q^t_ug|m · log p^t_ug|m  q^t_ug|m = n^t_ug / n^t_um
//! L = Σ_u Σ_t w^t_u [ L₁(u,t) + Σ_k (c^t_k/K)·L₂(u, t, m_k) ] + ridge
//! ```
//!
//! # The SUPPORT rule
//!
//! Every softmax runs over the track's own support `S_t`, never over the whole
//! gene axis. A track's rows ARE its feature axis — the producer emits a row
//! only where that channel was observed — so a gene with no row on track `t` is
//! outside the track's axis, not a gene with probability zero there. A gene
//! outside `S_t` takes NO gradient from track `t`, base or offset, and a module
//! outside `M_t` takes none either. A track that has a row for every gene is
//! *full* and carries no restriction; a one-track axis is exactly that case.
//! [`TrackSupport`] precomputes `S_t` / `M_t` once per fit.
//!
//! `m_k ~ q^t_u·` with replacement (K draws); `c^t_k` is module `m_k`'s draw
//! multiplicity, so `Σ_k (c_k/K)·L₂` is an unbiased estimator of
//! `Σ_m q_um·L₂`. [`StepPlan`] carries each pair's already-computed weight
//! `c_k/K`; the step never re-derives it from `q_um`.
//!
//! The base track (`t == 0`) IS the model; every other track is an additive
//! OFFSET from it: `μ^t = μ + Δ^t`, `b^t_m = b_m + β^t`, `r^t = r + δ^t`,
//! `b^t_g = b_g + γ^t`, with the base tables used directly at `t == 0`. The
//! gene offset is low-rank, `δ^t_g = δ₀_g + u^t_g · V^t` (see
//! [`super::params::TrackOffset`]), so a track moves its genes inside one
//! shared subspace. The base tables take the sum over tracks and each offset
//! its own track's term — autograd does that split by linearity.
//!
//! # How a step is laid out
//!
//! The host builds ids only: the plan's units, each track's scored modules,
//! and per `(track, module)` group the member genes the track has rows for,
//! the units drawn into it with their weights, and the `(unit, gene)` targets.
//! The module level is one `[B, |M_t|]` product per track. The gene level is
//! batched: a track's groups are padded to a common unit count and gene count
//! and scored by one batched matmul, one masked row log-softmax, and one
//! gather of the target positions — so a step is a few dozen kernels whatever
//! the number of modules, and no `[G, ·]` table is formed. Groups are bucketed
//! by member count first, so padding never exceeds a factor of two.
//!
//! # Ridge on the offsets
//!
//! `λ_step Σ_{t≥1} [ (1/M) Σ_m ‖Δ^t_m‖² + (1/G) Σ_g ‖u^t_g · V^t‖² ]` (biases
//! and a given `δ₀` free), exact on the FULL tables every step — the gene
//! term in Gram form ([`legume_numeric::candle::lora::PinnedLora::ridge`]), so no
//! `[G, H]` residual is formed. `λ_step` is the PER-STEP weight the caller
//! passes (`HierConfig::offset_l2 / steps_per_epoch`, see
//! [`super::train::per_step_offset_l2`]).

use super::params::HierParams;
use super::partition::{Partition, TrackSupport, UnitModules};
use super::units::UnitTable;
use legume_numeric::candle::candle_core::backprop::GradStore;
use legume_numeric::candle::candle_core::{DType, Result as CResult, Tensor, Var, D};
use legume_numeric::candle::candle_nn::ops::log_softmax;
use legume_numeric::candle::convert::{add_into, to_1d};
use legume_numeric::candle::fast_index::gather_rows;
use legume_numeric::candle::lora::PinnedLoraOpt;
use legume_numeric::candle::masking::additive_pad_mask;
use legume_numeric::candle::optim::RowAdagrad;

/// Groups are batched together while the largest member count is at most this
/// multiple of the smallest, so padding stays bounded.
const BUCKET_RATIO: usize = 2;

/// The units of one step and, per `(track, module)`, the `(unit, weight)`
/// pairs drawn into that module on that track — `weight = c_k/K`, the module's
/// draw multiplicity over the draws.
///
/// Invariants: `units` has no duplicates; each module appears at most once
/// PER TRACK in `pairs_by_module`; every unit named in a `pairs_by_module`
/// entry is also present in `units`.
pub struct StepPlan {
    pub units: Vec<u32>,
    /// Grouped by `(track, module)`.
    pub pairs_by_module: TrackModulePairs,
}

/// One `(unit, draw weight)` list per `(track, module)` key.
pub type TrackModulePairs = Vec<((u32, u32), Vec<(u32, f32)>)>;

#[derive(Default, Debug, Clone)]
pub struct StepStats {
    pub loss_module: f64,
    pub loss_gene: f64,
    /// The offset ridge and the LoRA ridge AT THIS STEP's weight; exactly `0`
    /// on a one-track axis without a residual.
    pub loss_ridge: f64,
}

/// What every step reads and never writes: the unit table, the partition and
/// the per-fit views built from them. Built once per fit; the plan is per step.
pub struct StepCtx<'a> {
    pub units: &'a UnitTable,
    pub um: &'a UnitModules,
    pub part: &'a Partition,
    pub sup: &'a TrackSupport,
}

/// `‖t‖²_F / n`: a table's mean row norm² over `n` rows.
fn mean_row_sq(t: &Tensor, n: usize) -> CResult<Tensor> {
    t.sqr()?.sum_all()?.affine(1.0 / n.max(1) as f64, 0.0)
}

/// `e_b · μ_effᵀ + b`, softmaxed over the track's scored modules, weighted by
/// the units' track weights and their module shares. `mu_lora` is the module
/// residual, already formed once for the step.
fn module_level(
    params: &HierParams,
    ctx: &StepCtx<'_>,
    plan: &StepPlan,
    e_b: &Tensor,
    mu_lora: Option<&Tensor>,
    t: usize,
) -> CResult<Tensor> {
    let (units, um, sup) = (ctx.units, ctx.um, ctx.sup);
    let dev = &params.dev;
    let n_m = um.n_modules;
    let all: Vec<u32>;
    let mods: &[u32] = if sup.is_full(t) {
        all = (0..n_m as u32).collect();
        &all
    } else {
        sup.modules_of(t)
    };
    let b = plan.units.len();
    // w_u · q_um on the scored modules, host-built: [B, |M_t|].
    let mut wq = vec![0f32; b * mods.len()];
    for (i, &u) in plan.units.iter().enumerate() {
        let w = units.weight_of(u as usize, t);
        let base = um.idx(u as usize, t, 0);
        for (j, &m) in mods.iter().enumerate() {
            wq[i * mods.len() + j] = w * um.q[base + m as usize];
        }
    }
    let wq = Tensor::from_vec(wq, (b, mods.len()), dev)?;
    let (mut mu_eff, mut b_eff) = (
        params.mu.as_tensor().clone(),
        params.b_m.as_tensor().clone(),
    );
    if let Some(o) = params.offset(t) {
        mu_eff = (mu_eff + o.d_mu.as_tensor())?;
        b_eff = (b_eff + o.d_b_m.as_tensor())?;
    }
    if let Some(l) = mu_lora {
        mu_eff = (mu_eff + l)?;
    }
    if !sup.is_full(t) {
        let m_ids = to_1d(mods, dev)?;
        mu_eff = gather_rows(&mu_eff, &m_ids)?;
        b_eff = gather_rows(&b_eff, &m_ids)?;
    }
    let s = e_b
        .matmul(&mu_eff.t()?)?
        .broadcast_add(&b_eff.unsqueeze(0)?)?;
    let logp = log_softmax(&s, D::Minus1)?;
    (wq * logp)?.sum_all()?.neg()
}

/// One `(track, module)` group to score: its member genes on the track, the
/// module id, and the `(unit, weight)` pairs drawn into it.
type Group<'a> = (Vec<u32>, usize, &'a [(u32, f32)]);

/// One padded batch of gene-level groups on one track: `P` groups, each with
/// up to `n_max` units and `d_max` genes.
struct GeneBatch {
    n_max: usize,
    d_max: usize,
    unit_ids: Vec<u32>,
    gene_ids: Vec<u32>,
    col_valid: Vec<f32>,
    /// Flat positions into the `[P, n_max, d_max]` log-probabilities, and the
    /// weighted target share at each: `w_u · (c_k/K) · q_ug|m`.
    target_pos: Vec<u32>,
    target_val: Vec<f32>,
}

/// Track `t`'s groups: member genes on the track's support, module id, drawn
/// units; sorted by member count so bucketing is a linear walk.
fn track_groups<'a>(ctx: &'a StepCtx<'a>, plan: &'a StepPlan, t: usize) -> Vec<Group<'a>> {
    let (part, sup) = (ctx.part, ctx.sup);
    let mut groups: Vec<Group<'a>> = plan
        .pairs_by_module
        .iter()
        .filter(|((tt, _), _)| *tt as usize == t)
        .map(|((_, m), pairs)| {
            let members = &part.members[*m as usize];
            let genes: Vec<u32> = if sup.is_full(t) {
                members.clone()
            } else {
                sup.slots_of(t, *m as usize)
                    .iter()
                    .map(|&j| members[j as usize])
                    .collect()
            };
            (genes, *m as usize, pairs.as_slice())
        })
        .collect();
    groups.sort_by_key(|(genes, _, _)| genes.len());
    groups
}

/// Consecutive runs of sorted groups whose member counts stay within
/// [`BUCKET_RATIO`] of the run's smallest.
fn bucket<'a, 'g>(groups: &'g [Group<'a>]) -> Vec<&'g [Group<'a>]> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < groups.len() {
        let d_min = groups[start].0.len().max(1);
        let len = groups[start..]
            .iter()
            .take_while(|(g, _, _)| g.len() <= d_min * BUCKET_RATIO)
            .count();
        chunks.push(&groups[start..start + len]);
        start += len;
    }
    chunks
}

/// One padded batch from a bucket of groups on track `t`.
fn fill_batch(ctx: &StepCtx<'_>, chunk: &[Group<'_>], t: usize) -> GeneBatch {
    let (units, um, sup) = (ctx.units, ctx.um, ctx.sup);
    let d_max = chunk
        .iter()
        .map(|(g, _, _)| g.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let n_max = chunk
        .iter()
        .map(|(_, _, p)| p.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let p_n = chunk.len();
    let n_pairs: usize = chunk.iter().map(|(_, _, p)| p.len()).sum();
    let mut b = GeneBatch {
        n_max,
        d_max,
        unit_ids: vec![0; p_n * n_max],
        gene_ids: vec![0; p_n * d_max],
        col_valid: vec![0.0; p_n * d_max],
        target_pos: Vec::with_capacity(n_pairs * 4),
        target_val: Vec::with_capacity(n_pairs * 4),
    };
    for (p, (genes, m, pairs)) in chunk.iter().enumerate() {
        for (j, &g) in genes.iter().enumerate() {
            b.gene_ids[p * d_max + j] = g;
            b.col_valid[p * d_max + j] = 1.0;
        }
        let local = (!sup.is_full(t)).then(|| sup.local_of(t, *m));
        for (i, &(u, wt)) in pairs.iter().enumerate() {
            b.unit_ids[p * n_max + i] = u;
            let scale = units.weight_of(u as usize, t) * wt;
            let n_um = um.n_um[um.idx(u as usize, t, *m)];
            for &(slot, c) in um.counts_of(u as usize, t, *m) {
                let col = local.map_or(slot as usize, |l| l[slot as usize] as usize);
                b.target_pos.push(((p * n_max + i) * d_max + col) as u32);
                b.target_val.push(scale * c / n_um);
            }
        }
    }
    b
}

/// Track `t`'s groups, bucketed by member count and padded.
fn build_gene_batches(ctx: &StepCtx<'_>, plan: &StepPlan, t: usize) -> Vec<GeneBatch> {
    let groups = track_groups(ctx, plan, t);
    bucket(&groups)
        .into_iter()
        .map(|chunk| fill_batch(ctx, chunk, t))
        .collect()
}

/// The gene-level loss of track `t`'s batches: one batched matmul, one masked
/// row log-softmax and one gather of the targets per batch.
fn score_gene_batches(
    params: &HierParams,
    batches: &[GeneBatch],
    t: usize,
) -> CResult<Option<Tensor>> {
    let dev = &params.dev;
    let h = params.h;
    let mut total: Option<Tensor> = None;
    for b in batches {
        if b.target_pos.is_empty() {
            continue;
        }
        let p_n = b.unit_ids.len() / b.n_max;
        let u_ids = to_1d(&b.unit_ids, dev)?;
        let g_ids = to_1d(&b.gene_ids, dev)?;
        let e = gather_rows(params.e_u.as_tensor(), &u_ids)?.reshape((p_n, b.n_max, h))?;
        let mut r = gather_rows(params.r.as_tensor(), &g_ids)?;
        let mut bias = gather_rows(params.b_g.as_tensor(), &g_ids)?;
        if let Some(o) = params.offset(t) {
            r = (r + o.residual_rows(&g_ids)?)?;
            bias = (bias + gather_rows(o.d_b_g.as_tensor(), &g_ids)?)?;
        }
        if let Some(l) = params.lora.as_ref() {
            r = (r + l.gene.residual_rows(&g_ids)?)?;
        }
        let r = r.reshape((p_n, b.d_max, h))?;
        let pad = additive_pad_mask(&to_1d(&b.col_valid, dev)?.reshape((p_n, 1, b.d_max))?)?;
        let s = e
            .matmul(&r.transpose(1, 2)?)?
            .broadcast_add(&bias.reshape((p_n, 1, b.d_max))?)?
            .broadcast_add(&pad)?;
        let logp = log_softmax(&s, D::Minus1)?.flatten_all()?;
        let picked = gather_rows(&logp, &to_1d(&b.target_pos, dev)?)?;
        add_into(
            &mut total,
            (picked * to_1d(&b.target_val, dev)?)?.sum_all()?.neg()?,
        )?;
    }
    Ok(total)
}

/// The step's loss as one tensor to differentiate, plus its parts as numbers
/// for the epoch log. Pure in the parameters: nothing is updated here.
pub fn step_loss(
    params: &HierParams,
    ctx: &StepCtx<'_>,
    plan: &StepPlan,
    offset_l2_step: f32,
    lora_ridge_step: f32,
) -> anyhow::Result<(StepStats, Tensor)> {
    let n_t = ctx.units.n_tracks();
    anyhow::ensure!(
        params.offsets.len() == n_t.saturating_sub(1),
        "one offset table per non-base track"
    );
    let dev = &params.dev;
    let e_b = gather_rows(params.e_u.as_tensor(), &to_1d(&plan.units, dev)?)?;
    // The module residual is `[M, H]` and track-free: once per step.
    let mu_lora = match params.lora.as_ref() {
        Some(l) => Some(l.module.residual()?),
        None => None,
    };
    let mut loss_module: Option<Tensor> = None;
    let mut loss_gene: Option<Tensor> = None;
    for t in 0..n_t {
        add_into(
            &mut loss_module,
            module_level(params, ctx, plan, &e_b, mu_lora.as_ref(), t)?,
        )?;
        let batches = build_gene_batches(ctx, plan, t);
        if let Some(l) = score_gene_batches(params, &batches, t)? {
            add_into(&mut loss_gene, l)?;
        }
    }
    let mut loss_ridge: Option<Tensor> = None;
    if offset_l2_step > 0.0 {
        let n_m = params.b_m.dims()[0];
        let n_g = params.b_g.dims()[0];
        for o in &params.offsets {
            let mu2 = mean_row_sq(o.d_mu.as_tensor(), n_m)?;
            let gene2 = o.d_r.ridge()?.affine(1.0 / n_g as f64, 0.0)?;
            add_into(
                &mut loss_ridge,
                (mu2 + gene2)?.affine(f64::from(offset_l2_step), 0.0)?,
            )?;
        }
    }
    // The same per-row shrinkage on the two LoRA residuals at this step's
    // weight — in Gram form, so no residual is formed.
    // Their gradient is how the shared factors are kept from marching off
    // the anchor.
    if let (Some(l), true) = (params.lora.as_ref(), lora_ridge_step > 0.0) {
        add_into(
            &mut loss_ridge,
            (l.gene.ridge()? + l.module.ridge()?)?.affine(f64::from(lora_ridge_step), 0.0)?,
        )?;
    }
    // One host sync for the three numbers.
    let zero = || Tensor::zeros((), DType::F32, dev);
    let parts: Vec<Tensor> = [&loss_module, &loss_gene, &loss_ridge]
        .into_iter()
        .map(|p| match p {
            Some(x) => Ok(x.clone()),
            None => zero(),
        })
        .collect::<CResult<_>>()?;
    let vals = Tensor::stack(&parts, 0)?.to_vec1::<f32>()?;
    let stats = StepStats {
        loss_module: f64::from(vals[0]),
        loss_gene: f64::from(vals[1]),
        loss_ridge: f64::from(vals[2]),
    };
    let total = parts
        .into_iter()
        .reduce(|a, b| (a + b).expect("same shape"))
        .expect("three parts");
    Ok((stats, total))
}

pub struct Optimizers {
    pub e_u: RowAdagrad,
    pub mu: RowAdagrad,
    pub r: RowAdagrad,
    /// Tracks `1..T`, in order: `(M rows for Δ/β, G rows for γ, the pair for
    /// the gene residual's factors)`.
    pub offsets: Vec<(RowAdagrad, RowAdagrad, PinnedLoraOpt)>,
    /// Under LoRA: the module residual's pair, then the gene residual's.
    pub lora: Option<[PinnedLoraOpt; 2]>,
}

impl Optimizers {
    pub fn new(params: &HierParams, lr: f32) -> CResult<Self> {
        let dev = &params.dev;
        let lr = f64::from(lr);
        let (n_u, n_m, n_g) = (
            params.e_u.dims()[0],
            params.mu.dims()[0],
            params.r.dims()[0],
        );
        Ok(Self {
            e_u: RowAdagrad::new(n_u, lr, dev)?,
            mu: RowAdagrad::new(n_m, lr, dev)?,
            r: RowAdagrad::new(n_g, lr, dev)?,
            offsets: params
                .offsets
                .iter()
                .map(|o| {
                    Ok((
                        RowAdagrad::new(n_m, lr, dev)?,
                        RowAdagrad::new(n_g, lr, dev)?,
                        o.d_r.optimizers(lr, params.offset_lr_ratio, dev)?,
                    ))
                })
                .collect::<CResult<_>>()?,
            lora: match params.lora.as_ref() {
                Some(l) => Some([
                    l.module.optimizers(lr, l.lr_ratio, dev)?,
                    l.gene.optimizers(lr, l.lr_ratio, dev)?,
                ]),
                None => None,
            },
        })
    }
}

/// Apply the gradients of one step with the row optimizers. Weight decay `wd`
/// multiplies a touched row by `1 − lr·wd` before its Adagrad step; biases
/// never decay, pinned rows never move, and the offset tables never decay at
/// all — their shrinkage is the exact ridge already in the gradient. A table
/// the loss never reached takes no step.
pub fn apply(
    params: &mut HierParams,
    opt: &mut Optimizers,
    grads: &GradStore,
    lr: f32,
    wd: f32,
) -> CResult<()> {
    let decay = if wd > 0.0 {
        1.0 - f64::from(lr) * f64::from(wd)
    } else {
        1.0
    };
    if let Some(g) = grads.get(&params.e_u) {
        let row_sq = g.sqr()?.sum(1)?;
        if decay != 1.0 {
            let touched = row_sq.gt(0f32)?.to_dtype(DType::F32)?;
            let factor = touched.affine(decay - 1.0, 1.0)?.unsqueeze(1)?;
            params
                .e_u
                .set(&params.e_u.as_tensor().broadcast_mul(&factor)?)?;
        }
        opt.e_u.step_with_row_sq(&params.e_u, g, &row_sq)?;
    }
    let pair = |opt: &mut RowAdagrad,
                row: &Var,
                bias: &Var,
                mask: Option<&Tensor>,
                decay: f64|
     -> CResult<()> {
        if let (Some(g_row), Some(g_bias)) = (grads.get(row), grads.get(bias)) {
            opt.step_with_bias(row, bias, g_row, g_bias, mask, decay)?;
        }
        Ok(())
    };
    // The whole dictionary is pinned or none of it: an all-zero mask.
    let mu_mask = params
        .mu_frozen
        .then(|| Tensor::zeros((params.mu.dims()[0], 1), DType::F32, &params.dev))
        .transpose()?;
    pair(
        &mut opt.mu,
        &params.mu,
        &params.b_m,
        mu_mask.as_ref(),
        decay,
    )?;
    pair(
        &mut opt.r,
        &params.r,
        &params.b_g,
        params.r_mask.as_ref(),
        decay,
    )?;
    for (o, (opt_mu, opt_b, opt_r)) in params.offsets.iter().zip(&mut opt.offsets) {
        pair(opt_mu, &o.d_mu, &o.d_b_m, None, 1.0)?;
        if let Some(g) = grads.get(&o.d_b_g) {
            opt_b.step_bias(&o.d_b_g, g)?;
        }
        o.d_r.step(opt_r, grads)?;
    }
    if let (Some(l), Some([opt_m, opt_g])) = (params.lora.as_ref(), opt.lora.as_mut()) {
        l.module.step(opt_m, grads)?;
        l.gene.step(opt_g, grads)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "step_tests.rs"]
mod step_tests;
