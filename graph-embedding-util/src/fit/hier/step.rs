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
//! base tables take the sum over tracks and each offset table its own track's
//! term — autograd does that split by linearity.
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
//! `λ_step Σ_{t≥1} [ (1/M) Σ_m ‖Δ^t_m‖² + (1/G) Σ_g ‖δ^t_g‖² ]` (biases free),
//! exact on the FULL tables every step. `λ_step` is the PER-STEP weight the
//! caller passes (`HierConfig::offset_l2 / steps_per_epoch`, see
//! [`super::train::per_step_offset_l2`]).

use super::params::{HierParams, ADAGRAD_EPS};
use super::partition::{Partition, TrackSupport, UnitModules};
use super::units::UnitTable;
use candle_util::candle_core::backprop::GradStore;
use candle_util::candle_core::{DType, Device, Result as CResult, Tensor, Var, D};
use candle_util::candle_nn::ops::log_softmax;
use candle_util::fast_index::gather_rows;

/// PBG's mask on a column that must not compete: `exp(−1e9)` is exactly zero.
const MASK_NEG: f64 = -1e9;
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
    /// The offset ridge AT THIS STEP's weight; exactly `0` on a one-track axis.
    pub loss_ridge: f64,
}

fn ids(v: &[u32], dev: &Device) -> CResult<Tensor> {
    Tensor::from_slice(v, v.len(), dev)
}

fn floats(v: &[f32], dev: &Device) -> CResult<Tensor> {
    Tensor::from_slice(v, v.len(), dev)
}

/// `e_b · μ_effᵀ + b`, softmaxed over the track's scored modules, weighted by
/// the units' track weights and their module shares.
fn module_level(
    params: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    sup: &TrackSupport,
    plan: &StepPlan,
    e_b: &Tensor,
    t: usize,
) -> CResult<Tensor> {
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
    let (mut mu_eff, b_eff) = match params.offset(t) {
        None => (
            params.mu.as_tensor().clone(),
            params.b_m.as_tensor().clone(),
        ),
        Some(o) => (
            (params.mu.as_tensor() + o.d_mu.as_tensor())?,
            (params.b_m.as_tensor() + o.d_b_m.as_tensor())?,
        ),
    };
    if let Some(l) = params.lora.as_ref() {
        // The module residual: `[M, H]`, the size of the dictionary itself.
        mu_eff = (mu_eff + l.module_factors().residual()?)?;
    }
    let (mu_eff, b_eff) = if sup.is_full(t) {
        (mu_eff, b_eff)
    } else {
        let m_ids = ids(mods, dev)?;
        (gather_rows(&mu_eff, &m_ids)?, gather_rows(&b_eff, &m_ids)?)
    };
    let s = e_b
        .matmul(&mu_eff.t()?)?
        .broadcast_add(&b_eff.unsqueeze(0)?)?;
    let logp = log_softmax(&s, D::Minus1)?;
    (wq * logp)?.sum_all()?.neg()
}

/// One `(track, module)` group to score: its member genes on the track, the
/// module id, and the `(unit, weight)` pairs drawn into it.
type Group<'a> = (Vec<u32>, usize, &'a [(u32, f32)]);

/// One padded batch of gene-level groups on one track.
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

/// The gene-level loss of every group on track `t`, batched.
fn gene_level(
    params: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    sup: &TrackSupport,
    plan: &StepPlan,
    t: usize,
) -> CResult<Option<Tensor>> {
    let dev = &params.dev;
    // Each group's member genes (restricted to the track's support) and its
    // `(unit, weight)` pairs; sorted by member count for bucketing.
    let mut groups: Vec<Group<'_>> = plan
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
    if groups.is_empty() {
        return Ok(None);
    }
    groups.sort_by_key(|(genes, _, _)| genes.len());
    let mut batches: Vec<GeneBatch> = Vec::new();
    let mut start = 0;
    while start < groups.len() {
        let d_min = groups[start].0.len().max(1);
        let mut end = start;
        while end < groups.len() && groups[end].0.len() <= d_min * BUCKET_RATIO {
            end += 1;
        }
        let chunk = &groups[start..end];
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
        let mut unit_ids = vec![0u32; p_n * n_max];
        let mut gene_ids = vec![0u32; p_n * d_max];
        let mut col_valid = vec![0f32; p_n * d_max];
        let mut target_pos: Vec<u32> = Vec::new();
        let mut target_val: Vec<f32> = Vec::new();
        for (p, (genes, m, pairs)) in chunk.iter().enumerate() {
            for (j, &g) in genes.iter().enumerate() {
                gene_ids[p * d_max + j] = g;
                col_valid[p * d_max + j] = 1.0;
            }
            let restricted = (!sup.is_full(t)).then(|| sup.local_of(t, *m));
            for (i, &(u, wt)) in pairs.iter().enumerate() {
                unit_ids[p * n_max + i] = u;
                let scale = units.weight_of(u as usize, t) * wt;
                let n_um = um.n_um[um.idx(u as usize, t, *m)];
                let counts = um.by_module[u as usize]
                    .iter()
                    .find(|((tr, k), _)| *tr as usize == t && *k as usize == *m)
                    .map(|(_, v)| v.as_slice())
                    .unwrap_or(&[]);
                for &(slot, c) in counts {
                    let col = match restricted {
                        None => slot as usize,
                        Some(local) => local[slot as usize] as usize,
                    };
                    target_pos.push(((p * n_max + i) * d_max + col) as u32);
                    target_val.push(scale * c / n_um);
                }
            }
        }
        batches.push(GeneBatch {
            n_max,
            d_max,
            unit_ids,
            gene_ids,
            col_valid,
            target_pos,
            target_val,
        });
        start = end;
    }

    let h = params.h;
    let lora = params.lora.as_ref().map(|l| l.gene_factors());
    let mut total: Option<Tensor> = None;
    for b in batches {
        let p_n = b.unit_ids.len() / b.n_max;
        let u_ids = ids(&b.unit_ids, dev)?;
        let g_ids = ids(&b.gene_ids, dev)?;
        let e = gather_rows(params.e_u.as_tensor(), &u_ids)?.reshape((p_n, b.n_max, h))?;
        let mut r = gather_rows(params.r.as_tensor(), &g_ids)?;
        let mut bias = gather_rows(params.b_g.as_tensor(), &g_ids)?;
        if let Some(o) = params.offset(t) {
            r = (r + gather_rows(o.d_r.as_tensor(), &g_ids)?)?;
            bias = (bias + gather_rows(o.d_b_g.as_tensor(), &g_ids)?)?;
        }
        if let Some(l) = lora.as_ref() {
            r = (r + l.residual_rows(&g_ids)?)?;
        }
        let r = r.reshape((p_n, b.d_max, h))?;
        // Pad columns take `−1e9`: exactly zero probability, no gradient.
        let pad = floats(&b.col_valid, dev)?
            .reshape((p_n, 1, b.d_max))?
            .affine(-MASK_NEG, MASK_NEG)?;
        let s = e
            .matmul(&r.transpose(1, 2)?)?
            .broadcast_add(&bias.reshape((p_n, 1, b.d_max))?)?
            .broadcast_add(&pad)?;
        let logp = log_softmax(&s, D::Minus1)?.flatten_all()?;
        if b.target_pos.is_empty() {
            continue;
        }
        let picked = gather_rows(&logp, &ids(&b.target_pos, dev)?)?;
        let loss = (picked * floats(&b.target_val, dev)?)?.sum_all()?.neg()?;
        total = Some(match total {
            None => loss,
            Some(acc) => (acc + loss)?,
        });
    }
    Ok(total)
}

/// The step's loss as one tensor to differentiate, plus its parts as numbers
/// for the epoch log. Pure in the parameters: nothing is updated here.
pub fn step_loss(
    params: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    sup: &TrackSupport,
    plan: &StepPlan,
    offset_l2_step: f32,
) -> anyhow::Result<(StepStats, Tensor)> {
    let n_t = units.n_tracks();
    anyhow::ensure!(
        params.offsets.len() == n_t.saturating_sub(1),
        "one offset table per non-base track"
    );
    let dev = &params.dev;
    let e_b = gather_rows(params.e_u.as_tensor(), &ids(&plan.units, dev)?)?;
    let mut loss_module: Option<Tensor> = None;
    let mut loss_gene: Option<Tensor> = None;
    let add = |acc: &mut Option<Tensor>, x: Tensor| -> CResult<()> {
        *acc = Some(match acc.take() {
            None => x,
            Some(a) => (a + x)?,
        });
        Ok(())
    };
    for t in 0..n_t {
        add(
            &mut loss_module,
            module_level(params, units, um, sup, plan, &e_b, t)?,
        )?;
        if let Some(l) = gene_level(params, units, um, part, sup, plan, t)? {
            add(&mut loss_gene, l)?;
        }
    }
    let mut loss_ridge: Option<Tensor> = None;
    if offset_l2_step > 0.0 {
        let n_m = params.b_m.dims()[0] as f64;
        let n_g = params.b_g.dims()[0] as f64;
        for o in &params.offsets {
            let mu2 = o
                .d_mu
                .as_tensor()
                .sqr()?
                .sum_all()?
                .affine(1.0 / n_m, 0.0)?;
            let r2 = o.d_r.as_tensor().sqr()?.sum_all()?.affine(1.0 / n_g, 0.0)?;
            add(
                &mut loss_ridge,
                (mu2 + r2)?.affine(f64::from(offset_l2_step), 0.0)?,
            )?;
        }
    }
    let scalar = |t: &Option<Tensor>| -> CResult<f64> {
        Ok(match t {
            None => 0.0,
            Some(x) => f64::from(x.to_scalar::<f32>()?),
        })
    };
    let stats = StepStats {
        loss_module: scalar(&loss_module)?,
        loss_gene: scalar(&loss_gene)?,
        loss_ridge: scalar(&loss_ridge)?,
    };
    let mut total = loss_module.unwrap_or(Tensor::zeros((), DType::F32, dev)?);
    if let Some(g) = loss_gene {
        total = (total + g)?;
    }
    if let Some(r) = loss_ridge {
        total = (total + r)?;
    }
    Ok((stats, total))
}

/// Adagrad with one accumulator per row, where a row carries a scalar bias
/// alongside it: the accumulator sees the mean of `grad²` over the row AND
/// the bias, and both move by the same row step. A row whose gradient is
/// masked to zero still moves its bias, with the accumulator seeing the bias
/// alone — a pinned row's bias trains as if it had no row.
pub struct RowAdagradBias {
    acc: Tensor,
    lr: f64,
}

impl RowAdagradBias {
    pub fn new(n_rows: usize, lr: f32, dev: &Device) -> CResult<Self> {
        Ok(Self {
            acc: Tensor::zeros(n_rows, DType::F32, dev)?,
            lr: f64::from(lr),
        })
    }

    /// One step of `(row, bias)` from their gradients. `row_mask` is `[n, 1]`
    /// with `0` on pinned rows; `decay` is the per-step weight-decay factor
    /// applied to every row whose gradient is nonzero (`1.0` for none).
    pub fn step(
        &mut self,
        row: &Var,
        bias: &Var,
        g_row: &Tensor,
        g_bias: &Tensor,
        row_mask: Option<&Tensor>,
        decay: f64,
    ) -> CResult<()> {
        let h = row.dims()[1] as f64;
        let g_row = match row_mask {
            None => g_row.detach(),
            Some(m) => g_row.detach().broadcast_mul(m)?,
        };
        let g_bias = g_bias.detach();
        let row_sq = g_row.sqr()?.sum(1)?; // [n]
                                           // Over the row and the bias where the row trains, the bias alone where
                                           // it is pinned.
        let n_eff = match row_mask {
            None => Tensor::full((h + 1.0) as f32, row_sq.dims()[0], row_sq.device())?,
            Some(m) => m.squeeze(1)?.affine(h, 1.0)?,
        };
        let g2 = (&row_sq + g_bias.sqr()?)?.div(&n_eff)?;
        self.acc = (&self.acc + g2)?.detach();
        let step = self
            .acc
            .sqrt()?
            .affine(1.0 / self.lr, f64::from(ADAGRAD_EPS) / self.lr)?
            .recip()?; // lr / (sqrt(acc) + eps)   [n]
        let mut new_row = row.as_tensor().clone();
        if decay != 1.0 {
            // `row *= decay` on touched rows only: a row the step never scored
            // keeps its value, as it always has.
            let touched = row_sq.gt(0f32)?.to_dtype(DType::F32)?;
            let factor = touched.affine(decay - 1.0, 1.0)?; // 1 on untouched, decay on touched
            new_row = new_row.broadcast_mul(&factor.unsqueeze(1)?)?;
        }
        let new_row = new_row.sub(&g_row.broadcast_mul(&step.unsqueeze(1)?)?)?;
        row.set(&new_row)?;
        bias.set(&bias.as_tensor().sub(&(g_bias * step)?)?)
    }
}

pub struct Optimizers {
    pub e_u: crate::fne::RowAdagrad,
    pub mu: RowAdagradBias,
    pub r: RowAdagradBias,
    /// Tracks `1..T`, in order: `(M rows for Δ/β, G rows for δ/γ)`.
    pub offsets: Vec<(RowAdagradBias, RowAdagradBias)>,
    /// Under LoRA, in order: `a` (M rows), `V_M`, `u` (G rows), `V_G`; the
    /// `V`s' learning rate carries the LoRA+ ratio.
    pub lora: Option<[crate::fne::RowAdagrad; 4]>,
}

impl Optimizers {
    pub fn new(params: &HierParams, lr: f32) -> CResult<Self> {
        let dev = &params.dev;
        let (n_u, n_m, n_g) = (
            params.e_u.dims()[0],
            params.mu.dims()[0],
            params.r.dims()[0],
        );
        Ok(Self {
            e_u: crate::fne::RowAdagrad::new(n_u, f64::from(lr), dev)?,
            mu: RowAdagradBias::new(n_m, lr, dev)?,
            r: RowAdagradBias::new(n_g, lr, dev)?,
            offsets: params
                .offsets
                .iter()
                .map(|_| {
                    Ok((
                        RowAdagradBias::new(n_m, lr, dev)?,
                        RowAdagradBias::new(n_g, lr, dev)?,
                    ))
                })
                .collect::<CResult<_>>()?,
            lora: match params.lora.as_ref() {
                Some(l) => {
                    let fast = f64::from(lr * l.lr_ratio);
                    Some([
                        crate::fne::RowAdagrad::new(n_m, f64::from(lr), dev)?,
                        crate::fne::RowAdagrad::new(l.v_m.dims()[0], fast, dev)?,
                        crate::fne::RowAdagrad::new(n_g, f64::from(lr), dev)?,
                        crate::fne::RowAdagrad::new(l.v_g.dims()[0], fast, dev)?,
                    ])
                }
                None => None,
            },
        })
    }
}

/// Apply the gradients of one step with the row optimizers. Weight decay `wd`
/// multiplies a touched row by `1 − lr·wd` before its Adagrad step; biases
/// never decay, pinned rows never move, and the offset tables never decay at
/// all — their shrinkage is the exact ridge already in the gradient.
pub fn apply(
    params: &mut HierParams,
    opt: &mut Optimizers,
    grads: &GradStore,
    lr: f32,
    wd: f32,
) -> CResult<()> {
    let dev = params.dev.clone();
    let decay = if wd > 0.0 {
        1.0 - f64::from(lr) * f64::from(wd)
    } else {
        1.0
    };
    let zeros_like = |v: &Var| Tensor::zeros(v.dims(), DType::F32, &dev);
    let grad = |v: &Var| -> CResult<Tensor> {
        match grads.get(v) {
            Some(g) => Ok(g.clone()),
            None => zeros_like(v),
        }
    };
    // Units: no bias, decay on touched rows.
    let g_e = grad(&params.e_u)?;
    if decay != 1.0 {
        let touched = g_e.sqr()?.sum(1)?.gt(0f32)?.to_dtype(DType::F32)?;
        let factor = touched.affine(decay - 1.0, 1.0)?.unsqueeze(1)?;
        params
            .e_u
            .set(&params.e_u.as_tensor().broadcast_mul(&factor)?)?;
    }
    opt.e_u.step(&params.e_u, &g_e)?;
    // Modules: every row is offered; the optimizer skips a zero gradient.
    let mu_mask = if params.mu_frozen {
        Some(Tensor::zeros((params.mu.dims()[0], 1), DType::F32, &dev)?)
    } else {
        None
    };
    opt.mu.step(
        &params.mu,
        &params.b_m,
        &grad(&params.mu)?,
        &grad(&params.b_m)?,
        mu_mask.as_ref(),
        decay,
    )?;
    opt.r.step(
        &params.r,
        &params.b_g,
        &grad(&params.r)?,
        &grad(&params.b_g)?,
        params.r_mask.as_ref(),
        decay,
    )?;
    for (o, (opt_mu, opt_r)) in params.offsets.iter().zip(&mut opt.offsets) {
        opt_mu.step(
            &o.d_mu,
            &o.d_b_m,
            &grad(&o.d_mu)?,
            &grad(&o.d_b_m)?,
            None,
            1.0,
        )?;
        opt_r.step(
            &o.d_r,
            &o.d_b_g,
            &grad(&o.d_r)?,
            &grad(&o.d_b_g)?,
            None,
            1.0,
        )?;
    }
    if let (Some(l), Some([opt_a, opt_vm, opt_u, opt_vg])) =
        (params.lora.as_ref(), opt.lora.as_mut())
    {
        if let Some(g) = grads.get(&l.a) {
            opt_a.step(&l.a, g)?;
        }
        if let Some(g) = grads.get(&l.v_m) {
            opt_vm.step(&l.v_m, g)?;
        }
        if let Some(g) = grads.get(&l.u) {
            opt_u.step(&l.u, &g.broadcast_mul(&l.u_mask)?)?;
        }
        if let Some(g) = grads.get(&l.v_g) {
            opt_vg.step(&l.v_g, g)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "step_tests.rs"]
mod step_tests;
