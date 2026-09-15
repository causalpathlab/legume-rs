//! One optimizer step of the exact two-level softmax, over one or more TRACKS
//! of the same genes.
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
//! not "a gene with probability zero on track `t`", it is outside the track's
//! axis. Scoring it there would make a structural zero a permanent negative:
//! with one channel carrying rows for a few hundred of tens of thousands of
//! genes, those negatives would dominate the track's gradient and drive its
//! per-gene biases toward −∞. So a gene outside `S_t` takes NO gradient from
//! track `t`, base or offset, and a module outside `M_t` takes none either.
//! `q^t_um` and `q^t_ug|m` are unchanged in value: counts only exist on the
//! support, so the restricted sums are the same numbers.
//!
//! A track that has a row for EVERY gene is *full* and carries no restriction —
//! it keeps the plain model's columns, empty modules included. A one-track axis
//! is exactly that case, so the one-track path never sees the rule at all.
//! [`TrackSupport`] precomputes `S_t` / `M_t` once per fit.
//!
//! `m_k ~ q^t_u·` with replacement (K draws); `c^t_k` is module `m_k`'s draw
//! multiplicity, so `Σ_k (c_k/K)·L₂` is an unbiased estimator of
//! `Σ_m q_um·L₂`. `StepPlan` carries each pair's already-computed weight
//! `c_k/K` — the step never re-derives it from `q_um`.
//!
//! The base track (`t == 0`) IS the model; every other track is an additive
//! OFFSET from it, so a track's effective tables are
//!
//! ```text
//! μ^t = μ + Δ^t    b^t_m = b_m + β^t    r^t = r + δ^t    b^t_g = b_g + γ^t
//! ```
//!
//! with `Δ⁰ = β⁰ = δ⁰ = γ⁰ ≡ 0` — never materialized: at `t == 0` the base
//! tables are used directly, so a one-track step is the plain model, statement
//! for statement.
//!
//! With `δ¹ = w^t (p − q)` at the module level and `δ²` within a module:
//!
//! ```text
//! ∂L/∂e_u = Σ_t w^t_u [ Σ_m δ¹_um μ^t_m + Σ_k (c_k/K) Σ_{g∈m_k} δ²_ug r^t_g ]
//! ∂L/∂μ_m = Σ_t Σ_u δ¹_um e_u      ∂L/∂Δ^t_m = Σ_u δ¹_um e_u   (that track alone)
//! ∂L/∂r_g = Σ_t Σ_u δ²_ug e_u      ∂L/∂δ^t_g = Σ_u δ²_ug e_u   (that track alone)
//! ```
//!
//! — the BASE tables take the sum over tracks and each offset table takes its
//! own track's term, by linearity of `base + offset`. The biases follow the
//! same split.
//!
//! Positives are the unit's own shares; negatives are everything else in each
//! partition, in proportion to how far the prediction exceeds the share. No
//! negative is ever sampled — only the modules a unit is scored in at the gene
//! level, weighted by its draw multiplicity. The within-module work is grouped
//! by `(track, module)`: one gemm per group over the units drawn into it, so a
//! step touches a gene row at most once PER TRACK — and only on the tracks whose
//! support holds it.
//!
//! # Ridge on the offsets
//!
//! ```text
//! λ_step Σ_{t≥1} [ (1/M) Σ_m ‖Δ^t_m‖² + (1/G) Σ_g ‖δ^t_g‖² ]    (biases free)
//! ```
//!
//! exact on the FULL tables every step, not only on the rows the plan touched,
//! so its gradient reaches every row: `2 λ_step Δ^t_m / M`, `2 λ_step δ^t_g / G`.
//! `λ_step` is the PER-STEP weight the caller passes: because the penalty lands
//! at full strength on every step, the trainer hands down
//! `HierConfig::offset_l2 / steps_per_epoch` (see
//! [`super::train::per_step_offset_l2`]), so an epoch's steps sum to exactly
//! `offset_l2 · (mean_m ‖Δ‖² + mean_g ‖δ‖²)` and the knob means the same thing
//! at every batch size.
//! **Representation chosen: every `TrackGrads` table is DENSE and flat** —
//! `r` is a `[G × H]` buffer indexed by gene id, with the ridge already summed
//! into it, rather than a sparse touched-row list plus a separate ridge plane.
//! One representation means `apply` has a single loop and the finite-difference
//! test can read any gene's offset gradient straight off `Grads`; the row
//! optimizer skips an all-zero gradient row anyway, so an untouched row whose
//! offset is still zero costs a comparison. Flat rather than keyed because the
//! key IS the index: a `Vec<(gene, Vec<f32>)>` would allocate `G` row vectors
//! per track per step on top of the buffer the gradient is accumulated in.

use super::params::{HierParams, RowAdagrad};
use super::partition::{Partition, TrackSupport, UnitModules};
use super::units::UnitTable;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::borrow::Cow;

/// The (unit, module) pairs one step evaluates at the gene level, grouped by
/// `(track, module)`. Each pair carries the module's per-unit draw weight
/// `c_k/K` — the multiplicity of that module among the unit's K draws on that
/// track, over K; an exhaustive plan lists every `(track, module)` the unit has
/// counts in at weight 1.0.
///
/// Invariants: `units` has no duplicates; each module appears at most once
/// PER TRACK in `pairs_by_module`; every unit named in a `pairs_by_module`
/// entry is also present in `units`.
pub struct StepPlan {
    pub units: Vec<u32>,
    /// Grouped by `(track, module)`.
    pub pairs_by_module: TrackModulePairs,
}

/// Transparent alias for `StepPlan::pairs_by_module`'s spelled-out type — one
/// `(unit, draw weight)` list per `(track, module)` key. Named only so the
/// nesting stays readable.
pub type TrackModulePairs = Vec<((u32, u32), Vec<(u32, f32)>)>;

#[derive(Default, Debug, Clone)]
pub struct StepStats {
    pub loss_module: f64,
    pub loss_gene: f64,
    /// The offset ridge AT THIS STEP's weight; exactly `0` on a one-track axis.
    pub loss_ridge: f64,
}

pub struct Optimizers {
    pub e_u: RowAdagrad,
    pub mu: RowAdagrad,
    pub r: RowAdagrad,
    /// Tracks `1..T`, in order: `(M rows for Δ/β, G rows for δ/γ)`.
    pub offsets: Vec<(RowAdagrad, RowAdagrad)>,
}

/// One non-base track's own gradients, every table DENSE and flat: `mu` is
/// `[M × H]` row-major, `b_m` `[M]`, `r` `[G × H]` row-major, `b_g` `[G]` —
/// indexed by module or gene id, no keys (see the module docs on the ridge
/// representation, which is what makes them dense).
pub struct TrackGrads {
    pub mu: Vec<f32>,
    pub b_m: Vec<f32>,
    pub r: Vec<f32>,
    pub b_g: Vec<f32>,
}

pub struct Grads {
    pub e_u: Vec<f32>,
    pub mu: Vec<f32>,
    pub b_m: Vec<f32>,
    pub r: Vec<(u32, Vec<f32>)>,
    pub b_g: Vec<(u32, f32)>,
    /// Tracks `1..T`, in order; the ridge gradient is already included. Empty
    /// on a one-track axis.
    pub offsets: Vec<TrackGrads>,
}

/// Row-major `[rows × h]` gather of `table` at `idx`.
fn gather(table: &[f32], h: usize, idx: &[u32]) -> DMatrix<f32> {
    let mut m = DMatrix::<f32>::zeros(idx.len(), h);
    for (i, &r) in idx.iter().enumerate() {
        let r = r as usize;
        m.row_mut(i).copy_from_slice(&table[r * h..(r + 1) * h]);
    }
    m
}

/// [`gather`] of `base + offset`, both row-major `[· × h]` on the same axis.
fn gather_sum(base: &[f32], offset: &[f32], h: usize, idx: &[u32]) -> DMatrix<f32> {
    let mut m = DMatrix::<f32>::zeros(idx.len(), h);
    for (i, &r) in idx.iter().enumerate() {
        let r = r as usize;
        let (b, o) = (&base[r * h..(r + 1) * h], &offset[r * h..(r + 1) * h]);
        for (k, dst) in m.row_mut(i).iter_mut().enumerate() {
            *dst = b[k] + o[k];
        }
    }
    m
}

/// `acc = acc + x`, MOVING `x` in when `acc` is still empty. `0 + x == x`
/// exactly, so this is the same arithmetic as adding into a zero matrix — it
/// just skips allocating and walking one when there is nothing to add to yet.
fn accumulate(acc: &mut Option<DMatrix<f32>>, x: DMatrix<f32>) {
    match acc {
        None => *acc = Some(x),
        Some(a) => *a += &x,
    }
}

/// In-place row softmax of `s`.
fn softmax_rows(s: &mut DMatrix<f32>) {
    for mut row in s.row_iter_mut() {
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = row.iter().map(|v| (v - mx).exp()).sum();
        let l = mx + z.ln();
        row.iter_mut().for_each(|v| *v = (*v - l).exp());
    }
}

/// Pure: loss + gradients for `plan`. Each gene-level pair in `plan` already
/// carries its own draw weight (`c_k/K`); this function never re-derives an
/// importance weight from `q_um`. `offset_l2` is the ridge on the non-base
/// tracks' offset tables for THIS step (module docs — the caller divides its
/// per-epoch weight by `steps_per_epoch`); it is ignored at `T == 1`, where
/// there are no offsets. `sup` is the precomputed per-track support (module
/// docs); a full track takes the unrestricted path, which at `t == 0` is the
/// plain model's.
pub fn loss_and_grads(
    params: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    sup: &TrackSupport,
    plan: &StepPlan,
    offset_l2_step: f32,
) -> (StepStats, Grads) {
    let n_t = units.n_tracks();
    debug_assert_eq!(
        params.offsets.len(),
        n_t.saturating_sub(1),
        "one offset table per non-base track"
    );
    let (h, n_m) = (params.h, part.n_modules());
    let n_g = params.b_g.len();
    let b = plan.units.len();

    //////////////////
    // Module level //
    //////////////////

    let e_b = gather(&params.e_u, h, &plan.units); // [B × H]
    let mu_base = DMatrix::<f32>::from_row_slice(n_m, h, &params.mu); // [M × H]
    let mut loss_module = 0f64;
    // Summed over tracks, but the FIRST track's result is moved in rather than
    // added to a zero matrix: at one track that is a plain assignment, with no
    // extra `[B × H]` / `[M × H]` allocation or pass.
    let mut g_e_module: Option<DMatrix<f32>> = None;
    let mut g_mu: Option<DMatrix<f32>> = None;
    let mut g_b_m = vec![0f32; n_m];
    // Per non-base track, in track order.
    let mut off_mu: Vec<Vec<f32>> = Vec::with_capacity(n_t.saturating_sub(1));
    let mut off_b_m: Vec<Vec<f32>> = Vec::with_capacity(n_t.saturating_sub(1));

    for t in 0..n_t {
        let w: Vec<f32> = plan
            .units
            .iter()
            .map(|&u| units.weight_of(u as usize, t))
            .collect();
        let offset = params.offset(t);
        if sup.is_full(t) {
            // Unrestricted: every module is a column, empty ones included. At
            // `t == 0` these are the plain model's statements, unchanged.
            // `Borrowed` at t == 0: the base dictionary itself, never `μ + 0`.
            // Same module-level softmax/gradient math as the restricted arm
            // below, gathered/scattered through `mods` there instead of run
            // dense here: a change to one side's math belongs on both.
            let mu_eff: Cow<DMatrix<f32>> = match offset {
                None => Cow::Borrowed(&mu_base),
                Some(o) => Cow::Owned(DMatrix::<f32>::from_fn(n_m, h, |m, k| {
                    params.mu[m * h + k] + o.d_mu[m * h + k]
                })),
            };
            let mut s = &e_b * mu_eff.transpose(); // [B × M]
            match offset {
                None => {
                    for mut row in s.row_iter_mut() {
                        row.iter_mut().zip(&params.b_m).for_each(|(v, b)| *v += b);
                    }
                }
                Some(o) => {
                    for mut row in s.row_iter_mut() {
                        row.iter_mut()
                            .zip(params.b_m.iter().zip(&o.d_b_m))
                            .for_each(|(v, (b, d))| *v += b + d);
                    }
                }
            }
            softmax_rows(&mut s); // s is now p (softmax output)
            let mut delta1 = DMatrix::<f32>::zeros(b, n_m); // w_u (p − q)
            for (i, &u) in plan.units.iter().enumerate() {
                let base = um.idx(u as usize, t, 0);
                let q = &um.q[base..base + n_m];
                for m in 0..n_m {
                    let p = s[(i, m)];
                    if q[m] > 0.0 {
                        // the module-level loss uses ln p, p being softmax_rows' output
                        // clamped: an underflowed p would report +inf; the gradient does not use ln p
                        loss_module -= f64::from(w[i] * q[m] * (p.max(f32::MIN_POSITIVE).ln()));
                    }
                    delta1[(i, m)] = w[i] * (p - q[m]);
                }
            }
            let g_e_t = &delta1 * &*mu_eff; // [B × H]
            let g_mu_t = delta1.tr_mul(&e_b); // [M × H]
            let g_b_m_t: Vec<f32> = (0..n_m).map(|m| delta1.column(m).sum()).collect();
            for (acc, x) in g_b_m.iter_mut().zip(&g_b_m_t) {
                *acc += x;
            }
            accumulate(&mut g_e_module, g_e_t);
            if offset.is_some() {
                off_mu.push(g_mu_t.transpose().as_slice().to_vec());
                off_b_m.push(g_b_m_t);
            }
            accumulate(&mut g_mu, g_mu_t);
            continue;
        }

        // Restricted: the softmax runs over M_t, the modules holding at least
        // one gene this track has a row for. A module outside M_t is not a
        // negative here — it is off this track's axis — so it gets no column and
        // no gradient. Rows are scattered back onto the full `[M × H]` tables.
        // Same module-level softmax/gradient math as the unrestricted arm
        // above, run dense there over every module instead of gathered
        // through `mods`: a change to one side's math belongs on both.
        let mods = sup.modules_of(t);
        let k_m = mods.len();
        let mu_eff = DMatrix::<f32>::from_fn(k_m, h, |j, k| {
            let m = mods[j] as usize;
            params.mu[m * h + k] + offset.map_or(0.0, |o| o.d_mu[m * h + k])
        });
        let mut s = &e_b * mu_eff.transpose(); // [B × |M_t|]
        for mut row in s.row_iter_mut() {
            for (j, &m) in mods.iter().enumerate() {
                let m = m as usize;
                row[j] += params.b_m[m] + offset.map_or(0.0, |o| o.d_b_m[m]);
            }
        }
        softmax_rows(&mut s);
        let mut delta1 = DMatrix::<f32>::zeros(b, k_m);
        for (i, &u) in plan.units.iter().enumerate() {
            let base = um.idx(u as usize, t, 0);
            for (j, &m) in mods.iter().enumerate() {
                let q = um.q[base + m as usize];
                let p = s[(i, j)];
                if q > 0.0 {
                    loss_module -= f64::from(w[i] * q * (p.max(f32::MIN_POSITIVE).ln()));
                }
                delta1[(i, j)] = w[i] * (p - q);
            }
        }
        accumulate(&mut g_e_module, &delta1 * &mu_eff);
        let g_mu_sub = delta1.tr_mul(&e_b); // [|M_t| × H]
        let mut mu_dense = vec![0f32; n_m * h];
        let mut b_m_dense = vec![0f32; n_m];
        let g_mu_acc = g_mu.get_or_insert_with(|| DMatrix::<f32>::zeros(n_m, h));
        for (j, &m) in mods.iter().enumerate() {
            let m = m as usize;
            for k in 0..h {
                let x = g_mu_sub[(j, k)];
                g_mu_acc[(m, k)] += x;
                mu_dense[m * h + k] = x;
            }
            let bb = delta1.column(j).sum();
            g_b_m[m] += bb;
            b_m_dense[m] = bb;
        }
        if offset.is_some() {
            off_mu.push(mu_dense);
            off_b_m.push(b_m_dense);
        }
    }

    let g_e_module = g_e_module.unwrap_or_else(|| DMatrix::<f32>::zeros(b, h));
    let g_mu = g_mu.unwrap_or_else(|| DMatrix::<f32>::zeros(n_m, h));

    ////////////////
    // Gene level //
    ////////////////

    // Position of each unit in the plan, for scattering e_u gradients.
    let pos_of: rustc_hash::FxHashMap<u32, usize> = plan
        .units
        .iter()
        .enumerate()
        .map(|(i, &u)| (u, i))
        .collect();
    struct ModuleOut {
        track: usize,
        module: u32,
        loss: f64,
        e_rows: Vec<(usize, Vec<f32>)>, // (position in plan, grad row)
        /// `(gene, grad row)`. Only the genes THIS track scored, ascending: a
        /// subset of the module's members on a restricted track, all of them
        /// otherwise.
        r_rows: Vec<(u32, Vec<f32>)>,
        b_rows: Vec<(u32, f32)>,
    }
    let outs: Vec<ModuleOut> = plan
        .pairs_by_module
        .par_iter()
        .map(|(key, pairs)| {
            let (t, m) = (key.0 as usize, key.1);
            let offset = params.offset(t);
            let members = &part.members[m as usize];
            // `None` on an unrestricted track: its columns ARE `members`, in
            // members order, so nothing is gathered, indexed or allocated.
            // `Some(slots)` restricts to the member slots this track has rows
            // for, and `local` maps a full slot back onto those columns.
            let restricted: Option<(&[u32], &[u32])> = (!sup.is_full(t))
                .then(|| (sup.slots_of(t, m as usize), sup.local_of(t, m as usize)));
            let genes: Cow<[u32]> = match restricted {
                None => Cow::Borrowed(members.as_slice()),
                Some((slots, _)) => {
                    Cow::Owned(slots.iter().map(|&j| members[j as usize]).collect())
                }
            };
            let d_m = genes.len();
            let r_m = match offset {
                None => gather(&params.r, h, &genes), // [d_m × H]
                Some(o) => gather_sum(&params.r, &o.d_r, h, &genes),
            };
            let ids: Vec<u32> = pairs.iter().map(|&(u, _)| u).collect();
            let e_m = gather(&params.e_u, h, &ids); // [n × H]
            let mut s = &e_m * r_m.transpose(); // [n × d_m]
            match offset {
                None => {
                    for mut row in s.row_iter_mut() {
                        for (j, &g) in genes.iter().enumerate() {
                            row[j] += params.b_g[g as usize];
                        }
                    }
                }
                Some(o) => {
                    for mut row in s.row_iter_mut() {
                        for (j, &g) in genes.iter().enumerate() {
                            row[j] += params.b_g[g as usize] + o.d_b_g[g as usize];
                        }
                    }
                }
            }
            softmax_rows(&mut s);
            let mut loss = 0f64;
            let mut delta2 = DMatrix::<f32>::zeros(pairs.len(), d_m); // w_u·(c_k/K) (p − q_g|m)
                                                                      // One target buffer per module, cleared through the slots it touched.
            let mut target = vec![0f32; d_m];
            for (i, &(u, wt)) in pairs.iter().enumerate() {
                let scale = units.weight_of(u as usize, t) * wt;
                // target shares within the module, on this track
                let counts = um.by_module[u as usize]
                    .iter()
                    .find(|((tr, k), _)| *tr as usize == t && *k == m)
                    .map(|(_, v)| v.as_slice())
                    .unwrap_or(&[]);
                let n_um = um.n_um[um.idx(u as usize, t, m as usize)];
                // Bucket slots index the FULL member list. Two loops rather
                // than a per-element branch: on an unrestricted track the slot
                // IS the column.
                match restricted {
                    None => {
                        for &(slot, c) in counts {
                            target[slot as usize] += c / n_um;
                        }
                    }
                    Some((_, local)) => {
                        for &(slot, c) in counts {
                            let col = local[slot as usize];
                            debug_assert_ne!(col, u32::MAX, "a count exists only where a row does");
                            target[col as usize] += c / n_um;
                        }
                    }
                }
                for j in 0..d_m {
                    let p = s[(i, j)];
                    if target[j] > 0.0 {
                        // clamped: an underflowed p would report +inf; the gradient does not use ln p
                        loss -= f64::from(scale * target[j] * p.max(f32::MIN_POSITIVE).ln());
                    }
                    delta2[(i, j)] = scale * (p - target[j]);
                }
                match restricted {
                    None => {
                        for &(slot, _) in counts {
                            target[slot as usize] = 0.0;
                        }
                    }
                    Some((_, local)) => {
                        for &(slot, _) in counts {
                            target[local[slot as usize] as usize] = 0.0;
                        }
                    }
                }
            }
            let g_e = &delta2 * &r_m; // [n × H]
            let g_r = delta2.tr_mul(&e_m); // [d_m × H]
            let e_rows = pairs
                .iter()
                .enumerate()
                .map(|(i, &(u, _))| {
                    let pos = *pos_of
                        .get(&u)
                        .expect("every unit drawn into a module is in plan.units");
                    (pos, g_e.row(i).iter().copied().collect())
                })
                .collect();
            let r_rows = genes
                .iter()
                .enumerate()
                .map(|(j, &g)| (g, g_r.row(j).iter().copied().collect()))
                .collect();
            let b_rows = genes
                .iter()
                .enumerate()
                .map(|(j, &g)| (g, delta2.column(j).sum()))
                .collect();
            ModuleOut {
                track: t,
                module: m,
                loss,
                e_rows,
                r_rows,
                b_rows,
            }
        })
        .collect();

    ////////////
    // Reduce //
    ////////////

    // The base gene tables take the SUM over tracks, so a module scored on more
    // than one track has to be merged. The common case — and the WHOLE of the
    // one-track case — is a module claimed by exactly one group, whose rows are
    // MOVED into the output untouched, in the group's own order: no buffer, no
    // copy, byte for byte what the plain model emits. Only a module a second
    // group also claims is merged, by gene id, and only then is its gene →
    // position index built. A member no scored track has a row for never
    // appears at all — so it takes no gradient AND no weight decay, since
    // `apply` decays what it is handed.
    let mut g_e_u: Vec<f32> = g_e_module.transpose().as_slice().to_vec(); // row-major [B × H]
                                                                          // (nalgebra is column-major: transpose().as_slice() walks row-major of the original)
    let mut loss_gene = 0f64;
    let mut g_r: Vec<(u32, Vec<f32>)> = Vec::new();
    let mut g_b_g: Vec<(u32, f32)> = Vec::new();
    // module → (first index in `g_r`, how many rows) for the group that claimed it.
    let mut claimed: rustc_hash::FxHashMap<u32, (usize, usize)> = rustc_hash::FxHashMap::default();
    // (module, gene) → index in `g_r`; filled lazily, only for merged modules.
    let mut pos_of_gene: rustc_hash::FxHashMap<(u32, u32), usize> =
        rustc_hash::FxHashMap::default();
    let mut indexed: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();

    // Each non-base track's own gene-side gradient, dense over all G genes.
    let n_off = n_t.saturating_sub(1);
    let mut off_r: Vec<Vec<f32>> = (0..n_off).map(|_| vec![0f32; n_g * h]).collect();
    let mut off_b_g: Vec<Vec<f32>> = (0..n_off).map(|_| vec![0f32; n_g]).collect();

    for o in outs {
        loss_gene += o.loss;
        for (i, row) in o.e_rows {
            for k in 0..h {
                g_e_u[i * h + k] += row[k];
            }
        }
        if o.track > 0 {
            let i = o.track - 1;
            for (gene, row) in &o.r_rows {
                let g = *gene as usize;
                for k in 0..h {
                    off_r[i][g * h + k] += row[k];
                }
            }
            for &(gene, x) in &o.b_rows {
                off_b_g[i][gene as usize] += x;
            }
        }
        match claimed.get(&o.module).copied() {
            None => {
                claimed.insert(o.module, (g_r.len(), o.r_rows.len()));
                g_r.extend(o.r_rows);
                g_b_g.extend(o.b_rows);
            }
            Some((first, len)) => {
                if indexed.insert(o.module) {
                    for (i, (gene, _)) in g_r.iter().enumerate().skip(first).take(len) {
                        pos_of_gene.insert((o.module, *gene), i);
                    }
                }
                debug_assert_eq!(o.r_rows.len(), o.b_rows.len());
                for ((gene, row), (_, x)) in o.r_rows.into_iter().zip(o.b_rows) {
                    match pos_of_gene.get(&(o.module, gene)).copied() {
                        Some(i) => {
                            for (acc, v) in g_r[i].1.iter_mut().zip(&row) {
                                *acc += v;
                            }
                            g_b_g[i].1 += x;
                        }
                        // The claiming group's track had no row for this gene.
                        None => {
                            pos_of_gene.insert((o.module, gene), g_r.len());
                            g_r.push((gene, row));
                            g_b_g.push((gene, x));
                        }
                    }
                }
            }
        }
    }

    ///////////////////////////
    // Ridge on the offsets  //
    ///////////////////////////

    let mut loss_ridge = 0f64;
    let (ridge_m, ridge_g) = (
        2.0 * offset_l2_step / n_m.max(1) as f32,
        2.0 * offset_l2_step / n_g.max(1) as f32,
    );
    let sq = |v: &[f32]| v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
    let offsets: Vec<TrackGrads> = params
        .offsets
        .iter()
        .enumerate()
        .map(|(i, o)| {
            loss_ridge += f64::from(offset_l2_step)
                * (sq(&o.d_mu) / n_m.max(1) as f64 + sq(&o.d_r) / n_g.max(1) as f64);
            let mut mu = std::mem::take(&mut off_mu[i]);
            for (x, d) in mu.iter_mut().zip(&o.d_mu) {
                *x += ridge_m * d;
            }
            // Dense over every gene row: the ridge reaches rows the plan never
            // touched (module docs). Added into the accumulator in place — the
            // buffer IS the gradient, never copied out row by row.
            let mut r = std::mem::take(&mut off_r[i]);
            for (x, d) in r.iter_mut().zip(&o.d_r) {
                *x += ridge_g * d;
            }
            TrackGrads {
                mu,
                b_m: std::mem::take(&mut off_b_m[i]),
                r,
                b_g: std::mem::take(&mut off_b_g[i]),
            }
        })
        .collect();

    (
        StepStats {
            loss_module,
            loss_gene,
            loss_ridge,
        },
        Grads {
            e_u: g_e_u,
            mu: g_mu.transpose().as_slice().to_vec(),
            b_m: g_b_m,
            r: g_r,
            b_g: g_b_g,
            offsets,
        },
    )
}

/// Apply `grads` with the row optimizers (weight decay `wd` on touched rows:
/// `row *= 1 − lr·wd` before the Adagrad step; biases never decay). The offset
/// tables never decay at all — their shrinkage is the exact ridge already
/// carried in `grads.offsets`.
pub fn apply(
    params: &mut HierParams,
    opt: &mut Optimizers,
    grads: &Grads,
    plan: &StepPlan,
    wd: f32,
) {
    let h = params.h;
    let decay = |row: &mut [f32], lr: f32| {
        if wd > 0.0 {
            let f = 1.0 - lr * wd;
            row.iter_mut().for_each(|x| *x *= f);
        }
    };
    for (i, &u) in plan.units.iter().enumerate() {
        let u = u as usize;
        let row = &mut params.e_u[u * h..(u + 1) * h];
        decay(row, opt.e_u.lr);
        opt.e_u.update(u, row, &grads.e_u[i * h..(i + 1) * h]);
    }
    for m in 0..params.b_m.len() {
        let row = &mut params.mu[m * h..(m + 1) * h];
        decay(row, opt.mu.lr);
        opt.mu.update_with_bias(
            m,
            row,
            &mut params.b_m[m],
            &grads.mu[m * h..(m + 1) * h],
            grads.b_m[m],
        );
    }
    // `grads.r` and `grads.b_g` are emitted in lockstep, one entry per gene
    // touched by at least one scored track, so they zip.
    debug_assert_eq!(grads.r.len(), grads.b_g.len());
    for ((g, gr), &(gb_gene, gb)) in grads.r.iter().zip(&grads.b_g) {
        debug_assert_eq!(*g, gb_gene);
        let gi = *g as usize;
        let row = &mut params.r[gi * h..(gi + 1) * h];
        decay(row, opt.r.lr);
        opt.r.update_with_bias(gi, row, &mut params.b_g[gi], gr, gb);
    }
    // Non-base tracks: no weight decay (the ridge is already exact in the
    // gradient), and every row is offered — the row optimizer skips a row whose
    // gradient is all zero, which is what an untouched, still-zero offset has.
    debug_assert_eq!(grads.offsets.len(), params.offsets.len());
    for ((tg, (opt_mu, opt_r)), off) in grads
        .offsets
        .iter()
        .zip(&mut opt.offsets)
        .zip(&mut params.offsets)
    {
        for m in 0..off.d_b_m.len() {
            opt_mu.update_with_bias(
                m,
                &mut off.d_mu[m * h..(m + 1) * h],
                &mut off.d_b_m[m],
                &tg.mu[m * h..(m + 1) * h],
                tg.b_m[m],
            );
        }
        debug_assert_eq!(tg.r.len(), tg.b_g.len() * h);
        for g in 0..tg.b_g.len() {
            opt_r.update_with_bias(
                g,
                &mut off.d_r[g * h..(g + 1) * h],
                &mut off.d_b_g[g],
                &tg.r[g * h..(g + 1) * h],
                tg.b_g[g],
            );
        }
    }
}

#[cfg(test)]
#[path = "step_tests.rs"]
mod step_tests;
