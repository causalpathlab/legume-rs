//! The warm-started polish on a **multi-track** feature axis: one Poisson
//! partition and one intercept per track, against one shared latent.
//!
//! A track is its own composition over its own rows with its own sequencing
//! depth, so the single-partition polish ([`super::pass`]) is the wrong model for
//! it twice over: it normalises `exp(s)` across rows that belong to different
//! compositions, and it gives the whole cell one depth. Here each track `t`
//! carries its own rows, its own null normaliser and its own intercept `c_t`,
//! and only the latent `Θ` is shared.
//!
//! # The objective
//!
//! For a block of `Bc` cells, with `E^t [F_t, H]` / `β^t [F_t]` the frozen
//! dictionary restricted to track `t`'s rows and `N^t` the block's counts on them:
//!
//! ```text
//! S^t    = Θ·E^tᵀ + c_t + β^t                                  [Bc, F_t]
//! loss   = Σ_t [ Σ_{f∈t} exp(S^t_f) − Σ_{f∈t} n_f·S^t_f ] + (λ/2)‖Θ‖²
//! ∂/∂Θ   = Σ_t (μ^t − N^t)·E^t + λΘ
//! ∂/∂c_t = Σ_{f∈t} μ^t_f − Σ_{f∈t} n_f
//! ```
//!
//! with `μ^t = exp(S^t)`. The ridge is on the latent only; the intercepts are
//! unpenalised, exactly as the single-partition solve leaves its one intercept.
//!
//! # A track the cell has no counts on is skipped
//!
//! Its intercept is not identified — the data term is empty, so the partition
//! `Σ_f exp(S^t)` would drive `c_t → −∞` and drag `Θ` along the way. Such a
//! cell's track is therefore **masked out of the objective entirely**: `μ^t` is
//! zeroed on that row, which zeroes its contribution to both gradients, so `Θ`
//! never sees the track and `c_t` stays exactly where it was initialised — at
//! the score-clamp floor, which is what [`super::solve`] already reports for a
//! cell with no counts at all.
//!
//! # One parameter, not `T + 1` of them
//!
//! The intercepts ride as **extra columns** of the parameter,
//! `Θ̃ = [Θ | c_0 | … | c_{T−1}]`, against a per-track design whose row `H + t`
//! is ones and whose other intercept rows are zero. That is the same trick the
//! single-partition solve plays with its one intercept, generalised: each `c_t`'s
//! gradient falls out of its own track's matmul (the ones row sums `μ^t` over
//! that track's features, which is exactly `∂/∂c_t`), there is one Adam state
//! rather than `T + 1`, and — because Adam is per-coordinate — the numbers are
//! the ones the separate-parameter form would produce.

use super::edges::{block_cells, EdgeTable};
use super::pass::{adam_step_size, poisson_deviance, BlockProgress, PassOut, PassStats};
use super::{
    Phase2Input, BETA1, BETA2, CHECK_EVERY, EPS, GATE_FOLD_EPS, POLISH_STEPS, TARGET_DELTA_S, TOL,
};
use crate::cell_projection::SCORE_CLAMP;
use crate::fit::config::TrackSpec;
use crate::fit::projection::CellBatchFold;
use candle_util::candle_core::{DType, Tensor};
use log::info;
use matrix_util::traits::FusedTensorOps;

///////////////////////////////////
// One track's frozen design     //
///////////////////////////////////

/// One track's share of the frozen dictionary, plus the cells' counts on it.
///
/// Unlike [`super::pass::PassDict`] there is no single augmented design: each
/// track's `Ẽ` is `[H + T, F_live]`, ones on **its own** intercept row and zero
/// on the others, so one shared parameter carries every track's intercept.
struct TrackDict {
    /// Track id — which intercept column of `Θ̃` this track drives.
    id: usize,
    /// This track's rows on the global feature axis, ascending.
    rows: Vec<u32>,
    /// Those rows' counts, flattened once (the batch fold applied here).
    edges: EdgeTable,
    /// Global feature id → this track's live-local id; `u32::MAX` off it.
    to_live: Vec<u32>,
    /// `Ẽᵀ [H + T, F_live]` and `Ẽ [F_live, H + T]` — both orientations are
    /// needed every step (forward `Θ̃·Ẽᵀ`, gradient `μ·Ẽ`).
    e_aug: Tensor,
    e_aug_t: Tensor,
    /// `β^t [1, F_live]`.
    b_row: Tensor,
    /// `[1, H + T]` selector with 1.0 in this track's intercept slot.
    intercept_mask: Tensor,
    /// Gate-folded partition mass `Σ_dead exp(β_f)`; 0 at the default
    /// [`GATE_FOLD_EPS`], where nothing is folded.
    dead_mass: f64,
    f_live: usize,
}

/// Build one track's design over `rows`, with `n_tracks` intercept columns in the
/// shared parameter.
fn build_track(
    input: &Phase2Input,
    cells: &[(u32, &[u32], &[f32])],
    batch_fold: Option<CellBatchFold>,
    rows: Vec<u32>,
    id: usize,
    n_tracks: usize,
) -> anyhow::Result<TrackDict> {
    let (h, dev) = (input.h, input.dev);
    let n_features = input.b_feat.len();

    // Live rows go into the matmul; a gate-folded row's score is `β_f + c_t`
    // however `Θ` moves, so its partition mass is a scalar.
    let mut live: Vec<u32> = Vec::with_capacity(rows.len());
    let mut dead_mass = 0f64;
    for &g in &rows {
        let e = &input.feat[g as usize * h..(g as usize + 1) * h];
        if e.iter().map(|x| x * x).sum::<f32>().sqrt() > GATE_FOLD_EPS {
            live.push(g);
        } else {
            dead_mass += f64::from(input.b_feat[g as usize]).exp();
        }
    }
    anyhow::ensure!(
        !live.is_empty(),
        "phase-2 polish: track {id} has no live features — the frozen dictionary \
         carries no signal to project its rows onto"
    );

    let f_live = live.len();
    let d = h + n_tracks;
    let mut e_aug = vec![0f32; d * f_live];
    let mut b_live = vec![0f32; f_live];
    for (l, &g) in live.iter().enumerate() {
        let row = &input.feat[g as usize * h..(g as usize + 1) * h];
        for (k, &v) in row.iter().enumerate() {
            e_aug[k * f_live + l] = v;
        }
        e_aug[(h + id) * f_live + l] = 1.0; // this track's intercept row
        b_live[l] = input.b_feat[g as usize];
    }
    let e_aug = Tensor::from_vec(e_aug, (d, f_live), dev)?;
    let e_aug_t = e_aug.t()?.contiguous()?;
    let b_row = Tensor::from_vec(b_live, (1, f_live), dev)?;

    let intercept_mask = {
        let mut v = vec![0f32; d];
        v[h + id] = 1.0;
        Tensor::from_vec(v, (1, d), dev)?
    };

    let mut to_live = vec![u32::MAX; n_features];
    for (l, &g) in live.iter().enumerate() {
        to_live[g as usize] = l as u32;
    }

    let edges = EdgeTable::build(cells, &rows, n_features, batch_fold);

    Ok(TrackDict {
        id,
        rows,
        edges,
        to_live,
        e_aug,
        e_aug_t,
        b_row,
        intercept_mask,
        dead_mass,
        f_live,
    })
}

//////////////////////////
// The per-track polish //
//////////////////////////

/// Finish every cell on the exact per-track Poisson objective, started from
/// `init` (`[n_kept × h]`, indexed by position in `cells`) and capped at
/// [`POLISH_STEPS`] Adam steps per block — the multi-track counterpart of
/// [`super::polish_cells`]'s single-partition pass, which it dispatches to at
/// `T == 1`.
pub(super) fn polish_tracks(
    input: &Phase2Input,
    cells: &[(u32, &[u32], &[f32])],
    batch_fold: Option<CellBatchFold>,
    init: &[f32],
    tracks: &TrackSpec,
) -> anyhow::Result<PassOut> {
    let (h, n_kept) = (input.h, cells.len());
    let n_features = input.b_feat.len();
    anyhow::ensure!(
        input.feat.len() == n_features * h,
        "phase-2 polish: e_feat has {} entries, expected {n_features} × {h}",
        input.feat.len(),
    );
    anyhow::ensure!(
        init.len() == n_kept * h,
        "phase-2 polish: init has {} entries, expected {n_kept} × {h}",
        init.len(),
    );
    tracks.validate(n_features)?;
    let n_tracks = tracks.n_tracks();

    let dicts: Vec<TrackDict> = (0..n_tracks)
        .map(|t| {
            build_track(
                input,
                cells,
                batch_fold,
                tracks.rows_of_track(t),
                t,
                n_tracks,
            )
        })
        .collect::<anyhow::Result<_>>()?;

    // Every track's `[Bc, F_t]` counts are resident at once, so size the block
    // from the SUM of the partitions — which is the whole feature axis.
    let n_rows: usize = dicts.iter().map(|d| d.rows.len()).sum();
    let bc = block_cells(n_rows);

    // One learning rate for one shared `Θ`, so the dictionary scale it is
    // calibrated on is every track's rows together (see `TARGET_DELTA_S`; the
    // estimator is the arithmetic mean `|e|`, not the RMS — the note in
    // [`super::pass::PassDict::build`] says why).
    let f_live_total: usize = dicts.iter().map(|d| d.f_live).sum();
    let lr0 = {
        let sa: f64 = dicts
            .iter()
            .flat_map(|d| {
                d.rows
                    .iter()
                    .filter(|&&g| d.to_live[g as usize] != u32::MAX)
            })
            .flat_map(|&g| &input.feat[g as usize * h..(g as usize + 1) * h])
            .map(|x| f64::from(*x).abs())
            .sum();
        let e_mean = (sa / (f_live_total * h) as f64).max(1e-12);
        TARGET_DELTA_S / (h as f64 * e_mean)
    };

    info!(
        "{} [polish · {n_tracks} tracks] — {f_live_total} live features (of {n_rows}), \
         blocks of {bc}, lr {:.4} (auto: Δs≈{TARGET_DELTA_S}), ≤{POLISH_STEPS} steps, \
         ridge λ={}",
        input.label, lr0, input.lambda,
    );

    let bar = crate::progress::new_progress_bar(n_kept as u64);
    bar.enable_steady_tick(std::time::Duration::from_millis(200));

    let mut latent = vec![0f32; n_kept * h];
    let mut intercepts = vec![vec![0f32; n_kept]; n_tracks];
    let mut stats = PassStats::default();
    let n_blocks = n_kept.div_ceil(bc);

    for (b, start) in (0..n_kept).step_by(bc).enumerate() {
        let end = (start + bc).min(n_kept);
        let out = solve_tracks_block(TrackBlockArgs {
            input,
            dicts: &dicts,
            init,
            lr0,
            start,
            end,
            progress: &BlockProgress {
                bar: &bar,
                stats: &stats,
                label: "polish · tracks",
                block: b + 1,
                n_blocks,
                max_steps: POLISH_STEPS,
            },
        })?;
        latent[start * h..end * h].copy_from_slice(&out.latent);
        for (t, c) in out.intercepts.iter().enumerate() {
            intercepts[t][start..end].copy_from_slice(c);
        }
        stats.fold(
            out.steps,
            out.converged,
            out.clamped,
            out.deviance,
            out.n_edges,
            out.loop_secs,
        );
    }
    bar.finish_and_clear();

    info!(
        "{} [polish · {n_tracks} tracks] — {n_kept} node(s) done: ⌀{:.0} steps/block, {} of {} \
         block(s) hit the {POLISH_STEPS}-step cap, mean per-edge deviance {:.4}, {:.0}s total \
         ({:.1} ms/step){}",
        input.label,
        stats.mean_steps(),
        stats.at_cap,
        stats.blocks,
        stats.mean_deviance(),
        stats.secs,
        stats.ms_per_step(),
        if stats.clamped > 0 {
            format!(
                " [WARNING: score clamp bound on {} block(s)]",
                stats.clamped
            )
        } else {
            String::new()
        },
    );

    let mut intercepts = intercepts.into_iter();
    let intercept = intercepts
        .next()
        .expect("a validated TrackSpec has a base track");
    Ok(PassOut {
        latent,
        intercept,
        other_intercepts: intercepts.collect(),
    })
}

///////////////
// One block //
///////////////

struct TrackBlockArgs<'a> {
    input: &'a Phase2Input<'a>,
    /// The frozen per-track design every block of this pass shares.
    dicts: &'a [TrackDict],
    /// Warm start for the latent, host-side `[n_kept × h]`.
    init: &'a [f32],
    lr0: f64,
    start: usize,
    end: usize,
    progress: &'a BlockProgress<'a>,
}

struct TrackBlockOut {
    latent: Vec<f32>,
    /// One `[Bc]` intercept per track, in track order.
    intercepts: Vec<Vec<f32>>,
    steps: usize,
    converged: bool,
    clamped: bool,
    deviance: f64,
    n_edges: usize,
    loop_secs: f64,
}

/// One track's counts for one block, densely, plus what the null intercept and
/// the presence mask need.
struct TrackBlock {
    /// `N^t [Bc, F_live]`.
    n_t: Tensor,
    /// Folded rows' count mass, `[Bc, 1]`; `None` when nothing was folded.
    n_dead: Option<Tensor>,
    /// Per-cell total count on this track.
    n_tot: Vec<f64>,
    /// `[Bc, 1]` 1.0 where the cell has counts on this track; `None` when every
    /// cell in the block does, so the common case costs no extra op.
    mask: Option<Tensor>,
    n_edges: usize,
}

fn gather_track(dict: &TrackDict, a: &TrackBlockArgs, bc: usize) -> anyhow::Result<TrackBlock> {
    let dev = a.input.dev;
    let f_live = dict.f_live;
    let mut n_dense = vec![0f32; bc * f_live];
    let mut n_tot = vec![0f64; bc];
    let mut n_dead = vec![0f32; bc];
    let mut n_edges = 0usize;
    for i in a.start..a.end {
        let local = i - a.start;
        let (feats, counts) = dict.edges.cell_slice(i);
        for (&f_pass, &n) in feats.iter().zip(counts) {
            // `edges` are indexed on this track's partition; map to its live subset.
            let g = dict.rows[f_pass as usize];
            let l = dict.to_live[g as usize];
            n_tot[local] += f64::from(n);
            if l == u32::MAX {
                n_dead[local] += n;
            } else {
                n_dense[local * f_live + l as usize] = n;
                n_edges += 1;
            }
        }
    }
    let present: Vec<f32> = n_tot.iter().map(|&n| f32::from(n > 0.0)).collect();
    let mask = if present.iter().all(|&p| p == 1.0) {
        None
    } else {
        Some(Tensor::from_vec(present, (bc, 1), dev)?)
    };
    Ok(TrackBlock {
        n_t: Tensor::from_vec(n_dense, (bc, f_live), dev)?.detach(),
        n_dead: if dict.dead_mass > 0.0 {
            Some(Tensor::from_vec(n_dead, (bc, 1), dev)?)
        } else {
            None
        },
        n_tot,
        mask,
        n_edges,
    })
}

/// The Adam loop for one block against every track's design: the closed-form
/// gradient the module docs give, over `Θ̃ = [Θ | c_0 … c_{T−1}]`.
fn solve_tracks_block(a: TrackBlockArgs) -> anyhow::Result<TrackBlockOut> {
    let (h, dev) = (a.input.h, a.input.dev);
    let bc = a.end - a.start;
    let n_tracks = a.dicts.len();
    let d = h + n_tracks;

    let blocks: Vec<TrackBlock> = a
        .dicts
        .iter()
        .map(|t| gather_track(t, &a, bc))
        .collect::<anyhow::Result<_>>()?;
    let n_edges: usize = blocks.iter().map(|b| b.n_edges).sum();

    ///////////////////////////////
    // Null-model initialisation //
    ///////////////////////////////

    // `Θ` at its warm start, and each track's intercept at the exact conditional
    // MLE given it:
    //     c_t = ln(Σ_{f∈t} n_cf) − ln(Σ_{f∈t} exp(⟨e_f, θ_c⟩ + β_f) + dead_mass_t)
    // so step 1 already sits at each track's right depth. A cell with no counts
    // on the track starts — and, being masked out of the objective, stays — at
    // the score-clamp floor.
    let mut theta = vec![0f32; bc * d];
    for (i, row) in a.init[a.start * h..a.end * h].chunks_exact(h).enumerate() {
        theta[i * d..i * d + h].copy_from_slice(row);
    }
    let init_block = Tensor::from_slice(&a.init[a.start * h..a.end * h], (bc, h), dev)?;
    for (t, dict) in a.dicts.iter().enumerate() {
        let log_norm: Vec<f64> = init_block
            .matmul(&dict.e_aug.narrow(0, 0, h)?.contiguous()?)?
            .broadcast_add(&dict.b_row)?
            .clamp(-SCORE_CLAMP, SCORE_CLAMP)?
            .exp()?
            .sum(1)?
            .to_vec1::<f32>()?
            .iter()
            .map(|x| (f64::from(*x) + dict.dead_mass).max(f64::MIN_POSITIVE).ln())
            .collect();
        for (i, (&n, &lz)) in blocks[t].n_tot.iter().zip(&log_norm).enumerate() {
            theta[i * d + h + t] = if n > 0.0 {
                (n.ln() - lz).clamp(-SCORE_CLAMP, SCORE_CLAMP) as f32
            } else {
                -SCORE_CLAMP as f32
            };
        }
    }
    let mut theta = Tensor::from_vec(theta, (bc, d), dev)?;

    ////////////////////////////////////
    // Loop-invariant gradient pieces //
    ////////////////////////////////////

    // The data term is LINEAR in the parameters, so `∂/∂Θ̃ Σ_t Σ_f n·s = −Σ_t N^t·Ẽ^t`
    // is a constant, computed once. An absent cell's `N^t` row is zero, so it
    // contributes nothing here either. A track's folded rows still owe `−n_dead·c_t`,
    // which lands on that track's intercept column.
    let mut ne = Tensor::zeros((bc, d), DType::F32, dev)?;
    for (dict, blk) in a.dicts.iter().zip(&blocks) {
        ne = (ne + blk.n_t.matmul(&dict.e_aug_t)?)?;
        if let Some(n_dead) = &blk.n_dead {
            ne = (ne + n_dead.broadcast_mul(&dict.intercept_mask)?)?;
        }
    }
    // The ridge is on the latent only — every intercept column is unpenalised.
    let lam_row = {
        let mut v = vec![a.input.lambda as f32; d];
        for x in v[h..].iter_mut() {
            *x = 0.0;
        }
        Tensor::from_vec(v, (1, d), dev)?
    };

    ///////////////////
    // The Adam loop //
    ///////////////////

    let mut m = Tensor::zeros((bc, d), DType::F32, dev)?;
    let mut v = Tensor::zeros((bc, d), DType::F32, dev)?;

    // Convergence is `‖ΔΘ‖/‖Θ‖` over the LATENT columns: the intercepts start at
    // their conditional MLE and barely move, so including them would let a latent
    // still in flight look settled.
    let mut prev = theta.narrow(1, 0, h)?.contiguous()?;
    let mut steps = 0usize;
    let mut converged = false;
    let mut emitted = 0usize;
    let loop_start = std::time::Instant::now();
    for step in 0..POLISH_STEPS {
        let mut g = theta.broadcast_mul(&lam_row)?;
        for (dict, blk) in a.dicts.iter().zip(&blocks) {
            // Upper clamp only: `exp` overflows f32 at 88, while `exp(−large)`
            // underflowing to 0 is the right answer for a feature the cell does
            // not express. The matmul result is fresh and unaliased, which is
            // what lets the fused clamp-exp-add consume it in place.
            let mu = theta
                .matmul(&dict.e_aug)?
                .clamped_exp_add_inplace(&dict.b_row, SCORE_CLAMP)?;
            // A cell with no counts here is out of the objective: zeroing `μ`
            // zeroes the track's contribution to `∂/∂Θ` and to `∂/∂c_t` alike.
            let mu = match &blk.mask {
                Some(m) => mu.broadcast_mul(m)?,
                None => mu,
            };
            g = (g + mu.matmul(&dict.e_aug_t)?)?;
            if dict.dead_mass > 0.0 {
                let dead = theta
                    .narrow(1, h + dict.id, 1)?
                    .exp()?
                    .affine(dict.dead_mass, 0.0)?;
                let dead = match &blk.mask {
                    Some(m) => (dead * m)?,
                    None => dead,
                };
                g = (g + dead.broadcast_mul(&dict.intercept_mask)?)?;
            }
        }
        let g = (g - &ne)?;

        // AdamW with `weight_decay = 0` — the ridge is already in `g`, and a
        // decoupled decay would double-count it. The decayed, bias-corrected step
        // multiplier is [`adam_step_size`], shared with the single-partition loop.
        m = ((&m * BETA1)? + (&g * (1.0 - BETA1))?)?;
        v = ((&v * BETA2)? + (g.sqr()? * (1.0 - BETA2))?)?;
        let step_size = adam_step_size(a.lr0, step, POLISH_STEPS);
        theta = (&theta - (&m * step_size)?.broadcast_div(&(v.sqrt()? + EPS)?)?)?;

        steps = step + 1;
        if steps.is_multiple_of(CHECK_EVERY) {
            emitted = a.progress.advance(bc, steps, emitted);
            a.progress.describe(steps);
            let cur = theta.narrow(1, 0, h)?.contiguous()?;
            // One readback, not two: each is a blocking device→host copy.
            let ds = Tensor::stack(
                &[(&cur - &prev)?.sqr()?.sum_all()?, cur.sqr()?.sum_all()?],
                0,
            )?
            .to_vec1::<f32>()?;
            prev = cur;
            if ds[1] > 0.0 && f64::from(ds[0] / ds[1]).sqrt() < TOL {
                converged = true;
                break;
            }
        }
    }
    let loop_secs = loop_start.elapsed().as_secs_f64();
    a.progress.finish_block(bc, emitted);

    /////////////////////////////
    // Read back + diagnostics //
    /////////////////////////////

    // The deviance of every track's observed edges, through the same
    // [`poisson_deviance`] the single-partition loop reduces with. An absent cell
    // contributes nothing: its `n` is zero throughout, which zeroes both terms.
    let mut deviance = 0f64;
    let mut clamped = false;
    for (dict, blk) in a.dicts.iter().zip(&blocks) {
        if blk.n_edges == 0 {
            continue;
        }
        let s = theta.matmul(&dict.e_aug)?.broadcast_add(&dict.b_row)?;
        clamped |= s.max_all()?.to_scalar::<f32>()? >= SCORE_CLAMP as f32;
        // Two-sided here (unlike the loop): the deviance takes `ln(n/μ)`, so a
        // rate that underflowed to 0 would report an infinite one.
        let s = s.clamp(-SCORE_CLAMP, SCORE_CLAMP)?;
        deviance += poisson_deviance(&blk.n_t, &s)?;
    }

    // Split `Θ̃` back into the latent and one intercept column per track.
    let mut intercepts = Vec::with_capacity(n_tracks);
    for t in 0..n_tracks {
        intercepts.push(theta.narrow(1, h + t, 1)?.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok(TrackBlockOut {
        latent: theta
            .narrow(1, 0, h)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?,
        intercepts,
        steps,
        converged,
        clamped,
        deviance,
        n_edges,
        loop_secs,
    })
}
