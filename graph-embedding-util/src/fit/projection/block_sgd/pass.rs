//! One pass over one feature partition: the frozen augmented design `Ẽ`, the block
//! loop that solves every node against it, and the running stats and progress the
//! pass reports. Also the [`BlockArgs`]/[`BlockOut`] pair that is the whole
//! interface to [`super::solve`] — built here, filled there.
//!
//! The split between [`PassDict`] and [`run_pass`] is the load-bearing one. The
//! design matrix, the live/gate-folded split, the null normalizer, the learning
//! rate and the block size depend **only on the frozen dictionary** — not on which
//! nodes are being projected against it. Holding them apart is what lets a caller
//! stream nodes past one dictionary group by group
//! ([`crate::fit::projection::FrozenProjector`]) and size those groups for its own
//! memory, instead of large enough to hide a per-call setup.

use super::edges::{block_cells, EdgeTable};
use super::solve::solve_block;
use super::{Phase2Input, GATE_FOLD_EPS, GROUP_BLOCKS, MAX_STEPS, TARGET_DELTA_S};
use candle_util::candle_core::{Device, Tensor};
use log::info;

////////////////////////////////////
// The frozen design for one pass //
////////////////////////////////////

/// The dictionary side of a pass: what [`PassDict::build`] reads, and how the pass
/// names itself in the log.
pub(crate) struct DictSpec<'a> {
    /// Frozen dictionary, row-major `[n_features × h]`.
    pub(crate) feat: &'a [f32],
    /// Frozen feature bias, `[n_features]`.
    pub(crate) b_feat: &'a [f32],
    pub(crate) h: usize,
    /// Ridge `λ`. Reported here; *applied* per block from [`Phase2Input`], which is
    /// where the solve reads it.
    pub(crate) lambda: f64,
    pub(crate) dev: &'a Device,
    /// Log prefix — `"Phase 2"`, `"Projection"`, `"pb velocity readout"`.
    pub(crate) label: &'static str,
    /// This pass's own name within that: `"identity"`, `"velocity"`, `"nodes"`.
    pub(crate) pass: &'static str,
}

/// Everything a pass needs that depends **only** on the frozen dictionary and the
/// feature partition it runs over.
///
/// Every field below used to be rebuilt inside `run_pass`. That was free while the
/// only callers projected a whole run's cells in one go, and a per-group tax the
/// moment one started streaming: the transposed dictionary, the live-feature scan,
/// the `Σ exp(β)` reduction, the learning-rate estimate and an info line, once per
/// handful of cells. Built once here, a group is sized by what the *caller* can
/// hold rather than by what it has to amortize.
pub(crate) struct PassDict {
    /// The pass's feature partition on the global feature axis. **Owned**: the dict
    /// outlives the call that built it, so it cannot borrow the caller's rows.
    rows: Vec<u32>,
    /// Global → live-local feature id; `u32::MAX` for a gate-folded row.
    pub(super) to_live: Vec<u32>,
    /// Augmented dictionary `Ẽᵀ [H+1, F_live]` — the live rows transposed, with a
    /// **row of ones** appended so the per-cell intercept rides in as the last
    /// column of the parameter matrix rather than as a separate broadcast add. That
    /// removes a whole `[Bc, F]` op and, more to the point, makes the intercept's
    /// gradient fall out of the same matmul as the latent's (the ones row sums `μ`
    /// over features, which is exactly `∂/∂c`).
    pub(super) e_aug: Tensor,
    /// `Ẽ [F_live, H+1]`. Both orientations are needed every step — `Θ̃·Ẽᵀ` forward
    /// and `μ·Ẽ` for the gradient — so the transpose is materialized once here.
    pub(super) e_aug_t: Tensor,
    /// Frozen feature bias over the live rows, `[1, F_live]`.
    pub(super) b_row: Tensor,
    /// `[1, H+1]` selector with 1.0 in the intercept slot, for the terms that land
    /// on the intercept alone.
    pub(super) intercept_mask: Tensor,
    /// `ln(Σ_f exp(β_f) + dead_mass)` over this pass's rows — a per-pass constant,
    /// so it is summed here rather than over every live feature in every block.
    pub(super) null_log_norm: f64,
    /// Gate-folded partition mass `Σ_dead exp(β_f)`; 0 at the default
    /// [`GATE_FOLD_EPS`], where nothing is folded.
    pub(super) dead_mass: f64,
    /// Auto-scaled initial learning rate — a property of the dictionary's scale (see
    /// [`TARGET_DELTA_S`]), not of the nodes.
    pub(super) lr0: f64,
    /// Cells per block, from the activation budget and this pass's feature count.
    block_cells: usize,
    /// Carried so the block loop and any streaming caller report the same pass the
    /// dictionary was built for.
    label: &'static str,
    pass: &'static str,
}

impl PassDict {
    /// Build the frozen design for one pass over `rows`.
    pub(crate) fn build(spec: &DictSpec, rows: Vec<u32>) -> anyhow::Result<Self> {
        let (h, dev) = (spec.h, spec.dev);

        // Split the partition into the live rows (which go into the matmul) and the
        // gate-folded rows (whose partition mass is a constant).
        let mut live: Vec<u32> = Vec::with_capacity(rows.len());
        let mut dead_mass = 0f64;
        for &g in &rows {
            let e = &spec.feat[g as usize * h..(g as usize + 1) * h];
            if e.iter().map(|x| x * x).sum::<f32>().sqrt() > GATE_FOLD_EPS {
                live.push(g);
            } else {
                dead_mass += f64::from(spec.b_feat[g as usize]).exp();
            }
        }
        anyhow::ensure!(
            !live.is_empty(),
            "phase-2 {}: every feature in this pass is gate-folded — the frozen \
             dictionary carries no signal to project onto",
            spec.pass
        );

        let f_live = live.len();
        let d = h + 1;
        let mut e_aug = vec![0f32; d * f_live];
        let mut b_live = vec![0f32; f_live];
        for (l, &g) in live.iter().enumerate() {
            let row = &spec.feat[g as usize * h..(g as usize + 1) * h];
            for (k, &v) in row.iter().enumerate() {
                e_aug[k * f_live + l] = v;
            }
            e_aug[h * f_live + l] = 1.0; // intercept row
            b_live[l] = spec.b_feat[g as usize];
        }
        let e_aug = Tensor::from_vec(e_aug, (d, f_live), dev)?;
        let e_aug_t = e_aug.t()?.contiguous()?;
        let b_row = Tensor::from_vec(b_live, (1, f_live), dev)?;

        let null_log_norm = (f64::from(b_row.exp()?.sum_all()?.to_scalar::<f32>()?) + dead_mass)
            .max(f64::MIN_POSITIVE)
            .ln();

        let intercept_mask = {
            let mut v = vec![0f32; d];
            v[h] = 1.0;
            Tensor::from_vec(v, (1, d), dev)?
        };

        // Global → live-local feature id, for remapping the pass's edges.
        let mut to_live = vec![u32::MAX; spec.b_feat.len()];
        for (l, &g) in live.iter().enumerate() {
            to_live[g as usize] = l as u32;
        }

        // Auto-scale the learning rate to the dictionary. Adam's per-coordinate step
        // is ≈ lr, so a step moves the linear predictor by `Δs ≈ lr · H · rms(e)`;
        // pinning `Δs` makes the schedule invariant to the `β ↔ θ` scale duality that
        // otherwise leaves `lr` either useless or divergent.
        // `ẽ` is the ARITHMETIC mean |e|, not the RMS. A step `Δθ` moves gene `f`'s score
        // by `Δs_f = ⟨e_f, Δθ⟩ ≈ lr·h·E|e|`, so inverting THAT is what pins `Δs` at the
        // target. Using `√E[e²]` instead delivers `TARGET_DELTA_S/κ` with
        // `κ = √E[e²]/E|e| ≥ 1` — it calibrates on the loudest genes while most of a
        // cell's likelihood rides on the median one. Measured on real fits `κ` ran
        // 2.8–66, i.e. 2–35% of the intended step, and `cor(log κ, kNN purity) = −0.93`
        // over 12 fits: large `κ` ⇒ blocks hit the step cap ⇒ under-converged `θ`.
        // Both estimators are scale-free (a uniform `E → αE` cancels), so this is about
        // the TAIL, not the magnitude.
        let e_mean = {
            let sa: f64 = live
                .iter()
                .flat_map(|&g| &spec.feat[g as usize * h..(g as usize + 1) * h])
                .map(|x| f64::from(*x).abs())
                .sum();
            (sa / (f_live * h) as f64).max(1e-12)
        };
        let lr0 = TARGET_DELTA_S / (h as f64 * e_mean);

        // Sized from the WHOLE partition, not the live subset: the gate fold shrinks
        // the matmul but the budget it is bounding is the block's, and a dictionary
        // whose fold is disabled must land on the same `Bc`.
        let block_cells = block_cells(rows.len());

        info!(
            "{} [{}] — {f_live} live features (of {}; {} gate-folded), blocks of \
             {block_cells}, lr {:.4} (auto: Δs≈{TARGET_DELTA_S}), ≤{MAX_STEPS} steps, \
             ridge λ={}",
            spec.label,
            spec.pass,
            rows.len(),
            rows.len() - f_live,
            lr0,
            spec.lambda,
        );

        Ok(Self {
            rows,
            to_live,
            e_aug,
            e_aug_t,
            b_row,
            intercept_mask,
            null_log_norm,
            dead_mass,
            lr0,
            block_cells,
            label: spec.label,
            pass: spec.pass,
        })
    }

    /// The feature partition this dictionary was built over — what an [`EdgeTable`]
    /// must be flattened against for its local ids to line up.
    pub(crate) fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// Nodes a streaming caller should hand one [`super::project_prepared`] call.
    ///
    /// A whole number of blocks, so a caller that cuts its groups here never leaves
    /// the engine a short trailing block; [`GROUP_BLOCKS`] of them, so the per-call
    /// host work either side of the solve is spread over a run of blocks rather than
    /// one. Both bounds live here because both derive from `Bc`, which is derived in
    /// turn from an activation budget only this module can see — a caller guessing
    /// at it in its own currency is guessing against numbers it cannot read.
    pub(crate) fn group_nodes(&self) -> usize {
        self.block_cells * GROUP_BLOCKS
    }
}

//////////////////////
// One phase-2 pass //
//////////////////////

/// The per-call half of a pass: which nodes' edges, and what they are solved
/// against. The dictionary half is [`PassDict`].
pub(super) struct PassSpec<'a> {
    pub(super) edges: &'a EdgeTable,
    /// Fixed identity `θ` (host, `[n_kept × h]`) folded into the per-edge offset —
    /// `Some` only on the velocity pass.
    pub(super) base_theta: Option<&'a [f32]>,
}

/// One pass's per-cell result, indexed by position in `cells` (not by global id).
pub(super) struct PassOut {
    pub(super) latent: Vec<f32>,
    pub(super) intercept: Vec<f32>,
}

pub(super) fn run_pass(
    input: &Phase2Input,
    dict: &PassDict,
    spec: &PassSpec,
    cells: &[(u32, &[u32], &[f32])],
    bar: &indicatif::ProgressBar,
) -> anyhow::Result<PassOut> {
    let h = input.h;
    let n_kept = cells.len();
    let bc = dict.block_cells;

    let mut latent = vec![0f32; n_kept * h];
    let mut intercept = vec![0f32; n_kept];
    let mut stats = PassStats::default();
    let n_blocks = n_kept.div_ceil(bc);

    for (b, start) in (0..n_kept).step_by(bc).enumerate() {
        let end = (start + bc).min(n_kept);
        let block = solve_block(BlockArgs {
            input,
            dict,
            spec,
            start,
            end,
            progress: &BlockProgress {
                bar,
                stats: &stats,
                label: dict.pass,
                block: b + 1,
                n_blocks,
            },
        })?;
        latent[start * h..end * h].copy_from_slice(&block.latent);
        intercept[start..end].copy_from_slice(&block.intercept);
        stats.absorb(&block);
    }

    info!(
        "{} [{}] — {n_kept} node(s) done: ⌀{:.0} steps/block, {} of {} block(s) hit the \
         {MAX_STEPS}-step cap, mean per-edge deviance {:.4}, {:.0}s total ({:.1} ms/step){}",
        dict.label,
        dict.pass,
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
    Ok(PassOut { latent, intercept })
}

#[derive(Default)]
pub(super) struct PassStats {
    pub(super) blocks: usize,
    pub(super) steps: usize,
    pub(super) at_cap: usize,
    pub(super) clamped: usize,
    pub(super) dev_sum: f64,
    pub(super) dev_n: f64,
    pub(super) secs: f64,
}

impl PassStats {
    fn absorb(&mut self, b: &BlockOut) {
        self.blocks += 1;
        self.steps += b.steps;
        self.at_cap += usize::from(!b.converged);
        self.clamped += usize::from(b.clamped);
        self.dev_sum += b.deviance;
        self.dev_n += b.n_edges as f64;
        self.secs += b.loop_secs;
    }
    pub(super) fn ms_per_step(&self) -> f64 {
        1e3 * self.secs / self.steps.max(1) as f64
    }
    pub(super) fn mean_steps(&self) -> f64 {
        self.steps as f64 / self.blocks.max(1) as f64
    }
    pub(super) fn mean_deviance(&self) -> f64 {
        self.dev_sum / self.dev_n.max(1.0)
    }
}

/// The shared progress bar plus what the in-flight block needs to describe itself.
///
/// The bar counts **cells** across every pass that shares it — including, on the
/// streaming path, the caller's own bar across every group. A block advances it *as
/// it steps*, pro-rata on its step budget, so the bar keeps moving through a block
/// that takes hundreds of Adam steps instead of jumping once per block — with `Bc`
/// sized for speed a whole pass is only a handful of blocks, and per-block ticks
/// would be nearly no feedback at all.
pub(super) struct BlockProgress<'a> {
    pub(super) bar: &'a indicatif::ProgressBar,
    /// Stats from the blocks already finished in this pass.
    pub(super) stats: &'a PassStats,
    pub(super) label: &'static str,
    /// 1-based index of the block in flight, and how many this pass has.
    pub(super) block: usize,
    pub(super) n_blocks: usize,
}

impl BlockProgress<'_> {
    /// Advance the bar to the fraction of `bc` this block's `steps` have earned,
    /// given `emitted` cells already reported for it. Returns the new `emitted`.
    pub(super) fn advance(&self, bc: usize, steps: usize, emitted: usize) -> usize {
        let want = (bc * steps / MAX_STEPS).min(bc);
        if want > emitted {
            self.bar.inc((want - emitted) as u64);
        }
        want.max(emitted)
    }

    pub(super) fn describe(&self, steps: usize) {
        self.bar.set_message(format!(
            "{} · block {}/{} step {}/{} · {} at cap · dev {:.3}",
            self.label,
            self.block,
            self.n_blocks,
            steps,
            MAX_STEPS,
            self.stats.at_cap,
            self.stats.mean_deviance(),
        ));
    }

    /// Snap the bar to this block's exact end — a block that converged early has
    /// earned the rest of its cells.
    pub(super) fn finish_block(&self, bc: usize, emitted: usize) {
        self.bar.inc((bc - emitted) as u64);
    }
}

///////////////
// One block //
///////////////

pub(super) struct BlockArgs<'a> {
    pub(super) input: &'a Phase2Input<'a>,
    /// The frozen design every block of this pass shares.
    pub(super) dict: &'a PassDict,
    pub(super) spec: &'a PassSpec<'a>,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) progress: &'a BlockProgress<'a>,
}

pub(super) struct BlockOut {
    pub(super) latent: Vec<f32>,
    pub(super) intercept: Vec<f32>,
    pub(super) steps: usize,
    pub(super) converged: bool,
    pub(super) clamped: bool,
    pub(super) deviance: f64,
    pub(super) n_edges: usize,
    /// Wall time in the Adam loop. Reported as ms/step because that is the number
    /// that says whether a pass is arithmetic-bound or overhead-bound — the whole
    /// cost is `n_blocks × steps × (one step)`, so a regression shows up here and
    /// nowhere else.
    pub(super) loop_secs: f64,
}
