//! Phase-2 per-cell projection as a **cell-block Poisson SGD** (candle).
//!
//! With the feature side frozen, the phase-2 objective is *separable per cell*:
//! a cell's embedding depends only on its own edges. A **block** of cells is
//! therefore an independent optimization problem with its own parameters — which
//! is what lets every block carry its own convergence test, and what makes the
//! whole pass a sequence of two dense matmuls per step instead of a per-cell
//! Newton solve.
//!
//! # The objective
//!
//! For a block of `Bc` cells against the frozen dictionary `E [F, H]` / `β [F]`,
//! with `Θ̃ = [Θ | c]` and `Ẽᵀ = [Eᵀ ; 1]` carrying the per-cell intercept:
//!
//! ```text
//! S      = Θ̃_b · Ẽᵀ + β(bcast)                       [Bc, F]
//! loss   = Σ_{all f} exp(S)  −  Σ_{observed} n·S  +  (λ/2)‖Θ_b‖²
//! ∂/∂Θ̃  = (exp(S) − N)·Ẽ + λΘ̃                        [Bc, H+1]
//! ```
//!
//! The first term is the **full log-partition over every feature**, which the
//! Newton solver in [`crate::cell_projection`] explicitly gives up on ("the exact
//! softmax MLE would normalise over *all* features … at scale that's
//! infeasible"). It is what makes the MAP identified: fitting only the features a
//! cell expressed leaves `θ` free to inflate the rate of every feature it did
//! *not* express, bounded by nothing but the ridge — which is how `‖θ‖` became the
//! dominant axis of the embedding rather than biology. Here the partition is one
//! side of a matmul that is being computed anyway, so it is affordable.
//!
//! # The gradient is taken in closed form, not by `loss.backward()`
//!
//! `∂L/∂s = exp(s) − n`, and `s` is one matmul away from the parameters, so the
//! whole gradient is two matmuls and an `exp`. Autograd costs ~7× that here: every
//! candle op's backward materialises a full `[Bc, F]` `zeros_like` for **both**
//! operands before discarding the untracked one, so `Op::Matmul` computes the
//! frozen dictionary's gradient — a third full-size matmul, half the step's FLOPs —
//! and each `Op::Binary` does the same against the constant count matrix and the
//! broadcast bias. `.detach()` does not prevent this: it decides whether a *node*
//! is visited, and the node is visited because the parameter is.
//!
//! Two consequences shape the code. The intercept is carried as an extra column of
//! the parameter against a ones row of the dictionary, so it costs no separate
//! broadcast and its gradient falls out of the same matmul. And the **data term is
//! linear in the parameters**, so `∂/∂Θ̃ (Σ n·s) = −N·Ẽ` is a constant: it is
//! computed once per block and never enters the loop at all.
//!
//! `N` is still materialised densely (~80 % zeros) for that one product and for the
//! deviance, rather than gathered at the observed pairs. That is not a stylistic
//! choice: a gather's backward is `index_add`, whose CUDA kernel
//! (`candle-kernels/src/indexing.cu`) parallelises over `left_size * right_size`
//! and loops *serially* over the index list. Scattering into a flattened `[Bc·F]`
//! score makes both of those 1, so it runs on **one CUDA thread** walking millions
//! of indices — measured at ~1 s per step, ~400× off bandwidth, and invariant to
//! block size (bigger blocks carry proportionally more indices).
//!
//! # What is shared and what is per block
//!
//! `Ẽᵀ` **is** the design matrix. It is built once per pass (in both orientations,
//! since the forward and the gradient each need one) and every block matmuls against
//! the same tensors — candle's `Tensor(Arc<Tensor_>)` makes that a refcount bump.
//! Contrast the Newton path, which allocates a fresh `m × (H+1)` f64 design matrix
//! *per cell, per solve* whose every row is a copy of a row of `e_feat` that is
//! already in memory. Per block, only the parameters and the `[Bc, F]` activations
//! are new; a block's slice of the edge arrays is a `narrow`, not a copy.
//!
//! # Precision
//!
//! f32 throughout, matching the rest of the crate — and there was never f64
//! information to begin with (`e_feat`/`b_feat`/counts are all f32 upstream; the
//! Newton path widens them purely for its Cholesky's conditioning, which SGD does
//! not have).
//!
//! The loss value is never formed: only its gradient is, and that has no
//! cancellation (`exp(s) − n` per entry). Convergence therefore keys on
//! `‖ΔΘ‖/‖Θ‖` rather than on a loss difference — which would have been a poor
//! ruler anyway, since `partition − data` is a difference of two nearly-equal
//! ~1e7 numbers at the optimum, where an f32 ulp is ~1. The one reduction that is
//! read back, the reported deviance, is accumulated in f64 on the host.

use super::{cell_edges, CellBatchDivisor};
use crate::cell_projection::SCORE_CLAMP;
use crate::progress::new_progress_bar;
use candle_util::candle_core::{DType, Device, Tensor};
use log::info;

/////////////////////////
// Schedule / tolerance //
/////////////////////////

/// Adam steps a block may take before it is reported as un-converged. Blocks are
/// convex, so this is a backstop, not the normal exit — the run logs how many
/// blocks actually hit it.
const MAX_STEPS: usize = 400;

/// Converged when the relative parameter change `‖ΔΘ‖/‖Θ‖` over the last
/// [`CHECK_EVERY`] steps drops below this. A parameter criterion, not a loss
/// criterion: the loss is `partition − data`, which at the optimum is a difference
/// of two nearly-equal ~1e7 numbers where an f32 ulp is ~1.
const TOL: f64 = 1e-3;

/// Steps between convergence checks. Each check reads one scalar back from the
/// device, which forces a sync — cheap at this stride, a stall at every step.
const CHECK_EVERY: usize = 10;

/// Target per-step movement of the linear predictor `s`, used to auto-scale the
/// learning rate. Adam's per-coordinate step is ≈ `lr`, so `Δs ≈ lr · H · rms(e)`
/// and `lr = TARGET_DELTA_S / (H · rms(e))`. This matters: `‖β_g‖` is ~0.013 on
/// real fits, so `θ`'s natural scale is ~1/0.013 ≈ 77 and any fixed learning rate
/// is either hopelessly slow or divergent depending on the dictionary's scale.
const TARGET_DELTA_S: f64 = 0.05;

/// Learning rate floor as a fraction of the initial rate, decayed linearly across
/// a block's steps so it settles instead of dithering around the optimum.
const LR_FLOOR_FRAC: f64 = 0.05;

/// Activation budget per block. `Bc` is sized from this and the pass's feature
/// count so a block's `[Bc, F]` tensors stay bounded regardless of `F`.
///
/// Total work is `n_blocks × steps_per_block` Adam steps over `Bc × F` elements, so
/// the *arithmetic* is invariant to `Bc` — what a bigger block buys is amortizing
/// the fixed per-step cost (kernel launches, the optimizer's own bookkeeping, the
/// convergence sync) over more cells. Allocation is not part of that: cudarc uses
/// `cuMemAllocAsync`/`cuMemFreeAsync` against the driver's pool when the device
/// supports it, so same-shape buffers are recycled rather than re-`cudaMalloc`ed.
///
/// Bigger is therefore mildly better, up to memory — but only mildly. The thing
/// that actually made this pass slow was the gather backward described in the
/// module docs, which was *invariant* to `Bc`; do not expect this constant to
/// rescue a per-step regression.
const BLOCK_ACTIVATION_BYTES: usize = 1536 << 20;

/// Live `[Bc, F]` f32 tensors in flight at once: the dense count matrix `N` and the
/// velocity pass's offset (both held for the whole block), plus the step's `s`,
/// `μ` and one temporary. With the gradient taken in closed form there is no
/// retained autograd graph, so this is far lower than it would be for
/// `loss.backward()` — which materialises a full-size buffer per operand per op.
const LIVE_BLOCK_TENSORS: usize = 8;

/// Ceiling on `Bc` regardless of the budget: past this the per-step overhead is
/// already amortized and a bigger block only coarsens the progress bar and the
/// per-block convergence report.
const MAX_BLOCK_CELLS: usize = 4096;

/// A feature whose frozen `‖e_f‖` is at or below this contributes `exp(β_f + c)`
/// independent of `Θ`, so it is folded into a scalar partition mass and dropped
/// from the matmul entirely.
///
/// **Zero by default**, which makes the fold exact: only a feature the gate has
/// driven to a true zero is removed. Raising it turns the fold into an
/// approximation with a score error bounded by `eps · ‖θ‖`, which is why the
/// threshold and the number folded are logged rather than silent.
const GATE_FOLD_EPS: f32 = 0.0;

/////////////////
// Public entry //
/////////////////

/// Everything the phase-2 solve needs that is not per pass.
pub(crate) struct Phase2Input<'a> {
    /// Frozen dictionary, row-major `[n_features × h]`.
    pub feat: &'a [f32],
    /// Frozen feature bias, `[n_features]`.
    pub b_feat: &'a [f32],
    pub h: usize,
    pub n_cells: usize,
    /// Ridge `λ` on the cell latent.
    pub lambda: f64,
    pub dev: &'a Device,
}

/// Phase-2 result on the host, in global cell-id order.
pub(crate) struct Phase2Out {
    /// Identity `θ`, `[n_cells × h]` row-major, **gauge-fixed to mean zero** over
    /// the solved cells (see [`GaugeShift`]). Cells with no edges stay at the
    /// origin — which, after centring, *is* the population mean, i.e. the right
    /// "no information" position rather than an arbitrary corner of the space.
    pub theta: Vec<f32>,
    /// Fitted per-cell intercept, `[n_cells]`.
    pub b_cell: Vec<f32>,
    /// Velocity increment `δ`, `[n_cells × h]`, likewise mean-zero; `None` off the
    /// splice path.
    pub velocity: Option<Vec<f32>>,
    /// The means that were removed. The caller **must** fold these into `b_feat`
    /// or the model is changed rather than re-gauged.
    pub gauge: GaugeShift,
}

/// The population means removed from the latents, so the caller can put them back
/// into the per-feature bias where they belong.
///
/// # Why this exists
///
/// `s_cf = ⟨e_f, θ_c⟩ + β_f + c_c` has an exact gauge freedom: for any fixed `v`,
///
/// ```text
/// θ_c ← θ_c − v      β_f ← β_f + ⟨e_f, v⟩
/// ```
///
/// leaves **every score identical**. The likelihood cannot pin `v` at all; only the
/// ridge can, and `λ = 1` against a data term of ~10⁵ counts per cell does not.
/// Left alone, `θ` drifts far along that flat direction — measured at
/// `median cos(θ_c, θ̄) = 0.999`, i.e. every cell nearly collinear.
///
/// That is not merely cosmetic. Cell–cell distances are invariant to it (a shared
/// offset cancels), so kNN, Leiden and UMAP are unaffected — but
/// [`crate::postprocess::feature_coembedding`] places each *gene* at a weighted
/// average of *cell* positions, so the common mode lands on the gene side too and
/// collapses every marker centroid onto one point. Measured on the reference fit:
/// nearest-centroid marker assignment put **100 %** of cells on a single type.
///
/// Fixing `v = θ̄` also *lowers* `‖θ‖`, so the centred point satisfies the ridge
/// strictly better — this is not a gauge preference, it is the correct MAP on a
/// direction the objective is flat in.
pub(crate) struct GaugeShift {
    /// Mean identity `θ̄` removed from every solved cell, `[h]`.
    pub theta_mean: Vec<f32>,
    /// Mean increment `δ̄` removed on the splice path, `[h]`; empty otherwise.
    pub delta_mean: Vec<f32>,
}

/// Project every cell onto the frozen dictionary.
///
/// `cells` is the flattened sampler view (`super::collect_sampler_cells`):
/// `(global cell id, feature ids, counts)`. `batch_divisor`, when set, applies the
/// `μ_residual` fold-factor divide — **once**, while the edges are flattened,
/// rather than on every solve.
///
/// Without `unspliced_rows` (bge) there is one pass over every feature row. With
/// it (gem β-sharing) there are two: identity `θ` from the spliced edges with the
/// partition over spliced rows, then — holding `θ` fixed — the velocity increment
/// `δ` from the unspliced edges with the partition over unspliced rows. That
/// mirrors [`crate::cell_projection::solve_cell_increment`]'s semantics exactly:
/// `δ` is a directed residual in `θ`'s own frame, with its own throwaway
/// intercept, not a second independent projection.
pub(crate) fn project_cells(
    input: &Phase2Input,
    cells: &[(u32, &[u32], &[f32])],
    batch_divisor: Option<CellBatchDivisor>,
    unspliced_rows: Option<&[bool]>,
) -> anyhow::Result<Phase2Out> {
    let h = input.h;
    let n_features = input.b_feat.len();
    anyhow::ensure!(
        input.feat.len() == n_features * h,
        "phase-2: e_feat has {} entries, expected {n_features} × {h}",
        input.feat.len()
    );

    // Per-pass feature partitions on the global feature axis. One pass (all rows)
    // off the splice path; spliced / unspliced rows otherwise.
    let (rows_a, rows_b) = match unspliced_rows {
        None => ((0..n_features as u32).collect::<Vec<_>>(), Vec::new()),
        Some(un) => {
            anyhow::ensure!(
                un.len() == n_features,
                "phase-2: unspliced mask has {} entries, expected {n_features}",
                un.len()
            );
            let mut spliced = Vec::with_capacity(n_features);
            let mut unspl = Vec::with_capacity(n_features);
            for (f, &is_un) in un.iter().enumerate() {
                if is_un {
                    unspl.push(f as u32);
                } else {
                    spliced.push(f as u32);
                }
            }
            (spliced, unspl)
        }
    };

    // Flatten the sampler edges once, applying the batch divisor here so no solve
    // ever re-derives them. Grouped by the cell's position in `cells`.
    let edges_a = EdgeTable::build(cells, &rows_a, n_features, batch_divisor);
    let edges_b = (!rows_b.is_empty())
        .then(|| EdgeTable::build(cells, &rows_b, n_features, batch_divisor));

    let blocks = block_partition(&rows_a, &rows_b);
    // The bar counts **cells**, across both passes, and advances *within* a block
    // in proportion to that block's Adam steps. Counting whole blocks would tick
    // maybe 16 times for an entire phase — and the better `Bc` gets for speed, the
    // coarser that becomes. Cells stay meaningful and the bar keeps moving.
    let bar = new_progress_bar((cells.len() * (1 + usize::from(blocks.two_pass))) as u64);
    bar.enable_steady_tick(std::time::Duration::from_millis(200));

    // Pass 1 — identity θ (and the kept per-cell intercept).
    let pass_a = run_pass(
        input,
        &PassSpec {
            label: "identity",
            rows: &rows_a,
            edges: &edges_a,
            base_theta: None,
            block_cells: blocks.block_cells_a,
        },
        cells,
        &bar,
    )?;

    // Pass 2 — velocity increment δ, with θ held fixed.
    let pass_b = match &edges_b {
        Some(eb) => Some(run_pass(
            input,
            &PassSpec {
                label: "velocity",
                rows: &rows_b,
                edges: eb,
                base_theta: Some(&pass_a.latent),
                block_cells: blocks.block_cells_b,
            },
            cells,
            &bar,
        )?),
        None => None,
    };
    bar.finish_and_clear();

    // Fix the gauge (see `GaugeShift`): remove the population mean from each
    // latent. Taken over the SOLVED cells only — a cell with no edges was never
    // placed by the likelihood, and after centring the origin is the population
    // mean, which is exactly where a no-information cell belongs.
    let theta_mean = mean_rows(&pass_a.latent, h);
    let delta_mean = pass_b
        .as_ref()
        .map_or_else(Vec::new, |p| mean_rows(&p.latent, h));
    info!(
        "Phase 2 — gauge fix: removed ‖θ̄‖={:.3}{} from the latents into b_feat \
         (exact reparametrisation: every score is unchanged)",
        norm(&theta_mean),
        if delta_mean.is_empty() {
            String::new()
        } else {
            format!(", ‖δ̄‖={:.3}", norm(&delta_mean))
        },
    );

    // Scatter the pass results (indexed by position in `cells`) back onto the
    // global cell axis. Cells the samplers never saw keep the zero row.
    let mut theta = vec![0f32; input.n_cells * h];
    let mut b_cell = vec![0f32; input.n_cells];
    let mut velocity = pass_b
        .as_ref()
        .map(|_| vec![0f32; input.n_cells * h]);
    for (i, &(cell, _, _)) in cells.iter().enumerate() {
        let (g, l) = (cell as usize * h, i * h);
        for k in 0..h {
            theta[g + k] = pass_a.latent[l + k] - theta_mean[k];
        }
        b_cell[cell as usize] = pass_a.intercept[i];
        if let (Some(v), Some(p)) = (velocity.as_mut(), pass_b.as_ref()) {
            for k in 0..h {
                v[g + k] = p.latent[l + k] - delta_mean[k];
            }
        }
    }

    Ok(Phase2Out {
        theta,
        b_cell,
        velocity,
        gauge: GaugeShift {
            theta_mean,
            delta_mean,
        },
    })
}

/// Column means of a row-major `[n × h]` buffer.
fn mean_rows(v: &[f32], h: usize) -> Vec<f32> {
    let n = v.len() / h.max(1);
    if n == 0 {
        return vec![0f32; h];
    }
    let mut acc = vec![0f64; h];
    for row in v.chunks_exact(h) {
        for (a, x) in acc.iter_mut().zip(row) {
            *a += f64::from(*x);
        }
    }
    acc.iter().map(|a| (a / n as f64) as f32).collect()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

////////////////////////////
// Edge table (host, once) //
////////////////////////////

/// Every cell's edges restricted to one pass's feature partition, remapped to that
/// pass's local feature ids, grouped by cell position and flattened.
///
/// Built once per pass. A block takes a *slice* of these — no per-block copy, and
/// no per-cell `Vec` churn of the kind `cell_edges` does on the Newton path.
struct EdgeTable {
    /// `offsets[i]..offsets[i + 1]` is cell `i`'s slice of `feat`/`count`.
    offsets: Vec<usize>,
    /// Pass-local feature ids (index into the pass's live dictionary).
    feat: Vec<u32>,
    count: Vec<f32>,
}

impl EdgeTable {
    fn build(
        cells: &[(u32, &[u32], &[f32])],
        rows: &[u32],
        n_features: usize,
        batch_divisor: Option<CellBatchDivisor>,
    ) -> Self {
        // Global feature id → pass-local id, or `u32::MAX` when the feature is not
        // in this pass's partition.
        let mut local = vec![u32::MAX; n_features];
        for (l, &g) in rows.iter().enumerate() {
            local[g as usize] = l as u32;
        }
        let mut offsets = Vec::with_capacity(cells.len() + 1);
        let mut feat = Vec::new();
        let mut count = Vec::new();
        offsets.push(0);
        for &(cell, feats, counts) in cells {
            // The `μ_residual` divide happens here, once, instead of per solve.
            for (f, c) in cell_edges(cell, feats, counts, batch_divisor) {
                let l = local[f as usize];
                if l != u32::MAX {
                    feat.push(l);
                    count.push(c);
                }
            }
            offsets.push(feat.len());
        }
        Self {
            offsets,
            feat,
            count,
        }
    }

    fn cell_slice(&self, i: usize) -> (&[u32], &[f32]) {
        let (s, e) = (self.offsets[i], self.offsets[i + 1]);
        (&self.feat[s..e], &self.count[s..e])
    }
}

//////////////////////
// Block partition //
//////////////////////

struct BlockPlan {
    block_cells_a: usize,
    block_cells_b: usize,
    two_pass: bool,
}

/// Size `Bc` per pass so a block's `[Bc, F_pass]` activations stay inside
/// [`BLOCK_ACTIVATION_BYTES`].
fn block_partition(rows_a: &[u32], rows_b: &[u32]) -> BlockPlan {
    BlockPlan {
        block_cells_a: block_cells(rows_a.len()),
        block_cells_b: block_cells(rows_b.len()),
        two_pass: !rows_b.is_empty(),
    }
}

/// Cells per block for a pass over `f` live features.
fn block_cells(f: usize) -> usize {
    if f == 0 {
        return 1;
    }
    (BLOCK_ACTIVATION_BYTES / (f * 4 * LIVE_BLOCK_TENSORS)).clamp(1, MAX_BLOCK_CELLS)
}

/////////////////////
// One phase-2 pass //
/////////////////////

struct PassSpec<'a> {
    label: &'static str,
    /// This pass's feature partition on the global feature axis.
    rows: &'a [u32],
    edges: &'a EdgeTable,
    /// Fixed identity `θ` (host, `[n_kept × h]`) folded into the per-edge offset —
    /// `Some` only on the velocity pass.
    base_theta: Option<&'a [f32]>,
    block_cells: usize,
}

/// One pass's per-cell result, indexed by position in `cells` (not by global id).
struct PassOut {
    latent: Vec<f32>,
    intercept: Vec<f32>,
}

fn run_pass(
    input: &Phase2Input,
    spec: &PassSpec,
    cells: &[(u32, &[u32], &[f32])],
    bar: &indicatif::ProgressBar,
) -> anyhow::Result<PassOut> {
    let (h, dev) = (input.h, input.dev);
    let n_kept = cells.len();

    // Split this pass's partition into the live rows (which go into the matmul)
    // and the gate-folded rows (whose partition mass is a constant).
    let mut live: Vec<u32> = Vec::with_capacity(spec.rows.len());
    let mut dead_mass = 0f64;
    for &g in spec.rows {
        let e = &input.feat[g as usize * h..(g as usize + 1) * h];
        if e.iter().map(|x| x * x).sum::<f32>().sqrt() > GATE_FOLD_EPS {
            live.push(g);
        } else {
            dead_mass += f64::from(input.b_feat[g as usize]).exp();
        }
    }
    anyhow::ensure!(
        !live.is_empty(),
        "phase-2 {}: every feature in this pass is gate-folded — the frozen \
         dictionary carries no signal to project onto",
        spec.label
    );

    // Frozen, shared by every block: the transposed live dictionary, and its bias.
    //
    // **Augmented with a row of ones**, `Ẽᵀ = [Eᵀ ; 1]` of shape `[H+1, F]`, so the
    // per-cell intercept rides in as the last column of the parameter matrix rather
    // than as a separate broadcast add. That removes a whole `[Bc, F]` op (and, more
    // to the point, makes the intercept's gradient fall out of the same matmul as
    // the latent's — the ones row sums `μ` over features, which is exactly `∂/∂c`).
    let f_live = live.len();
    let d = h + 1;
    let mut e_aug = vec![0f32; d * f_live];
    let mut b_live = vec![0f32; f_live];
    for (l, &g) in live.iter().enumerate() {
        let row = &input.feat[g as usize * h..(g as usize + 1) * h];
        for (k, &v) in row.iter().enumerate() {
            e_aug[k * f_live + l] = v;
        }
        e_aug[h * f_live + l] = 1.0; // intercept row
        b_live[l] = input.b_feat[g as usize];
    }
    // Both orientations are needed every step — `Θ̃·Ẽᵀ` forward and `μ·Ẽ` for the
    // gradient — so materialize the transpose once here rather than per step.
    let e_aug = Tensor::from_vec(e_aug, (d, f_live), dev)?;
    let e_aug_t = e_aug.t()?.contiguous()?;
    let b_row = Tensor::from_vec(b_live, (1, f_live), dev)?;

    // `Σ_f exp(β_f)` over the live rows: a per-PASS constant, so it is summed here
    // rather than over every live feature in every block.
    let null_log_norm = (f64::from(b_row.exp()?.sum_all()?.to_scalar::<f32>()?) + dead_mass)
        .max(f64::MIN_POSITIVE)
        .ln();

    // `[1, H+1]` selector for the intercept slot.
    let intercept_mask = {
        let mut v = vec![0f32; d];
        v[h] = 1.0;
        Tensor::from_vec(v, (1, d), dev)?
    };

    // Global → live-local feature id, for remapping the pass's edges.
    let mut to_live = vec![u32::MAX; input.b_feat.len()];
    for (l, &g) in live.iter().enumerate() {
        to_live[g as usize] = l as u32;
    }

    // Auto-scale the learning rate to the dictionary. Adam's per-coordinate step
    // is ≈ lr, so a step moves the linear predictor by `Δs ≈ lr · H · rms(e)`;
    // pinning `Δs` makes the schedule invariant to the `β ↔ θ` scale duality that
    // otherwise leaves `lr` either useless or divergent.
    let e_rms = {
        let ss: f64 = live
            .iter()
            .flat_map(|&g| &input.feat[g as usize * h..(g as usize + 1) * h])
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum();
        (ss / (f_live * h) as f64).sqrt().max(1e-12)
    };
    let lr0 = TARGET_DELTA_S / (h as f64 * e_rms);

    info!(
        "Phase 2 [{}] — {n_kept} cells × {f_live} live features (of {}; {} gate-folded), \
         blocks of {}, lr {:.4} (auto: Δs≈{TARGET_DELTA_S}), ≤{MAX_STEPS} steps, ridge λ={}",
        spec.label,
        spec.rows.len(),
        spec.rows.len() - f_live,
        spec.block_cells,
        lr0,
        input.lambda,
    );

    let mut latent = vec![0f32; n_kept * h];
    let mut intercept = vec![0f32; n_kept];
    let mut stats = PassStats::default();
    let n_blocks = n_kept.div_ceil(spec.block_cells);

    for (b, start) in (0..n_kept).step_by(spec.block_cells).enumerate() {
        let end = (start + spec.block_cells).min(n_kept);
        let block = solve_block(BlockArgs {
            input,
            spec,
            to_live: &to_live,
            e_aug: &e_aug,
            e_aug_t: &e_aug_t,
            b_row: &b_row,
            intercept_mask: &intercept_mask,
            null_log_norm,
            dead_mass,
            lr0,
            start,
            end,
            progress: &BlockProgress {
                bar,
                stats: &stats,
                label: spec.label,
                block: b + 1,
                n_blocks,
            },
        })?;
        latent[start * h..end * h].copy_from_slice(&block.latent);
        intercept[start..end].copy_from_slice(&block.intercept);
        stats.absorb(&block);
    }

    info!(
        "Phase 2 [{}] — done: ⌀{:.0} steps/block, {} of {} block(s) hit the {MAX_STEPS}-step cap, \
         mean per-edge deviance {:.4}, {:.0}s total ({:.1} ms/step){}",
        spec.label,
        stats.mean_steps(),
        stats.at_cap,
        stats.blocks,
        stats.mean_deviance(),
        stats.secs,
        stats.ms_per_step(),
        if stats.clamped > 0 {
            format!(" [WARNING: score clamp bound on {} block(s)]", stats.clamped)
        } else {
            String::new()
        },
    );
    Ok(PassOut { latent, intercept })
}

#[derive(Default)]
struct PassStats {
    blocks: usize,
    steps: usize,
    at_cap: usize,
    clamped: usize,
    dev_sum: f64,
    dev_n: f64,
    secs: f64,
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
    fn ms_per_step(&self) -> f64 {
        1e3 * self.secs / self.steps.max(1) as f64
    }
    fn mean_steps(&self) -> f64 {
        self.steps as f64 / self.blocks.max(1) as f64
    }
    fn mean_deviance(&self) -> f64 {
        self.dev_sum / self.dev_n.max(1.0)
    }
}

/// The shared progress bar plus what the in-flight block needs to describe itself.
///
/// The bar counts **cells** across both passes. A block advances it *as it steps*,
/// pro-rata on its step budget, so the bar keeps moving through a block that takes
/// hundreds of Adam steps instead of jumping once per block — with `Bc` sized for
/// speed a whole pass is only a handful of blocks, and per-block ticks would be
/// nearly no feedback at all.
struct BlockProgress<'a> {
    bar: &'a indicatif::ProgressBar,
    /// Stats from the blocks already finished in this pass.
    stats: &'a PassStats,
    label: &'static str,
    /// 1-based index of the block in flight, and how many this pass has.
    block: usize,
    n_blocks: usize,
}

impl BlockProgress<'_> {
    /// Advance the bar to the fraction of `bc` this block's `steps` have earned,
    /// given `emitted` cells already reported for it. Returns the new `emitted`.
    fn advance(&self, bc: usize, steps: usize, emitted: usize) -> usize {
        let want = (bc * steps / MAX_STEPS).min(bc);
        if want > emitted {
            self.bar.inc((want - emitted) as u64);
        }
        want.max(emitted)
    }

    fn describe(&self, steps: usize) {
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
    fn finish_block(&self, bc: usize, emitted: usize) {
        self.bar.inc((bc - emitted) as u64);
    }
}

//////////////////
// One block //
//////////////////

struct BlockArgs<'a> {
    input: &'a Phase2Input<'a>,
    spec: &'a PassSpec<'a>,
    to_live: &'a [u32],
    /// Augmented dictionary `Ẽᵀ [H+1, F]` (ones row last) and its transpose
    /// `Ẽ [F, H+1]` — both built once per pass, shared by every block.
    e_aug: &'a Tensor,
    e_aug_t: &'a Tensor,
    /// Frozen feature bias as a `[1, F]` row, shared by every block.
    b_row: &'a Tensor,
    /// `[1, H+1]` selector with 1.0 in the intercept slot, for the terms that land
    /// on the intercept alone.
    intercept_mask: &'a Tensor,
    /// `ln(Σ_f exp(β_f) + dead_mass)` — a per-pass constant.
    null_log_norm: f64,
    dead_mass: f64,
    lr0: f64,
    start: usize,
    end: usize,
    progress: &'a BlockProgress<'a>,
}

struct BlockOut {
    latent: Vec<f32>,
    intercept: Vec<f32>,
    steps: usize,
    converged: bool,
    clamped: bool,
    deviance: f64,
    n_edges: usize,
    /// Wall time in the Adam loop. Reported as ms/step because that is the number
    /// that says whether a pass is arithmetic-bound or overhead-bound — the whole
    /// cost is `n_blocks × steps × (one step)`, so a regression shows up here and
    /// nowhere else.
    loop_secs: f64,
}

fn solve_block(a: BlockArgs) -> anyhow::Result<BlockOut> {
    let (h, dev) = (a.input.h, a.input.dev);
    let bc = a.end - a.start;
    let d = h + 1; // [Θ | c]
    let f_live = a.b_row.dim(1)?;
    let (e_aug, e_aug_t, intercept_mask) = (a.e_aug, a.e_aug_t, a.intercept_mask);

    ///////////////////////////////////////
    // Gather the block's observed edges //
    ///////////////////////////////////////

    // Flat index into the block's `[Bc, F]` score matrix, so the data term is a
    // dense product against the score. See the module docs: the gather's backward
    // (`index_add`) degenerates to a single-threaded serial walk of the index list
    // on CUDA, so it is far cheaper to carry the zeros.
    let mut n_dense = vec![0f32; bc * f_live];
    // Per-cell total count, for the null-model intercept; and the folded-feature
    // count mass, which still owes `−n·c` to the loss.
    let mut n_tot = vec![0f64; bc];
    let mut n_dead = vec![0f32; bc];
    let mut n_edges = 0usize;
    for i in a.start..a.end {
        let local = i - a.start;
        let (feats, counts) = a.spec.edges.cell_slice(i);
        for (&f_pass, &n) in feats.iter().zip(counts) {
            // `edges` are indexed on the pass partition; map to the live subset.
            let g = a.spec.rows[f_pass as usize];
            let l = a.to_live[g as usize];
            n_tot[local] += f64::from(n);
            if l == u32::MAX {
                n_dead[local] += n;
            } else {
                n_dense[local * f_live + l as usize] = n;
                n_edges += 1;
            }
        }
    }
    let n_t = Tensor::from_vec(n_dense, (bc, f_live), dev)?.detach();
    let n_dead_t = Tensor::from_vec(n_dead, bc, dev)?;

    ///////////////////////////////////////////////////////
    // Fixed per-edge offset, materialized once per block //
    ///////////////////////////////////////////////////////

    // Everything in the score that does not depend on the trainable parameters,
    // pre-added into ONE `[Bc, F]` tensor: the frozen feature bias `β`, plus (on
    // the velocity pass) the frozen identity contribution `⟨e_f, θ_c⟩`.
    //
    // `⟨e_f, θ_c⟩` is what makes `δ` a directed residual in `θ`'s own frame rather
    // than a second projection, and it is constant across the Adam loop, so it is
    // one matmul *outside* it. Fusing `β` in here too costs nothing extra and
    // removes a broadcast add from every step — per-op overhead is the binding
    // constraint (see `BLOCK_ACTIVATION_BYTES`), so op count is worth spending
    // block memory on.
    // `[Bc, F]` on the velocity pass (it carries a per-cell term), but only the
    // `[1, F]` bias row on the identity pass — where materializing the broadcast
    // would cost a whole extra block-sized tensor and an extra block-sized read on
    // every step, for no fewer ops. `score` broadcast-adds either shape.
    let offset = match a.spec.base_theta {
        Some(theta) => {
            let t = Tensor::from_slice(&theta[a.start * h..a.end * h], (bc, h), dev)?;
            // Only the latent rows of `Ẽᵀ` — the ones row is the intercept's, and
            // the fixed identity carries no intercept of its own.
            t.matmul(&e_aug.narrow(0, 0, h)?.contiguous()?)?
                .broadcast_add(a.b_row)?
        }
        None => a.b_row.clone(),
    };

    ///////////////////////////////
    // Null-model initialisation //
    ///////////////////////////////

    // Θ = 0 and the exact intercept at Θ = 0:
    //     c = ln(Σ_f n_cf) − ln(Σ_f exp(offset_cf))
    // so step 1 already sits at the right depth and the optimiser only has to
    // learn the deviation. (The Newton path starts from a randn `e_cell` and a
    // zero intercept, which is why its first steps have to move so far.)
    let log_norm: Vec<f64> = match a.spec.base_theta {
        // Velocity pass: `offset` already carries `⟨e_f, θ⟩ + β`, so sum it there.
        Some(_) => offset
            .clamp(-SCORE_CLAMP, SCORE_CLAMP)?
            .exp()?
            .sum(1)?
            .to_vec1::<f32>()?
            .iter()
            .map(|x| f64::from(*x).max(f64::MIN_POSITIVE).ln())
            .collect(),
        // Identity pass: Θ = 0 ⇒ every row shares the same Σ_f exp(β_f), hoisted to
        // the pass rather than re-summed over every live feature in every block.
        None => vec![a.null_log_norm; bc],
    };
    // Θ̃ = [Θ | c] — the latent and its intercept in ONE `[Bc, H+1]` parameter, with
    // the intercept initialised at the null model and the latent at zero.
    let mut theta = vec![0f32; bc * d];
    for (i, (&n, &lz)) in n_tot.iter().zip(&log_norm).enumerate() {
        theta[i * d + h] = if n > 0.0 {
            (n.ln() - lz).clamp(-SCORE_CLAMP, SCORE_CLAMP) as f32
        } else {
            -SCORE_CLAMP as f32
        };
    }
    let mut theta = Tensor::from_vec(theta, (bc, d), dev)?;

    ////////////////////////////////////
    // Loop-invariant gradient pieces //
    ////////////////////////////////////

    // **The data term is LINEAR in the parameters**, so its gradient is a constant:
    // `Σ_cf n_cf·s_cf` with `s = Θ̃·Ẽᵀ + offset` gives `∂/∂Θ̃ = −N·Ẽ`, computed once
    // here instead of rebuilt (and back-propagated through) every step. The
    // intercept column of `N·Ẽ` is `Σ_f n_cf`, so the folded rows' `−n·c` term folds
    // straight into it.
    let ne = {
        let mut ne = n_t.matmul(e_aug_t)?; // [Bc, H+1]
        // Folded rows still owe `−n_dead·c`, which lands on the intercept column.
        if a.dead_mass > 0.0 {
            ne = (ne + n_dead_t.reshape((bc, 1))?.broadcast_mul(intercept_mask)?)?;
        }
        ne
    };
    // Ridge applies to the latent only — the intercept is unpenalised, so the last
    // entry of the row is zero and one broadcast multiply covers both.
    let lam_row = {
        let mut v = vec![a.input.lambda as f32; d];
        v[h] = 0.0;
        Tensor::from_vec(v, (1, d), dev)?
    };

    ///////////////////
    // The Adam loop //
    ///////////////////

    // Hand-rolled gradient, hand-rolled Adam — **not** `loss.backward()`.
    //
    // The objective's gradient is three lines (`∂L/∂s = exp(s) − n`, chain through
    // one matmul), and taking it directly instead of through autograd is worth ~7×
    // here. candle's backward for this graph emits ~29 extra `[Bc, F]`-sized kernels
    // on top of the 8 the forward needs, because every op's backward materialises a
    // full-size `zeros_like` for BOTH operands and only afterwards discards the one
    // that isn't tracked — `Op::Matmul` computes the frozen dictionary's gradient
    // (a third full matmul, 50 % of the step's FLOPs) and `Op::Binary` does the same
    // for the constant count matrix and the broadcast bias. `.detach()` does not
    // prevent any of it: it only controls whether the *node* is visited, and the
    // node is visited because the parameter is.
    //
    // `GradStore::new` is private, so candle's `AdamW` cannot be driven from a
    // hand-built gradient; Adam on a `[Bc, H+1]` parameter is a handful of
    // elementwise ops on a tensor ~30 000× smaller than the score block, so it is
    // cheaper to write than to work around.
    let (beta1, beta2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut m = Tensor::zeros((bc, d), DType::F32, dev)?;
    let mut v = Tensor::zeros((bc, d), DType::F32, dev)?;

    // Convergence is `‖ΔΘ‖/‖Θ‖` over the LATENT columns, so the intercept — which
    // starts at the null model and barely moves — cannot mask a latent still in
    // flight.
    let mut prev = theta.narrow(1, 0, h)?.contiguous()?;
    let mut steps = 0usize;
    let mut converged = false;
    let mut emitted = 0usize; // cells this block has already reported to the bar
    let loop_start = std::time::Instant::now();
    for step in 0..MAX_STEPS {
        // Linear decay to a floor so the block settles rather than dithers.
        let frac = step as f64 / MAX_STEPS as f64;
        let lr = a.lr0 * (1.0 - frac * (1.0 - LR_FLOOR_FRAC));

        // Upper bound only. `exp` overflows f32 at 88 so the ceiling is a real
        // guard; the floor is not, since `exp(−large)` underflowing to 0 is the
        // right answer for a feature the cell does not express.
        let s = theta
            .matmul(e_aug)?
            .broadcast_add(&offset)?
            .minimum(SCORE_CLAMP)?;
        let mu = s.exp()?;

        // ∂L/∂Θ̃ = (μ − N)·Ẽ + λΘ̃, with the constant `N·Ẽ` hoisted above. The
        // intercept column comes out of the same matmul via `Ẽ`'s ones row, and
        // picks up the gate-folded partition mass `exp(c)·Σ_dead exp(β_f)`.
        let mut g = ((mu.matmul(e_aug_t)? - &ne)? + theta.broadcast_mul(&lam_row)?)?;
        if a.dead_mass > 0.0 {
            let dead = theta.narrow(1, h, 1)?.exp()?.affine(a.dead_mass, 0.0)?;
            g = (g + dead.broadcast_mul(intercept_mask)?)?;
        }

        // AdamW with `weight_decay = 0` — the ridge is already in `g` above, and a
        // decoupled decay would double-count it.
        let t = (step + 1) as f64;
        m = ((&m * beta1)? + (&g * (1.0 - beta1))?)?;
        v = ((&v * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
        let step_size = lr * (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));
        theta = (&theta - (&m * step_size)?.broadcast_div(&(v.sqrt()? + eps)?)?)?;

        steps = step + 1;
        // Converged on `‖ΔΘ‖/‖Θ‖` — a parameter criterion, immune to the
        // `partition − data` cancellation that makes the loss a poor ruler here.
        //
        // The bar is advanced on this same stride, not per step: this is where the
        // loop already pays a device sync, so the update rides along for free, and
        // ~`MAX_STEPS/CHECK_EVERY` ticks per block is plenty of motion without
        // hammering the bar's lock and reformatting its message 400 times.
        if steps.is_multiple_of(CHECK_EVERY) {
            emitted = a.progress.advance(bc, steps, emitted);
            a.progress.describe(steps);
            let cur = theta.narrow(1, 0, h)?.contiguous()?;
            // One readback, not two: each `to_scalar` is a blocking device→host copy.
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

    let s = theta.matmul(e_aug)?.broadcast_add(&offset)?;
    // One scalar, not the `[Bc, F]` block: did the overflow guard ever bind?
    let clamped = s.max_all()?.to_scalar::<f32>()? >= SCORE_CLAMP as f32;
    // Two-sided here (unlike the training loop): the deviance takes `ln(n/μ)`, so a
    // rate that underflowed to 0 would report an infinite one.
    let s = s.clamp(-SCORE_CLAMP, SCORE_CLAMP)?;
    // Poisson deviance over the observed edges, `2·[ n·ln(n/μ) − (n − μ) ]`, reduced
    // on device so the block's fitted values never cross the bus.
    //
    // Computed densely against `N` like the data term, which needs the unobserved
    // entries — where `n = 0` — to contribute nothing: `n·ln(n/μ)` has the
    // `n log n → 0` limit there, and `−(n − μ)` is not part of a deviance taken over
    // observed edges only. `n.max(1)` inside the log keeps `ln 0` out of the graph,
    // and multiplying by `n` zeroes the term anyway; the mask does the same for the
    // second piece.
    let deviance = if n_edges > 0 {
        let mu = s.exp()?;
        let mask = n_t.gt(0f32)?.to_dtype(DType::F32)?;
        let log_n = n_t.clamp(1f32, f32::MAX)?.log()?;
        let term = ((&n_t * (log_n - &s)?)? - ((&n_t - &mu)? * &mask)?)?;
        f64::from(term.sum_all()?.to_scalar::<f32>()?) * 2.0
    } else {
        0.0
    };

    // Split `Θ̃` back into the latent and its intercept column.
    Ok(BlockOut {
        latent: theta.narrow(1, 0, h)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?,
        intercept: theta.narrow(1, h, 1)?.flatten_all()?.to_vec1::<f32>()?,
        steps,
        converged,
        clamped,
        deviance,
        n_edges,
        loop_secs,
    })
}

#[cfg(test)]
#[path = "cell_sgd_tests.rs"]
mod tests;
