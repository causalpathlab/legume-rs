//! Block **Poisson-MAP SGD** over a frozen dictionary (candle) — the shared solver
//! behind both the per-cell Phase 2 ([`super::cells`]) and the per-pseudobulk
//! velocity readout ([`super::pseudobulk`]); a "node" below is a cell or a pb
//! aggregate, whichever the caller passes.
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
//!
//! # Layout
//!
//! [`edges`] flattens the sampler's edges once per pass and sizes the blocks;
//! [`pass`] drives one pass (θ, or δ with θ fixed) block by block and owns the
//! per-block argument/result types; [`solve`] is the Adam loop for one block;
//! [`joint`] is the alternative single-solve θ+δ path. The tuning constants, the
//! caller-facing types and the entry point stay here.

use super::CellBatchDivisor;
use crate::progress::new_progress_bar;
use candle_util::candle_core::Device;
use log::info;

mod edges;
mod joint;
mod pass;
mod solve;

use edges::{block_partition, EdgeTable};
use joint::run_joint_pass;
use pass::{run_pass, PassSpec};

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
    /// Log prefix, so a caller reusing this solver reads honestly: `"Phase 2"`
    /// for the per-cell projection, `"pb velocity readout"` for the pseudobulk one.
    pub label: &'static str,
    /// Remove the population mean from the latents and report it in [`GaugeShift`].
    /// **Cells set this `true`**: the common mode must leave `θ` and be folded into
    /// `b_feat`, or it lands on the gene centroids and collapses marker annotation.
    /// **The pb readout sets it `false`**: its landmarks are never co-embedded, and
    /// the cell-lift differences cells against *raw* pb `θ` — so pb latents stay in
    /// the as-trained frame and nothing is folded (`GaugeShift` comes back zero).
    pub gauge_fix: bool,
    /// Joint θ+δ solve (β-sharing only). When `true` and `unspliced_rows` is given, one
    /// SGD estimates identity `θ` and velocity `δ` **together** — θ pulled by both the
    /// spliced and unspliced tracks — instead of the default sequential θ-then-δ (δ with
    /// θ held fixed). Ignored without `unspliced_rows`.
    pub joint: bool,
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
/// `cells` is the flattened per-node view `(global id, feature ids, counts)` — the
/// per-cell sampler flattening on the `super::cells` path, one entry per pb node on
/// the `super::pseudobulk` path. `batch_divisor`, when set, applies the
/// `μ_residual` fold-factor divide — **once**, while the edges are flattened,
/// rather than on every solve.
///
/// Without `unspliced_rows` (bge) there is one pass over every feature row. With
/// it (gem β-sharing) there are two: identity `θ` from the spliced edges with the
/// partition over spliced rows, then — holding `θ` fixed — the velocity increment
/// `δ` from the unspliced edges with the partition over unspliced rows. That
/// mirrors the retired analytical increment's semantics exactly:
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
    let edges_b =
        (!rows_b.is_empty()).then(|| EdgeTable::build(cells, &rows_b, n_features, batch_divisor));

    let blocks = block_partition(&rows_a, &rows_b);
    // The bar counts **cells**, across both passes, and advances *within* a block
    // in proportion to that block's Adam steps. Counting whole blocks would tick
    // maybe 16 times for an entire phase — and the better `Bc` gets for speed, the
    // coarser that becomes. Cells stay meaningful and the bar keeps moving.
    let bar = new_progress_bar((cells.len() * (1 + usize::from(blocks.two_pass))) as u64);
    bar.enable_steady_tick(std::time::Duration::from_millis(200));

    // Estimate θ (and, on the splice path, δ). Two modes:
    //   • **joint** (β-sharing + `input.joint`): one solve over both partitions with θ
    //     pulled by the spliced AND unspliced tracks ([`run_joint_pass`]);
    //   • **sequential** (default): identity θ from the spliced edges, then δ as a
    //     directed residual with θ held fixed.
    let (pass_a, pass_b) = match (&edges_b, input.joint) {
        (Some(eb), true) => {
            let (pa, pb) = run_joint_pass(input, &rows_a, &edges_a, &rows_b, eb, cells, &bar)?;
            (pa, Some(pb))
        }
        _ => {
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
            (pass_a, pass_b)
        }
    };
    bar.finish_and_clear();

    // Fix the gauge (see `GaugeShift`): remove the population mean from each
    // latent. Taken over the SOLVED cells only — a cell with no edges was never
    // placed by the likelihood, and after centring the origin is the population
    // mean, which is exactly where a no-information cell belongs.
    let (theta_mean, delta_mean) = if input.gauge_fix {
        let tm = mean_rows(&pass_a.latent, h);
        let dm = pass_b
            .as_ref()
            .map_or_else(Vec::new, |p| mean_rows(&p.latent, h));
        info!(
            "{} — gauge fix: removed ‖θ̄‖={:.3}{} from the latents into b_feat \
             (exact reparametrisation: every score is unchanged)",
            input.label,
            norm(&tm),
            if dm.is_empty() {
                String::new()
            } else {
                format!(", ‖δ̄‖={:.3}", norm(&dm))
            },
        );
        (tm, dm)
    } else {
        // pb readout: no re-gauge — nothing leaves the latents. Zero means make the
        // scatter a no-op; the caller ignores the reported (zero) `GaugeShift`.
        (vec![0f32; h], vec![0f32; h])
    };

    // Scatter the pass results (indexed by position in `cells`) back onto the
    // global cell axis. Cells the samplers never saw keep the zero row.
    let mut theta = vec![0f32; input.n_cells * h];
    let mut b_cell = vec![0f32; input.n_cells];
    let mut velocity = pass_b.as_ref().map(|_| vec![0f32; input.n_cells * h]);
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

#[cfg(test)]
#[path = "block_sgd_tests.rs"]
mod tests;
