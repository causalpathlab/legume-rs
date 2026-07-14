//! Post-hoc projection of **held-out features** onto the frozen pseudobulk
//! embedding.
//!
//! `faba gem --n-hvg 5000` trains a good cell embedding, but only those 5000
//! genes ever get a `β_g`. Raising `--n-hvg` is not the fix — low-information
//! genes then shape `E_feat` (and therefore `θ`) and cell-type spread degrades.
//! Instead, keep training on the HVG subset and afterwards solve `β_g` for the
//! held-out genes against the embedding that already exists. Nothing on the cell
//! side is read or written, so the cell output is unchanged.
//!
//! **Why the pseudobulk side is the right anchor.** With the default
//! `phase1_cells_per_pb == 0` the phase-1 cell axis is suppressed entirely, so
//! `β` is shaped *only* by the per-level pseudobulk axes, combined with
//! `CompositeMode::Sum` at uniform `DEFAULT_AXIS_LAMBDA`. Stacking every level's
//! `θ_pb` into one frozen table therefore reconstructs the exact objective `β`
//! was trained under. Cells enter only at phase 2, against an already-frozen
//! feature side, and are irrelevant here.
//!
//! **The solve is the phase-2 solve, transposed.** [`solve_one_cell`] takes an
//! `(index, count)` edge list against a frozen table; it does not care whether
//! the frozen table holds features or pseudobulks. Feeding it a gene's per-pb
//! counts with `frozen_e = θ_pb`, `frozen_b = b_pb` returns `(β_g, b_g)` under
//! `μ_p = exp(⟨β_g, θ_p⟩ + b_p + b_g)`. Likewise [`solve_cell_increment`] with
//! `e_base = β_g` returns the splice offset `δ_g` from the unspliced edges. No
//! new math lives in this module — only bookkeeping and calibration.
//!
//! **Calibration.** Trained `β` came out of NCE-SGD; a projected `β̂` comes out
//! of a Poisson MAP. Re-solving the *trained* genes with the identical solver
//! gives paired `(β̂, β)`, from which a ridge `H×H` map `M` is fit and applied to
//! the held-out genes. One map absorbs the NCE-vs-MAP discrepancy, the
//! mean-vs-sum count scale, and any residual axis mismatch at once. `M ≈ I`
//! means the frames already agreed; a low `mean_cosine` means they did not, and
//! the projection should be distrusted rather than patched.
//!
//! **Null gate.** A held-out gene with no cell-state structure is zeroed, not
//! given a direction. The test is a likelihood ratio against an intercept-only
//! fit — see [`FeatureProjectionConfig::null_fdr`] for why `‖β̂‖²` is the wrong
//! statistic on this side even though it is the right one for trained genes.
//!
//! **How well does it work?** On simulated data, comparing each gene's cell-loading
//! profile `θ·β_g` (what `faba annotate` scores, and invariant to the frame) against
//! a run where every gene *was* trained: projected genes recover ≈60% of the
//! above-chance agreement that trained genes achieve, ≈65% among high-contrast
//! genes. Useful, and clearly not a free lunch — a projected `β_g` is a weaker
//! estimate than a trained one, which is what `lrt` / `live` are there to expose.

use crate::cell_projection::{solve_cell_increment, solve_one_cell};
use crate::null_call::chi2_null_call;
use log::{info, warn};
use nalgebra::{DMatrix, DVector, RowDVector};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

/// Relative margin above a pseudobulk column's floor at which a gene counts as
/// "detected" there. The collapse emits Gamma posterior *means*, so an absent
/// gene still reads `a0 / (b0 + size_s) > 0`; the column minimum over backend
/// rows is that floor.
const DETECTION_MARGIN: f32 = 1e-3;

/// Below this many live trained genes the `H×H` map is under-determined, so the
/// calibration falls back to the identity and says so. `H` pairs would exactly
/// determine `M`; require a few times that for a stable ridge fit.
const MIN_CALIB_GENES_PER_DIM: usize = 3;

/// Default ridge `λ` on the per-gene Poisson-MAP solve.
///
/// Deliberately 100× looser than the phase-2 cell ridge
/// ([`crate::fit::projection::PHASE2_RIDGE`] `= 1.0`). A cell is solved from its
/// own sparse edge list; a gene is solved from ~1350 dense pseudobulk observations
/// against `H ≤ 32` parameters, so it needs far less regularization. A paired
/// ablation — one fixed trained `β`, ridge the only knob — shows agreement with the
/// trained frame rising monotonically as `λ` falls, saturating below `0.01`:
///
/// | `λ`  | weighted cosine | `R²` |
/// |------|-----------------|------|
/// | 1.0  | 0.624           | 0.29 |
/// | 0.01 | 0.674           | 0.35 |
/// | 1e-3 | 0.692           | 0.37 |
///
/// `0.01` takes most of that gain while keeping the `(H+1)×(H+1)` Newton solve
/// well-conditioned for genes whose counts sit near the pseudobulk floor: those
/// carry almost no Fisher information, and `λ → 0` lets `β̂` run off toward whatever
/// direction the noise points. The LRT null gate zeroes such genes anyway, but a
/// bounded solve is the cheaper guarantee.
pub const DEFAULT_PROJECTION_RIDGE: f64 = 0.01;

/// Default ridge `γ` on the `H×H` map carrying projected `β̂` into the trained frame.
pub const DEFAULT_PROJECTION_CALIB_RIDGE: f64 = 1.0;

/// Factor by which a gene's ridge is raised when its Poisson-MAP solve diverges.
const RIDGE_ESCALATION: f64 = 100.0;

/// Escalation attempts before giving up on a gene. Three steps of `×100` take the
/// default `λ = 0.01` to `10⁴`, by which point `β̂ ≈ 0` and the statistic is valid.
const MAX_RIDGE_ESCALATIONS: u8 = 3;

//////////////////////
// Caller-facing IO //
//////////////////////

/// Caller-provided spec for the held-out feature projection. Both index vectors
/// are on the **full backend** feature axis (`unified.count_backend()` rows),
/// not the HVG-narrowed compact axis.
pub struct FeatureProjectionConfig {
    /// Ridge `λ` on the per-gene Poisson-MAP solve.
    pub ridge: f64,
    /// Ridge `γ` on the `H×H` calibration map.
    pub calib_ridge: f64,
    /// Backend row → gene id. Identity (each row its own gene) for a free,
    /// non-factored model such as `senna bge`.
    pub backend_row_to_gene: Vec<u32>,
    /// Per-backend-row unspliced flag. All-false for a non-factored model.
    pub backend_unspliced_rows: Vec<bool>,
    /// Also solve the per-gene splice offset `δ_g` from the unspliced edges.
    pub with_velocity: bool,
    /// FDR for the empirical-Bayes null call that gates the **projected** genes.
    /// `0` disables it. The calibration set is not gated — see [`fit_calibration`].
    ///
    /// The statistic is a χ² call on the likelihood ratio `D_null − D_fit`, *not*
    /// on `‖β̂_g‖²`. `‖β‖²` is the right statistic for a *trained* gene, whose null
    /// sits at its random init; a Poisson-MAP gene has no init. With near-floor
    /// counts it carries almost no Fisher information, so the ridge leaves `β̂`
    /// pointing at noise with a norm set by `λ`. Measured on simulated data, `‖β̂‖²`
    /// ranks genes *anti-correlated* with how well the projection reproduces the
    /// trained embedding (Spearman ≈ −0.16); the LRT ranks them correctly (≈ +0.17).
    ///
    /// A gene that fails is zeroed rather than handed a fabricated direction: null
    /// genes never shaped the cell embedding, so inventing a gene-space coordinate
    /// for them only adds noise to marker scoring.
    pub null_fdr: f32,
}

/// How well the Poisson-MAP frame lines up with the trained NCE frame, measured
/// on the live trained genes (which have both). `mean_cosine ≈ 1` and
/// `norm_ratio ≈ 1` with `M ≈ I` means the frames agreed before calibration.
///
/// Every statistic is `‖β_trained‖²`-weighted. Unweighted, they would be dominated
/// by rows the model never moved off random init — typically the majority — and
/// would report a frame mismatch that says nothing about the marker genes carrying
/// cell-type identity.
pub struct CalibrationDiag {
    /// Trained genes paired to fit `M` (0 = fell back to the identity).
    pub n_trained: usize,
    /// Which map was affordable. `Scalar` or `Identity` means the projected genes'
    /// frame is uncorrected — read `beta` with that in mind.
    pub kind: CalibrationKind,
    /// `‖β‖²`-weighted mean cosine between the calibrated re-solve `β̂·M` and `β`.
    pub mean_cosine: f32,
    /// Weighted median `‖β̂·M‖ / ‖β‖` over the live trained genes.
    pub norm_ratio: f32,
    /// `‖β‖²`-weighted `R²` of `β̂·M` against `β` (the calibration objective).
    pub r2: f32,
}

/// The trained per-gene embedding the calibration maps onto, keyed to the caller's
/// **backend** gene axis.
///
/// For a β-sharing (factored) model this must be the `beta` Var, *not* the
/// `e_feat` rows. An unspliced row's `e_feat` is `β_g + δ_g`, and the feature-null
/// QC drops spliced and unspliced rows independently — in practice it keeps
/// unspliced rows preferentially, since `‖β_g + δ_g‖² > ‖β_g‖²`. Keying the
/// calibration on trained *spliced rows* therefore empties the set on exactly the
/// runs where the QC bit hardest. Every gene with any trained row has a `β_g`, and
/// that is the target.
pub(crate) struct TrainedBeta {
    /// `[n_trained_genes × H]` row-major.
    pub beta: Vec<f32>,
    /// Trained gene index → backend gene id.
    pub backend_gene_id: Vec<u32>,
}

/// Held-out gene embeddings, aligned row-for-row with `gene_ids`.
pub struct FeatureProjection {
    /// Held-out gene ids, ascending, into the caller's backend gene axis (the
    /// codomain of `backend_row_to_gene`).
    pub gene_ids: Vec<u32>,
    /// Calibrated `β_g`, `[n_heldout × H]` row-major.
    pub beta: Vec<f32>,
    /// Calibrated `δ_g`, `[n_heldout × H]` row-major; `None` when velocity is
    /// off or the model has no unspliced rows.
    pub delta: Option<Vec<f32>>,
    /// Pseudobulk samples (summed over levels) where the gene reads above the
    /// column floor. The honest confidence signal: held-out genes are exactly
    /// the low-detection genes HVG dropped.
    pub n_detected_pb: Vec<u32>,
    /// Poisson deviance of the spliced solve.
    pub deviance: Vec<f32>,
    /// `D_null − D_fit`: how much of the gene's pseudobulk profile `θ` explains,
    /// over an intercept-only fit. The statistic the null gate tests (`χ²_H`).
    ///
    /// **Can be negative**, which is not a small LRT but a *failed solve*: at the
    /// Poisson-MAP optimum `D_fit ≤ D_null` always (see `deviance_and_lrt`), so a
    /// negative value means the per-gene IRLS did not converge. Such genes are
    /// excluded from the null fit and zeroed. Exposed rather than clamped so the
    /// failure is visible in `gene_qc.parquet`.
    pub lrt: Vec<f32>,
    /// `true` = the solved `β̂_g` rose above the estimated null and was kept;
    /// `false` = indistinguishable from a gene the model never moved, so `beta`
    /// (and `delta`) were zeroed rather than fabricated. All `true` when
    /// `null_fdr == 0`.
    pub live: Vec<bool>,
    pub calib: CalibrationDiag,
}

///////////////////////
// Pseudobulk counts //
///////////////////////

/// The stacked pseudobulk view the solve runs against: every collapse level's
/// `θ_pb` / `b_pb` concatenated into one frozen table, with the matching count
/// matrices kept on the **full backend** feature axis.
///
/// **Exposure.** The collapse emits Gamma-posterior *rates* (per-cell means), not
/// counts. A rate has `Var(n) = μ / size_p`, so a Poisson fit to it is a
/// quasi-Poisson with a per-column dispersion — its deviance scales with the
/// pseudobulk's cell count and with the gene's expression, and is therefore *not*
/// `χ²`-calibrated. Measured on real data that broke the null gate
/// outright: `Spearman(LRT, detection) = +0.60`, the lower quantiles collapsed to
/// zero, `σ̂² → 0`, and 59% of dropped genes were called live against an estimated
/// `π̂₀ = 0.81`.
///
/// So this view converts to the count scale: an edge carries `rate · size_p`, and
/// `bias` carries `b_pb + log(size_p)` — the standard Poisson exposure offset. The
/// modelled rate `exp(⟨β_g, θ_p⟩ + b_p + b_g)` is unchanged, so the frame still
/// matches training; only the likelihood's scale is now correct, which restores
/// both the ridge's meaning and the LRT's `χ²_H` calibration.
pub(crate) struct StackedPb<'a> {
    /// `[Σ n_pb × H]` row-major, levels concatenated in `counts` order.
    pub theta: Vec<f32>,
    /// `[Σ n_pb]`, same order. Already includes the `log(size_p)` exposure offset.
    pub bias: Vec<f32>,
    /// One `[backend_rows × n_pb^(l)]` rate matrix per level.
    pub counts: Vec<&'a DMatrix<f32>>,
    /// Cells per pseudobulk, per level — the exposure. Aligned with `counts`.
    pub sizes: Vec<Vec<f32>>,
    /// `offsets[l]` = global pb index of level `l`'s first column.
    pub offsets: Vec<usize>,
}

impl StackedPb<'_> {
    /// Dense edge list for one backend row: every pseudobulk column of every
    /// level, on the **count** scale (`rate · size_p`). `n_pb` is ~1350 stacked at
    /// the defaults, so dense is both affordable and a genuinely better-posed
    /// regression than positives-only.
    fn edges_for_row(&self, row: usize) -> Vec<(u32, f32)> {
        let mut edges = Vec::with_capacity(self.bias.len());
        for (l, m) in self.counts.iter().enumerate() {
            let base = self.offsets[l];
            for s in 0..m.ncols() {
                edges.push(((base + s) as u32, m[(row, s)] * self.sizes[l][s]));
            }
        }
        edges
    }

    /// Edge list summed over several backend rows (a gene's tracks within one
    /// modality). Rows share a pseudobulk axis, so counts add.
    fn edges_for_rows(&self, rows: &[usize]) -> Vec<(u32, f32)> {
        let Some((&first, rest)) = rows.split_first() else {
            return Vec::new();
        };
        let mut edges = self.edges_for_row(first);
        for &r in rest {
            for (l, m) in self.counts.iter().enumerate() {
                let base = self.offsets[l];
                for s in 0..m.ncols() {
                    edges[base + s].1 += m[(r, s)] * self.sizes[l][s];
                }
            }
        }
        edges
    }
}

////////////////////
// The projection //
////////////////////

/// Rows of one gene, split by modality, on the backend axis.
#[derive(Default)]
struct GeneRows {
    spliced: Vec<usize>,
    unspliced: Vec<usize>,
}

/// Solve `β_g` (and optionally `δ_g`) for every backend gene absent from the
/// trained feature axis, then map the result into the trained frame and gate it
/// against the estimated null.
///
/// `trained` carries the per-gene `β` the calibration maps onto, keyed to the
/// backend gene axis (see [`TrainedBeta`]); every backend gene it does not name is
/// held out.
pub(crate) fn project_held_out_features(
    pb: &StackedPb,
    h: usize,
    trained: &TrainedBeta,
    cfg: &FeatureProjectionConfig,
) -> FeatureProjection {
    let n_backend_rows = cfg.backend_row_to_gene.len();
    let n_genes = cfg
        .backend_row_to_gene
        .iter()
        .map(|&g| g as usize + 1)
        .max()
        .unwrap_or(0);

    // Bucket backend rows per gene, split by modality.
    let mut gene_rows: Vec<GeneRows> = (0..n_genes).map(|_| GeneRows::default()).collect();
    for row in 0..n_backend_rows {
        let g = cfg.backend_row_to_gene[row] as usize;
        if cfg.backend_unspliced_rows[row] {
            gene_rows[g].unspliced.push(row);
        } else {
            gene_rows[g].spliced.push(row);
        }
    }

    // Held-out at GENE granularity: a gene with any surviving trained row keeps
    // its model β_g. `--feature-null-fdr` drops feature *rows*, and under
    // β-sharing a gene's spliced and unspliced rows can be dropped separately —
    // re-projecting such a gene would silently overwrite a trained embedding.
    let trained_genes: FxHashSet<u32> = trained.backend_gene_id.iter().copied().collect();
    let held_out: Vec<u32> = (0..n_genes as u32)
        .filter(|g| !trained_genes.contains(g))
        .collect();

    let calib = fit_calibration(pb, h, &gene_rows, trained, cfg);
    info!(
        "Held-out feature projection: {} genes ({:?} calibration on {} live trained genes; \
         weighted cosine {:.3}, norm ratio {:.3}, R² {:.3})",
        held_out.len(),
        calib.diag.kind,
        calib.diag.n_trained,
        calib.diag.mean_cosine,
        calib.diag.norm_ratio,
        calib.diag.r2
    );

    let pbs = PbScalars::new(pb);
    let want_delta = cfg.with_velocity
        && held_out
            .iter()
            .any(|&g| !gene_rows[g as usize].unspliced.is_empty());

    let solved: Vec<SolvedGene> = held_out
        .par_iter()
        .map(|&g| {
            let rows = &gene_rows[g as usize];
            let spliced = pb.edges_for_rows(&rows.spliced);
            if spliced.is_empty() {
                return SolvedGene::empty(h);
            }
            // Escalate the ridge until the solve is well-posed. `lrt <= 0` is
            // impossible at the MAP optimum, so it means the undamped Newton
            // iteration diverged — on real data the worst offenders overshot by 15
            // orders of magnitude. That happens where the Hessian is near-singular
            // (near-floor counts) and `λ = 0.01` barely bounds the step. Rather than
            // discard the gene, re-solve it with the regularization it actually
            // needs. As `λ → ∞`, `β̂ → 0` and `lrt → 0⁺`, so the loop always lands on
            // a valid statistic — a genuinely uninformative gene simply converges to
            // "no evidence" and the null gate zeroes it on the merits.
            let mut ridge = cfg.ridge;
            let mut escalations = 0u8;
            let (beta, deviance, lrt) = loop {
                let (beta, b_g) = solve_one_cell(&spliced, &pb.theta, &pb.bias, h, ridge);
                let (deviance, lrt) =
                    deviance_and_lrt(&spliced, &pb.theta, &pb.bias, &beta, b_g, h, &pbs);
                if lrt >= 0.0 || escalations == MAX_RIDGE_ESCALATIONS {
                    break (beta, deviance, lrt);
                }
                ridge *= RIDGE_ESCALATION;
                escalations += 1;
            };
            let delta = if want_delta && !rows.unspliced.is_empty() {
                let unspliced = pb.edges_for_rows(&rows.unspliced);
                // `beta` here is the UNCALIBRATED solve — it lives in θ_pb's
                // frame, which is the frame the increment is solved in. Both are
                // pushed through `M` together at the end. The increment reuses the
                // gene's escalated ridge, so it is conditioned like its own base.
                solve_cell_increment(&unspliced, &beta, &pb.theta, &pb.bias, h, ridge).0
            } else {
                vec![0f32; h]
            };
            SolvedGene {
                n_detected: count_detected(&spliced, &pbs.floor_scaled),
                deviance,
                lrt,
                escalations,
                beta,
                delta,
            }
        })
        .collect();

    /////////////////////////////////////////////////////////
    // Push both β̂ and δ̂ through the calibration map `M` //
    /////////////////////////////////////////////////////////

    let n = solved.len();
    let mut beta = vec![0f32; n * h];
    let mut delta = want_delta.then(|| vec![0f32; n * h]);
    let mut n_detected_pb = vec![0u32; n];
    let mut deviance = vec![0f32; n];
    for (i, s) in solved.iter().enumerate() {
        let b = calib.apply(&s.beta, h);
        beta[i * h..(i + 1) * h].copy_from_slice(&b);
        if let Some(d) = delta.as_mut() {
            let dv = calib.apply(&s.delta, h);
            d[i * h..(i + 1) * h].copy_from_slice(&dv);
        }
        n_detected_pb[i] = s.n_detected;
        deviance[i] = s.deviance;
    }
    let lrt: Vec<f32> = solved.iter().map(|s| s.lrt).collect();

    ///////////////////////////////////////////////////////////
    // Null gate: don't hand a degenerate gene a fake vector //
    ///////////////////////////////////////////////////////////
    //
    // The statistic is the LRT, not `‖β̂_g‖²` — see [`FeatureProjectionConfig::null_fdr`].
    // `chi2_null_call` fits the null's scale and effective dof from the lower
    // quantiles, so over-dispersion is absorbed rather than assumed away.
    //
    // That estimator is what forces the two exclusions below: it reads the *lower*
    // quantiles, so anything piled at zero drags `σ̂² → 0` and then every positive
    // LRT clears the FDR. Only genes with a well-posed statistic enter the fit;
    // both exclusions are failed directly (zeroed) rather than clamped and tested.
    //
    // * **Undetected** (`n_detected_pb == 0`) — flat on the pseudobulk floor, so
    //   `lrt == 0` structurally. Null by definition.
    // * **Non-converged** (`lrt <= 0`) — mathematically impossible at the MAP
    //   optimum (see `deviance_and_lrt`), so it flags a failed IRLS solve, not a
    //   null gene. Its `β̂` is untrustworthy and its LRT is not a χ² draw. Measured
    //   at ~4.5% of detected genes on a real 34k-gene run at `λ = 0.01`.
    let live = if cfg.null_fdr > 0.0 && n > 0 {
        let tested: Vec<usize> = (0..n)
            .filter(|&i| n_detected_pb[i] > 0 && lrt[i] > 0.0)
            .collect();
        let lrt_tested: Vec<f64> = tested.iter().map(|&i| f64::from(lrt[i])).collect();
        let mut live = vec![false; n];
        let null = chi2_null_call(&lrt_tested, h, cfg.null_fdr);
        for (k, &i) in tested.iter().enumerate() {
            live[i] = null.live[k];
        }
        let n_live = live.iter().filter(|&&l| l).count();
        let n_undetected = (0..n).filter(|&i| n_detected_pb[i] == 0).count();
        // Still negative after escalation: a diverged solve, not a null gene. A gene
        // that lands at exactly 0 has converged to "no evidence" and is simply not live.
        let n_unconverged = (0..n)
            .filter(|&i| n_detected_pb[i] > 0 && lrt[i] < 0.0)
            .count();
        info!(
            "  projected-gene LRT null call — σ̂²={:.4}, ν̂={:.1}/{h}, π̂₀={:.2}; \
             {n_live} / {n} live at FDR {} ({n_undetected} undetected + {n_unconverged} \
             non-converged excluded from the null fit; the rest zeroed)",
            null.sigma2, null.eff_dof, null.pi0, cfg.null_fdr,
        );
        let n_escalated = solved.iter().filter(|s| s.escalations > 0).count();
        if n_escalated > 0 {
            info!(
                "  {n_escalated} / {n} gene(s) needed a raised ridge to converge \
                 ({n_unconverged} still diverged after {MAX_RIDGE_ESCALATIONS} escalations \
                 and were zeroed)"
            );
        }
        if n_unconverged * 20 > n {
            warn!(
                "{n_unconverged} of {n} projected genes ({:.1}%) still fail to beat an \
                 intercept-only fit after ridge escalation, which is impossible at the \
                 Poisson-MAP optimum — their IRLS diverged. They were zeroed. Something is \
                 wrong with the pseudobulk side or the counts.",
                100.0 * n_unconverged as f64 / n as f64
            );
        }
        for (i, &l) in live.iter().enumerate() {
            if !l {
                beta[i * h..(i + 1) * h].fill(0.0);
                if let Some(d) = delta.as_mut() {
                    d[i * h..(i + 1) * h].fill(0.0);
                }
            }
        }
        live
    } else {
        vec![true; n]
    };

    FeatureProjection {
        gene_ids: held_out,
        beta,
        delta,
        n_detected_pb,
        deviance,
        lrt,
        live,
        calib: calib.diag,
    }
}

struct SolvedGene {
    beta: Vec<f32>,
    delta: Vec<f32>,
    n_detected: u32,
    /// Ridge escalations this gene needed before its solve converged.
    escalations: u8,
    deviance: f32,
    /// `D_null − D_fit`, raw. Negative ⇒ the IRLS solve did not converge.
    lrt: f32,
}

impl SolvedGene {
    fn empty(h: usize) -> Self {
        Self {
            beta: vec![0f32; h],
            delta: vec![0f32; h],
            n_detected: 0,
            escalations: 0,
            deviance: f32::NAN,
            lrt: 0.0,
        }
    }
}

/// Pseudobulk samples where the gene reads above the column floor.
///
/// `edges` is the gene's already-gathered edge list, so this costs one linear scan
/// over a cache-hot buffer. Re-reading the count matrix here instead would stride
/// by `nrows` (`DMatrix` is column-major), which at ~34k backend rows is a fresh
/// cache line per element — ~46M of them across the gene loop, for values we are
/// already holding.
///
/// The comparison is scale-invariant: an edge carries `Σ_rows rate · size_p`, so
/// testing it against `floor_p · size_p` is the same test as `Σ rate > floor_p`.
fn count_detected(edges: &[(u32, f32)], floor_scaled: &[f32]) -> u32 {
    edges
        .iter()
        .filter(|&&(p, v)| v > floor_scaled[p as usize] * (1.0 + DETECTION_MARGIN))
        .count() as u32
}

/// Gene-independent per-pseudobulk quantities, computed once and shared across the
/// ~34k per-gene solves. Every one of these was previously recomputed inside the
/// rayon map, at ~1350 columns per gene.
struct PbScalars {
    /// `exp(b_p)` per global pb index — the intercept-only rate before `b⁰`.
    exp_b: Vec<f64>,
    /// `Σ_p exp(b_p)`. Constant because every gene's edge list spans all columns.
    sum_exp_b: f64,
    /// Detection floor per global pb index, already multiplied by `size_p` so it
    /// compares directly against an edge value.
    floor_scaled: Vec<f32>,
}

impl PbScalars {
    fn new(pb: &StackedPb) -> Self {
        let exp_b: Vec<f64> = pb
            .bias
            .iter()
            .map(|&b| f64::from(b).clamp(-SCORE_CLAMP, SCORE_CLAMP).exp())
            .collect();
        let sum_exp_b = exp_b.iter().sum();
        let mut floor_scaled = Vec::with_capacity(pb.bias.len());
        for (l, m) in pb.counts.iter().enumerate() {
            let floors: Vec<f32> = (0..m.ncols())
                .into_par_iter()
                .map(|s| m.column(s).iter().copied().fold(f32::INFINITY, f32::min))
                .collect();
            floor_scaled.extend(floors.iter().zip(&pb.sizes[l]).map(|(f, s)| f * s));
        }
        Self {
            exp_b,
            sum_exp_b,
            floor_scaled,
        }
    }
}

/// Clamp on the linear predictor before `exp`, mirroring `cell_projection`'s.
const SCORE_CLAMP: f64 = 30.0;

/// Poisson deviance `D = 2 Σ [ n log(n/μ) − (n − μ) ]` at the fitted `(β_g, b_g)`,
/// and the likelihood-ratio statistic against the intercept-only null `β_g = 0`.
///
/// Returns `(deviance, lrt)` with `lrt = D_null − D_fit ≥ 0`, asymptotically `χ²_h`
/// when the gene carries no cell-state structure. The intercept-only MLE is closed
/// form: `μ⁰_p = exp(b_p + b⁰)` with `b⁰ = log(Σ n_p / Σ exp(b_p))`.
fn deviance_and_lrt(
    edges: &[(u32, f32)],
    theta: &[f32],
    bias: &[f32],
    beta: &[f32],
    b_g: f32,
    h: usize,
    pbs: &PbScalars,
) -> (f32, f32) {
    let (mut d_fit, mut sum_n) = (0f64, 0f64);
    for &(p, n) in edges {
        let p = p as usize;
        let dot: f64 = theta[p * h..(p + 1) * h]
            .iter()
            .zip(beta)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let mu = (dot + f64::from(bias[p]) + f64::from(b_g))
            .clamp(-SCORE_CLAMP, SCORE_CLAMP)
            .exp();
        let n = f64::from(n);
        d_fit += 2.0 * (dev_term(n, mu) - (n - mu));
        sum_n += n;
    }
    if sum_n <= 0.0 || pbs.sum_exp_b <= 0.0 {
        return (d_fit as f32, 0.0);
    }
    // `μ⁰_p = exp(b_p + b⁰) = exp_b[p] · exp(b⁰)`: one `exp` per gene, not one per
    // pseudobulk column. Note `Σ_p μ⁰_p = Σ_p n_p` exactly, as the intercept-only
    // MLE requires — clamping the exponent here would break that identity.
    let exp_b0 = sum_n / pbs.sum_exp_b;
    let mut d_null = 0f64;
    for &(p, n) in edges {
        let mu0 = pbs.exp_b[p as usize] * exp_b0;
        let n = f64::from(n);
        d_null += 2.0 * (dev_term(n, mu0) - (n - mu0));
    }
    // Returned RAW, possibly negative. At the MAP optimum it cannot be: `(0, b⁰)` is
    // feasible for the penalised objective, so `ℓ(β̂,b̂) − (λ/2)‖β̂‖² ≥ ℓ(0,b⁰)` and
    // hence `D_fit ≤ D_null`. A negative value therefore does *not* mean "this gene
    // fits worse than a constant" — it means [`solve_poisson_map`]'s 8 undamped
    // Newton steps did not converge, which happens on near-floor counts where the
    // Hessian is near-singular and the small ridge barely bounds the step.
    //
    // Clamping to 0 would launder that numerical failure into a legitimate
    // "no evidence" statistic *and* pile exact zeros into the lower tail that
    // `chi2_null_call` fits its null scale from — the same `σ̂² → 0` collapse the
    // undetected-gene exclusion exists to prevent. The caller must exclude these.
    (d_fit as f32, (d_null - d_fit) as f32)
}

/// `n·log(n/μ)`, with the `n log n → 0` limit at `n = 0`.
fn dev_term(n: f64, mu: f64) -> f64 {
    if n > 0.0 && mu > 0.0 {
        n * (n / mu).ln()
    } else {
        0.0
    }
}

/////////////////////////
// Calibration mapping //
/////////////////////////

/// Which map the calibration could afford to fit. Reported so a reader knows how
/// much to trust `beta`, and — crucially — whether projected `β` is on the same
/// scale as trained `β`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CalibrationKind {
    /// Full `H×H` ridge map. Needs `≥ 3H` live trained genes.
    Linear,
    /// A single scalar `s`, `M = s·I`, from the weighted median `‖β‖/‖β̂‖`.
    /// Cannot correct rotation or per-dimension shrinkage, but it does put
    /// projected genes on the trained scale — which is all that stops them from
    /// dominating every cross-gene marker score.
    Scalar,
    /// Nothing to fit against. `β̂` is left on the raw Poisson-MAP scale — measured
    /// at 4–5× the trained norm — and is NOT comparable to trained `β`.
    Identity,
}

struct Calibration {
    /// `[H × H]`; `β_calibrated = β̂ · M`.
    m: DMatrix<f32>,
    diag: CalibrationDiag,
}

impl Calibration {
    /// `β_calibrated = β̂ᵀ · M` (row vector times the map).
    fn apply(&self, v: &[f32], _h: usize) -> Vec<f32> {
        (RowDVector::from_row_slice(v) * &self.m)
            .iter()
            .copied()
            .collect()
    }

    fn identity(h: usize) -> Self {
        Self {
            m: DMatrix::identity(h, h),
            diag: CalibrationDiag {
                n_trained: 0,
                kind: CalibrationKind::Identity,
                mean_cosine: f32::NAN,
                norm_ratio: f32::NAN,
                r2: f32::NAN,
            },
        }
    }
}

/// Re-solve the trained genes with the same stacked-pb Poisson MAP, then fit
/// `M = (B̂ᵀ W B̂ + γI)⁻¹ B̂ᵀ W B` with `W = diag(‖β_trained‖²)`.
///
/// **The `‖β_trained‖²` weight is the only filter, and it is enough.** A gene the
/// model never moved off random init has a tiny `‖β‖²` and so contributes almost
/// nothing to `M` — no hard gate is needed. Applying `embedding_null_call` here as
/// well is actively harmful: by the time a caller reaches this stage its trained
/// genes have usually *already* survived a `--feature-null-fdr` pass, so re-testing
/// them estimates a null from the survivors' own lower tail and rejects most of
/// them. Measured on a real bone-marrow gem run: 1358 QC'd trained genes collapsed
/// to 18 "live", far below the `3H = 96` needed for the `H×H` map, silently
/// demoting the calibration to a scalar and leaving the frame uncorrected.
///
/// Each gene is re-solved from its **spliced** backend rows — the same edges a
/// held-out gene's `β̂` comes from — so `M` maps like onto like. This holds even
/// for a gene whose spliced row was dropped from training: its trained `β_g` is
/// still in the frame, and the counts are still on disk.
fn fit_calibration(
    pb: &StackedPb,
    h: usize,
    gene_rows: &[GeneRows],
    trained: &TrainedBeta,
    cfg: &FeatureProjectionConfig,
) -> Calibration {
    let n_trained_genes = trained.backend_gene_id.len();
    if n_trained_genes == 0 {
        warn!("Held-out projection: no trained genes at all — β̂ left uncalibrated.");
        return Calibration::identity(h);
    }

    // Every trained gene with a non-degenerate β and spliced counts to re-solve
    // against. The regression weight grades them; nothing is hard-gated.
    let calib_genes: Vec<usize> = (0..n_trained_genes)
        .filter(|&i| {
            crate::null_call::live_row(&trained.beta, i, h).is_some()
                && !gene_rows[trained.backend_gene_id[i] as usize]
                    .spliced
                    .is_empty()
        })
        .collect();

    let m_rows = calib_genes.len();
    if m_rows == 0 {
        warn!("Held-out projection: no live trained gene to calibrate against — β̂ left raw.");
        return Calibration::identity(h);
    }

    let resolved: Vec<Vec<f32>> = calib_genes
        .par_iter()
        .map(|&i| {
            let rows = &gene_rows[trained.backend_gene_id[i] as usize].spliced;
            let edges = pb.edges_for_rows(rows);
            solve_one_cell(&edges, &pb.theta, &pb.bias, h, cfg.ridge).0
        })
        .collect();

    let b_hat = DMatrix::from_fn(m_rows, h, |i, k| resolved[i][k]);
    let b_ref = DMatrix::from_fn(m_rows, h, |i, k| trained.beta[calib_genes[i] * h + k]);
    let w: DVector<f32> = DVector::from_fn(m_rows, |i, _| b_ref.row(i).iter().map(|x| x * x).sum());

    // An H×H map needs enough paired genes to be determined; a scalar needs one.
    // Never skip straight to the identity: `faba annotate` scores markers ACROSS
    // genes, so leaving projected β on the raw scale is what silently corrupts
    // downstream results. See [`CalibrationKind`].
    let want = MIN_CALIB_GENES_PER_DIM * h;
    let (m, kind) = if m_rows >= want {
        // (B̂ᵀ W B̂ + γI)⁻¹ B̂ᵀ W B
        let bw = DMatrix::from_fn(m_rows, h, |i, k| b_hat[(i, k)] * w[i]);
        let mut lhs = b_hat.transpose() * &bw;
        for k in 0..h {
            lhs[(k, k)] += cfg.calib_ridge as f32;
        }
        let rhs = bw.transpose() * &b_ref;
        match lhs.cholesky().map(|c| c.solve(&rhs)) {
            Some(m) => (m, CalibrationKind::Linear),
            None => (scalar_map(&b_hat, &b_ref, &w, h), CalibrationKind::Scalar),
        }
    } else {
        warn!(
            "Held-out projection: only {m_rows} live trained gene(s) — too few for an {h}×{h} map \
             (want ≥ {want}). Falling back to a SCALAR rescale: projected β lands on the trained \
             scale, but its frame is uncorrected. Treat projected genes as coarse."
        );
        (scalar_map(&b_hat, &b_ref, &w, h), CalibrationKind::Scalar)
    };

    /////////////////
    // Diagnostics //
    /////////////////

    // Every statistic is ‖β‖²-weighted, matching the fit: the map is set by the
    // high-contrast genes, so it should be judged on them too.
    let fitted = &b_hat * &m;
    let (mut cos_sum, mut w_sum, mut sse, mut sst) = (0f64, 0f64, 0f64, 0f64);
    let mut ratios: Vec<(f32, f32)> = Vec::with_capacity(m_rows);
    for i in 0..m_rows {
        let (f, r) = (fitted.row(i), b_ref.row(i));
        let nf: f32 = f.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nr: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        let wi = f64::from(w[i]);
        if nf > 0.0 && nr > 0.0 {
            let cos = f.iter().zip(r.iter()).map(|(a, b)| a * b).sum::<f32>() / (nf * nr);
            cos_sum += wi * f64::from(cos);
            w_sum += wi;
            ratios.push((nf / nr, w[i]));
        }
        sse += wi
            * f64::from(
                f.iter()
                    .zip(r.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>(),
            );
        sst += wi * f64::from(nr * nr);
    }

    Calibration {
        m,
        diag: CalibrationDiag {
            n_trained: m_rows,
            kind,
            mean_cosine: if w_sum > 0.0 {
                (cos_sum / w_sum) as f32
            } else {
                f32::NAN
            },
            norm_ratio: weighted_median(&mut ratios),
            r2: if sst > 0.0 {
                (1.0 - sse / sst) as f32
            } else {
                f32::NAN
            },
        },
    }
}

/// `M = s·I` with `s` the `‖β‖²`-weighted median of `‖β_g‖ / ‖β̂_g‖`. Determined by
/// a single paired gene, so it survives the runs where the null QC leaves almost
/// nothing trained. A median (not a mean) so one degenerate `β̂ ≈ 0` cannot blow `s`
/// up.
fn scalar_map(
    b_hat: &DMatrix<f32>,
    b_ref: &DMatrix<f32>,
    w: &DVector<f32>,
    h: usize,
) -> DMatrix<f32> {
    let mut ratios: Vec<(f32, f32)> = Vec::with_capacity(b_hat.nrows());
    for i in 0..b_hat.nrows() {
        let nh: f32 = b_hat.row(i).iter().map(|x| x * x).sum::<f32>().sqrt();
        let nr: f32 = b_ref.row(i).iter().map(|x| x * x).sum::<f32>().sqrt();
        if nh > 1e-9 && nr > 0.0 {
            ratios.push((nr / nh, w[i]));
        }
    }
    let s = weighted_median(&mut ratios);
    DMatrix::identity(h, h) * if s.is_finite() && s > 0.0 { s } else { 1.0 }
}

/// Weighted median of `(value, weight)` pairs: the smallest value at which the
/// cumulative weight reaches half the total.
fn weighted_median(pairs: &mut [(f32, f32)]) -> f32 {
    if pairs.is_empty() {
        return f32::NAN;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let half = pairs.iter().map(|p| f64::from(p.1)).sum::<f64>() / 2.0;
    let mut acc = 0f64;
    for &(v, wt) in pairs.iter() {
        acc += f64::from(wt);
        if acc >= half {
            return v;
        }
    }
    pairs[pairs.len() - 1].0
}

#[cfg(test)]
mod tests;
