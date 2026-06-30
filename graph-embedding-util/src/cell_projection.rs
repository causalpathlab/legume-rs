//! Analytical per-cell projection onto a **frozen** feature dictionary.
//!
//! Both `senna bge` and `faba gem` train in two phases: phase 1 fits the
//! shared feature side, phase 2 re-estimates the per-cell embedding. With
//! the feature side frozen, each cell's embedding is independent of every
//! other cell — so phase 2 is a per-cell projection, embarrassingly
//! parallel, and (near) closed-form. This module is the shared solver both
//! callers use instead of SGD over a frozen-feature `e_cell` `VarMap`.
//!
//! **Objective — Poisson MAP on observed features.** For a cell with frozen
//! feature embeddings `e_f` / biases `b_f`, model its observed counts `n_f`
//! as Poisson with rate `μ_f = exp(⟨e_f, e_c⟩ + b_f + b_c)` and put a
//! Gaussian (ridge) prior `N(0, 1/λ)` on `e_c`. The exact softmax MLE would
//! normalise over *all* features (the partition NCE only ever approximated);
//! at scale that's infeasible, so we fit the cell's observed features and
//! let the ridge prior stand in for the partition (bounding `e_c`, which
//! fitting positives alone would push to ∞). The per-cell intercept `b_c`
//! absorbs library size.
//!
//! Each Newton/IRLS step is a small `(H+1)×(H+1)` SPD solve:
//! ```text
//! θ = [e_c; b_c],  ẽ_f = [e_f; 1],  s_f = ⟨θ, ẽ_f⟩ + b_f,  μ_f = exp(s_f)
//! θ ← θ + (Σ_f μ_f ẽ_f ẽ_fᵀ + λP)⁻¹ (Σ_f (n_f − μ_f) ẽ_f − λP θ)
//! ```
//! with `P = diag(1,…,1, 0)` (ridge on `e_c`, not the intercept). The
//! Hessian is symmetric, so only its upper triangle is accumulated.

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Max IRLS/Newton steps per cell (converges in a few given the ridge).
const MAX_IRLS_ITERS: usize = 8;
/// Clamp the linear predictor before `exp` to avoid overflow.
const SCORE_CLAMP: f64 = 30.0;
/// PD jitter added to the Hessian diagonal so the Cholesky never fails on a
/// degenerate (few-feature) cell.
const PD_JITTER: f64 = 1e-6;
/// Stop early when the Newton update is this small (max-abs component).
const CONVERGE_TOL: f64 = 1e-5;

/// Project every cell onto the frozen dictionary, in parallel.
///
/// `frozen_e` is row-major `[n_features × h]`, `frozen_b` is `[n_features]`.
/// `per_cell[c]` is cell `c`'s observed `(feature_index, count)` list
/// (indices into `frozen_e`/`frozen_b`). Returns `(e_cell [n_cells × h]
/// row-major, b_cell [n_cells])`. The per-cell bias `b_c` is **always**
/// fitted (it absorbs library size, keeping `e_c` depth-corrected and
/// well-scaled) and kept by both callers (bge and gem).
///
/// `progress`, when `Some`, is incremented by one per solved cell — pass a
/// bar pre-sized to the *total* cell count so a caller that invokes this
/// once per chunk gets a single bar spanning all chunks.
#[must_use]
pub fn project_cells(
    frozen_e: &[f32],
    frozen_b: &[f32],
    per_cell: &[Vec<(u32, f32)>],
    h: usize,
    lambda: f64,
    progress: Option<&indicatif::ProgressBar>,
) -> (Vec<f32>, Vec<f32>) {
    let n_cells = per_cell.len();
    let solved: Vec<(Vec<f32>, f32)> = per_cell
        .par_iter()
        .map(|feats| {
            let solved = solve_one_cell(feats, frozen_e, frozen_b, h, lambda);
            if let Some(pb) = progress {
                pb.inc(1);
            }
            solved
        })
        .collect();
    let mut e = vec![0f32; n_cells * h];
    let mut b = vec![0f32; n_cells];
    for (c, (e_c, b_c)) in solved.iter().enumerate() {
        e[c * h..(c + 1) * h].copy_from_slice(e_c);
        b[c] = *b_c;
    }
    (e, b)
}

/// An optional second likelihood folded into a cell's MAP solve, sharing the
/// same `e_c` as the Poisson expression term. The implementor lives in a
/// downstream crate (e.g. faba's m6A binomial term); it owns its per-cell data
/// and frozen feature side, and contributes its Gauss-Newton grad/Hessian into
/// the shared accumulators each IRLS step. Generic seam: geu never knows the
/// modality.
///
/// `θ` layout is `[e_c (h); b_c (1); extra (n_extra)]` — the term reads `e_c`
/// (`theta[0..h]`) and its own extra slots (`theta[h+1 ..]`), and writes into
/// the **upper triangle** of `hess` over those same indices (the caller mirrors
/// once). It must leave the expression intercept slot `h` untouched unless it
/// deliberately couples to library size.
pub trait PerCellAuxTerm: Send + Sync {
    /// Extra parameters this term appends to `θ` after `[e_c; b_c]` (e.g. a
    /// dedicated per-cell intercept). `0` to share none.
    fn n_extra(&self) -> usize;
    /// Accumulate grad/Hessian (Gauss-Newton, SPD; upper triangle) for `cell`
    /// at the current `theta`. `extra_offset == h + 1` is where this term's
    /// extra params start in `θ`.
    fn accumulate(
        &self,
        cell: usize,
        theta: &DVector<f64>,
        h: usize,
        extra_offset: usize,
        grad: &mut DVector<f64>,
        hess: &mut DMatrix<f64>,
    );
    /// Ridge added to each extra param's Hessian diagonal (PD + shrinkage).
    fn extra_ridge(&self) -> f64 {
        0.0
    }
}

/// Poisson MAP IRLS for one cell. `θ = [e_c; b_c]`; the ridge `λ` applies to
/// `e_c` only (the intercept `b_c` is unpenalised and absorbs library size).
/// Returns `(e_c, b_c)`. A cell with no observed features gets the zero
/// embedding. Thin wrapper over [`solve_one_cell_aux`] with no auxiliary term —
/// the no-aux path is byte-identical to the original solver.
#[must_use]
pub fn solve_one_cell(
    feats: &[(u32, f32)],
    frozen_e: &[f32],
    frozen_b: &[f32],
    h: usize,
    lambda: f64,
) -> (Vec<f32>, f32) {
    let (e_c, b_c, _extra) = solve_one_cell_aux(0, feats, frozen_e, frozen_b, h, lambda, None);
    (e_c, b_c)
}

/// Joint MAP IRLS for one cell with an optional auxiliary likelihood ([`PerCellAuxTerm`])
/// sharing `e_c`. `θ = [e_c (h); b_c (1); extra (n_extra)]`; the ridge `λ`
/// applies to `e_c` only. Returns `(e_c, b_c, extra_params)`. With `aux == None`
/// this is exactly the Poisson-only solver (`n_extra == 0`, `extra` empty).
///
/// The objective is convex (Poisson NLL + the aux term's convex NLL + ridge),
/// so the damped Newton step converges; the aux term adds SPD Gauss-Newton
/// blocks so the Hessian stays PD.
#[must_use]
pub fn solve_one_cell_aux(
    cell: usize,
    feats: &[(u32, f32)],
    frozen_e: &[f32],
    frozen_b: &[f32],
    h: usize,
    lambda: f64,
    aux: Option<&dyn PerCellAuxTerm>,
) -> (Vec<f32>, f32, Vec<f32>) {
    let n_extra = aux.map_or(0, PerCellAuxTerm::n_extra);
    let extra_offset = h + 1;
    let d = extra_offset + n_extra; // [e_c; b_c; extra...]
                                    // No expression edges AND no aux data → the zero embedding (preserves the
                                    // original empty-cell contract).
    if feats.is_empty() && n_extra == 0 {
        return (vec![0f32; h], 0.0, Vec::new());
    }
    let mut theta = DVector::<f64>::zeros(d);
    let mut grad = DVector::<f64>::zeros(d); // reused across iterations
    for _ in 0..MAX_IRLS_ITERS {
        grad.fill(0.0);
        // Fresh Hessian (consumed by the Cholesky below, so no clone). Only
        // the upper triangle is filled, then mirrored once.
        let mut hess = DMatrix::<f64>::zeros(d, d);
        for &(idx, n) in feats {
            let ef = &frozen_e[idx as usize * h..(idx as usize + 1) * h];
            let bf = f64::from(frozen_b[idx as usize]);
            // s = ⟨e_c, e_f⟩ + b_f + b_c
            let mut s = bf + theta[h];
            for (k, &efk) in ef.iter().enumerate() {
                s += theta[k] * f64::from(efk);
            }
            let mu = s.clamp(-SCORE_CLAMP, SCORE_CLAMP).exp();
            let resid = f64::from(n) - mu;
            // grad += resid · ẽ ;  hess (upper triangle) += μ · ẽ ẽᵀ
            for (a, &efa) in ef.iter().enumerate() {
                let efa = f64::from(efa);
                grad[a] += resid * efa;
                let row = mu * efa;
                for (bb, &efb) in ef.iter().enumerate().skip(a) {
                    hess[(a, bb)] += row * f64::from(efb);
                }
                hess[(a, h)] += row; // cross column with the intercept (1)
            }
            grad[h] += resid;
            hess[(h, h)] += mu;
        }
        // Auxiliary likelihood (e.g. m6A binomial): adds its grad/Hess over
        // [e_c; extra] at the current θ, upper-triangle, same convention.
        if let Some(a) = aux {
            a.accumulate(cell, &theta, h, extra_offset, &mut grad, &mut hess);
        }
        // Ridge prior on e_c (not the intercepts) + PD jitter on the diagonal.
        for k in 0..h {
            hess[(k, k)] += lambda;
            grad[k] -= lambda * theta[k];
        }
        // Optional ridge/shrinkage on the aux term's extra params.
        if let Some(a) = aux {
            let r = a.extra_ridge();
            if r > 0.0 {
                for k in extra_offset..d {
                    hess[(k, k)] += r;
                    grad[k] -= r * theta[k];
                }
            }
        }
        for k in 0..d {
            hess[(k, k)] += PD_JITTER;
        }
        // Mirror upper → lower to complete the symmetric Hessian.
        for a in 0..d {
            for bb in (a + 1)..d {
                hess[(bb, a)] = hess[(a, bb)];
            }
        }
        let Some(chol) = hess.cholesky() else {
            break; // degenerate; keep the current estimate
        };
        let delta = chol.solve(&grad);
        theta += &delta;
        if delta.amax() < CONVERGE_TOL {
            break;
        }
    }
    let e_c: Vec<f32> = (0..h).map(|k| theta[k] as f32).collect();
    let extra: Vec<f32> = (extra_offset..d).map(|k| theta[k] as f32).collect();
    (e_c, theta[h] as f32, extra)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic: a few frozen features with known e_f, a planted cell e_c*,
    // Poisson counts at the noiseless rate. IRLS should recover e_c* closely.
    #[test]
    fn irls_recovers_planted_cell() {
        let h = 4;
        let n_id = 12;
        let mut e = vec![0f32; n_id * h];
        let mut b = vec![0f32; n_id];
        for f in 0..n_id {
            for k in 0..h {
                e[f * h + k] = (((f * 7 + k * 13) % 11) as f32 / 11.0) - 0.5;
            }
            b[f] = (((f * 5) % 7) as f32 / 7.0) - 0.3;
        }
        let e_star = [0.8f32, -0.6, 0.4, 0.2];
        let b_star = 0.5f32;
        let feats: Vec<(u32, f32)> = (0..n_id)
            .map(|f| {
                let ef = &e[f * h..(f + 1) * h];
                let s: f32 =
                    ef.iter().zip(&e_star).map(|(a, b)| a * b).sum::<f32>() + b[f] + b_star;
                (f as u32, s.exp())
            })
            .collect();
        let (e_c, _b_c) = solve_one_cell(&feats, &e, &b, h, 1e-3);
        let dot: f32 = e_c.iter().zip(&e_star).map(|(a, b)| a * b).sum();
        let na: f32 = e_c.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = e_star.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(
            cos > 0.97,
            "recovered cell embedding misaligned (cos={cos:.3})"
        );
    }

    #[test]
    fn empty_cell_is_zero() {
        let (e_c, b_c) = solve_one_cell(&[], &[0.0; 4], &[0.0; 1], 4, 1.0);
        assert_eq!(e_c, vec![0.0; 4]);
        assert_eq!(b_c, 0.0);
    }

    #[test]
    fn project_cells_assembles_rows() {
        // Two cells, h=2, trivial single-feature each → shapes + placement.
        let e = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 features
        let b = vec![0.0f32, 0.0];
        let per_cell = vec![vec![(0u32, 2.0f32)], vec![(1u32, 3.0f32)]];
        let (ec, bc) = project_cells(&e, &b, &per_cell, 2, 1.0, None);
        assert_eq!(ec.len(), 4); // 2 cells × h=2
        assert_eq!(bc.len(), 2);
    }
}
