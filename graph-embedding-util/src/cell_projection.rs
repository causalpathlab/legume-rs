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
/// well-scaled) — a caller whose model scores without a per-cell bias
/// (e.g. bge) simply discards `b_cell`; a global per-cell offset doesn't
/// change `e_c`'s direction.
pub fn project_cells(
    frozen_e: &[f32],
    frozen_b: &[f32],
    per_cell: &[Vec<(u32, f32)>],
    h: usize,
    lambda: f64,
) -> (Vec<f32>, Vec<f32>) {
    let n_cells = per_cell.len();
    let solved: Vec<(Vec<f32>, f32)> = per_cell
        .par_iter()
        .map(|feats| solve_one_cell(feats, frozen_e, frozen_b, h, lambda))
        .collect();
    let mut e = vec![0f32; n_cells * h];
    let mut b = vec![0f32; n_cells];
    for (c, (e_c, b_c)) in solved.iter().enumerate() {
        e[c * h..(c + 1) * h].copy_from_slice(e_c);
        b[c] = *b_c;
    }
    (e, b)
}

/// Poisson MAP IRLS for one cell. `θ = [e_c; b_c]`; the ridge `λ` applies to
/// `e_c` only (the intercept `b_c` is unpenalised and absorbs library size).
/// Returns `(e_c, b_c)`. A cell with no observed features gets the zero
/// embedding.
pub fn solve_one_cell(
    feats: &[(u32, f32)],
    frozen_e: &[f32],
    frozen_b: &[f32],
    h: usize,
    lambda: f64,
) -> (Vec<f32>, f32) {
    let d = h + 1; // trailing dim is the intercept b_c
    if feats.is_empty() {
        return (vec![0f32; h], 0.0);
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
            let bf = frozen_b[idx as usize] as f64;
            // s = ⟨e_c, e_f⟩ + b_f + b_c
            let mut s = bf + theta[h];
            for (k, &efk) in ef.iter().enumerate() {
                s += theta[k] * efk as f64;
            }
            let mu = s.clamp(-SCORE_CLAMP, SCORE_CLAMP).exp();
            let resid = n as f64 - mu;
            // grad += resid · ẽ ;  hess (upper triangle) += μ · ẽ ẽᵀ
            for (a, &efa) in ef.iter().enumerate() {
                let efa = efa as f64;
                grad[a] += resid * efa;
                let row = mu * efa;
                for (bb, &efb) in ef.iter().enumerate().skip(a) {
                    hess[(a, bb)] += row * efb as f64;
                }
                hess[(a, h)] += row; // cross column with the intercept (1)
            }
            grad[h] += resid;
            hess[(h, h)] += mu;
        }
        // Ridge prior on e_c (not the intercept) + PD jitter on the diagonal.
        for k in 0..h {
            hess[(k, k)] += lambda;
            grad[k] -= lambda * theta[k];
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
    (e_c, theta[h] as f32)
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
        let (ec, bc) = project_cells(&e, &b, &per_cell, 2, 1.0);
        assert_eq!(ec.len(), 4); // 2 cells × h=2
        assert_eq!(bc.len(), 2);
    }
}
