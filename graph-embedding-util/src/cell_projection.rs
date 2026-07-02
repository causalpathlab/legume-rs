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

/// Poisson MAP IRLS for one cell. `θ = [e_c; b_c]`; the ridge `λ` applies to
/// `e_c` only (the intercept `b_c` is unpenalised and absorbs library size).
/// Returns `(e_c, b_c)`. A cell with no observed features gets the zero
/// embedding.
///
/// The objective is convex (Poisson NLL + ridge), so the damped Newton step
/// converges; the ridge + PD jitter keep the Hessian PD.
#[must_use]
pub fn solve_one_cell(
    feats: &[(u32, f32)],
    frozen_e: &[f32],
    frozen_b: &[f32],
    h: usize,
    lambda: f64,
) -> (Vec<f32>, f32) {
    if feats.is_empty() {
        return (vec![0f32; h], 0.0);
    }
    // Per-edge offset = the feature's frozen bias b_f (no fixed-latent term).
    let edge_offset: Vec<f64> = feats
        .iter()
        .map(|&(idx, _)| f64::from(frozen_b[idx as usize]))
        .collect();
    solve_poisson_map(feats, frozen_e, &edge_offset, h, lambda)
}

/// Analytic per-cell **velocity increment**. Holding the identity latent `e_base`
/// (the spliced solve `θ`) and the frozen feature side fixed, estimate the latent
/// shift `δ` that best explains the cell's UNSPLICED edges under the rate
/// `μ_f = exp(⟨e_f, e_base + δ⟩ + b_f + b_c)`. Returns `(δ, b_c)`.
///
/// This is the same IRLS as [`solve_one_cell`] with the per-feature identity
/// contribution `⟨e_f, e_base⟩` folded into the offset (constant across Newton
/// steps, so precomputed once), and the unknown initialised at `δ = 0`. Unlike a
/// second independent projection (`φ`), `δ` is the *directed residual* — how far
/// the identity must move to explain nascent transcription — so it isolates real
/// dynamics from a noisier re-measurement of the same state and keeps its
/// magnitude (speed). A cell with no unspliced edges gets `δ = 0`.
#[must_use]
pub fn solve_cell_increment(
    feats: &[(u32, f32)],
    e_base: &[f32],
    frozen_e: &[f32],
    frozen_b: &[f32],
    h: usize,
    lambda: f64,
) -> (Vec<f32>, f32) {
    if feats.is_empty() {
        return (vec![0f32; h], 0.0);
    }
    // Per-edge offset = b_f + ⟨e_f, e_base⟩ (the fixed-identity log-rate). Constant
    // over the Newton loop since `e_base` is held fixed, so compute it once.
    let edge_offset: Vec<f64> = feats
        .iter()
        .map(|&(idx, _)| {
            let ef = &frozen_e[idx as usize * h..(idx as usize + 1) * h];
            let dot: f64 = ef
                .iter()
                .zip(e_base)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            f64::from(frozen_b[idx as usize]) + dot
        })
        .collect();
    solve_poisson_map(feats, frozen_e, &edge_offset, h, lambda)
}

/// Shared IRLS core for the identity and velocity-increment solves. Fits
/// `θ = [v; b_c]` maximizing `Σ_k [ n_k·s_k − exp(s_k) ] − (λ/2)‖v‖²` with
/// `s_k = ⟨e_{f_k}, v⟩ + edge_offset[k] + b_c`; `edge_offset` is aligned with
/// `feats` (the frozen bias, plus any fixed-latent log-rate folded in). The ridge
/// `λ` applies to `v` only; the intercept `b_c` is unpenalised. Returns `(v, b_c)`.
///
/// Each Newton step is a small `(h+1)×(h+1)` SPD solve; only the Hessian's upper
/// triangle is accumulated, then mirrored.
fn solve_poisson_map(
    feats: &[(u32, f32)],
    frozen_e: &[f32],
    edge_offset: &[f64],
    h: usize,
    lambda: f64,
) -> (Vec<f32>, f32) {
    let d = h + 1; // [v; b_c]
    let mut theta = DVector::<f64>::zeros(d);
    let mut grad = DVector::<f64>::zeros(d); // reused across iterations
    for _ in 0..MAX_IRLS_ITERS {
        grad.fill(0.0);
        // Fresh Hessian (consumed by the Cholesky below, so no clone). Only
        // the upper triangle is filled, then mirrored once.
        let mut hess = DMatrix::<f64>::zeros(d, d);
        for (k, &(idx, n)) in feats.iter().enumerate() {
            let ef = &frozen_e[idx as usize * h..(idx as usize + 1) * h];
            // s = ⟨e_f, v⟩ + offset_k + b_c
            let mut s = edge_offset[k] + theta[h];
            for (j, &efj) in ef.iter().enumerate() {
                s += theta[j] * f64::from(efj);
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
        // Ridge prior on v (not the intercept) + PD jitter on the diagonal.
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
    let v: Vec<f32> = (0..h).map(|k| theta[k] as f32).collect();
    (v, theta[h] as f32)
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

    // Plant an identity e_base and a velocity δ*, generate unspliced counts at the
    // noiseless rate exp(⟨e_f, e_base+δ*⟩ + b_f + b_c), and check the increment
    // solve recovers δ* (direction) holding e_base fixed.
    #[test]
    fn increment_recovers_planted_velocity() {
        let h = 4;
        let n_id = 14;
        let mut e = vec![0f32; n_id * h];
        let mut b = vec![0f32; n_id];
        for f in 0..n_id {
            for k in 0..h {
                e[f * h + k] = (((f * 5 + k * 17) % 13) as f32 / 13.0) - 0.5;
            }
            b[f] = (((f * 3) % 5) as f32 / 5.0) - 0.3;
        }
        let e_base = [0.5f32, 0.3, -0.4, 0.1];
        let delta_star = [0.3f32, -0.5, 0.2, 0.4];
        let b_c = 0.2f32;
        let feats: Vec<(u32, f32)> = (0..n_id)
            .map(|f| {
                let ef = &e[f * h..(f + 1) * h];
                let s: f32 = ef
                    .iter()
                    .enumerate()
                    .map(|(k, ev)| ev * (e_base[k] + delta_star[k]))
                    .sum::<f32>()
                    + b[f]
                    + b_c;
                (f as u32, s.exp())
            })
            .collect();
        let (delta, _b_c) = solve_cell_increment(&feats, &e_base, &e, &b, h, 1e-3);
        let dot: f32 = delta.iter().zip(&delta_star).map(|(a, b)| a * b).sum();
        let na: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = delta_star.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(cos > 0.97, "recovered velocity misaligned (cos={cos:.3})");
    }

    // With δ* = 0 (unspliced explained by the identity alone), the increment is ≈ 0.
    #[test]
    fn increment_zero_when_no_velocity() {
        let h = 3;
        let e = [0.4f32, -0.2, 0.6, 0.1, 0.5, -0.3, -0.5, 0.2, 0.4];
        let b = [0.1f32, -0.2, 0.0];
        let e_base = [0.3f32, 0.4, -0.2];
        let feats: Vec<(u32, f32)> = (0..3)
            .map(|f| {
                let ef = &e[f * h..(f + 1) * h];
                let s: f32 = ef.iter().zip(&e_base).map(|(a, c)| a * c).sum::<f32>() + b[f];
                (f as u32, s.exp())
            })
            .collect();
        let (delta, _) = solve_cell_increment(&feats, &e_base, &e, &b, h, 1e-2);
        let mag: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            mag < 0.2,
            "increment should be ~0 with no velocity (‖δ‖={mag:.3})"
        );
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
