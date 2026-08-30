//! Analytical Poisson-MAP projection onto a **frozen** feature dictionary.
//!
//! Both `senna bge` and `senna gem` train in two phases: phase 1 fits the
//! shared feature side, phase 2 re-estimates the cell side. With the feature
//! side frozen each node's embedding is independent of every other node, so
//! the projection is embarrassingly parallel and (near) closed-form.
//!
//! # Scope: pseudobulks and held-out genes, **not** cells
//!
//! The per-**cell** phase-2 path no longer comes through here — it is a
//! cell-block SGD ([`crate::fit::projection`]) that can afford the full
//! all-feature log-partition this solver approximates away (see below), which
//! is what keeps `‖θ_c‖` from running away. What remains on this solver are
//! the caller where the node count is small and Newton is the better tool: the
//! per-pseudobulk velocity readout (`project_pbs_phase2`, a few hundred nodes).
//! It also served the held-out gene projection, which is gone — the softmax gate
//! selects in one pass, so no gene is held out to be solved separately.
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

use nalgebra::DMatrix;
use rayon::prelude::*;

/// Clamp on the linear predictor before `exp`.
///
/// Shared crate-wide: every Poisson fit here exponentiates the same linear
/// predictor in f32 (which overflows at 88), so the bound must move as one.
pub const SCORE_CLAMP: f64 = 30.0;

pub fn velocity_operator(
    beta_g: &[f32],
    delta_g: &[f32],
    n_genes: usize,
    h: usize,
    lambda: f64,
) -> Vec<f32> {
    debug_assert_eq!(beta_g.len(), n_genes * h);
    debug_assert_eq!(delta_g.len(), n_genes * h);
    let bs = DMatrix::<f64>::from_fn(n_genes, h, |i, j| f64::from(beta_g[i * h + j]));
    let d = DMatrix::<f64>::from_fn(n_genes, h, |i, j| f64::from(delta_g[i * h + j]));
    let mut gram = bs.tr_mul(&bs); // Bₛᵀ Bₛ  [h×h]
    let rhs = bs.tr_mul(&d); //       Bₛᵀ D   [h×h]
                             // Ridge scaled to the mean Gram diagonal keeps `lambda` dimensionless and the solve PD.
    let scale = (gram.diagonal().sum() / h as f64).max(1e-12);
    for k in 0..h {
        gram[(k, k)] += lambda * scale;
    }
    // Solve gram·P = rhs by Cholesky (gram is SPD after the ridge); LU only if it somehow
    // is not. Never an explicit inverse.
    let p = gram
        .clone()
        .cholesky()
        .map(|c| c.solve(&rhs))
        .or_else(|| gram.lu().solve(&rhs))
        .unwrap_or(rhs);
    let mut out = vec![0f32; h * h];
    for i in 0..h {
        for j in 0..h {
            out[i * h + j] = p[(i, j)] as f32;
        }
    }
    out
}

/// Apply the [`velocity_operator`] `p` (`[h × h]` row-major) to per-cell identities:
/// `v_c = P·θ_c`. `theta` is row-major `[n_cells × h]`; returns `[n_cells × h]` row-major.
#[must_use]
pub fn apply_velocity_operator(theta: &[f32], p: &[f32], n_cells: usize, h: usize) -> Vec<f32> {
    debug_assert_eq!(theta.len(), n_cells * h);
    debug_assert_eq!(p.len(), h * h);
    let mut v = vec![0f32; n_cells * h];
    v.par_chunks_mut(h)
        .zip(theta.par_chunks(h))
        .for_each(|(vc, tc)| {
            for i in 0..h {
                let row = &p[i * h..(i + 1) * h];
                vc[i] = row
                    .iter()
                    .zip(tc)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>() as f32;
            }
        });
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic: a few frozen features with known e_f, a planted cell e_c*,
    // Poisson counts at the noiseless rate. IRLS should recover e_c* closely.
    // The same planted cell at ~55× the depth the frozen biases describe — the
    // regime that carried a small tail of cells into the score clamp under
    // the undamped solve. Every fitted score must stay inside the clamp, and the
    // fitted rates must reproduce the counts, not overshoot them by orders of
    // magnitude.
    // Plant an identity e_base and a velocity δ*, generate unspliced counts at the
    // noiseless rate exp(⟨e_f, e_base+δ*⟩ + b_f + b_c), and check the increment
    // solve recovers δ* (direction) holding e_base fixed.
    // With δ* = 0 (unspliced explained by the identity alone), the increment is ≈ 0.
    // The operator solves P = (BₛᵀBₛ+λI)⁻¹ BₛᵀD. With D = Bₛ·M (each gene's δ_g the same
    // linear image of its β_s), the least-squares map must recover M (up to the tiny ridge).
    #[test]
    fn velocity_operator_recovers_planted_map() {
        let (g, h) = (40usize, 5usize);
        let bs: Vec<f32> = (0..g * h)
            .map(|i| (((i * 7 + 3) % 23) as f32 / 23.0) - 0.5)
            .collect();
        let m: Vec<f32> = (0..h * h)
            .map(|i| (((i * 11 + 5) % 17) as f32 / 17.0) - 0.5)
            .collect();
        let mut d = vec![0f32; g * h]; // D = Bₛ · M, row-major
        for r in 0..g {
            for c in 0..h {
                d[r * h + c] = (0..h).map(|k| bs[r * h + k] * m[k * h + c]).sum();
            }
        }
        let p = velocity_operator(&bs, &d, g, h, 1e-8);
        let err = p
            .iter()
            .zip(&m)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            err < 1e-2,
            "operator did not recover the planted map (max err={err:.4})"
        );
    }

    #[test]
    fn apply_velocity_operator_is_matvec() {
        let h = 3;
        let p = vec![1.0f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]; // diag(1,2,3)
        let theta = vec![1.0f32, 1.0, 1.0, 2.0, 0.0, -1.0]; // two cells
        let v = apply_velocity_operator(&theta, &p, 2, h);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 2.0, 0.0, -3.0]);
    }
}
