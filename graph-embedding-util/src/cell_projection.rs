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
/// Vectorised as iteratively-reweighted least squares: the design matrix `E = [m × d]`
/// (row `k = [e_{f_k} | 1]`, `m = feats.len()`) is FIXED across Newton steps, so it is
/// built once and each step is two matmuls — the scores `s = Eθ` and the weighted Gram
/// `EᵀWE` (`W = diag(μ)`). This hands the `O(m·h²)` Hessian to a SIMD-blocked `matmul`
/// instead of a scalar rank-1 triangle accumulation; the result is algebraically identical,
/// just reassociated. `d = h+1` SPD Cholesky solve as before.
fn solve_poisson_map(
    feats: &[(u32, f32)],
    frozen_e: &[f32],
    edge_offset: &[f64],
    h: usize,
    lambda: f64,
) -> (Vec<f32>, f32) {
    let d = h + 1; // [v; b_c]
    let m = feats.len();
    // Design matrix E and the response n, gathered once. E's last column is the intercept.
    let mut e_mat = DMatrix::<f64>::zeros(m, d);
    let mut n_vec = DVector::<f64>::zeros(m);
    let offset = DVector::<f64>::from_iterator(m, edge_offset.iter().copied());
    for (k, &(idx, n)) in feats.iter().enumerate() {
        let ef = &frozen_e[idx as usize * h..(idx as usize + 1) * h];
        for (j, &efj) in ef.iter().enumerate() {
            e_mat[(k, j)] = f64::from(efj);
        }
        e_mat[(k, h)] = 1.0;
        n_vec[k] = f64::from(n);
    }

    let mut theta = DVector::<f64>::zeros(d);
    let mut we = DMatrix::<f64>::zeros(m, d); // E with rows reweighted by μ; reused
    for _ in 0..MAX_IRLS_ITERS {
        // s = Eθ + offset ; μ = exp(clamp s) ; resid = n − μ
        let mut s = &e_mat * &theta;
        s += &offset;
        let mu = s.map(|x| x.clamp(-SCORE_CLAMP, SCORE_CLAMP).exp());
        let resid = &n_vec - &mu;
        // grad = Eᵀ resid − λ P θ   (ridge on v only, not the intercept b_c)
        let mut grad = e_mat.tr_mul(&resid);
        for k in 0..h {
            grad[k] -= lambda * theta[k];
        }
        // Hessian = Eᵀ diag(μ) E + λ P + jitter. Reweight E's rows by μ on the contiguous
        // column-major backing (no per-element bounds checks), then one gemm for the Gram.
        let (we_s, e_s, mu_s) = (we.as_mut_slice(), e_mat.as_slice(), mu.as_slice());
        for j in 0..d {
            let base = j * m;
            for k in 0..m {
                we_s[base + k] = mu_s[k] * e_s[base + k];
            }
        }
        let mut hess = e_mat.tr_mul(&we);
        for k in 0..h {
            hess[(k, k)] += lambda;
        }
        for k in 0..d {
            hess[(k, k)] += PD_JITTER;
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

/// Embedding-space velocity **operator** — the fate-faithful alternative to the per-cell
/// increment [`solve_cell_increment`]. The increment fits each cell's sparse *unspliced*
/// counts *absolutely*, so it is dominated by a shrinkage-toward-origin common-mode
/// (empirically `δ ≈ −0.5·θ` in ~all cells) and carries little fate direction.
///
/// This reads velocity off the DENOISED dictionaries instead. A cell's future spliced
/// state is where its nascent (unspliced) content points, so the velocity is the latent
/// shift `v` that makes the spliced prediction catch up to the nascent one, least-squares
/// over genes:
/// ```text
/// minᵥ Σ_g ( β_s,g·(θ+v) − β_u,g·θ )²  ⟹  v = P·θ,   P = (BₛᵀBₛ + λI)⁻¹ Bₛᵀ D
/// ```
/// with `Bₛ = β_g` (the spliced dictionary) and `D = β_u − β_s = δ_g` (the delta
/// dictionary). It touches only model parameters — **no raw U−S count differencing** — and
/// is solved by a ridge-conditioned Cholesky lin-solve, **never an explicit matrix inverse**
/// (`BₛᵀBₛ` is easily ill-conditioned once the latent axes correlate).
///
/// `beta_g`/`delta_g` are row-major `[n_genes × h]` and gene-aligned. Returns the operator
/// `P` row-major `[h × h]`; apply it per cell with [`apply_velocity_operator`]. The ridge
/// `lambda` is scaled to the Gram trace, so it is dimensionless.
#[must_use]
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
