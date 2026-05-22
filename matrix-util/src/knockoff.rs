//! Gaussian model-X knockoff `s`-vectors.
//!
//! For a feature correlation matrix `Σ` (symmetric PD, unit diagonal), Gaussian
//! model-X knockoffs need a diagonal `D = diag(s)` such that the joint
//! covariance of originals and knockoffs
//!
//! ```text
//!   G = [[ Σ,    Σ − D ],
//!        [ Σ − D,   Σ  ]]   ⪰ 0
//! ```
//!
//! is positive semidefinite. Block-diagonalizing `G` with `[[I,I],[I,−I]]/√2`
//! (valid since `Σ − D` is symmetric) shows `G ⪰ 0 ⟺ 2Σ − D ⪰ 0 and D ⪰ 0`,
//! i.e. `0 ⪯ diag(s) ⪯ 2Σ`. Power grows with `s` (a larger `s_j` decorrelates
//! the knockoff from the original), so we want `s` as large as the PSD
//! constraint allows.
//!
//! - [`knockoff_s_equicorrelated`]: the cheap closed form `s_j = min(1,
//!   2·λ_min(Σ))`. One scalar set by the worst eigenvalue, so a single
//!   near-zero eigenvalue (rank-deficient `Σ`) collapses `s` for every feature.
//! - [`knockoff_s_mvr`] / [`knockoff_s_me`]: minimum-variance-reconstructability
//!   and maximum-entropy knockoffs (Spector & Janson 2022), solved by
//!   coordinate descent with closed-form 1-D updates and Sherman–Morrison
//!   inverse maintenance. Per-coordinate `s_j`, so they stay powerful even when
//!   `Σ` is ill-conditioned — at no extra dependency (nalgebra only) and
//!   empirically `≥` the SDP construction.
//!
//! Refs: Candès et al. (2018, JRSS-B) model-X knockoffs; Spector & Janson
//! (2022, Ann. Statist.) "Powerful knockoffs via minimizing reconstructability".

use nalgebra::{DMatrix, DVector};

/// Method for the knockoff diagonal `s`-vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnockoffS {
    /// Equicorrelated: `s_j = min(1, 2·λ_min(Σ))` for all `j`.
    Equicorrelated,
    /// Minimum variance-based reconstructability (per-coordinate `s`).
    Mvr,
    /// Maximum entropy.
    Me,
}

/// Solve the knockoff `s`-vector for a symmetric PD correlation matrix `sigma`
/// by the chosen [`KnockoffS`] method. Returns `s` with `0 ⪯ diag(s)` and
/// `2Σ − diag(s) ⪰ 0`.
pub fn knockoff_s(sigma: &DMatrix<f64>, method: KnockoffS) -> DVector<f64> {
    match method {
        KnockoffS::Equicorrelated => knockoff_s_equicorrelated(sigma),
        KnockoffS::Mvr => knockoff_s_mvr(sigma),
        KnockoffS::Me => knockoff_s_me(sigma),
    }
}

/// Equicorrelated `s`: constant `s_j = min(1, 2·λ_min(Σ))` (clamped to `≥ 0`).
pub fn knockoff_s_equicorrelated(sigma: &DMatrix<f64>) -> DVector<f64> {
    let p = sigma.nrows();
    if p == 0 {
        return DVector::zeros(0);
    }
    let lambda_min = min_eig(sigma);
    let s = (2.0 * lambda_min).clamp(0.0, 1.0);
    DVector::from_element(p, s)
}

/// Minimum variance-based reconstructability (MVR) knockoffs: minimize
/// `tr((2Σ − D)⁻¹) + Σ_j 1/s_j` over `0 ≺ D ≺ 2Σ`.
pub fn knockoff_s_mvr(sigma: &DMatrix<f64>) -> DVector<f64> {
    solve_coordinate(sigma, Objective::Mvr)
}

/// Maximum-entropy (ME) knockoffs: minimize `−logdet(2Σ − D) − Σ_j log s_j`
/// over `0 ≺ D ≺ 2Σ`.
pub fn knockoff_s_me(sigma: &DMatrix<f64>) -> DVector<f64> {
    solve_coordinate(sigma, Objective::Me)
}

#[derive(Clone, Copy)]
enum Objective {
    Mvr,
    Me,
}

/// Coordinate descent shared by MVR and ME. Both objectives separate into a
/// `tr/logdet` term in `M = 2Σ − D` plus a barrier in `s`, and the per-feature
/// 1-D minimizer is closed form — MVR: `s_j ← (1 + s_j·m_jj) / (√c_j + m_jj)`;
/// ME: `s_j ← (1 + s_j·m_jj) / (2·m_jj)` — with `m_jj = (M⁻¹)_jj` and
/// `c_j = ‖M⁻¹ e_j‖²`. `M⁻¹` is refreshed exactly once per sweep (kills
/// Sherman–Morrison drift) and rank-1 updated within the sweep so later
/// coordinates see the current iterate (Gauss–Seidel).
fn solve_coordinate(sigma: &DMatrix<f64>, obj: Objective) -> DVector<f64> {
    let p = sigma.nrows();
    if p == 0 {
        return DVector::zeros(0);
    }
    let two_sigma = sigma * 2.0;
    let lambda_min = min_eig(sigma);
    if lambda_min <= 1e-10 {
        // Σ not PD enough to start an interior point; fall back.
        return knockoff_s_equicorrelated(sigma);
    }
    // Feasible interior start: M = 2Σ − D ≻ 0 with margin.
    let s0 = (2.0 * lambda_min).clamp(1e-6, 1.0) * 0.5;
    let mut s = DVector::from_element(p, s0);

    const MAX_ITER: usize = 50;
    const TOL: f64 = 1e-8;
    for _ in 0..MAX_ITER {
        let mut m = two_sigma.clone();
        for j in 0..p {
            m[(j, j)] -= s[j];
        }
        let mut minv = match m.try_inverse() {
            Some(inv) => inv,
            None => break, // hit the boundary; keep the last feasible s
        };

        let mut max_delta = 0.0f64;
        for j in 0..p {
            let m_jj = minv[(j, j)];
            if m_jj.is_nan() || m_jj <= 1e-12 {
                continue;
            }
            let s_old = s[j];
            let s_target = match obj {
                Objective::Me => (1.0 + m_jj * s_old) / (2.0 * m_jj),
                Objective::Mvr => {
                    let col = minv.column(j);
                    let c_j = col.dot(&col);
                    (1.0 + m_jj * s_old) / (c_j.sqrt() + m_jj)
                }
            };
            // δ keeps M = M − δ e_je_jᵀ ≻ 0 iff δ < 1/m_jj; clamp with margin
            // and keep s_j > 0.
            let mut delta = s_target - s_old;
            let max_step = 0.99 / m_jj;
            if delta > max_step {
                delta = max_step;
            }
            if s_old + delta < 1e-8 {
                delta = 1e-8 - s_old;
            }
            if delta.abs() < 1e-15 {
                continue;
            }
            let denom = 1.0 - delta * m_jj;
            if denom <= 1e-12 {
                continue;
            }
            // Sherman–Morrison: (M − δ e_je_jᵀ)⁻¹ = M⁻¹ + (δ/denom)·u uᵀ, u = M⁻¹e_j.
            let u = minv.column(j).into_owned();
            minv.ger(delta / denom, &u, &u, 1.0);
            s[j] = s_old + delta;
            max_delta = max_delta.max(delta.abs());
        }
        if max_delta < TOL {
            break;
        }
    }
    s
}

/// Smallest eigenvalue of a symmetric matrix.
fn min_eig(sigma: &DMatrix<f64>) -> f64 {
    sigma
        .symmetric_eigenvalues()
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    /// Build a random PD correlation matrix `Σ = corr(F Fᵀ + εI)` over `p`
    /// features from `k` latent factors (so it is genuinely low-rank + ridge).
    fn random_corr(p: usize, k: usize, ridge: f64, seed: u64) -> DMatrix<f64> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let f = DMatrix::<f64>::from_fn(p, k, |_, _| StandardNormal.sample(&mut rng));
        let mut cov = &f * f.transpose();
        for i in 0..p {
            cov[(i, i)] += ridge;
        }
        let d: Vec<f64> = (0..p).map(|i| cov[(i, i)].sqrt()).collect();
        DMatrix::from_fn(p, p, |i, j| cov[(i, j)] / (d[i] * d[j]))
    }

    fn min_eig_2sigma_minus_d(sigma: &DMatrix<f64>, s: &DVector<f64>) -> f64 {
        let p = sigma.nrows();
        let mut m = sigma * 2.0;
        for j in 0..p {
            m[(j, j)] -= s[j];
        }
        min_eig(&m)
    }

    /// MVR and ME must return feasible `s`: `s_j > 0` and `2Σ − diag(s) ⪰ 0`.
    #[test]
    fn s_vectors_are_feasible() {
        for &seed in &[1u64, 2, 3] {
            let sigma = random_corr(40, 6, 0.2, seed);
            for method in [KnockoffS::Mvr, KnockoffS::Me, KnockoffS::Equicorrelated] {
                let s = knockoff_s(&sigma, method);
                assert!(s.iter().all(|&v| v > -1e-9), "{method:?}: negative s");
                let lam = min_eig_2sigma_minus_d(&sigma, &s);
                assert!(
                    lam > -1e-6,
                    "{method:?}: 2Σ−D not PSD (λ_min={lam:.3e}) seed={seed}"
                );
            }
        }
    }

    /// MVR should achieve a smaller reconstructability objective than the
    /// equicorrelated `s` — i.e. it is at least as powerful.
    #[test]
    fn mvr_beats_equicorrelated_objective() {
        let sigma = random_corr(50, 8, 0.1, 7);
        let obj = |s: &DVector<f64>| -> f64 {
            let p = sigma.nrows();
            let mut m = &sigma * 2.0;
            for j in 0..p {
                m[(j, j)] -= s[j];
            }
            let minv = m.try_inverse().unwrap();
            minv.diagonal().sum() + s.iter().map(|&v| 1.0 / v).sum::<f64>()
        };
        let s_mvr = knockoff_s_mvr(&sigma);
        let s_equi = knockoff_s_equicorrelated(&sigma);
        assert!(
            obj(&s_mvr) < obj(&s_equi),
            "MVR objective {:.4} not below equicorrelated {:.4}",
            obj(&s_mvr),
            obj(&s_equi)
        );
    }

    /// Under heterogeneous conditioning — a few tight clusters amid otherwise
    /// near-independent features — a single tight cluster drags `λ_min(Σ)` down,
    /// so equicorrelated assigns the same small `s` to EVERY feature. MVR
    /// instead lifts the near-independent features toward `s ≈ 1` (more power),
    /// only shrinking `s` for the truly collinear ones. This redistribution —
    /// not beating rank-deficiency, which no `s`-method can — is MVR's win.
    #[test]
    fn mvr_outpowers_equicorrelated_with_tight_clusters() {
        let p = 20;
        let mut sigma = DMatrix::<f64>::identity(p, p);
        for &(a, b) in &[(0usize, 1usize), (2, 3)] {
            sigma[(a, b)] = 0.985;
            sigma[(b, a)] = 0.985;
        }
        let s_equi = knockoff_s_equicorrelated(&sigma);
        let s_mvr = knockoff_s_mvr(&sigma);
        assert!(
            min_eig_2sigma_minus_d(&sigma, &s_mvr) > -1e-6,
            "MVR infeasible"
        );
        // the tight pairs (λ_min ≈ 0.015) drag equicorrelated down for all.
        assert!(s_equi[0] < 0.05, "equicorrelated s = {}", s_equi[0]);
        // MVR lifts the near-independent features close to their optimum s ≈ 1.
        let mean_indep_mvr = (4..p).map(|j| s_mvr[j]).sum::<f64>() / (p - 4) as f64;
        assert!(
            mean_indep_mvr > 0.7,
            "MVR independent-feature s too small: {mean_indep_mvr:.3}"
        );
        assert!(
            s_mvr.mean() > 5.0 * s_equi.mean(),
            "MVR mean {:.3} not >> equicorrelated {:.3}",
            s_mvr.mean(),
            s_equi.mean()
        );
    }
}
