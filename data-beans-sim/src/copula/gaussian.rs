//! Low-rank Gaussian copula covariance: fit a rank-`r` factor `F = U·diag(σ)/√N`
//! directly from the per-cell z-score matrix `Z` via RSVD, then **per-row
//! rescale** so the sampled `z*[h] = F[h,:]·η + ridge_sd[h]·ε` has unit
//! marginal variance.
//!
//! Without the per-row rescale, fitting `Σ̂ = (1/N)·Z·Zᵀ + λI` gives a
//! covariance, not a correlation. PIT-Z scores from discrete NB marginals are
//! **sub-unit-variance** (especially for sparse genes where the zero mass
//! piles around `Φ⁻¹(0.5·P(X=0))`), so `Σ̂[h,h] ≪ 1`, sampled `z*[h]` lives
//! near 0, and `Φ(z*[h])` never crosses `F(0)` for high-zero-mass marginals
//! → inverse-CDF returns 0 every time. A Gaussian copula needs the *latent*
//! variable to be standard normal per gene; the off-diagonals of the original
//! covariance carry the dependence structure, but the diagonal must be 1.
//!
//! Concretely: after `F = U·diag(σ)/√N` and ridge `λ`, the per-row variance
//! is `v_h = ‖F[h,:]‖² + λ`. We divide row `h` of `F` by `√v_h` and store a
//! per-row `ridge_sd[h] = √λ / √v_h`, so:
//!   - **diag** of `F'·F'ᵀ + diag(ridge_sd²)` is exactly 1,
//!   - **off-diag** correlations are `F[h,:]·F[k,:] / √(v_h·v_k)`, the
//!     standard covariance-to-correlation rescale,
//!   - degenerate rows (`v_h ≈ 0`, only possible when `λ = 0`) collapse to
//!     `z*[h] = 0`, falling back to median NB sampling.
//!
//! By construction this:
//!  - **never forms the `G × G` covariance** (memory is `G·r`, not `G²`),
//!  - **honors the empirical rank** — for `N < G` the empirical Σ̂ is rank ≤ N
//!    anyway, so a low-rank factor captures the real structure instead of
//!    padding it with regularization noise,
//!  - keeps the per-row ridge as **isotropic noise that fills directions
//!    outside the top-r subspace**, mirroring the previous Cholesky
//!    regularization but rescaled to maintain unit marginal variance.

use matrix_util::traits::RandomizedAlgs;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct CopulaCovariance {
    /// Per-row-rescaled factor: `F'[h, k] = (U[h, k]·σ_k/√N) / √(‖F[h,:]‖²+λ)`.
    /// Shape `(g, rank)`. After the rescale, `‖F'[h,:]‖² + ridge_sd[h]² = 1`,
    /// so sampled `z*[h]` is unit-variance — the latent of a *correlation*
    /// copula, not a covariance.
    pub factor: DMatrix<f32>,
    /// Per-row ridge SD: `ridge_sd[h] = √λ / √(‖F[h,:]‖² + λ)`. Length `g`.
    pub ridge_sd: DVector<f32>,
}

impl CopulaCovariance {
    pub fn dim(&self) -> usize {
        self.factor.nrows()
    }

    pub fn rank(&self) -> usize {
        self.factor.ncols()
    }

    /// Fit by running RSVD on `Z` (genes × cells), forming `F = U·diag(σ)/√N`,
    /// then **per-row rescaling** so each row has unit total variance under
    /// `Var(z*[h]) = ‖F[h,:]‖² + ridge_sd[h]²`. `rank` caps the factor rank;
    /// effective rank is `min(rank, g, n)`.
    pub fn fit(z: &DMatrix<f32>, rank: usize, regularization: f32) -> anyhow::Result<Self> {
        let g = z.nrows();
        let n = z.ncols();
        if n < 2 {
            anyhow::bail!("need ≥2 cells to fit a copula covariance");
        }
        let r_eff = rank.max(1).min(g).min(n);
        let (u, sigmas, _vt) = z.rsvd(r_eff)?;
        let actual_rank = u.ncols().min(sigmas.len());
        let scale = 1.0 / (n as f32).sqrt();
        let mut factor = if u.ncols() > actual_rank {
            u.columns(0, actual_rank).into_owned()
        } else {
            u
        };
        for k in 0..actual_rank {
            let w = sigmas[k] * scale;
            factor.column_mut(k).scale_mut(w);
        }
        // Per-row rescale to unit marginal variance (correlation copula).
        // v_h = ‖F[h,:]‖² + λ; F'[h,:] = F[h,:]/√v_h; ridge_sd[h] = √λ/√v_h.
        let lambda = regularization.max(0.0);
        let mut ridge_sd = DVector::<f32>::zeros(g);
        for h in 0..g {
            let row_norm_sq: f32 = (0..actual_rank).map(|k| factor[(h, k)].powi(2)).sum();
            let v_h = row_norm_sq + lambda;
            if v_h > 1e-12 {
                let inv_sd = 1.0 / v_h.sqrt();
                for k in 0..actual_rank {
                    factor[(h, k)] *= inv_sd;
                }
                ridge_sd[h] = lambda.sqrt() * inv_sd;
            }
            // else: degenerate row (F[h,:]=0 ∧ λ=0) → leave factor row at 0
            // and ridge_sd[h]=0 → z*[h]=0 → median NB fallback.
        }
        Ok(Self { factor, ridge_sd })
    }

    /// Sample `z*[h] = F[h,:]·η + ridge_sd[h]·ε_h`, `η ~ N(0, I_rank)`,
    /// `ε ~ N(0, I_g)`. After the per-row rescale in `fit`, `Var(z*[h]) = 1`.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f32> {
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let eta = DVector::from_fn(self.rank(), |_, _| normal.sample(rng));
        let mut z = &self.factor * eta;
        for i in 0..self.dim() {
            z[i] += self.ridge_sd[i] * normal.sample(rng);
        }
        z
    }

    /// Take the top `new_rank` columns of `self.factor` and re-fill the
    /// per-row residual to keep `‖F[h,:]‖² + ridge_sd[h]² = 1`. Used by
    /// the two-stage simulator's `--batch-program biology` mode: batch
    /// shifts ride the dominant gene-gene correlation axes from the
    /// reference-fitted copula.
    pub fn truncate_rank(&self, new_rank: usize) -> Self {
        let g = self.dim();
        let r_keep = new_rank.min(self.rank());
        let factor = if r_keep == 0 {
            DMatrix::<f32>::zeros(g, 0)
        } else {
            self.factor.columns(0, r_keep).into_owned()
        };
        let mut ridge_sd = DVector::<f32>::zeros(g);
        for h in 0..g {
            let row_norm_sq: f32 = (0..r_keep).map(|k| factor[(h, k)].powi(2)).sum();
            ridge_sd[h] = (1.0 - row_norm_sq).max(0.0).sqrt();
        }
        Self { factor, ridge_sd }
    }

    /// Construct a fresh low-rank factor from iid N(0, 1) entries, then
    /// per-row-rescale so each gene has unit marginal variance under
    /// `Var(z*[h]) = ‖F[h,:]‖² + ridge_sd[h]² = 1`. Used by the two-stage
    /// simulator's `--batch-program random` mode: batch shifts live on a
    /// random `rank`-dim subspace independent of biology.
    ///
    /// `rank = 0` produces an isotropic sampler (`F` empty, all weight on
    /// the per-gene ridge), equivalent to Splatter-style iid log-normal
    /// batch effects.
    pub fn random_low_rank<R: Rng + ?Sized>(g: usize, rank: usize, rng: &mut R) -> Self {
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut factor = DMatrix::<f32>::from_fn(g, rank, |_, _| normal.sample(rng));
        // Cap each row's norm² at 0.99 so the residual ε always has nonzero
        // variance (avoids degenerate `ridge_sd = 0` rows). The 0.99 ceiling
        // is well above what random-Gaussian rows produce for typical g, r.
        const ROW_NORM_SQ_CAP: f32 = 0.99;
        let mut ridge_sd = DVector::<f32>::zeros(g);
        for h in 0..g {
            let row_norm_sq: f32 = (0..rank).map(|k| factor[(h, k)].powi(2)).sum();
            // Rescale the row to unit norm, then attenuate by √ROW_NORM_SQ_CAP
            // so ‖F[h,:]‖² = ROW_NORM_SQ_CAP and the residual gets the rest.
            let target_norm = ROW_NORM_SQ_CAP.sqrt();
            let scale = if row_norm_sq > 1e-12 {
                target_norm / row_norm_sq.sqrt()
            } else {
                0.0
            };
            for k in 0..rank {
                factor[(h, k)] *= scale;
            }
            let new_norm_sq = if rank > 0 { ROW_NORM_SQ_CAP } else { 0.0 };
            ridge_sd[h] = (1.0 - new_norm_sq).max(0.0).sqrt();
        }
        Self { factor, ridge_sd }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn identity_recovery_full_rank() {
        let g = 5;
        let n = 5000;
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let z = DMatrix::from_fn(g, n, |_, _| normal.sample(&mut rng));
        // Full-rank fit (rank = g) reproduces the empirical Σ̂.
        let cov = CopulaCovariance::fit(&z, g, 0.0).unwrap();
        let recovered = &cov.factor * cov.factor.transpose();
        for i in 0..g {
            for j in 0..g {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (recovered[(i, j)] - expected).abs() < 0.1,
                    "Σ[{},{}]={} expected {}",
                    i,
                    j,
                    recovered[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn correlation_round_trip() {
        // Build a known Σ with a strong off-diagonal block.
        let g = 3;
        let mut sigma = DMatrix::<f32>::identity(g, g);
        sigma[(0, 1)] = 0.7;
        sigma[(1, 0)] = 0.7;
        let l = sigma.cholesky().unwrap().l();

        let n = 10_000;
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        let mut z = DMatrix::<f32>::zeros(g, n);
        for j in 0..n {
            let eta = DVector::from_fn(g, |_, _| normal.sample(&mut rng));
            z.set_column(j, &(&l * eta));
        }

        let cov = CopulaCovariance::fit(&z, g, 0.0).unwrap();
        let recovered = &cov.factor * cov.factor.transpose();
        assert!(
            (recovered[(0, 1)] - 0.7).abs() < 0.05,
            "off-diag={}",
            recovered[(0, 1)]
        );
        assert!(
            (recovered[(0, 0)] - 1.0).abs() < 0.05,
            "diag[0,0]={}",
            recovered[(0, 0)]
        );
    }

    #[test]
    fn sample_shape_and_rank_cap() {
        let g = 50;
        let n = 200;
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let z = DMatrix::from_fn(g, n, |_, _| normal.sample(&mut rng));
        // Request rank=200 but expect it capped at min(g, n) = 50.
        let cov = CopulaCovariance::fit(&z, 200, 1e-3).unwrap();
        assert!(cov.rank() <= g);
        assert_eq!(cov.dim(), g);
        let s = cov.sample(&mut rng);
        assert_eq!(s.len(), g);
    }

    #[test]
    fn rank_one_recovery() {
        // Σ = v vᵀ with v = (1, 1, 0, 0)ᵀ. A rank-1 fit should return a
        // factor proportional to v (up to sign), and the recovered Σ̂[0, 1]
        // should land near 1.
        let g = 4;
        let n = 8000;
        let v = DVector::<f32>::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let mut z = DMatrix::<f32>::zeros(g, n);
        for k in 0..n {
            let eta: f32 = normal.sample(&mut rng);
            z.set_column(k, &(eta * &v));
        }
        let cov = CopulaCovariance::fit(&z, 1, 0.0).unwrap();
        let recovered = &cov.factor * cov.factor.transpose();
        assert!(
            (recovered[(0, 0)] - 1.0).abs() < 0.1,
            "Σ̂[0,0]={}",
            recovered[(0, 0)]
        );
        assert!(
            (recovered[(0, 1)] - 1.0).abs() < 0.1,
            "Σ̂[0,1]={}",
            recovered[(0, 1)]
        );
        assert!(
            recovered[(2, 2)].abs() < 0.05,
            "Σ̂[2,2]={} (should be ~0)",
            recovered[(2, 2)]
        );
    }

    /// Sub-unit-variance input (mimics PIT-Z from sparse discrete NB) — the
    /// per-row rescale in `fit` must drag every marginal back to Var≈1.
    /// Without the rescale this test would catch the original bug:
    /// HVG-like rows produce `z*` with Var ≪ 1 → `Φ(z*) ≈ 0.5` → inverse-CDF
    /// always returns 0.
    #[test]
    fn unit_marginal_variance_on_subunit_input() {
        let g = 30;
        let n = 5000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2026);
        // Each row drawn N(0, σ_h²) with σ_h ranging over [0.05, 1.0].
        // Pre-fix code would carry these σ_h into z*; post-fix code must
        // rescale every row back to unit variance.
        let row_sds: Vec<f32> = (0..g)
            .map(|h| 0.05 + 0.95 * (h as f32) / (g as f32 - 1.0))
            .collect();
        let mut z = DMatrix::<f32>::zeros(g, n);
        for h in 0..g {
            let normal = Normal::new(0.0_f32, row_sds[h]).unwrap();
            for j in 0..n {
                z[(h, j)] = normal.sample(&mut rng);
            }
        }
        let cov = CopulaCovariance::fit(&z, 5, 1e-3).unwrap();

        // Empirically estimate Var(z*[h]) over many draws.
        let n_samples = 20_000;
        let mut s2 = vec![0.0_f64; g];
        for _ in 0..n_samples {
            let s = cov.sample(&mut rng);
            for h in 0..g {
                let v = s[h] as f64;
                s2[h] += v * v;
            }
        }
        for h in 0..g {
            let var_h = s2[h] / n_samples as f64;
            assert!(
                (var_h - 1.0).abs() < 0.08,
                "row {} (input σ={:.2}): Var(z*[{}]) = {:.3} (expected ≈1)",
                h,
                row_sds[h],
                h,
                var_h
            );
        }
    }

    /// Off-diagonal correlation must be preserved under the per-row rescale.
    /// We build Σ_in with strong off-diagonal block (rows 0,1 strongly
    /// correlated; rows 2,3 independent) then check Σ_out has corr(0,1)≈high,
    /// corr(0,2)≈0 — i.e. structure is conserved when only diag is normalized.
    #[test]
    fn correlation_preserved_under_rescale() {
        let g = 4;
        let n = 10_000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        // Latent: rows 0,1 share η₁ (scaled differently); rows 2,3 are independent.
        let mut z = DMatrix::<f32>::zeros(g, n);
        for j in 0..n {
            let eta1: f32 = normal.sample(&mut rng);
            let eps0: f32 = normal.sample(&mut rng);
            let eps1: f32 = normal.sample(&mut rng);
            z[(0, j)] = 0.2 * (0.9 * eta1 + 0.44 * eps0); // small σ but corr w/ 1
            z[(1, j)] = 1.0 * (0.9 * eta1 + 0.44 * eps1); // big σ but corr w/ 0
            z[(2, j)] = 0.5 * normal.sample(&mut rng);
            z[(3, j)] = 1.5 * normal.sample(&mut rng);
        }
        let cov = CopulaCovariance::fit(&z, 4, 1e-4).unwrap();

        // Empirical corr from samples.
        let n_samples = 30_000;
        let mut sum = vec![0.0_f64; g];
        let mut sum2 = vec![0.0_f64; g];
        let mut sxy = vec![vec![0.0_f64; g]; g];
        for _ in 0..n_samples {
            let s = cov.sample(&mut rng);
            for h in 0..g {
                let vh = s[h] as f64;
                sum[h] += vh;
                sum2[h] += vh * vh;
                for k in 0..g {
                    sxy[h][k] += vh * (s[k] as f64);
                }
            }
        }
        let nf = n_samples as f64;
        let mean: Vec<f64> = sum.iter().map(|s| s / nf).collect();
        let var: Vec<f64> = (0..g).map(|h| sum2[h] / nf - mean[h] * mean[h]).collect();
        let corr =
            |h: usize, k: usize| (sxy[h][k] / nf - mean[h] * mean[k]) / (var[h] * var[k]).sqrt();

        // Rows 0 and 1 share latent η₁ with weight 0.9 → cosine ≈ 0.81/(0.81+0.44²) ≈ 0.808.
        let c01 = corr(0, 1);
        assert!(c01 > 0.6, "corr(z*[0], z*[1])={:.3} expected >0.6", c01);
        // Rows 2 and 3 are independent.
        let c23 = corr(2, 3);
        assert!(c23.abs() < 0.1, "corr(z*[2], z*[3])={:.3} expected ≈0", c23);
        // Cross: row 0 vs row 2 — independent.
        let c02 = corr(0, 2);
        assert!(c02.abs() < 0.1, "corr(z*[0], z*[2])={:.3} expected ≈0", c02);
        // All marginals unit variance.
        for (h, &v) in var.iter().enumerate().take(g) {
            assert!(
                (v - 1.0).abs() < 0.05,
                "Var(z*[{}])={:.3}, expected ≈1",
                h,
                v
            );
        }
    }

    /// `truncate_rank` keeps the top-r columns of an existing rank-R fit
    /// and refills the per-row residual to maintain unit marginal variance.
    /// Used by the simulator's `--batch-program biology` mode.
    #[test]
    fn truncate_rank_preserves_unit_variance() {
        let g = 25;
        let n = 4_000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2027);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let z = DMatrix::from_fn(g, n, |_, _| normal.sample(&mut rng));
        let full = CopulaCovariance::fit(&z, 10, 1e-3).unwrap();
        for new_rank in [0usize, 1, 3, 7] {
            let truncated = full.truncate_rank(new_rank);
            assert_eq!(truncated.dim(), g);
            assert_eq!(truncated.rank(), new_rank.min(full.rank()));

            let n_samples = 15_000;
            let mut s2 = vec![0.0_f64; g];
            for _ in 0..n_samples {
                let s = truncated.sample(&mut rng);
                for h in 0..g {
                    let v = s[h] as f64;
                    s2[h] += v * v;
                }
            }
            for (h, ss) in s2.iter().enumerate().take(g) {
                let var_h = ss / n_samples as f64;
                assert!(
                    (var_h - 1.0).abs() < 0.08,
                    "rank={}, row {}: Var(z*) = {:.3} (expected ≈1)",
                    new_rank,
                    h,
                    var_h
                );
            }
        }
    }

    /// `random_low_rank` builds a fresh `(g, rank)` factor with iid Gaussian
    /// entries, rescaled to a fixed row-norm cap so the residual carries
    /// the rest of the unit-variance budget. Used by `--batch-program random`.
    #[test]
    fn random_low_rank_unit_variance() {
        let g = 30;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2028);
        for rank in [0usize, 1, 3, 5] {
            let cov = CopulaCovariance::random_low_rank(g, rank, &mut rng);
            assert_eq!(cov.dim(), g);
            assert_eq!(cov.rank(), rank);

            let n_samples = 15_000;
            let mut s2 = vec![0.0_f64; g];
            for _ in 0..n_samples {
                let s = cov.sample(&mut rng);
                for h in 0..g {
                    let v = s[h] as f64;
                    s2[h] += v * v;
                }
            }
            for (h, ss) in s2.iter().enumerate().take(g) {
                let var_h = ss / n_samples as f64;
                assert!(
                    (var_h - 1.0).abs() < 0.08,
                    "rank={}, row {}: Var(z*) = {:.3} (expected ≈1)",
                    rank,
                    h,
                    var_h
                );
            }
        }
    }
}
