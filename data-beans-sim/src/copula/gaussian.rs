//! Low-rank Gaussian copula covariance: fit a rank-`r` factor `F = U·diag(σ)/√N`
//! directly from the per-cell z-score matrix `Z` via RSVD, then sample
//! `z* = F·η + √λ·ε` with `η ~ N(0, I_r)`, `ε ~ N(0, I_G)`.
//!
//! By construction `Cov(z*) = (1/N) U Σ² Uᵀ + λI = Σ̂_rank-r + λI`, which:
//!  - **never forms the `G × G` covariance** (memory is `G·r`, not `G²`),
//!  - **honors the empirical rank** — for `N < G` the empirical Σ̂ is rank ≤ N
//!    anyway, so a low-rank factor captures the real structure instead of
//!    padding it with regularization noise,
//!  - keeps the `λ I` term as **per-gene isotropic noise** that fills in the
//!    directions outside the top-r subspace, mirroring the previous Cholesky
//!    regularization.

use matrix_util::traits::RandomizedAlgs;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct CopulaCovariance {
    /// Scaled left factor `F[:, k] = U[:, k] · σ_k / √N`. Shape `(g, rank)`.
    /// `F · Fᵀ ≈ Σ̂` (rank-r approximation).
    pub factor: DMatrix<f32>,
    /// Standard deviation of the per-gene isotropic ridge added at sample time.
    pub ridge_sd: f32,
}

impl CopulaCovariance {
    pub fn dim(&self) -> usize {
        self.factor.nrows()
    }

    pub fn rank(&self) -> usize {
        self.factor.ncols()
    }

    /// Fit by running RSVD on `Z` (genes × cells) and forming `F = U·diag(σ)/√N`.
    /// `rank` caps the factor rank; effective rank is `min(rank, g, n)`.
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
        Ok(Self {
            factor,
            ridge_sd: regularization.max(0.0).sqrt(),
        })
    }

    /// Sample `z* = F · η + √λ · ε`, `η ~ N(0, I_rank)`, `ε ~ N(0, I_dim)`.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f32> {
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let eta = DVector::from_fn(self.rank(), |_, _| normal.sample(rng));
        let mut z = &self.factor * eta;
        if self.ridge_sd > 0.0 {
            for i in 0..self.dim() {
                z[i] += self.ridge_sd * normal.sample(rng);
            }
        }
        z
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
}
