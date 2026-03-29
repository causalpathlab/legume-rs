//! Nalgebra-only RSS eigenspace projection for summary-statistics fine-mapping.
//!
//! This is the CPU-only equivalent of `candle_util::sgvb::RssSvd`, using
//! `matrix_util::traits::RandomizedAlgs::rsvd` for the randomized SVD.
//!
//! # Model
//!
//! Starting from the RSS likelihood (Zhu & Stephens, 2017):
//! ```text
//! z ~ N(R β, R)    where R = X'X / n
//! ```
//!
//! We avoid forming R explicitly via SVD of X/√n = U D V',
//! giving R = V D² V'. In the K-dimensional eigenspace:
//! ```text
//! ỹ = D̃⁻¹ V' z     (K × T response)
//! X̃ = D̃ V'           (K × p design)
//! ```
//! where D̃ = √(D² + λ) are the regularized singular values.
//! This gives a standard fixed-variance Gaussian regression in K-space.

use anyhow::Result;
use matrix_util::traits::RandomizedAlgs;
use nalgebra::{DMatrix, DVector};

/// Precomputed SVD of the reference genotype matrix for RSS likelihood (nalgebra).
pub struct RssSvdNal {
    /// X̃ = D̃ V', shape (K, p)
    x_tilde: DMatrix<f32>,
    /// 1/D̃, length K
    d_reg_inv: DVector<f32>,
    /// Original singular values D, length K
    singular_values: DVector<f32>,
    /// V, shape (p, K)
    v_mat: DMatrix<f32>,
    lambda: f64,
}

impl RssSvdNal {
    /// Compute the SVD of a standardized genotype matrix.
    ///
    /// # Arguments
    /// * `x` - Standardized genotype matrix, shape (n, p). Columns should be
    ///   mean-centered and unit-variance.
    /// * `max_rank` - Maximum rank for randomized SVD.
    /// * `lambda` - Regularization: D̃ = √(D² + λ).
    pub fn from_genotypes(x: &DMatrix<f32>, max_rank: usize, lambda: f64) -> Result<Self> {
        let n = x.nrows();
        let scale = 1.0 / (n as f32).sqrt();
        let x_scaled = x * scale;

        let (_u, d, v) = x_scaled.rsvd(max_rank)?;
        let k = d.len();

        let lambda_f = lambda as f32;
        let d_reg: DVector<f32> = DVector::from_fn(k, |i, _| (d[i] * d[i] + lambda_f).sqrt());
        let d_reg_inv: DVector<f32> = DVector::from_fn(k, |i, _| 1.0 / d_reg[i]);

        // X̃ = D̃ V' → (K, p), reading V transposed without allocating V'
        let x_tilde = DMatrix::from_fn(k, v.nrows(), |i, j| d_reg[i] * v[(j, i)]);

        Ok(Self {
            x_tilde,
            d_reg_inv,
            singular_values: d,
            v_mat: v,
            lambda,
        })
    }

    /// Project z-scores into the eigenspace: ỹ = D̃⁻¹ V' z.
    ///
    /// # Arguments
    /// * `z` - Z-score matrix, shape (p, T).
    ///
    /// # Returns
    /// Projected z-scores ỹ, shape (K, T).
    pub fn project_zscores(&self, z: &DMatrix<f32>) -> DMatrix<f32> {
        let k = self.effective_rank();
        let vt_z = self.v_mat.tr_mul(z); // (K, T)
        DMatrix::from_fn(k, z.ncols(), |i, j| self.d_reg_inv[i] * vt_z[(i, j)])
    }

    pub fn x_design(&self) -> &DMatrix<f32> {
        &self.x_tilde
    }

    pub fn singular_values(&self) -> &DVector<f32> {
        &self.singular_values
    }

    pub fn singular_values_sq(&self) -> Vec<f32> {
        self.singular_values.iter().map(|&d| d * d).collect()
    }

    pub fn v_mat(&self) -> &DMatrix<f32> {
        &self.v_mat
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn effective_rank(&self) -> usize {
        self.singular_values.len()
    }

    /// Estimate the LDSC intercept in eigenspace via OLS, per trait.
    ///
    /// Under z ~ N(0, h·R + a·I), projected scores satisfy
    /// E[(V'z)²_k] = h · d²_k + a.
    ///
    /// # Arguments
    /// * `d_sq` - Squared singular values d²_k, length K.
    /// * `y_raw` - Projected z-scores V'z, stored as y_raw\[k\]\[t\].
    /// * `num_traits` - Number of traits T.
    ///
    /// # Returns
    /// `(intercepts, slopes)` each of length T.
    /// Intercepts are clamped to >= 1.0 (no deflation).
    pub fn estimate_ldsc_intercept(
        d_sq: &[f32],
        y_raw: &[Vec<f32>],
        num_traits: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let k = d_sq.len();
        if k == 0 || num_traits == 0 {
            return (vec![1.0; num_traits], vec![0.0; num_traits]);
        }

        let mean_x: f32 = d_sq.iter().sum::<f32>() / k as f32;
        let var_x: f32 = d_sq.iter().map(|&x| (x - mean_x) * (x - mean_x)).sum();

        let mut intercepts = Vec::with_capacity(num_traits);
        let mut slopes = Vec::with_capacity(num_traits);

        for tt in 0..num_traits {
            let y2: Vec<f32> = (0..k).map(|kk| y_raw[kk][tt] * y_raw[kk][tt]).collect();
            let mean_y: f32 = y2.iter().sum::<f32>() / k as f32;

            let cov: f32 = (0..k)
                .map(|kk| (d_sq[kk] - mean_x) * (y2[kk] - mean_y))
                .sum();

            let slope = if var_x.abs() > 1e-12 {
                cov / var_x
            } else {
                0.0
            };
            let intercept = (mean_y - slope * mean_x).max(1.0);

            intercepts.push(intercept);
            slopes.push(slope);
        }

        (intercepts, slopes)
    }

    /// Estimate per-trait LDSC h² (slope) for a single LD block.
    ///
    /// Performs rSVD on the block genotypes and regresses (V'z)²_k on d²_k.
    /// Returns h²_block per trait, or zeros if K <= 2.
    pub fn estimate_block_h2(
        x_block: &DMatrix<f32>,
        z_block: &DMatrix<f32>,
        max_rank: usize,
        lambda: f64,
    ) -> Result<Vec<f32>> {
        let n = x_block.nrows() as f32;
        let t = z_block.ncols();

        let svd = Self::from_genotypes(x_block, max_rank, lambda)?;
        let kk = svd.effective_rank();

        if kk <= 2 {
            return Ok(vec![0.0; t]);
        }

        let vt_z = svd.v_mat().tr_mul(z_block); // (K, T)
        let d_sq = svd.singular_values_sq();

        let y_raw: Vec<Vec<f32>> = (0..kk)
            .map(|ki| (0..t).map(|tt| vt_z[(ki, tt)]).collect())
            .collect();

        let (_intercepts, slopes) = Self::estimate_ldsc_intercept(&d_sq, &y_raw, t);
        Ok(slopes.iter().map(|&s| (s / n).max(0.0)).collect())
    }

    /// Compute LDSC intercept and rescale z-scores in-place if inflated.
    ///
    /// Returns (intercepts, slopes). Modifies `z_block` directly when correction is needed.
    pub fn ldsc_correct_zscores_inplace(&self, z_block: &mut DMatrix<f32>) -> (Vec<f32>, Vec<f32>) {
        let kk = self.effective_rank();
        let t = z_block.ncols();

        if kk <= 2 {
            return (vec![1.0; t], vec![0.0; t]);
        }

        let vt_z = self.v_mat.tr_mul(&*z_block); // (K, T)
        let d_sq = self.singular_values_sq();

        let y_raw: Vec<Vec<f32>> = (0..kk)
            .map(|ki| (0..t).map(|tt| vt_z[(ki, tt)]).collect())
            .collect();

        let (intercepts, slopes) = Self::estimate_ldsc_intercept(&d_sq, &y_raw, t);

        let any_inflated = intercepts.iter().any(|&a| a > 1.0 + 1e-6);
        if any_inflated {
            for (tt, &intercept) in intercepts.iter().enumerate() {
                let scale = 1.0 / intercept.sqrt();
                z_block.column_mut(tt).scale_mut(scale);
            }
        }

        (intercepts, slopes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_matrix(n: usize, p: usize, seed: u64) -> DMatrix<f32> {
        let mut rng = SmallRng::seed_from_u64(seed);
        DMatrix::from_fn(n, p, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        })
    }

    #[test]
    fn test_rss_svd_shapes() {
        let n = 100;
        let p = 50;
        let t = 3;
        let max_rank = 20;
        let lambda = 0.1;

        let x = random_matrix(n, p, 42);
        let z = random_matrix(p, t, 43);

        let svd = RssSvdNal::from_genotypes(&x, max_rank, lambda).unwrap();

        assert_eq!(svd.effective_rank(), max_rank);
        assert_eq!(svd.x_design().nrows(), max_rank);
        assert_eq!(svd.x_design().ncols(), p);
        assert_eq!(svd.singular_values().len(), max_rank);

        let y_tilde = svd.project_zscores(&z);
        assert_eq!(y_tilde.nrows(), max_rank);
        assert_eq!(y_tilde.ncols(), t);
    }

    #[test]
    fn test_rss_regression_recovers_signal() {
        // Simulate: y = X*beta + noise, compute z = X'y/sqrt(n)
        let n = 200;
        let p = 30;
        let causal = 10;
        let beta_true = 2.0f32;

        let mut rng = SmallRng::seed_from_u64(42);
        let x = DMatrix::from_fn(n, p, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        });
        let noise = DVector::from_fn(n, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        });
        let y = x.column(causal) * beta_true + &noise;

        // Compute z-scores: z_j = x_j'y / sqrt(n)
        let n_sqrt = (n as f32).sqrt();
        let z = DMatrix::from_fn(p, 1, |j, _| x.column(j).dot(&y) / n_sqrt);

        let svd = RssSvdNal::from_genotypes(&x, n, 0.1).unwrap();
        let y_tilde = svd.project_zscores(&z);

        // The projected z-scores at the causal SNP direction should be large
        let x_tilde = svd.x_design();
        // Simple check: x_tilde * e_causal should correlate with y_tilde
        let pred = DVector::from_fn(x_tilde.nrows(), |i, _| x_tilde[(i, causal)]);
        let corr = pred.dot(&y_tilde.column(0)) / (pred.norm() * y_tilde.column(0).norm());
        assert!(
            corr.abs() > 0.3,
            "correlation with causal direction should be substantial, got {}",
            corr
        );
    }

    #[test]
    fn test_ldsc_intercept_no_inflation() {
        let d_sq = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Under null: E[(V'z)^2_k] ≈ 1 (no signal)
        let y_raw: Vec<Vec<f32>> = d_sq.iter().map(|_| vec![1.0]).collect();
        let (intercepts, slopes) = RssSvdNal::estimate_ldsc_intercept(&d_sq, &y_raw, 1);
        assert_eq!(intercepts.len(), 1);
        assert!(intercepts[0] >= 1.0); // clamped
        assert!(slopes[0].abs() < 0.5);
    }
}
