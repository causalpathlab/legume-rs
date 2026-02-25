//! RSS (Regression with Summary Statistics) likelihood.
//!
//! Implements the eigenspace projection approach for summary-statistics-based
//! variational inference.
//!
//! # References
//!
//! - Zhu, X. & Stephens, M. "Bayesian large-scale multiple regression with
//!   summary statistics from genome-wide association studies."
//!   Ann. Appl. Stat. 11(3): 1561-1592, 2017.
//!
//! - Park, Y. Eigenspace projection for summary-statistics-based QTL mapping.
//!   <https://github.com/YPARK/zqtl>
//!
//! # Model
//!
//! Starting from the RSS likelihood (Zhu & Stephens, 2017):
//!
//! ```text
//! z ~ N(R β, R)    where R = X'X / n
//! ```
//!
//! We avoid forming R explicitly via SVD of X/√n = U D V',
//! giving R = V D² V'. In the K-dimensional eigenspace
//! (Park, YPARK/zqtl), define α = V'β, ỹ = V'z, and the
//! RSS log-likelihood:
//!
//! ```text
//! ℓ(β) = z'β - ½ β'Rβ = ỹ'α - ½ α'D²α
//! ```
//!
//! This is equivalent to -½ ‖D̃⁻¹ỹ - D̃V'β‖² + const, a standard
//! fixed-variance Gaussian regression in K-space with:
//!
//! ```text
//! ỹ_rss = D̃⁻¹ V' z      (K × T)
//! X̃     = D̃ V'            (K × p)
//! ```
//!
//! where D̃ = diag(√(d_k² + λ)) are the regularized singular values.
//! The regularization λ ensures D̃⁻¹ is well-conditioned.
//!
//! # Usage
//!
//! ```ignore
//! // Compute SVD once from genotypes
//! let svd = RssSvd::from_genotypes(&x, max_rank, lambda, device)?;
//!
//! // Reuse SVD for different z-score sets
//! let rss1 = RssLikelihood::new(&svd, &z_trait1)?;
//! let rss2 = RssLikelihood::new(&svd, &z_trait2)?;
//!
//! // Use svd.x_design() as the design matrix in LinearModelSGVB
//! let model = LinearModelSGVB::from_variational(susie, svd.x_design().clone(), prior, config);
//! let loss = local_reparam_loss(&model, &rss1, num_samples, kl_weight)?;
//! ```

use candle_core::{DType, Device, Result, Tensor};
use matrix_util::traits::RandomizedAlgs;

use crate::sgvb::BlackBoxLikelihood;

/// Precomputed SVD of the reference genotype matrix for RSS likelihood.
///
/// Stores D̃, V, and the derived design matrix X̃ = D̃ V'. Computed once
/// and reused across different z-score sets (traits, blocks, etc.).
pub struct RssSvd {
    /// Fat design matrix X̃ = D̃ V', shape (K, p)
    x_tilde: Tensor,
    /// Regularized inverse D̃⁻¹, shape (K,)
    d_reg_inv: Tensor,
    /// Original singular values D, shape (K,)
    singular_values: Tensor,
    /// Regularized singular values D̃ = √(D² + λ), shape (K,)
    singular_values_reg: Tensor,
    /// Right singular vectors V, shape (p, K)
    v_mat: Tensor,
    /// V', shape (K, p) — cached for projecting z-scores
    vt: Tensor,
    /// Regularization parameter λ
    lambda: f64,
    /// Effective rank K
    effective_rank: usize,
}

impl RssSvd {
    /// Compute the SVD of a standardized genotype matrix.
    ///
    /// # Arguments
    /// * `x` - Standardized genotype matrix, shape (n, p).
    ///   Columns should be mean-centered and unit-variance.
    /// * `max_rank` - Maximum rank for randomized SVD.
    /// * `lambda` - Regularization: D̃ = √(D² + λ).
    /// * `device` - Candle device for output tensors.
    pub fn from_genotypes(
        x: &Tensor,
        max_rank: usize,
        lambda: f64,
        device: &Device,
    ) -> Result<Self> {
        let (n, p) = x.dims2()?;
        let dtype = x.dtype();

        let x_data: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let x_nal = nalgebra::DMatrix::from_row_slice(n, p, &x_data);
        let scale = 1.0 / (n as f32).sqrt();
        let x_scaled = x_nal * scale;

        let (_, d_nal, v_nal) = x_scaled
            .rsvd(max_rank)
            .map_err(|e| candle_core::Error::Msg(format!("rSVD failed: {}", e)))?;

        Self::from_raw_svd(d_nal, v_nal, p, lambda, dtype, device)
    }

    /// Project z-scores into the eigenspace: ỹ = D̃⁻¹ V' z.
    ///
    /// # Arguments
    /// * `z` - Z-score matrix, shape (p, T).
    ///
    /// # Returns
    /// Projected z-scores ỹ, shape (K, T).
    pub fn project_zscores(&self, z: &Tensor) -> Result<Tensor> {
        let vt_z = self.vt.matmul(z)?; // (K, T)
        vt_z.broadcast_mul(&self.d_reg_inv.unsqueeze(1)?) // (K, T)
    }

    /// Get the fat design matrix X̃ = D̃ V', shape (K, p).
    pub fn x_design(&self) -> &Tensor {
        &self.x_tilde
    }

    /// Get the original singular values D, shape (K,).
    pub fn singular_values(&self) -> &Tensor {
        &self.singular_values
    }

    /// Get the regularized singular values D̃ = √(D² + λ), shape (K,).
    pub fn singular_values_reg(&self) -> &Tensor {
        &self.singular_values_reg
    }

    /// Get V matrix (p, K).
    pub fn v_mat(&self) -> &Tensor {
        &self.v_mat
    }

    /// Get the regularization parameter λ.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Get the effective rank K.
    pub fn effective_rank(&self) -> usize {
        self.effective_rank
    }

    /// Estimate λ by MLE under the null model z ~ N(0, (1-λ)R + λI).
    ///
    /// In eigenspace: ỹ_kt ~ N(0, σ²_k(λ)) where σ²_k(λ) = (1-λ)d²_k + λ.
    /// The NLL is convex in λ, so we bisect on the derivative:
    ///
    /// ```text
    /// dNLL/dλ = ½ Σ_k (1 - d²_k) Σ_t [1/σ²_k - ỹ²_kt/σ⁴_k]
    /// ```
    ///
    /// # Arguments
    /// * `d_sq` - Squared singular values d²_k, length K.
    /// * `y_raw` - Projected z-scores V'z, stored as y_raw[k][t].
    ///
    /// # Returns
    /// Estimated λ ∈ [0, 1].
    pub fn estimate_lambda(d_sq: &[f32], y_raw: &[Vec<f32>]) -> f64 {
        let k = d_sq.len();
        let t = if k > 0 { y_raw[0].len() } else { return 0.0 };

        // Derivative of NLL w.r.t. λ
        let grad = |lam: f64| -> f64 {
            let mut g = 0.0f64;
            for kk in 0..k {
                let dk2 = d_sq[kk] as f64;
                let sigma_sq = (1.0 - lam) * dk2 + lam;
                if sigma_sq <= 1e-12 {
                    continue;
                }
                let coeff = 1.0 - dk2; // dσ²/dλ
                let inv_s = 1.0 / sigma_sq;
                let mut sum_y2 = 0.0f64;
                for tt in 0..t {
                    let y = y_raw[kk][tt] as f64;
                    sum_y2 += y * y;
                }
                g += coeff * (t as f64 * inv_s - sum_y2 * inv_s * inv_s);
            }
            0.5 * g
        };

        // Bisection on the derivative over [0, 1]
        let mut lo = 0.0f64;
        let mut hi = 1.0f64;

        // Check boundary: if gradient at lo >= 0, minimum is at λ=0
        if grad(lo) >= 0.0 {
            return 0.0;
        }
        // If gradient at hi <= 0, minimum is at λ=1
        if grad(hi) <= 0.0 {
            return 1.0;
        }

        for _ in 0..50 {
            let mid = (lo + hi) * 0.5;
            if grad(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        (lo + hi) * 0.5
    }

    /// Create an RssSvd with λ estimated from data via MLE.
    ///
    /// Computes the rSVD, then bisects for the optimal λ under the
    /// null marginal z ~ N(0, (1-λ)R + λI).
    ///
    /// # Arguments
    /// * `x` - Standardized genotype matrix, shape (n, p).
    /// * `z` - Z-score matrix, shape (p, T).
    /// * `max_rank` - Maximum rank for randomized SVD.
    /// * `device` - Candle device.
    pub fn from_genotypes_estimate_lambda(
        x: &Tensor,
        z: &Tensor,
        max_rank: usize,
        device: &Device,
    ) -> Result<Self> {
        let (n, p) = x.dims2()?;
        let dtype = x.dtype();

        let x_data: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let x_nal = nalgebra::DMatrix::from_row_slice(n, p, &x_data);
        let scale = 1.0 / (n as f32).sqrt();
        let x_scaled = x_nal * scale;

        let (_, d_nal, v_nal) = x_scaled
            .rsvd(max_rank)
            .map_err(|e| candle_core::Error::Msg(format!("rSVD failed: {}", e)))?;

        let k = d_nal.len();
        let d_sq: Vec<f32> = d_nal.iter().map(|&di| di * di).collect();

        // Compute V'z for lambda estimation
        let v_data: Vec<f32> = v_nal.iter().cloned().collect();
        let z_data: Vec<f32> = z.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let (_, t) = z.dims2()?;

        // y_raw[kk][tt] = Σ_j V[j,kk] * z[j,tt]
        // V is (p, K) column-major, z is (p, T) row-major from candle
        let y_raw: Vec<Vec<f32>> = (0..k)
            .map(|kk| {
                (0..t)
                    .map(|tt| {
                        (0..p)
                            .map(|j| v_data[kk * p + j] * z_data[j * t + tt])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        let lambda = Self::estimate_lambda(&d_sq, &y_raw);

        // Build with estimated lambda
        Self::from_raw_svd(d_nal, v_nal, p, lambda, dtype, device)
    }

    /// Build RssSvd from pre-computed SVD components and a given λ.
    fn from_raw_svd(
        d_nal: nalgebra::DVector<f32>,
        v_nal: nalgebra::DMatrix<f32>,
        p: usize,
        lambda: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let k = d_nal.len();

        let d_vec: Vec<f32> = d_nal
            .iter()
            .map(|&di| ((di as f64) * (di as f64) + lambda).sqrt() as f32)
            .collect();
        let d_orig: Vec<f32> = d_nal.iter().cloned().collect();
        let d_inv_vec: Vec<f32> = d_vec.iter().map(|&di| 1.0 / di).collect();

        let v_data: Vec<f32> = v_nal.iter().cloned().collect();
        let v_row_major: Vec<f32> = (0..p)
            .flat_map(|row| {
                let v_ref = &v_data;
                (0..k).map(move |col| v_ref[col * p + row])
            })
            .collect();

        let v_mat = Tensor::from_vec(v_row_major, (p, k), device)?.to_dtype(dtype)?;
        let vt = v_mat.t()?;

        let d_t = Tensor::from_vec(d_orig, (k,), device)?.to_dtype(dtype)?;
        let d_reg_t = Tensor::from_vec(d_vec, (k,), device)?.to_dtype(dtype)?;
        let d_reg_inv_t = Tensor::from_vec(d_inv_vec, (k,), device)?.to_dtype(dtype)?;

        let x_tilde = vt.broadcast_mul(&d_reg_t.unsqueeze(1)?)?;

        Ok(Self {
            x_tilde,
            d_reg_inv: d_reg_inv_t,
            singular_values: d_t,
            singular_values_reg: d_reg_t,
            v_mat,
            vt,
            lambda,
            effective_rank: k,
        })
    }
}

/// RSS likelihood via eigenspace projection.
///
/// Constructed from a precomputed [`RssSvd`] and z-scores.
/// Implements [`BlackBoxLikelihood`] for use with SGVB models.
///
/// References:
/// - Zhu & Stephens, Ann. Appl. Stat. 2017.
/// - <https://github.com/YPARK/zqtl>
pub struct RssLikelihood {
    /// Projected z-scores ỹ = D̃⁻¹ V' z, shape (K, T)
    y_tilde: Tensor,
}

impl RssLikelihood {
    /// Create an RSS likelihood from precomputed SVD and z-scores.
    ///
    /// # Arguments
    /// * `svd` - Precomputed SVD from [`RssSvd::from_genotypes`].
    /// * `z` - Z-score matrix, shape (p, T).
    pub fn new(svd: &RssSvd, z: &Tensor) -> Result<Self> {
        let y_tilde = svd.project_zscores(z)?;
        Ok(Self { y_tilde })
    }

    /// Create from pre-projected ỹ (already in eigenspace).
    ///
    /// Use this when you have already computed ỹ = D̃⁻¹ V' z
    /// and want to avoid recomputing it.
    pub fn from_projected(y_tilde: Tensor) -> Self {
        Self { y_tilde }
    }

    /// Get the projected z-scores ỹ = D̃⁻¹ V' z, shape (K, T).
    pub fn y_tilde(&self) -> &Tensor {
        &self.y_tilde
    }
}

impl BlackBoxLikelihood for RssLikelihood {
    /// Evaluate the RSS log-likelihood in K-space.
    ///
    /// ```text
    /// log p(z | β) = -½ ‖ỹ - Σ_j η_j‖²
    /// ```
    ///
    /// where η_j = X̃_j β_j (K × T) are additive components and
    /// ỹ = D̃⁻¹ V' z (K × T).
    ///
    /// # Arguments
    /// * `etas` - Each `etas[j]` has shape (S, K, T). All are summed
    ///   to form the combined predictor.
    ///
    /// # Returns
    /// Log-likelihood values, shape (S,).
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        // Sum all additive components: sparse + intercept [+ polygenic]
        let mut eta = etas[0].clone();
        for e in &etas[1..] {
            eta = eta.broadcast_add(e)?;
        }
        let diff_sq = eta.broadcast_sub(&self.y_tilde)?.sqr()?; // (S, K, T)
        diff_sq.sum(2)?.sum(1)? * (-0.5) // (S,)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rss_construction() -> Result<()> {
        let device = Device::Cpu;
        let n = 100;
        let p = 50;
        let t = 3;
        let k = 30;
        let lambda = 0.1 / k as f64;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let z = Tensor::randn(0f32, 1f32, (p, t), &device)?;

        let svd = RssSvd::from_genotypes(&x, k, lambda, &device)?;
        let rss = RssLikelihood::new(&svd, &z)?;

        assert_eq!(svd.effective_rank(), k);
        assert_eq!(svd.x_design().dims(), &[k, p]);
        assert_eq!(rss.y_tilde().dims(), &[k, t]);
        assert_eq!(svd.v_mat().dims(), &[p, k]);

        println!("n={}, p={}, K={}, T={}, λ={:.4e}", n, p, k, t, svd.lambda());
        Ok(())
    }

    #[test]
    fn test_svd_reuse() -> Result<()> {
        let device = Device::Cpu;
        let n = 100;
        let p = 40;
        let k = 20;
        let lambda = 0.1 / k as f64;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let svd = RssSvd::from_genotypes(&x, k, lambda, &device)?;

        // Same SVD, different z-scores
        let z1 = Tensor::randn(0f32, 1f32, (p, 3), &device)?;
        let z2 = Tensor::randn(0f32, 1f32, (p, 5), &device)?;

        let rss1 = RssLikelihood::new(&svd, &z1)?;
        let rss2 = RssLikelihood::new(&svd, &z2)?;

        assert_eq!(rss1.y_tilde().dims(), &[k, 3]);
        assert_eq!(rss2.y_tilde().dims(), &[k, 5]);

        // Both use the same design matrix
        assert_eq!(svd.x_design().dims(), &[k, p]);
        Ok(())
    }

    #[test]
    fn test_rss_evaluation() -> Result<()> {
        let device = Device::Cpu;
        let n = 100;
        let p = 30;
        let t = 2;
        let s = 5;
        let k = 20;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let z = Tensor::randn(0f32, 1f32, (p, t), &device)?;

        let svd = RssSvd::from_genotypes(&x, k, 0.1 / k as f64, &device)?;
        let rss = RssLikelihood::new(&svd, &z)?;

        let beta = Tensor::randn(0f32, 0.1f32, (s, p, t), &device)?;
        let eta = svd.x_design().unsqueeze(0)?.broadcast_matmul(&beta)?;
        let llik = rss.log_likelihood(&[&eta])?;

        assert_eq!(llik.dims(), &[s]);
        let vals: Vec<f32> = llik.to_vec1()?;
        for v in &vals {
            assert!(v.is_finite());
        }
        Ok(())
    }

    #[test]
    fn test_rss_susie_recovery() -> Result<()> {
        use crate::sgvb::{
            local_reparam_loss, GaussianPrior, LinearModelSGVB, SGVBConfig, SusieVar,
        };
        use candle_nn::{Optimizer, VarBuilder, VarMap};

        let device = Device::Cpu;
        let dtype = DType::F32;
        let n = 200;
        let p = 50;
        let t = 1;
        let l = 3;
        let k = 40;
        let lambda = 0.1 / k as f64;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        let mut beta_data = vec![0.0f32; p];
        beta_data[10] = 3.0;
        let true_beta = Tensor::from_vec(beta_data, (p, t), &device)?;

        let y = (x.matmul(&true_beta)? + Tensor::randn(0f32, 1f32, (n, t), &device)?)?;
        let z = (x.t()?.matmul(&y)? / (n as f64).sqrt())?;

        let svd = RssSvd::from_genotypes(&x, k, lambda, &device)?;
        let rss = RssLikelihood::new(&svd, &z)?;
        println!("K = {}, λ = {:.4e}", svd.effective_rank(), svd.lambda());

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let susie = SusieVar::new(vb.pp("susie"), l, p, t)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::new(50);
        let model = LinearModelSGVB::from_variational(susie, svd.x_design().clone(), prior, config);

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.01)?;
        for i in 0..300 {
            let loss = local_reparam_loss(&model, &rss, 50, 1.0)?;
            optimizer.backward_step(&loss)?;
            if i % 50 == 0 {
                let lv: f32 = loss.to_scalar()?;
                let pip10: f32 = model.variational.pip()?.get(10)?.get(0)?.to_scalar()?;
                println!("iter {}: loss={:.4}, PIP[10]={:.4}", i, lv, pip10);
            }
        }

        let pip = model.variational.pip()?;
        let pip_10: f32 = pip.get(10)?.get(0)?.to_scalar()?;
        let mut other_sum = 0.0f32;
        for j in 0..p {
            if j != 10 {
                other_sum += pip.get(j)?.get(0)?.to_scalar::<f32>()?;
            }
        }
        let other_mean = other_sum / (p - 1) as f32;

        println!("PIP[10]={:.4}, others mean={:.4}", pip_10, other_mean);
        assert!(pip_10 > other_mean * 3.0);
        Ok(())
    }

    #[test]
    fn test_regularization_effect() -> Result<()> {
        let device = Device::Cpu;
        let n = 50;
        let p = 100;
        let k = 30;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        let svd_small = RssSvd::from_genotypes(&x, k, 1e-6, &device)?;
        let svd_default = RssSvd::from_genotypes(&x, k, 0.1 / k as f64, &device)?;

        let d_small: Vec<f32> = svd_small.singular_values_reg().to_vec1()?;
        let d_default: Vec<f32> = svd_default.singular_values_reg().to_vec1()?;

        let last = d_small.len() - 1;
        println!(
            "Smallest D̃: small_λ={:.4e}, default_λ={:.4e}",
            d_small[last], d_default[last]
        );
        assert!(d_default[last] >= d_small[last]);
        Ok(())
    }
}
