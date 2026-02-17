//! Coordinate Ascent Variational Inference (CAVI) for SuSiE.
//!
//! Implements the Iterative Bayesian Stepwise Selection (IBSS) algorithm
//! with closed-form Bayes factor updates. Unlike the SGVB SuSiE which uses
//! gradient-based optimization, CAVI directly computes optimal variational
//! parameters via conjugate Bayesian updates, giving reliable feature selection
//! even when p is large.
//!
//! Reference: Wang, Sarber, Blei, Carbonetto, Stephens (2020).
//! "Variational inference for the Sum of Single Effects model"

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use log::info;

/// Parameters for CAVI SuSiE.
pub struct CaviSusieParams {
    /// Number of single-effect components L
    pub num_components: usize,
    /// Maximum IBSS iterations
    pub max_iter: usize,
    /// ELBO convergence tolerance
    pub tol: f64,
    /// Prior effect size variance σ²_0
    pub prior_variance: f64,
    /// Whether to estimate residual variance
    pub estimate_residual_variance: bool,
}

impl Default for CaviSusieParams {
    fn default() -> Self {
        Self {
            num_components: 5,
            max_iter: 100,
            tol: 1e-3,
            prior_variance: 0.2,
            estimate_residual_variance: true,
        }
    }
}

/// Results from CAVI SuSiE fit.
pub struct CaviSusieResult {
    /// Selection probabilities, shape (L, p)
    pub alpha: Vec<Vec<f64>>,
    /// Posterior means, shape (L, p)
    pub mu: Vec<Vec<f64>>,
    /// Posterior variances, shape (L, p)
    pub s2: Vec<Vec<f64>>,
    /// Posterior inclusion probabilities, shape (p,)
    pub pip: Vec<f64>,
    /// Estimated residual variance
    pub sigma2: f64,
    /// Estimated intercept
    pub intercept: f64,
    /// ELBO trace per iteration
    pub elbo_trace: Vec<f64>,
}

impl CaviSusieResult {
    /// Convert alpha to a candle Tensor of shape (L, p).
    pub fn alpha_tensor(&self, device: &Device) -> Result<Tensor> {
        let l = self.alpha.len();
        let p = self.alpha[0].len();
        let flat: Vec<f32> = self
            .alpha
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();
        Ok(Tensor::from_vec(flat, (l, p), device)?)
    }

    /// Convert mu to a candle Tensor of shape (L, p).
    pub fn mu_tensor(&self, device: &Device) -> Result<Tensor> {
        let l = self.mu.len();
        let p = self.mu[0].len();
        let flat: Vec<f32> = self
            .mu
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();
        Ok(Tensor::from_vec(flat, (l, p), device)?)
    }

    /// Convert pip to a candle Tensor of shape (p,).
    pub fn pip_tensor(&self, device: &Device) -> Result<Tensor> {
        let flat: Vec<f32> = self.pip.iter().map(|&v| v as f32).collect();
        Ok(Tensor::from_vec(flat, self.pip.len(), device)?)
    }

    /// Compute posterior mean of beta: E[β] = Σ_l (α_l ⊙ μ_l), shape (p,).
    pub fn beta_mean(&self) -> Vec<f64> {
        let p = self.pip.len();
        let mut beta = vec![0.0; p];
        for l in 0..self.alpha.len() {
            for (j, b) in beta.iter_mut().enumerate() {
                *b += self.alpha[l][j] * self.mu[l][j];
            }
        }
        beta
    }
}

/// Run CAVI SuSiE on pre-loaded data.
///
/// # Arguments
/// * `x` - Design matrix tensor, shape (n, p), f32 on CPU
/// * `y` - Response tensor, shape (n,) or (n, 1), f32 on CPU
/// * `params` - Algorithm parameters
///
/// # Returns
/// CaviSusieResult with fitted variational parameters
pub fn cavi_susie(x: &Tensor, y: &Tensor, params: &CaviSusieParams) -> Result<CaviSusieResult> {
    // Extract dimensions and convert to f64 vecs for the inner loop
    let x = x.to_dtype(DType::F64)?;
    let y_tensor = if y.dims().len() == 2 && y.dim(1)? == 1 {
        y.squeeze(1)?
    } else {
        y.clone()
    };
    let y_tensor = y_tensor.to_dtype(DType::F64)?;

    let n = x.dim(0)?;
    let p = x.dim(1)?;
    let l = params.num_components;

    // Extract X as row-major (n, p) and y as (n,)
    let x_data: Vec<f64> = x.flatten_all()?.to_vec1()?;
    let y_data: Vec<f64> = y_tensor.flatten_all()?.to_vec1()?;

    // Center y
    let y_mean: f64 = y_data.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y_data.iter().map(|&v| v - y_mean).collect();

    // Precompute d_j = ||x_j||² (column norms squared)
    let mut d = vec![0.0f64; p];
    for i in 0..n {
        for j in 0..p {
            let xij = x_data[i * p + j];
            d[j] += xij * xij;
        }
    }

    // Initialize variational parameters
    let mut alpha = vec![vec![1.0 / p as f64; p]; l];
    let mut mu = vec![vec![0.0f64; p]; l];
    let mut s2 = vec![vec![0.0f64; p]; l];

    // Initialize residual variance from data
    let y_var: f64 = y_centered.iter().map(|&v| v * v).sum::<f64>() / n as f64;
    let mut sigma2 = y_var;

    // E[β] = Σ_l (α_l ⊙ μ_l), accumulated across components
    let mut fitted = vec![0.0f64; p]; // fitted[j] = E[β_j]

    // Workspace for X'r computation
    let mut xtr = vec![0.0f64; p];
    // Workspace for X @ E[β]
    let mut x_fitted = vec![0.0f64; n];

    let mut elbo_trace = Vec::with_capacity(params.max_iter);
    let mut prev_elbo = f64::NEG_INFINITY;

    for iter in 0..params.max_iter {
        for ll in 0..l {
            // Add back component l's contribution to fitted
            let mut old_contrib = vec![0.0f64; p];
            for (j, oc) in old_contrib.iter_mut().enumerate() {
                *oc = alpha[ll][j] * mu[ll][j];
            }

            // r_l = y_centered - X @ (E[β] - old_contrib_l)
            // Compute X @ fitted and X @ old_contrib simultaneously
            // residual_i = y_centered_i - Σ_j x_ij * (fitted_j - old_contrib_j)
            // X'r_j = Σ_i x_ij * residual_i

            // Update x_fitted: x_fitted = X @ fitted
            // But it's cheaper to compute X'r directly:
            // r_i = y_centered_i - Σ_j x_ij * fitted_j + Σ_j x_ij * old_contrib_j
            //      = y_centered_i - x_fitted_i + Σ_j x_ij * old_contrib_j

            // Recompute x_fitted = X @ fitted
            for i in 0..n {
                x_fitted[i] = 0.0;
                for j in 0..p {
                    x_fitted[i] += x_data[i * p + j] * fitted[j];
                }
            }

            // Compute X'r where r = y_centered - x_fitted + X @ old_contrib
            for v in xtr.iter_mut() {
                *v = 0.0;
            }
            for i in 0..n {
                // r_i = y_centered_i - x_fitted_i + Σ_j x_ij * old_contrib_j
                let mut x_old_i = 0.0;
                for j in 0..p {
                    x_old_i += x_data[i * p + j] * old_contrib[j];
                }
                let r_i = y_centered[i] - x_fitted[i] + x_old_i;
                for j in 0..p {
                    xtr[j] += x_data[i * p + j] * r_i;
                }
            }

            // SER update for each feature j
            let sigma2_0 = params.prior_variance;
            let mut log_bf = vec![0.0f64; p];

            for j in 0..p {
                // Posterior variance: s²_j = 1 / (d_j/σ² + 1/σ²_0)
                s2[ll][j] = 1.0 / (d[j] / sigma2 + 1.0 / sigma2_0);

                // Posterior mean: μ_j = s²_j * Xtr_j / σ²
                mu[ll][j] = s2[ll][j] * xtr[j] / sigma2;

                // Log Bayes factor: 0.5 * [-log(1 + d_j*σ²_0/σ²) + μ²_j/s²_j]
                log_bf[j] = 0.5
                    * (-(1.0 + d[j] * sigma2_0 / sigma2).ln() + mu[ll][j] * mu[ll][j] / s2[ll][j]);
            }

            // α_l = softmax(log_BF) using logsumexp trick
            let max_bf = log_bf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0f64;
            for (j, &lbf) in log_bf.iter().enumerate() {
                alpha[ll][j] = (lbf - max_bf).exp();
                sum_exp += alpha[ll][j];
            }
            for j in 0..p {
                alpha[ll][j] /= sum_exp;
            }

            // Update fitted: remove old contribution, add new
            for j in 0..p {
                fitted[j] -= old_contrib[j];
                fitted[j] += alpha[ll][j] * mu[ll][j];
            }
        }

        // Update σ² if estimating
        if params.estimate_residual_variance {
            // Recompute x_fitted = X @ fitted
            for i in 0..n {
                x_fitted[i] = 0.0;
                for j in 0..p {
                    x_fitted[i] += x_data[i * p + j] * fitted[j];
                }
            }

            // σ² = (||y - X @ E[β]||² + Σ_l Σ_j α_{l,j} * s²_{l,j} * d_j) / n
            let mut rss = 0.0f64;
            for i in 0..n {
                let r = y_centered[i] - x_fitted[i];
                rss += r * r;
            }

            let mut var_correction = 0.0f64;
            for ll in 0..l {
                for (j, &dj) in d.iter().enumerate() {
                    var_correction += alpha[ll][j] * s2[ll][j] * dj;
                }
            }

            sigma2 = (rss + var_correction) / n as f64;
        }

        // Compute ELBO for convergence check
        let elbo = compute_elbo_cavi(
            &y_centered,
            &x_data,
            n,
            p,
            l,
            sigma2,
            params.prior_variance,
            &alpha,
            &mu,
            &s2,
            &d,
            &fitted,
        );
        elbo_trace.push(elbo);

        if iter % 10 == 0 || iter == params.max_iter - 1 {
            // Find top PIP features for logging
            let pip = compute_pip(&alpha, p);
            let mut top: Vec<(usize, f64)> = pip.iter().enumerate().map(|(j, &v)| (j, v)).collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_str: Vec<String> = top
                .iter()
                .take(5)
                .map(|(j, v)| format!("{}:{:.3}", j, v))
                .collect();
            info!(
                "CAVI iter {:4}: ELBO = {:12.4}, σ² = {:.6}, top PIPs: {}",
                iter,
                elbo,
                sigma2,
                top_str.join(" ")
            );
        }

        // Check convergence
        if iter > 0 && (elbo - prev_elbo).abs() < params.tol {
            info!(
                "CAVI converged at iteration {} (ΔELBO = {:.2e})",
                iter,
                (elbo - prev_elbo).abs()
            );
            break;
        }
        prev_elbo = elbo;
    }

    let pip = compute_pip(&alpha, p);
    let intercept = y_mean;

    Ok(CaviSusieResult {
        alpha,
        mu,
        s2,
        pip,
        sigma2,
        intercept,
        elbo_trace,
    })
}

/// Compute PIP = 1 - Π_l(1 - α_l) for each feature.
fn compute_pip(alpha: &[Vec<f64>], p: usize) -> Vec<f64> {
    let mut pip = vec![0.0f64; p];
    for j in 0..p {
        let mut log_one_minus = 0.0f64;
        for al in alpha {
            log_one_minus += (1.0 - al[j]).max(1e-15).ln();
        }
        pip[j] = 1.0 - log_one_minus.exp();
    }
    pip
}

/// Compute ELBO for CAVI SuSiE.
#[allow(clippy::too_many_arguments)]
fn compute_elbo_cavi(
    y: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
    l: usize,
    sigma2: f64,
    sigma2_0: f64,
    alpha: &[Vec<f64>],
    mu: &[Vec<f64>],
    s2: &[Vec<f64>],
    d: &[f64],
    fitted: &[f64],
) -> f64 {
    // Expected log-likelihood: -n/2 * log(2πσ²) - 1/(2σ²) * E[||y - Xβ||²]
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();

    // Compute x_fitted = X @ fitted
    let mut rss = 0.0f64;
    for i in 0..n {
        let mut xf = 0.0;
        for j in 0..p {
            xf += x[i * p + j] * fitted[j];
        }
        let r = y[i] - xf;
        rss += r * r;
    }

    // E[||y - Xβ||²] = ||y - X E[β]||² + Σ_l Σ_j α_{l,j} * (s²_{l,j} + μ²_{l,j}) * d_j
    //                   - Σ_l Σ_j (α_{l,j} * μ_{l,j})² * d_j
    let mut expected_rss = rss;
    for ll in 0..l {
        for (j, &dj) in d.iter().enumerate() {
            expected_rss += alpha[ll][j] * (s2[ll][j] + mu[ll][j] * mu[ll][j]) * dj;
            expected_rss -= (alpha[ll][j] * mu[ll][j]).powi(2) * dj;
        }
    }

    let ell = -0.5 * n as f64 * (ln_2pi + sigma2.ln()) - 0.5 / sigma2 * expected_rss;

    // KL divergence for each component
    let mut kl = 0.0f64;
    for ll in 0..l {
        // KL(q(b_l) || p(b_l)) where q is the SER posterior and p is the spike-and-slab prior
        // For the Gaussian slab part:
        // Σ_j α_{l,j} * [0.5 * (s²_{l,j}/σ²_0 + μ²_{l,j}/σ²_0 - 1 - ln(s²_{l,j}/σ²_0))]
        let mut kl_gauss = 0.0f64;
        for j in 0..p {
            let ratio = s2[ll][j] / sigma2_0;
            kl_gauss +=
                alpha[ll][j] * 0.5 * (ratio + mu[ll][j] * mu[ll][j] / sigma2_0 - 1.0 - ratio.ln());
        }

        // Categorical KL: Σ_j α_{l,j} * ln(α_{l,j} * p)
        let mut kl_cat = 0.0f64;
        for j in 0..p {
            if alpha[ll][j] > 1e-15 {
                kl_cat += alpha[ll][j] * (alpha[ll][j] * p as f64).ln();
            }
        }

        kl += kl_gauss + kl_cat;
    }

    ell - kl
}

/// Write CAVI SuSiE results to parquet in melted format compatible with existing output.
pub fn write_cavi_result_parquet(
    result: &CaviSusieResult,
    file_path: &str,
    feature_names: Option<&[Box<str>]>,
) -> Result<()> {
    use parquet::basic::{Compression, ConvertedType, Type as ParquetType, ZstdLevel};
    use parquet::data_type::{ByteArray, ByteArrayType, FloatType};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type;
    use std::fs::File;
    use std::sync::Arc;

    let p = result.pip.len();
    let beta = result.beta_mean();

    // Build row names
    let rows: Vec<ByteArray> = (0..p)
        .map(|j| {
            if let Some(names) = feature_names {
                ByteArray::from(names[j].as_ref())
            } else {
                ByteArray::from(j.to_string().as_bytes())
            }
        })
        .collect();

    // Column name (single output "0")
    let cols: Vec<ByteArray> = (0..p).map(|_| ByteArray::from("0".as_bytes())).collect();

    // Value columns
    let mean_vals: Vec<f32> = beta.iter().map(|&v| v as f32).collect();
    let pip_vals: Vec<f32> = result.pip.iter().map(|&v| v as f32).collect();

    let schema = Arc::new(
        Type::group_type_builder("CaviSusie")
            .with_fields(vec![
                Arc::new(
                    Type::primitive_type_builder("row", ParquetType::BYTE_ARRAY)
                        .with_repetition(parquet::basic::Repetition::REQUIRED)
                        .with_converted_type(ConvertedType::UTF8)
                        .build()?,
                ),
                Arc::new(
                    Type::primitive_type_builder("column", ParquetType::BYTE_ARRAY)
                        .with_repetition(parquet::basic::Repetition::REQUIRED)
                        .with_converted_type(ConvertedType::UTF8)
                        .build()?,
                ),
                Arc::new(
                    Type::primitive_type_builder("mean", ParquetType::FLOAT)
                        .with_repetition(parquet::basic::Repetition::REQUIRED)
                        .build()?,
                ),
                Arc::new(
                    Type::primitive_type_builder("pip", ParquetType::FLOAT)
                        .with_repetition(parquet::basic::Repetition::REQUIRED)
                        .build()?,
                ),
            ])
            .build()?,
    );

    let file = File::create(file_path)?;
    let zstd_level = ZstdLevel::try_new(5)?;
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(zstd_level))
            .build(),
    );
    let mut writer = SerializedFileWriter::new(file, schema, props)?;
    let mut row_group = writer.next_row_group()?;

    // Write row names
    if let Some(mut col_writer) = row_group.next_column()? {
        col_writer
            .typed::<ByteArrayType>()
            .write_batch(&rows, None, None)?;
        col_writer.close()?;
    }

    // Write column names
    if let Some(mut col_writer) = row_group.next_column()? {
        col_writer
            .typed::<ByteArrayType>()
            .write_batch(&cols, None, None)?;
        col_writer.close()?;
    }

    // Write mean values
    if let Some(mut col_writer) = row_group.next_column()? {
        col_writer
            .typed::<FloatType>()
            .write_batch(&mean_vals, None, None)?;
        col_writer.close()?;
    }

    // Write pip values
    if let Some(mut col_writer) = row_group.next_column()? {
        col_writer
            .typed::<FloatType>()
            .write_batch(&pip_vals, None, None)?;
        col_writer.close()?;
    }

    row_group.close()?;
    writer.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cavi_susie_basic() -> Result<()> {
        // Simple test: y = 2*x_0 + noise, with p=10 features
        let n = 100;
        let p = 10;
        let device = Device::Cpu;

        // Create design matrix with random data
        let x = Tensor::randn(0f64, 1f64, (n, p), &device)?;
        let x_data: Vec<f64> = x.flatten_all()?.to_vec1()?;

        // True signal: y = 2 * x_0 + noise
        let mut y_data = vec![0.0f64; n];
        for i in 0..n {
            y_data[i] = 2.0 * x_data[i * p] + 0.1 * rand_noise(i as u64);
        }
        let y = Tensor::from_vec(y_data, n, &device)?;
        let x_f32 = x.to_dtype(DType::F32)?;
        let y_f32 = y.to_dtype(DType::F32)?;

        let params = CaviSusieParams {
            num_components: 3,
            max_iter: 50,
            prior_variance: 1.0,
            ..Default::default()
        };

        let result = cavi_susie(&x_f32, &y_f32, &params)?;

        // Feature 0 should have highest PIP
        let max_pip_idx = result
            .pip
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_pip_idx, 0, "Feature 0 should have highest PIP");
        assert!(
            result.pip[0] > 0.5,
            "PIP for causal feature should be > 0.5, got {}",
            result.pip[0]
        );

        // ELBO should be monotonically non-decreasing (approximately)
        for i in 1..result.elbo_trace.len() {
            assert!(
                result.elbo_trace[i] >= result.elbo_trace[i - 1] - 1e-6,
                "ELBO should be non-decreasing: {} < {} at iter {}",
                result.elbo_trace[i],
                result.elbo_trace[i - 1],
                i
            );
        }

        Ok(())
    }

    /// Simple deterministic pseudo-random noise for testing.
    fn rand_noise(seed: u64) -> f64 {
        let x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-1, 1]
        (x as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
}
