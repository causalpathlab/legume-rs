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
    /// Prior inclusion weights π_j (length p). If None, uniform 1/p.
    /// Will be normalized to sum to 1.
    pub prior_weights: Option<Vec<f64>>,
}

impl Default for CaviSusieParams {
    fn default() -> Self {
        Self {
            num_components: 5,
            max_iter: 100,
            tol: 1e-3,
            prior_variance: 0.2,
            estimate_residual_variance: true,
            prior_weights: None,
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
    let dev = x.device();
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

    let log_prior: Vec<f64> = match &params.prior_weights {
        Some(w) => {
            let sum: f64 = w.iter().sum();
            w.iter().map(|&v| (v / sum).max(1e-30).ln()).collect()
        }
        None => vec![-(p as f64).ln(); p],
    };

    // Center y
    let y_mean = y_tensor.mean_all()?.to_scalar::<f64>()?;
    let y_centered = (y_tensor - y_mean)?;

    // Precompute d_j = ||x_j||² (column norms squared) — tensor op
    let d_tensor = x.sqr()?.sum(0)?;
    let d: Vec<f64> = d_tensor.to_vec1()?;

    // x_t = X' precomputed for X'r products
    let x_t = x.t()?;

    // Initialize variational parameters
    let mut alpha = vec![vec![1.0 / p as f64; p]; l];
    let mut mu = vec![vec![0.0f64; p]; l];
    let mut s2 = vec![vec![0.0f64; p]; l];

    let y_var: f64 = y_centered.sqr()?.mean_all()?.to_scalar()?;
    let mut sigma2 = y_var;

    // Pre-shape y as column vector for matmul residual computation
    let y_col = y_centered.reshape((n, 1))?;

    let mut fitted = vec![0.0f64; p];

    let mut elbo_trace = Vec::with_capacity(params.max_iter);
    let mut prev_elbo = f64::NEG_INFINITY;

    /// Helper: &[f64] → Tensor column vector [len, 1]
    fn to_col(v: &[f64], dev: &Device) -> Result<Tensor> {
        Ok(Tensor::from_slice(v, (v.len(), 1), dev)?)
    }

    for iter in 0..params.max_iter {
        for ll in 0..l {
            let old_contrib: Vec<f64> = (0..p).map(|j| alpha[ll][j] * mu[ll][j]).collect();

            // r = y - X @ (fitted - old_contrib): fuse into one matmul
            let net: Vec<f64> = (0..p).map(|j| fitted[j] - old_contrib[j]).collect();
            let r = (&y_col - x.matmul(&to_col(&net, dev)?)?)?;

            // xtr = X' @ r  [p, 1] → Vec
            let xtr: Vec<f64> = x_t.matmul(&r)?.flatten_all()?.to_vec1()?;

            // SER update (scalar — involves log/exp/branching, not worth tensorizing)
            let sigma2_0 = params.prior_variance;
            let mut log_bf = vec![0.0f64; p];

            for j in 0..p {
                s2[ll][j] = 1.0 / (d[j] / sigma2 + 1.0 / sigma2_0);
                mu[ll][j] = s2[ll][j] * xtr[j] / sigma2;
                log_bf[j] = 0.5
                    * (-(1.0 + d[j] * sigma2_0 / sigma2).ln() + mu[ll][j] * mu[ll][j] / s2[ll][j])
                    + log_prior[j];
            }

            // Softmax: α_l = softmax(log_bf)
            let max_bf = log_bf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0f64;
            for (j, &lbf) in log_bf.iter().enumerate() {
                alpha[ll][j] = (lbf - max_bf).exp();
                sum_exp += alpha[ll][j];
            }
            for a in alpha[ll].iter_mut().take(p) {
                *a /= sum_exp;
            }

            // Update fitted
            for j in 0..p {
                fitted[j] += alpha[ll][j] * mu[ll][j] - old_contrib[j];
            }
        }

        // Update σ²
        if params.estimate_residual_variance {
            let residual = (&y_col - x.matmul(&to_col(&fitted, dev)?)?)?;
            let rss: f64 = residual.sqr()?.sum_all()?.to_scalar()?;

            let var_correction: f64 = (0..l)
                .map(|ll| {
                    d.iter()
                        .enumerate()
                        .map(|(j, &dj)| alpha[ll][j] * s2[ll][j] * dj)
                        .sum::<f64>()
                })
                .sum();

            sigma2 = (rss + var_correction) / n as f64;
        }

        // ELBO
        let elbo = compute_elbo_cavi(
            &x,
            &y_col,
            n,
            l,
            sigma2,
            params.prior_variance,
            &alpha,
            &mu,
            &s2,
            &d,
            &fitted,
            &log_prior,
        )?;
        elbo_trace.push(elbo);

        if iter % 10 == 0 || iter == params.max_iter - 1 {
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
    x: &Tensor,
    y_col: &Tensor,
    n: usize,
    l: usize,
    sigma2: f64,
    sigma2_0: f64,
    alpha: &[Vec<f64>],
    mu: &[Vec<f64>],
    s2: &[Vec<f64>],
    d: &[f64],
    fitted: &[f64],
    log_prior: &[f64],
) -> Result<f64> {
    let dev = x.device();
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();

    // RSS via tensor matmul: ||y - X @ fitted||²
    let fitted_t = Tensor::from_slice(fitted, (fitted.len(), 1), dev)?;
    let residual = (y_col - x.matmul(&fitted_t)?)?;
    let rss: f64 = residual.sqr()?.sum_all()?.to_scalar()?;

    // E[||y - Xβ||²] correction terms (scalar — small per-component work)
    let mut expected_rss = rss;
    for ll in 0..l {
        for (j, &dj) in d.iter().enumerate() {
            expected_rss += alpha[ll][j] * (s2[ll][j] + mu[ll][j] * mu[ll][j]) * dj;
            expected_rss -= (alpha[ll][j] * mu[ll][j]).powi(2) * dj;
        }
    }

    let ell = -0.5 * n as f64 * (ln_2pi + sigma2.ln()) - 0.5 / sigma2 * expected_rss;

    // KL divergence (scalar — involves log/branching per element)
    let mut kl = 0.0f64;
    for ll in 0..l {
        let mut kl_gauss = 0.0f64;
        for j in 0..d.len() {
            let ratio = s2[ll][j] / sigma2_0;
            kl_gauss +=
                alpha[ll][j] * 0.5 * (ratio + mu[ll][j] * mu[ll][j] / sigma2_0 - 1.0 - ratio.ln());
        }

        let mut kl_cat = 0.0f64;
        for (j, &lp) in log_prior.iter().enumerate() {
            if alpha[ll][j] > 1e-15 {
                kl_cat += alpha[ll][j] * (alpha[ll][j].ln() - lp);
            }
        }

        kl += kl_gauss + kl_cat;
    }

    Ok(ell - kl)
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
