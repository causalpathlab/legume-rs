//! Von Mises-Fisher likelihood for directional data on the unit hypersphere.
//!
//! The vMF distribution models unit vectors where cosine similarity (angle) matters
//! more than magnitude. Useful for comparing normalized embeddings or topic distributions.
//!
//! # Model
//! ```text
//! p(y | μ, κ) = C_d(κ) * exp(κ * μᵀy)
//! ```
//! where:
//! - y: observed unit vector (n-dimensional)
//! - μ: mean direction (unit vector)
//! - κ: concentration parameter (κ > 0, higher = more concentrated)
//! - C_d(κ): normalizing constant involving Bessel function
//!
//! # References
//! - Banerjee et al. (2005) "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"
//! - Hasnat et al. (2017) "vMF Mixture Model-based Deep Learning" (arXiv:1706.04264)

use candle_core::{Result, Tensor};

use crate::sgvb::BlackBoxLikelihood;

/// L2 normalize a tensor along the specified dimension.
pub fn l2_normalize_dim(x: &Tensor, dim: usize) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(dim)?.sqrt()?;
    let norm_safe = (norm + 1e-8)?; // Avoid division by zero
    x.broadcast_div(&norm_safe)
}

/// Fast lgamma approximation (Paul Mineiro's fastlgamma).
///
/// lgamma(x) ≈ -2.081061466 - x + 0.0833333/(x+3) - log(x*(1+x)*(2+x)) + (2.5+x)*log(x+3)
#[inline]
fn fast_lgamma(x: f64) -> f64 {
    let logterm = (x * (1.0 + x) * (2.0 + x)).ln();
    let xp3 = 3.0 + x;
    -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * xp3.ln()
}

/// Softplus: log(1 + exp(x)) with numerical stability.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 10.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Log-sum-exp: log(exp(a) + exp(b)) with numerical stability.
#[inline]
fn log_sum_exp(log_a: f64, log_b: f64) -> f64 {
    if log_a > log_b {
        log_a + softplus(log_b - log_a)
    } else {
        log_b + softplus(log_a - log_b)
    }
}

/// Compute log I_p(x) - logarithm of modified Bessel function of first kind.
///
/// Uses series expansion with log-sum-exp for numerical stability:
/// log I_p(x) = p * log(x/2) + log(Σ_j f(x,j))
/// where log f(x,j) = 2j * log(x/2) - lgamma(j+1) - lgamma(p+j+1)
pub fn log_bessel_i(p: f64, x: f64) -> f64 {
    if x < 1e-10 {
        // I_p(0) = 0 for p > 0
        return f64::NEG_INFINITY;
    }

    let log_x_half = (x * 0.5).ln();

    // Number of terms in series (heuristic: 3*p is usually enough)
    let n_terms = ((3.0 * p).max(30.0)).min(200.0) as usize;

    // Initialize with j=0 term: log f(x,0) = -lgamma(p+1)
    let mut log_sum_series = -fast_lgamma(p + 1.0);

    for j in 1..n_terms {
        let jf = j as f64;
        // log f(x,j) = 2j * log(x/2) - lgamma(j+1) - lgamma(p+j+1)
        let log_f = 2.0 * jf * log_x_half - fast_lgamma(jf + 1.0) - fast_lgamma(p + jf + 1.0);
        log_sum_series = log_sum_exp(log_sum_series, log_f);
    }

    log_sum_series + p * log_x_half
}

/// Compute the log normalizer for the von Mises-Fisher distribution.
///
/// ```text
/// log C_d(κ) = (d/2 - 1) * log(κ) - (d/2) * log(2π) - log I_{d/2-1}(κ)
/// ```
pub fn vmf_log_normalizer(dim: usize, kappa: f64) -> f64 {
    let d = dim as f64;
    let v = d / 2.0 - 1.0; // Order of Bessel function (d/2 - 1)

    if kappa < 1e-10 {
        // For very small kappa, distribution approaches uniform on sphere
        let log_surface_area = (d / 2.0) * std::f64::consts::TAU.ln() - fast_lgamma(d / 2.0);
        return -log_surface_area;
    }

    let log_bessel = log_bessel_i(v, kappa);

    // log C_d(κ) = (d/2 - 1) * log(κ) - (d/2) * log(2π) - log I_{d/2-1}(κ)
    v * kappa.ln() - (d / 2.0) * std::f64::consts::TAU.ln() - log_bessel
}

/// Estimate optimal kappa using MLE approximation (Banerjee et al. 2005).
///
/// Given mean resultant length r̄ (mean cosine similarity), estimate:
/// ```text
/// κ̂ ≈ r̄(d - r̄²) / (1 - r̄²)
/// ```
///
/// This is the closed-form M-step in EM for vMF mixture models.
pub fn estimate_kappa_mle(mean_cos_sim: f64, dim: usize) -> f64 {
    let r_bar = mean_cos_sim.clamp(0.0, 0.9999); // Avoid division by zero
    let d = dim as f64;

    // Banerjee approximation
    let kappa = r_bar * (d - r_bar * r_bar) / (1.0 - r_bar * r_bar);

    // Clamp to reasonable range
    kappa.clamp(0.1, 1000.0)
}

/// Suggest initial kappa based on dimension (Hasnat et al. 2017).
///
/// Returns κ = √(d/2), which is a reasonable starting point.
pub fn suggest_kappa_init(dim: usize) -> f64 {
    ((dim as f64) / 2.0).sqrt()
}

/// Von Mises-Fisher likelihood with FIXED concentration parameter.
///
/// More stable than learning kappa during training. Includes proper log normalizer
/// for valid model comparison across different kappa values.
///
/// # Per-Gene Scaling (Option A)
/// When `gene_weights` is provided, each gene gets a scaled concentration:
/// ```text
/// κ_g = κ_base * w_g
/// ```
/// This allows different genes to have different "confidence" levels in the model.
/// Genes with higher weights contribute more strongly to the likelihood.
///
/// # Usage with SGVB
/// ```ignore
/// // Standard usage (uniform kappa)
/// let likelihood = VmfFixedKappaLikelihood::new(y_data, 16.0)?;
///
/// // With per-gene scaling
/// let weights = Tensor::from_vec(gene_weights, (n_genes,), &device)?;
/// let likelihood = VmfFixedKappaLikelihood::with_gene_weights(y_data, 16.0, weights)?;
/// ```
pub struct VmfFixedKappaLikelihood {
    /// Unit-normalized observed directions, shape (n, k)
    y: Tensor,
    /// Base concentration parameter
    kappa: f64,
    /// Dimension (number of features/genes)
    dim: usize,
    /// Number of topics/outputs
    n_topics: usize,
    /// Per-gene weights for scaling kappa, shape (n,). None = uniform.
    gene_weights: Option<Tensor>,
    /// Precomputed log normalizer(s).
    /// - If gene_weights is None: single f64 value
    /// - If gene_weights is Some: Tensor of shape (n,) with per-gene normalizers
    log_normalizer_uniform: f64,
    /// Per-gene log normalizers when using gene_weights, shape (n,)
    log_normalizer_per_gene: Option<Tensor>,
}

impl VmfFixedKappaLikelihood {
    /// Create a new vMF likelihood with fixed kappa.
    ///
    /// # Arguments
    /// * `y` - Observed directions, shape (n, k). Will be L2-normalized along dim 0.
    /// * `kappa` - Fixed concentration parameter (must be > 0)
    pub fn new(y: Tensor, kappa: f64) -> Result<Self> {
        let y_normalized = l2_normalize_dim(&y, 0)?;
        Self::from_normalized(y_normalized, kappa)
    }

    /// Create from pre-normalized data (skip normalization).
    pub fn from_normalized(y: Tensor, kappa: f64) -> Result<Self> {
        let dim = y.dim(0)?;
        let n_topics = y.dim(1)?;
        let log_normalizer = vmf_log_normalizer(dim, kappa);

        Ok(Self {
            y,
            kappa,
            dim,
            n_topics,
            gene_weights: None,
            log_normalizer_uniform: log_normalizer,
            log_normalizer_per_gene: None,
        })
    }

    /// Create from pre-normalized data with per-gene weights.
    ///
    /// Each gene gets a scaled concentration: κ_g = κ_base * w_g
    ///
    /// # Arguments
    /// * `y` - Pre-normalized observed directions, shape (n, k)
    /// * `kappa` - Base concentration parameter
    /// * `gene_weights` - Per-gene scaling weights, shape (n,). Should be > 0.
    pub fn with_gene_weights(y: Tensor, kappa: f64, gene_weights: Tensor) -> Result<Self> {
        let dim = y.dim(0)?;
        let n_topics = y.dim(1)?;
        let n_genes = gene_weights.dim(0)?;

        assert_eq!(
            dim, n_genes,
            "gene_weights length ({}) must match y dimension ({})",
            n_genes, dim
        );

        // Compute per-gene log normalizers
        let weights_vec: Vec<f32> = gene_weights.to_vec1()?;
        let mut log_norm_vec = Vec::with_capacity(n_genes);
        for &w in &weights_vec {
            let kappa_g = kappa * (w as f64);
            log_norm_vec.push(vmf_log_normalizer(dim, kappa_g) as f32);
        }
        let log_normalizer_per_gene =
            Tensor::from_vec(log_norm_vec, (n_genes,), gene_weights.device())?;

        let log_normalizer_uniform = vmf_log_normalizer(dim, kappa);

        Ok(Self {
            y,
            kappa,
            dim,
            n_topics,
            gene_weights: Some(gene_weights),
            log_normalizer_uniform,
            log_normalizer_per_gene: Some(log_normalizer_per_gene),
        })
    }

    /// Get the base kappa value.
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Get the dimension (number of features/genes).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check if per-gene weights are used.
    pub fn has_gene_weights(&self) -> bool {
        self.gene_weights.is_some()
    }

    /// Get the gene weights if set.
    pub fn gene_weights(&self) -> Option<&Tensor> {
        self.gene_weights.as_ref()
    }

    /// Get the uniform log normalizer C_d(κ).
    /// Note: when using per-gene weights, use total_log_normalizer() instead.
    pub fn log_normalizer(&self) -> f64 {
        self.log_normalizer_uniform
    }

    /// Compute total log normalizer contribution.
    /// - Uniform: log_normalizer * n_topics
    /// - Per-gene: sum over genes of per-gene log normalizers * n_topics
    pub fn total_log_normalizer(&self) -> Result<f64> {
        match &self.log_normalizer_per_gene {
            Some(per_gene) => {
                let sum: f32 = per_gene.sum_all()?.to_scalar()?;
                Ok((sum as f64) * self.n_topics as f64)
            }
            None => Ok(self.log_normalizer_uniform * self.n_topics as f64),
        }
    }

    /// Compute mean cosine similarity from predicted directions.
    ///
    /// Useful for M-step kappa re-estimation in EM-style algorithms.
    pub fn mean_cos_sim(&self, eta_dir: &Tensor) -> Result<f64> {
        let mu = l2_normalize_dim(eta_dir, 1)?;
        let cos_sim = mu.broadcast_mul(&self.y)?.sum(1)?;
        let mean: f32 = cos_sim.mean_all()?.to_scalar()?;
        Ok(mean as f64)
    }

    /// Estimate optimal kappa from current predictions (M-step).
    ///
    /// # Returns
    /// Tuple of (mean_cosine_similarity, suggested_kappa)
    pub fn estimate_kappa(&self, eta_dir: &Tensor) -> Result<(f64, f64)> {
        let mean_cos = self.mean_cos_sim(eta_dir)?;
        let suggested_kappa = estimate_kappa_mle(mean_cos, self.dim);
        Ok((mean_cos, suggested_kappa))
    }

    /// Create a new likelihood with updated kappa (for EM iteration).
    /// Preserves gene_weights if set.
    pub fn with_kappa(&self, new_kappa: f64) -> Result<Self> {
        match &self.gene_weights {
            Some(weights) => Self::with_gene_weights(self.y.clone(), new_kappa, weights.clone()),
            None => Self::from_normalized(self.y.clone(), new_kappa),
        }
    }
}

impl BlackBoxLikelihood for VmfFixedKappaLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        assert!(
            !etas.is_empty(),
            "VmfFixedKappaLikelihood requires at least 1 eta (direction)"
        );

        let eta_dir = etas[0]; // Direction predictor: (S, n, k)

        // L2 normalize eta_dir along feature dimension (dim 1) to get μ
        let mu = l2_normalize_dim(eta_dir, 1)?;

        // Cosine similarity: μᵀy per gene per topic
        // mu: (S, n, k), y: (n, k) -> (S, n, k)
        let cos_sim_per_gene = mu.broadcast_mul(&self.y)?; // (S, n, k)

        match (&self.gene_weights, &self.log_normalizer_per_gene) {
            (Some(weights), Some(log_norms)) => {
                // Per-gene kappa: κ_g = κ_base * w_g
                // log p(y_g | η_g) = κ_g * cos(y_g, η_g) + log C_d(κ_g) per topic
                //
                // weights: (n,) -> (1, n, 1) for broadcasting
                // log_norms: (n,) -> (1, n, 1) for broadcasting
                let weights_3d = weights.unsqueeze(0)?.unsqueeze(2)?; // (1, n, 1)
                let log_norms_3d = log_norms.unsqueeze(0)?.unsqueeze(2)?; // (1, n, 1)

                // Scaled cosine: κ_base * w_g * cos_sim (S, n, k)
                let scaled_cos = cos_sim_per_gene
                    .broadcast_mul(&weights_3d)?
                    .affine(self.kappa, 0.0)?;

                // Add per-gene log normalizers (broadcast over topics)
                let log_prob_per_gene = scaled_cos.broadcast_add(&log_norms_3d)?; // (S, n, k)

                // Sum over genes and topics: (S, n, k) -> (S,)
                log_prob_per_gene.sum(vec![1, 2])
            }
            _ => {
                // Uniform kappa: original behavior
                // Sum over genes to get (S, k)
                let cos_sim = cos_sim_per_gene.sum(1)?;

                // vMF log-likelihood: κ * (μᵀy) + log C_d(κ) per topic
                // Total normalizer contribution = log_normalizer * n_topics
                let total_normalizer = self.log_normalizer_uniform * self.n_topics as f64;
                let log_prob = cos_sim.affine(self.kappa, total_normalizer)?;

                // Sum over k (topics) to get (S,)
                log_prob.sum(1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_vmf_log_normalizer() {
        // Test that normalizer is computed without NaN/Inf
        let dims = [10, 100, 1000, 5000];
        let kappas = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0];

        for &d in &dims {
            for &k in &kappas {
                let log_c = vmf_log_normalizer(d, k);
                assert!(
                    log_c.is_finite(),
                    "log_normalizer not finite for d={}, kappa={}: {}",
                    d,
                    k,
                    log_c
                );
                println!("d={}, kappa={}: log_C = {:.4}", d, k, log_c);
            }
        }
    }

    #[test]
    fn test_kappa_estimation() {
        // Test MLE kappa estimation
        let dim = 100;

        // High concentration: r_bar close to 1
        let kappa_high = estimate_kappa_mle(0.95, dim);
        assert!(kappa_high > 50.0, "High r_bar should give high kappa");

        // Low concentration: r_bar close to 0
        let kappa_low = estimate_kappa_mle(0.3, dim);
        assert!(kappa_low < 50.0, "Low r_bar should give low kappa");

        println!("r_bar=0.95, d=100 -> kappa={:.2}", kappa_high);
        println!("r_bar=0.30, d=100 -> kappa={:.2}", kappa_low);
    }

    #[test]
    fn test_suggest_kappa_init() {
        assert!((suggest_kappa_init(100) - 7.07).abs() < 0.1);
        assert!((suggest_kappa_init(1000) - 22.36).abs() < 0.1);
    }

    #[test]
    fn test_vmf_fixed_kappa_likelihood() -> Result<()> {
        let device = Device::Cpu;

        // Simple test: y and mu are identical -> cos_sim = 1.0
        let y = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (3, 2), &device)?;

        let kappa = 10.0;
        let likelihood = VmfFixedKappaLikelihood::from_normalized(y.clone(), kappa)?;

        // eta_dir same as y (perfect match)
        let eta_dir = y.unsqueeze(0)?; // (1, 3, 2)

        let log_lik = likelihood.log_likelihood(&[&eta_dir])?;
        let val: f32 = log_lik.get(0)?.to_scalar()?;

        // With perfect alignment, cos_sim = 1.0 for each topic
        // log_prob ≈ κ * 1.0 * 2 + 2 * log_C_3(10)
        println!("Perfect alignment log_lik: {}", val);
        println!("log_normalizer: {}", likelihood.log_normalizer());

        // Should be positive for good alignment with reasonable kappa
        assert!(val.is_finite(), "Log likelihood should be finite");

        Ok(())
    }

    #[test]
    fn test_vmf_likelihood_stability() -> Result<()> {
        let device = Device::Cpu;

        // Test with realistic dimensions
        let n_genes = 500;
        let n_topics = 10;

        // Random normalized vectors
        let y_raw = Tensor::randn(0f32, 1f32, (n_genes, n_topics), &device)?;
        let y = l2_normalize_dim(&y_raw, 0)?;

        // Test across different kappa values
        let kappas = [5.0, 10.0, 16.0, 25.0, 50.0, 100.0];

        for &kappa in &kappas {
            let likelihood = VmfFixedKappaLikelihood::from_normalized(y.clone(), kappa)?;

            // Random predictions
            let eta_raw = Tensor::randn(0f32, 1f32, (10, n_genes, n_topics), &device)?;
            let log_lik = likelihood.log_likelihood(&[&eta_raw])?;

            // Check all values are finite
            let vals: Vec<f32> = log_lik.to_vec1()?;
            for (i, &v) in vals.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "Log likelihood not finite at sample {} for kappa={}: {}",
                    i,
                    kappa,
                    v
                );
            }

            let mean: f32 = log_lik.mean_all()?.to_scalar()?;
            println!("kappa={}: mean_log_lik={:.2}", kappa, mean);
        }

        Ok(())
    }

    #[test]
    fn test_em_style_kappa_update() -> Result<()> {
        let device = Device::Cpu;

        let n_genes = 100;
        let n_topics = 5;

        // Create well-aligned data
        let y_raw = Tensor::randn(0f32, 1f32, (n_genes, n_topics), &device)?;
        let y = l2_normalize_dim(&y_raw, 0)?;

        // Start with initial kappa
        let mut likelihood = VmfFixedKappaLikelihood::from_normalized(y.clone(), 16.0)?;

        // Predictions that are somewhat aligned with y
        let noise = Tensor::randn(0f32, 0.5f32, (1, n_genes, n_topics), &device)?;
        let eta = (&y.unsqueeze(0)? + &noise)?;

        // Estimate new kappa (M-step)
        let (mean_cos, new_kappa) = likelihood.estimate_kappa(&eta)?;
        println!(
            "mean_cos_sim={:.4}, estimated_kappa={:.2}",
            mean_cos, new_kappa
        );

        // Update likelihood with new kappa
        likelihood = likelihood.with_kappa(new_kappa)?;
        assert!((likelihood.kappa() - new_kappa).abs() < 1e-10);

        Ok(())
    }

    /// Test vMF likelihood with SuSiE for sparse recovery.
    ///
    /// Simulation:
    /// - n_genes = 100, n_annotations = 10, n_topics = 5
    /// - True assignment: sparse diagonal (annotation i -> topic i for i < 5)
    /// - Membership matrix X: block structure
    /// - Topic directions y: derived from X @ θ + noise, then normalized
    #[test]
    fn test_vmf_susie_sparse_recovery() -> Result<()> {
        use crate::sgvb::{direct_elbo_loss, GaussianPrior, LinearModelSGVB, SGVBConfig, SusieVar};
        use candle_core::DType;
        use candle_nn::{Optimizer, VarBuilder, VarMap};

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n_genes = 100;
        let n_annotations = 10;
        let n_topics = 5;
        let genes_per_annot = 10;
        let n_components = 3;
        let num_samples = 30;
        let kappa = 16.0;

        // 1. Create membership matrix X (gene × annotation)
        // Each annotation has genes_per_annot marker genes (block structure)
        let mut x_data = vec![0.0f32; n_genes * n_annotations];
        for a in 0..n_annotations {
            let start_gene = a * genes_per_annot;
            for g in start_gene..(start_gene + genes_per_annot).min(n_genes) {
                x_data[g * n_annotations + a] = 1.0;
            }
        }
        let x_ga = Tensor::from_vec(x_data, (n_genes, n_annotations), &device)?;
        let x_norm = l2_normalize_dim(&x_ga, 0)?;

        // 2. Create true effect matrix θ (annotation × topic)
        // Sparse diagonal: annotation i has effect on topic i (for i < n_topics)
        let mut theta_true_data = vec![0.0f32; n_annotations * n_topics];
        for i in 0..n_topics {
            theta_true_data[i * n_topics + i] = 3.0; // Strong effect on diagonal
        }
        let theta_true = Tensor::from_vec(theta_true_data, (n_annotations, n_topics), &device)?;

        // 3. Generate topic directions: y = normalize(X @ θ + noise)
        let eta_true = x_norm.matmul(&theta_true)?;
        let noise = Tensor::randn(0f32, 0.3f32, (n_genes, n_topics), &device)?;
        let y_noisy = (&eta_true + &noise)?;
        let y = l2_normalize_dim(&y_noisy, 0)?;

        // 4. Create model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = SusieVar::new(vb.pp("susie"), n_components, n_annotations, n_topics)?;
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::new(num_samples);
        let model = LinearModelSGVB::from_variational(susie, x_norm, prior, config.clone());

        let likelihood = VmfFixedKappaLikelihood::from_normalized(y, kappa)?;

        // 5. Train
        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.05)?;

        for i in 0..500 {
            let loss = direct_elbo_loss(&model, &likelihood, config.num_samples)?;
            optimizer.backward_step(&loss)?;

            if i % 100 == 0 {
                let loss_val: f32 = loss.to_scalar()?;
                println!("iter {}: loss = {:.4}", i, loss_val);
            }
        }

        // 6. Check recovery - diagonal elements should have high PIPs
        let pip = model.variational.pip()?;

        println!("\nPosterior Inclusion Probabilities (annotation × topic):");
        for a in 0..n_annotations {
            let mut row = Vec::new();
            for t in 0..n_topics {
                let val: f32 = pip.get(a)?.get(t)?.to_scalar()?;
                row.push(format!("{:.3}", val));
            }
            let marker = if a < n_topics { " <- true" } else { "" };
            println!("  annot {}: [{}]{}", a, row.join(", "), marker);
        }

        // Check diagonal dominance
        let mut diagonal_sum = 0.0f32;
        let mut off_diagonal_sum = 0.0f32;
        let mut off_diagonal_count = 0;

        for a in 0..n_annotations {
            for t in 0..n_topics {
                let val: f32 = pip.get(a)?.get(t)?.to_scalar()?;
                if a == t && a < n_topics {
                    diagonal_sum += val;
                } else {
                    off_diagonal_sum += val;
                    off_diagonal_count += 1;
                }
            }
        }

        let diagonal_mean = diagonal_sum / n_topics as f32;
        let off_diagonal_mean = off_diagonal_sum / off_diagonal_count as f32;

        println!("\nDiagonal mean PIP: {:.4}", diagonal_mean);
        println!("Off-diagonal mean PIP: {:.4}", off_diagonal_mean);

        // Diagonal should be significantly higher than off-diagonal
        assert!(
            diagonal_mean > off_diagonal_mean * 2.0,
            "Diagonal mean should be > 2x off-diagonal: {} vs {}",
            diagonal_mean,
            off_diagonal_mean
        );

        // Each diagonal element should be reasonably high
        for i in 0..n_topics {
            let pip_ii: f32 = pip.get(i)?.get(i)?.to_scalar()?;
            assert!(
                pip_ii > 0.1,
                "Diagonal PIP[{},{}] should be > 0.1, got {}",
                i,
                i,
                pip_ii
            );
        }

        Ok(())
    }

    /// Test vMF recovery with different kappa values to verify model comparison.
    #[test]
    fn test_vmf_kappa_comparison() -> Result<()> {
        use crate::sgvb::{direct_elbo_loss, GaussianPrior, LinearModelSGVB, SGVBConfig, SusieVar};
        use candle_core::DType;
        use candle_nn::{Optimizer, VarBuilder, VarMap};

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n_genes = 50;
        let n_annotations = 5;
        let n_topics = 3;
        let n_components = 2;
        let num_samples = 20;

        // Create simple data
        let x_ga = Tensor::randn(0f32, 1f32, (n_genes, n_annotations), &device)?;
        let x_norm = l2_normalize_dim(&x_ga, 0)?;

        let y_raw = Tensor::randn(0f32, 1f32, (n_genes, n_topics), &device)?;
        let y = l2_normalize_dim(&y_raw, 0)?;

        // Train with different kappa values and compare final losses
        let kappas = [5.0, 16.0, 50.0];
        let mut results: Vec<(f64, f32, f64)> = Vec::new(); // (kappa, loss, log_normalizer)

        for &kappa in &kappas {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

            let susie = SusieVar::new(vb.pp("susie"), n_components, n_annotations, n_topics)?;
            let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
            let config = SGVBConfig::new(num_samples);
            let model =
                LinearModelSGVB::from_variational(susie, x_norm.clone(), prior, config.clone());

            let likelihood = VmfFixedKappaLikelihood::from_normalized(y.clone(), kappa)?;

            let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.05)?;

            for _ in 0..200 {
                let loss = direct_elbo_loss(&model, &likelihood, config.num_samples)?;
                optimizer.backward_step(&loss)?;
            }

            let final_loss: f32 =
                direct_elbo_loss(&model, &likelihood, config.num_samples)?.to_scalar()?;
            let log_norm = likelihood.log_normalizer();

            results.push((kappa, final_loss, log_norm));
            println!(
                "kappa={}: final_loss={:.4}, log_normalizer={:.4}",
                kappa, final_loss, log_norm
            );
        }

        // Verify that all losses are finite
        for (kappa, loss, log_norm) in &results {
            assert!(
                loss.is_finite(),
                "Loss should be finite for kappa={}",
                kappa
            );
            assert!(
                log_norm.is_finite(),
                "Log normalizer should be finite for kappa={}",
                kappa
            );
        }

        Ok(())
    }

    /// Test per-gene kappa scaling (Option A).
    #[test]
    fn test_vmf_per_gene_weights() -> Result<()> {
        let device = Device::Cpu;

        let n_genes = 50;
        let n_topics = 5;
        let kappa = 16.0;

        // Create normalized data
        let y_raw = Tensor::randn(0f32, 1f32, (n_genes, n_topics), &device)?;
        let y = l2_normalize_dim(&y_raw, 0)?;

        // Create per-gene weights (some genes have higher weight)
        let mut weights_vec = vec![1.0f32; n_genes];
        for i in 0..10 {
            weights_vec[i] = 2.0; // First 10 genes have double weight
        }
        let gene_weights = Tensor::from_vec(weights_vec.clone(), (n_genes,), &device)?;

        // Create likelihood with per-gene weights
        let likelihood =
            VmfFixedKappaLikelihood::with_gene_weights(y.clone(), kappa, gene_weights)?;

        assert!(likelihood.has_gene_weights());
        assert_eq!(likelihood.kappa(), kappa);

        // Test log likelihood computation
        let eta_raw = Tensor::randn(0f32, 1f32, (5, n_genes, n_topics), &device)?;
        let log_lik = likelihood.log_likelihood(&[&eta_raw])?;

        let vals: Vec<f32> = log_lik.to_vec1()?;
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Log likelihood not finite at sample {}: {}",
                i,
                v
            );
        }

        println!(
            "Per-gene weights: mean_log_lik = {:.4}",
            log_lik.mean_all()?.to_scalar::<f32>()?
        );

        // Compare with uniform weights
        let likelihood_uniform = VmfFixedKappaLikelihood::from_normalized(y.clone(), kappa)?;
        assert!(!likelihood_uniform.has_gene_weights());

        let log_lik_uniform = likelihood_uniform.log_likelihood(&[&eta_raw])?;
        println!(
            "Uniform weights: mean_log_lik = {:.4}",
            log_lik_uniform.mean_all()?.to_scalar::<f32>()?
        );

        // With per-gene weights, effective kappa is higher for some genes
        // So total log-likelihood magnitude should be different
        let mean_weighted: f32 = log_lik.mean_all()?.to_scalar()?;
        let mean_uniform: f32 = log_lik_uniform.mean_all()?.to_scalar()?;

        // They should be different (weighted has higher effective kappa)
        assert!(
            (mean_weighted - mean_uniform).abs() > 0.1,
            "Weighted and uniform should differ: {} vs {}",
            mean_weighted,
            mean_uniform
        );

        Ok(())
    }

    /// Test that per-gene weights are preserved through kappa updates.
    #[test]
    fn test_vmf_per_gene_weights_kappa_update() -> Result<()> {
        let device = Device::Cpu;

        let n_genes = 20;
        let n_topics = 3;

        let y_raw = Tensor::randn(0f32, 1f32, (n_genes, n_topics), &device)?;
        let y = l2_normalize_dim(&y_raw, 0)?;

        let weights_vec = vec![1.5f32; n_genes];
        let gene_weights = Tensor::from_vec(weights_vec, (n_genes,), &device)?;

        let likelihood = VmfFixedKappaLikelihood::with_gene_weights(y, 16.0, gene_weights)?;
        assert!(likelihood.has_gene_weights());

        // Update kappa (simulating EM M-step)
        let updated = likelihood.with_kappa(25.0)?;

        assert!(
            updated.has_gene_weights(),
            "Gene weights should be preserved after kappa update"
        );
        assert!((updated.kappa() - 25.0).abs() < 1e-10);

        Ok(())
    }
}
