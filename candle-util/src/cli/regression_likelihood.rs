use candle_core::{Result, Tensor};

use crate::sgvb::BlackBoxLikelihood;

/// Fast lgamma approximation based on YPARK/fqtl fastgamma.h
///
/// Uses the formula:
///   lgamma(x) ≈ -2.081061466 - x + 0.0833333/xp3 - log(x*(1+x)*(2+x)) + (2.5+x)*log(xp3)
/// where xp3 = x + 3
fn lgamma_approx(x: &Tensor) -> Result<Tensor> {
    // Clamp x to avoid log(0) issues
    let x_safe = x.clamp(1e-6f32, f32::MAX)?;

    // logterm = log(x * (1 + x) * (2 + x))
    let x_plus_1 = (&x_safe + 1.0)?;
    let x_plus_2 = (&x_safe + 2.0)?;
    let product = ((&x_safe * &x_plus_1)? * &x_plus_2)?;
    let logterm = product.log()?;

    // xp3 = x + 3
    let xp3 = (&x_safe + 3.0)?;
    let log_xp3 = xp3.log()?;

    // lgamma(x) = -2.081061466 - x + 0.0833333/xp3 - logterm + (2.5 + x)*log(xp3)
    let recip_term = (xp3.recip()? * 0.0833333)?;
    let mult_term = ((&x_safe + 2.5)? * &log_xp3)?;

    (((recip_term - 2.081061466)? - &x_safe)? - &logterm)? + &mult_term
}

/// Poisson likelihood for count data: y ~ Poisson(exp(eta))
pub struct PoissonLikelihood {
    y: Tensor,
}

impl PoissonLikelihood {
    pub fn new(y: Tensor) -> Self {
        Self { y }
    }
}

impl BlackBoxLikelihood for PoissonLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> candle_core::Result<Tensor> {
        let eta = etas[0];
        // log p(y | lambda) = y * log(lambda) - lambda - log(y!)
        // with lambda = exp(eta):
        // log p(y | eta) = y * eta - exp(eta) - log(y!)
        // We ignore log(y!) as it's constant w.r.t. eta
        let y_eta = eta.broadcast_mul(&self.y)?;
        let exp_eta = eta.exp()?;
        let log_prob = (y_eta - exp_eta)?;

        log_prob.sum(2)?.sum(1)
    }
}

/// Gaussian likelihood: y ~ N(η₁, exp(η₂))
///
/// Requires two etas:
/// - etas[0]: mean
/// - etas[1]: log-variance
pub struct GaussianLikelihood {
    y: Tensor,
}

impl GaussianLikelihood {
    pub fn new(y: Tensor) -> Self {
        Self { y }
    }
}

impl BlackBoxLikelihood for GaussianLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> candle_core::Result<Tensor> {
        assert!(
            etas.len() >= 2,
            "GaussianLikelihood requires 2 etas (mean, log_var)"
        );
        let mu = etas[0]; // mean: (S, n, k)
        let log_var_raw = etas[1]; // log-variance: (S, n, k)

        // Clamp log_var to avoid numerical issues with exp()
        // Range [-10, 10] gives variance in [4.5e-5, 22026]
        let log_var = log_var_raw.clamp(-10.0, 10.0)?;

        // log N(y; μ, exp(log_var)) = -0.5 * [log(2π) + log_var + (y-μ)²/exp(log_var)]
        let ln_2pi: f64 = (2.0 * std::f64::consts::PI).ln();

        let diff = mu.broadcast_sub(&self.y)?;
        let diff_sq = diff.sqr()?;
        let var = log_var.exp()?;
        let scaled_diff_sq = (diff_sq / &var)?;

        let log_prob = ((scaled_diff_sq + &log_var)? + ln_2pi)? * (-0.5);

        // Sum over (n, k) dimensions
        log_prob?.sum(2)?.sum(1)
    }
}

/// Negative Binomial likelihood: y ~ NB(exp(η₁), exp(η₂))
///
/// Parameterization: μ = exp(η₁) is mean, r = exp(η₂) is dispersion.
/// Var(y) = μ + μ²/r
///
/// Requires two etas:
/// - etas[0]: log-mean
/// - etas[1]: log-dispersion
pub struct NegativeBinomialLikelihood {
    y: Tensor,
}

impl NegativeBinomialLikelihood {
    pub fn new(y: Tensor) -> Self {
        Self { y }
    }
}

impl BlackBoxLikelihood for NegativeBinomialLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        assert!(
            etas.len() >= 2,
            "NegativeBinomialLikelihood requires 2 etas (log_mean, log_dispersion)"
        );

        let log_mu_raw = etas[0]; // log-mean: (S, n, k)
        let log_r_raw = etas[1]; // log-dispersion: (S, n, k)

        // Clamp to avoid numerical issues
        let log_mu = log_mu_raw.clamp(-10.0, 10.0)?;
        let log_r = log_r_raw.clamp(-10.0, 10.0)?;

        let mu = log_mu.exp()?;
        let r = log_r.exp()?;

        // NB log-likelihood:
        // log P(y | μ, r) = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
        //                 + r*log(r/(r+μ)) + y*log(μ/(r+μ))
        //
        // Rearranging:
        // = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
        //   + r*log(r) - r*log(r+μ) + y*log(μ) - y*log(r+μ)
        // = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
        //   + r*log(r) + y*log(μ) - (r+y)*log(r+μ)

        let y_plus_r = self.y.broadcast_add(&r)?;
        let r_plus_mu = (&r + &mu)?;

        // Gamma terms
        let lgamma_y_plus_r = lgamma_approx(&y_plus_r)?;
        let lgamma_r = lgamma_approx(&r)?;
        let lgamma_y_plus_1 = lgamma_approx(&(&self.y + 1.0)?)?;

        // Log terms
        let r_log_r = (&r * &log_r)?;
        let y_log_mu = self.y.broadcast_mul(&log_mu)?;
        let r_plus_y_log_r_plus_mu = (&y_plus_r * r_plus_mu.log()?)?;

        // Combine
        let log_prob = (((((&lgamma_y_plus_r - &lgamma_r)? - &lgamma_y_plus_1)?
            + &r_log_r)?
            + &y_log_mu)?
            - &r_plus_y_log_r_plus_mu)?;

        // Sum over (n, k) dimensions
        log_prob.sum(2)?.sum(1)
    }
}

/// Von Mises-Fisher likelihood for directional data on the unit hypersphere.
///
/// Models the alignment between observed unit vectors and predicted directions.
/// Useful when comparing normalized embeddings or topic distributions where
/// cosine similarity (angle) matters more than magnitude.
///
/// Requires two etas:
/// - etas[0]: direction predictor (will be L2-normalized to get μ)
/// - etas[1]: log-concentration (κ = exp(η₂))
///
/// log p(y | μ, κ) ∝ κ * μᵀy  (ignoring normalization constant C_d(κ))
pub struct VonMisesFisherLikelihood {
    /// Unit-normalized observed directions, shape (n, k)
    y: Tensor,
}

impl VonMisesFisherLikelihood {
    /// Create a new vMF likelihood.
    ///
    /// # Arguments
    /// * `y` - Observed directions. Will be L2-normalized along dim 0 (genes/features).
    pub fn new(y: Tensor) -> Result<Self> {
        // L2 normalize y along the feature dimension (dim 0 for 2D, or handle 3D)
        let y_normalized = l2_normalize_dim(&y, 0)?;
        Ok(Self { y: y_normalized })
    }

    /// Create from pre-normalized data (skip normalization).
    pub fn from_normalized(y: Tensor) -> Self {
        Self { y }
    }
}

/// L2 normalize a tensor along the specified dimension.
fn l2_normalize_dim(x: &Tensor, dim: usize) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(dim)?.sqrt()?;
    let norm_safe = (norm + 1e-8)?; // Avoid division by zero
    x.broadcast_div(&norm_safe)
}

impl BlackBoxLikelihood for VonMisesFisherLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        assert!(
            etas.len() >= 2,
            "VonMisesFisherLikelihood requires 2 etas (direction, log_concentration)"
        );

        let eta_dir = etas[0]; // Direction predictor: (S, n, k)
        let log_kappa_raw = etas[1]; // Log-concentration: (S, n, k) or (S, 1, k)

        // Clamp log_kappa for numerical stability
        // κ in [exp(-5), exp(5)] ≈ [0.007, 148]
        let log_kappa = log_kappa_raw.clamp(-5.0, 5.0)?;
        let kappa = log_kappa.exp()?;

        // L2 normalize eta_dir along feature dimension (dim 1) to get μ
        // eta_dir shape: (S, n, k) -> normalize along n
        let mu = l2_normalize_dim(eta_dir, 1)?;

        // Cosine similarity: μᵀy = sum over features of (μ * y)
        // mu: (S, n, k), y: (n, k) -> broadcast_mul -> (S, n, k) -> sum(dim=1) -> (S, k)
        let cos_sim = mu.broadcast_mul(&self.y)?.sum(1)?;

        // If kappa is (S, 1, k), squeeze or handle broadcasting
        let kappa_squeezed = if kappa.dims().len() == 3 && kappa.dim(1)? == 1 {
            kappa.squeeze(1)? // (S, 1, k) -> (S, k)
        } else if kappa.dims().len() == 3 {
            // (S, n, k) -> mean over n to get per-topic kappa
            kappa.mean(1)?
        } else {
            kappa
        };

        // vMF log-likelihood (unnormalized, for relative scoring):
        // log p(y | μ, κ) ∝ κ * (μᵀy)
        let log_prob = (&cos_sim * &kappa_squeezed)?;

        // Sum over k (topics) to get (S,)
        log_prob.sum(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::{
        samples_direct_elbo_loss, GaussianPrior, LinearModelSGVB, LinearRegressionSGVB,
        SGVBConfig, SgvbModel, SusieVar,
    };
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Optimizer, VarBuilder, VarMap};

    /// Test vMF likelihood with simulation to verify recovery of true assignments.
    ///
    /// Simulation setup:
    /// - n_genes = 100, n_annotations = 5, n_topics = 5
    /// - True assignment: topic i matches annotation i (diagonal structure)
    /// - Membership matrix X: each annotation has ~20 marker genes
    /// - Topic directions y: derived from true assignment + noise
    #[test]
    fn test_vmf_recovery_simulation() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n_genes = 100;
        let n_annotations = 5;
        let n_topics = 5;
        let genes_per_annot = 20;
        let n_components = 3; // SuSiE components
        let num_samples = 30;

        // 1. Create membership matrix X (gene × annotation)
        // Each annotation has genes_per_annot marker genes
        let mut x_data = vec![0.0f32; n_genes * n_annotations];
        for a in 0..n_annotations {
            let start_gene = a * genes_per_annot;
            for g in start_gene..(start_gene + genes_per_annot).min(n_genes) {
                x_data[g * n_annotations + a] = 1.0;
            }
        }
        let x_ga = Tensor::from_vec(x_data, (n_genes, n_annotations), &device)?;

        // L2 normalize columns of X
        let x_norm = l2_normalize_dim(&x_ga, 0)?;

        // 2. Create true effect matrix θ (annotation × topic)
        // Diagonal: annotation i has effect on topic i only
        let mut theta_true_data = vec![0.0f32; n_annotations * n_topics];
        for i in 0..n_annotations.min(n_topics) {
            theta_true_data[i * n_topics + i] = 3.0; // Strong effect
        }
        let theta_true = Tensor::from_vec(theta_true_data, (n_annotations, n_topics), &device)?;

        // 3. Generate true topic directions: y = normalize(X @ θ + noise)
        let eta_true = x_norm.matmul(&theta_true)?;
        let noise = Tensor::randn(0f32, 0.1f32, (n_genes, n_topics), &device)?;
        let y_noisy = (&eta_true + &noise)?;
        let y = l2_normalize_dim(&y_noisy, 0)?;

        // 4. Create models (heterogeneous types, so we sample separately)
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        // Direction model: SuSiE for sparse selection
        let susie = SusieVar::new(vb.pp("susie"), n_components, n_annotations, n_topics)?;
        let prior_dir = GaussianPrior::new(vb.pp("prior_dir"), 1.0)?;
        let config = SGVBConfig::new(num_samples);
        let model_dir = LinearModelSGVB::from_variational(susie, x_norm.clone(), prior_dir, config.clone());

        // Concentration model: intercept-only for per-topic κ
        let x_intercept = Tensor::ones((n_genes, 1), dtype, &device)?;
        let prior_kappa = GaussianPrior::new(vb.pp("prior_kappa"), 1.0)?;
        let model_kappa = LinearRegressionSGVB::new(vb.pp("kappa"), x_intercept, n_topics, prior_kappa, config.clone())?;

        // 5. Create likelihood
        let likelihood = VonMisesFisherLikelihood::from_normalized(y);

        // 6. Train using samples_direct_elbo_loss for heterogeneous models
        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.05)?;

        for i in 0..500 {
            // Sample from both models separately
            let sample_dir = model_dir.sample(num_samples)?;
            let sample_kappa = model_kappa.sample(num_samples)?;

            // Combine samples and compute loss
            let samples = vec![sample_dir, sample_kappa];
            let loss = samples_direct_elbo_loss(&samples, &likelihood)?;
            optimizer.backward_step(&loss)?;

            if i % 100 == 0 {
                let loss_val: f32 = loss.to_scalar()?;
                println!("iter {}: loss = {:.4}", i, loss_val);
            }
        }

        // 7. Check recovery via coefficient mean - diagonal should dominate
        let coef_mean = model_dir.coef_mean()?;
        println!("\nLearned coefficients (annotation × topic):");
        for a in 0..n_annotations {
            let mut row = Vec::new();
            for t in 0..n_topics {
                let val: f32 = coef_mean.get(a)?.get(t)?.to_scalar()?;
                row.push(format!("{:6.2}", val));
            }
            println!("  annot {}: [{}]", a, row.join(", "));
        }

        // Check diagonal dominance: diagonal elements should be larger than off-diagonal
        let mut diagonal_sum = 0.0f32;
        let mut off_diagonal_sum = 0.0f32;
        let mut off_diagonal_count = 0;

        for a in 0..n_annotations {
            for t in 0..n_topics {
                let val: f32 = coef_mean.get(a)?.get(t)?.to_scalar()?;
                if a == t {
                    diagonal_sum += val.abs();
                } else {
                    off_diagonal_sum += val.abs();
                    off_diagonal_count += 1;
                }
            }
        }

        let diagonal_mean = diagonal_sum / n_annotations.min(n_topics) as f32;
        let off_diagonal_mean = off_diagonal_sum / off_diagonal_count as f32;

        println!("\nDiagonal mean abs: {:.4}", diagonal_mean);
        println!("Off-diagonal mean abs: {:.4}", off_diagonal_mean);

        // Diagonal should dominate
        assert!(
            diagonal_mean > off_diagonal_mean * 2.0,
            "Diagonal should be > 2x off-diagonal: {} vs {}",
            diagonal_mean,
            off_diagonal_mean
        );

        Ok(())
    }

    #[test]
    fn test_vmf_likelihood_basic() -> Result<()> {
        let device = Device::Cpu;

        // Simple test: y and mu are identical -> cos_sim = 1.0
        let y = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], // 2 unit vectors (3 genes, 2 topics)
            (3, 2),
            &device
        )?;

        let likelihood = VonMisesFisherLikelihood::from_normalized(y.clone());

        // eta_dir same as y (perfect match)
        let eta_dir = y.unsqueeze(0)?; // (1, 3, 2)
        // log_kappa = log(10) ≈ 2.3
        let log_kappa = Tensor::full(2.3f32, (1, 3, 2), &device)?;

        let log_lik = likelihood.log_likelihood(&[&eta_dir, &log_kappa])?;
        // Result is (S,) = (1,), get first element
        let val: f32 = log_lik.get(0)?.to_scalar()?;

        // With perfect alignment, cos_sim ≈ 1.0 for each topic
        // log_prob ≈ κ * cos_sim * 2 topics ≈ 10 * 2 = 20
        // (actual value slightly lower due to normalization numerics)
        println!("Perfect alignment log_lik: {}", val);
        assert!(val > 10.0, "Expected positive log_lik for alignment, got {}", val);

        Ok(())
    }
}
