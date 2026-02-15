use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{Optimizer, VarBuilder, VarMap};
use candle_util::sgvb::{
    BlackBoxLikelihood, FixedGaussianPrior, GaussianPrior, LinearModelSGVB,
    LinearRegressionSGVB, NegativeBinomialLikelihood, PoissonLikelihood, Prior, SGVBConfig,
    SgvbModel, SusieVar, direct_elbo_loss,
};
use special::Error as SpecialError;

/// Fixed-variance Gaussian likelihood for QTL mapping.
///
/// log p(y | eta) = -0.5 * [log(2π) + log(σ²) + (y - η)² / σ²]
pub struct SimpleGaussianLikelihood {
    y: Tensor,
    sigma: f64,
}

impl SimpleGaussianLikelihood {
    pub fn new(y: Tensor, sigma: f64) -> Self {
        Self { y, sigma }
    }
}

impl BlackBoxLikelihood for SimpleGaussianLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> candle_util::candle_core::Result<Tensor> {
        let eta = etas[0];
        let sigma_sq = self.sigma.powi(2);
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();
        let ln_sigma = self.sigma.ln();
        let const_term = 2.0 * ln_sigma + ln_2pi;

        let diff_sq = eta.broadcast_sub(&self.y)?.sqr()?;
        let log_prob = (((diff_sq / sigma_sq)? + const_term)? * (-0.5))?;

        log_prob.sum(2)?.sum(1)
    }
}

/// Configuration for SGVB QTL mapping.
#[derive(Debug, Clone)]
pub struct MappingConfig {
    pub model: String,
    pub likelihood: String,
    pub num_components: usize,
    pub num_samples: usize,
    pub num_iters: usize,
    pub learning_rate: f64,
    pub prior_tau: f32,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self {
            model: "susie".to_string(),
            likelihood: "gaussian".to_string(),
            num_components: 5,
            num_samples: 10,
            num_iters: 500,
            learning_rate: 0.02,
            prior_tau: 0.0,
        }
    }
}

/// Results from mapping a single gene.
#[derive(Debug, Clone)]
pub struct GeneQtlResult {
    pub gene_id: Box<str>,
    pub cell_type: Box<str>,
    pub snp_indices: Vec<usize>,
    pub effect_sizes: Vec<f32>,
    pub effect_stds: Vec<f32>,
    pub pips: Option<Vec<f32>>,
    pub z_scores: Vec<f32>,
    pub p_values: Vec<f32>,
    pub final_elbo: f32,
}

/// Extracted variational posterior summaries.
struct PosteriorSummary {
    means: Vec<f32>,
    stds: Vec<f32>,
    pips: Option<Vec<f32>>,
}

/// Run SGVB regression for a single gene.
pub fn map_gene_qtl(
    gene_id: &str,
    cell_type: &str,
    cis_snp_indices: &[usize],
    genotypes: &nalgebra::DMatrix<f32>,
    pseudobulk_y: &nalgebra::DVector<f32>,
    individual_mask: &[bool],
    config: &MappingConfig,
) -> Result<GeneQtlResult> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let p_cis = cis_snp_indices.len();

    // Subset to valid individuals
    let valid_indices: Vec<usize> = individual_mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(i, _)| i)
        .collect();
    let n_valid = valid_indices.len();

    if n_valid == 0 || p_cis == 0 {
        return Ok(empty_result(gene_id, cell_type, cis_snp_indices, config));
    }

    // Build design matrix X: (n_valid, p_cis) and y: (n_valid, 1)
    let mut x_data = Vec::with_capacity(n_valid * p_cis);
    for &ind in &valid_indices {
        for &snp_idx in cis_snp_indices {
            x_data.push(genotypes[(ind, snp_idx)]);
        }
    }
    let x_tensor = Tensor::from_vec(x_data, (n_valid, p_cis), &device)?.to_dtype(dtype)?;

    let y_data: Vec<f32> = valid_indices.iter().map(|&i| pseudobulk_y[i]).collect();
    let y_tensor = Tensor::from_vec(y_data, (n_valid, 1), &device)?.to_dtype(dtype)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let sgvb_config = SGVBConfig::new(config.num_samples);

    // Build model, run optimization, extract results
    // We use a macro to handle the combinatorial explosion of model × prior × likelihood
    // while still having access to the concrete model type for result extraction.
    let (final_elbo, summary) = if config.model == "susie" {
        run_susie_model(
            vb, &varmap, x_tensor, &y_tensor, sgvb_config, config,
        )?
    } else {
        run_gaussian_model(
            vb, &varmap, x_tensor, &y_tensor, sgvb_config, config,
        )?
    };

    let z_scores: Vec<f32> = summary
        .means
        .iter()
        .zip(&summary.stds)
        .map(|(&m, &s)| if s > 0.0 { m / s } else { 0.0 })
        .collect();
    let p_values: Vec<f32> = z_scores.iter().map(|&z| z_to_pvalue(z)).collect();

    Ok(GeneQtlResult {
        gene_id: Box::from(gene_id),
        cell_type: Box::from(cell_type),
        snp_indices: cis_snp_indices.to_vec(),
        effect_sizes: summary.means,
        effect_stds: summary.stds,
        pips: summary.pips,
        z_scores,
        p_values,
        final_elbo,
    })
}

fn run_susie_model(
    vb: VarBuilder,
    varmap: &VarMap,
    x_tensor: Tensor,
    y_tensor: &Tensor,
    sgvb_config: SGVBConfig,
    config: &MappingConfig,
) -> Result<(f32, PosteriorSummary)> {
    let p_cis = x_tensor.dim(1)?;
    let n = x_tensor.dim(0)?;
    let variational = SusieVar::new(vb.clone(), config.num_components, p_cis, 1)?;

    macro_rules! run_and_extract {
        ($prior:expr) => {{
            let model = LinearModelSGVB::from_variational(
                variational,
                x_tensor,
                $prior,
                sgvb_config,
            );
            let elbo = run_optimization(varmap, &model, y_tensor, n, config)?;
            let summary = extract_susie_results(&model)?;
            Ok((elbo, summary))
        }};
    }

    if config.prior_tau > 0.0 {
        run_and_extract!(FixedGaussianPrior::new(config.prior_tau))
    } else {
        run_and_extract!(GaussianPrior::new(vb.pp("prior"), 1.0)?)
    }
}

fn run_gaussian_model(
    vb: VarBuilder,
    varmap: &VarMap,
    x_tensor: Tensor,
    y_tensor: &Tensor,
    sgvb_config: SGVBConfig,
    config: &MappingConfig,
) -> Result<(f32, PosteriorSummary)> {
    let n = x_tensor.dim(0)?;

    macro_rules! run_and_extract {
        ($prior:expr) => {{
            let model = LinearRegressionSGVB::new(
                vb.clone(),
                x_tensor,
                1,
                $prior,
                sgvb_config,
            )?;
            let elbo = run_optimization(varmap, &model, y_tensor, n, config)?;
            let summary = extract_gaussian_results(&model)?;
            Ok((elbo, summary))
        }};
    }

    if config.prior_tau > 0.0 {
        run_and_extract!(FixedGaussianPrior::new(config.prior_tau))
    } else {
        run_and_extract!(GaussianPrior::new(vb.pp("prior"), 1.0)?)
    }
}

fn run_optimization<M: SgvbModel>(
    varmap: &VarMap,
    model: &M,
    y_tensor: &Tensor,
    n: usize,
    config: &MappingConfig,
) -> Result<f32> {
    match config.likelihood.as_str() {
        "poisson" => {
            let likelihood = PoissonLikelihood::new(y_tensor.clone());
            train_loop(varmap, model, &likelihood, config)
        }
        "nb" | "negative_binomial" => {
            // Create a learnable log-dispersion parameter (scalar, init = 0 → r = 1)
            let device = Device::Cpu;
            let dtype = DType::F32;
            let vb_disp = candle_util::candle_nn::VarBuilder::from_varmap(varmap, dtype, &device);
            let log_r = vb_disp.get_with_hints(
                (1,),
                "log_r",
                candle_util::candle_nn::Init::Const(0.0),
            )?;
            let likelihood = NegativeBinomialLikelihood::new(y_tensor.clone());
            train_loop_nb(varmap, model, &likelihood, &log_r, n, config)
        }
        _ => {
            let y_sd = y_std(y_tensor)?;
            let likelihood = SimpleGaussianLikelihood::new(y_tensor.clone(), y_sd);
            train_loop(varmap, model, &likelihood, config)
        }
    }
}

fn train_loop<M: SgvbModel, L: BlackBoxLikelihood>(
    varmap: &VarMap,
    model: &M,
    likelihood: &L,
    config: &MappingConfig,
) -> Result<f32> {
    let mut optimizer =
        candle_util::candle_nn::AdamW::new_lr(varmap.all_vars(), config.learning_rate)?;

    let mut final_elbo = f32::NEG_INFINITY;
    for _iter in 0..config.num_iters {
        let loss = direct_elbo_loss(model, likelihood, config.num_samples)?;
        optimizer.backward_step(&loss)?;
        final_elbo = -(loss.to_scalar::<f32>()?);
    }
    Ok(final_elbo)
}

/// Training loop for negative binomial likelihood.
///
/// NB requires two etas: log-mean (from the regression model) and log-dispersion
/// (a learnable scalar shared across individuals). We sample from the main model
/// to get eta1 (log-mean), then broadcast the learnable log_r as eta2.
fn train_loop_nb<M: SgvbModel>(
    varmap: &VarMap,
    model: &M,
    likelihood: &NegativeBinomialLikelihood,
    log_r: &Tensor,
    n: usize,
    config: &MappingConfig,
) -> Result<f32> {
    let mut optimizer =
        candle_util::candle_nn::AdamW::new_lr(varmap.all_vars(), config.learning_rate)?;

    let mut final_elbo = f32::NEG_INFINITY;
    for _iter in 0..config.num_iters {
        let sample = model.sample(config.num_samples)?;
        let s = config.num_samples;

        // Broadcast log_r to (S, n, 1) to match eta shape
        let log_r_broadcast = log_r
            .reshape((1, 1, 1))?
            .broadcast_as((s, n, 1))?
            .contiguous()?;

        // Compute log-likelihood with both etas
        let llik = likelihood.log_likelihood(&[&sample.eta, &log_r_broadcast])?;
        let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

        // ELBO = log_lik + log_prior - log_q
        let elbo = ((&llik + &sample.log_prior)? - &sample.log_q)?;
        let loss = elbo.mean(0)?.neg()?;

        optimizer.backward_step(&loss)?;
        final_elbo = -(loss.to_scalar::<f32>()?);
    }
    Ok(final_elbo)
}

fn extract_susie_results<P: Prior>(
    model: &LinearModelSGVB<SusieVar, P>,
) -> Result<PosteriorSummary> {
    let means_tensor = model.coef_mean()?; // (p, k)
    let var_tensor = model.coef_var()?; // (p, k)
    let std_tensor = var_tensor.clamp(0.0, f64::MAX)?.sqrt()?;
    let pip_tensor = model.variational.pip()?; // (p, k)

    let means: Vec<f32> = means_tensor.flatten_all()?.to_vec1()?;
    let stds: Vec<f32> = std_tensor.flatten_all()?.to_vec1()?;
    let pips: Vec<f32> = pip_tensor.flatten_all()?.to_vec1()?;

    Ok(PosteriorSummary {
        means,
        stds,
        pips: Some(pips),
    })
}

fn extract_gaussian_results<P: Prior>(
    model: &LinearRegressionSGVB<P>,
) -> Result<PosteriorSummary> {
    let means_tensor = model.coef_mean()?; // (p, k)
    let var_tensor = model.coef_var()?; // (p, k)
    let std_tensor = var_tensor.clamp(0.0, f64::MAX)?.sqrt()?;

    let means: Vec<f32> = means_tensor.flatten_all()?.to_vec1()?;
    let stds: Vec<f32> = std_tensor.flatten_all()?.to_vec1()?;

    Ok(PosteriorSummary {
        means,
        stds,
        pips: None,
    })
}

fn y_std(y: &Tensor) -> Result<f64> {
    let mean = y.mean_all()?;
    let diff = y.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_all()?;
    let std_val: f64 = var.sqrt()?.to_scalar::<f32>()? as f64;
    Ok(std_val.max(0.01))
}

fn z_to_pvalue(z: f32) -> f32 {
    let p = (z.abs() as f64 / std::f64::consts::SQRT_2).compl_error();
    p as f32
}

fn empty_result(
    gene_id: &str,
    cell_type: &str,
    cis_snp_indices: &[usize],
    config: &MappingConfig,
) -> GeneQtlResult {
    let p = cis_snp_indices.len();
    GeneQtlResult {
        gene_id: Box::from(gene_id),
        cell_type: Box::from(cell_type),
        snp_indices: cis_snp_indices.to_vec(),
        effect_sizes: vec![0.0; p],
        effect_stds: vec![f32::NAN; p],
        pips: if config.model == "susie" { Some(vec![0.0; p]) } else { None },
        z_scores: vec![0.0; p],
        p_values: vec![1.0; p],
        final_elbo: f32::NEG_INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use rand::SeedableRng;
    use rand_distr::{Distribution, Gamma, Normal, Poisson};

    #[test]
    fn test_simple_gaussian_likelihood() -> Result<()> {
        let device = Device::Cpu;
        let y = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3, 1), &device)?;
        let likelihood = SimpleGaussianLikelihood::new(y.clone(), 1.0);

        let eta_perfect = y.unsqueeze(0)?; // (1, 3, 1)
        let ll = likelihood.log_likelihood(&[&eta_perfect])?;
        let ll_val: f32 = ll.flatten_all()?.to_vec1::<f32>()?[0];

        let eta_zero = Tensor::zeros((1, 3, 1), DType::F32, &device)?;
        let ll_zero = likelihood.log_likelihood(&[&eta_zero])?;
        let ll_zero_val: f32 = ll_zero.flatten_all()?.to_vec1::<f32>()?[0];

        assert!(ll_val > ll_zero_val, "Perfect prediction should have higher LL");
        Ok(())
    }

    #[test]
    fn test_z_to_pvalue() {
        assert!((z_to_pvalue(0.0) - 1.0).abs() < 0.01);
        assert!(z_to_pvalue(3.0) < 0.01);
        assert!(z_to_pvalue(-3.0) < 0.01);
    }

    #[test]
    fn test_map_gene_qtl_gaussian() -> Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 50;
        let p = 5;
        let normal = Normal::new(0.0f32, 1.0).unwrap();

        let mut geno = DMatrix::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                geno[(i, j)] = (normal.sample(&mut rng) > 0.0) as u8 as f32
                    + (normal.sample(&mut rng) > 0.0) as u8 as f32;
            }
        }

        let mut y = DVector::<f32>::zeros(n);
        for i in 0..n {
            y[i] = 0.5 * geno[(i, 2)] + normal.sample(&mut rng) * 0.5;
        }

        let mask = vec![true; n];
        let cis_snps: Vec<usize> = (0..p).collect();

        let config = MappingConfig {
            model: "gaussian".to_string(),
            num_samples: 5,
            num_iters: 200,
            prior_tau: 1.0,
            ..Default::default()
        };

        let result = map_gene_qtl("gene_0", "T_cell", &cis_snps, &geno, &y, &mask, &config)?;

        assert_eq!(result.effect_sizes.len(), p);
        assert_eq!(result.z_scores.len(), p);
        let max_idx = result
            .effect_sizes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2, "Causal SNP should have largest effect size");

        Ok(())
    }

    /// Generate NB counts via Poisson-Gamma mixture: y ~ Poisson(Gamma(r, r/μ))
    fn sample_nb(rng: &mut impl rand::Rng, mu: f32, r: f32) -> f32 {
        let gamma = Gamma::new(r as f64, (mu / r) as f64).unwrap();
        let lambda = gamma.sample(rng).max(1e-10);
        let pois = Poisson::new(lambda).unwrap();
        pois.sample(rng) as f32
    }

    #[test]
    fn test_map_gene_qtl_nb() -> Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let n = 80;
        let p = 5;
        let normal = Normal::new(0.0f32, 1.0).unwrap();
        let r_true = 5.0f32; // dispersion

        // Simulate genotypes (0/1/2)
        let mut geno = DMatrix::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                geno[(i, j)] = (normal.sample(&mut rng) > 0.0) as u8 as f32
                    + (normal.sample(&mut rng) > 0.0) as u8 as f32;
            }
        }

        // Generate NB counts: log(mu) = 2.0 + 0.5 * X[:,2]
        // Causal SNP is index 2 with positive effect on log-mean
        let mut y = DVector::<f32>::zeros(n);
        for i in 0..n {
            let log_mu = 2.0 + 0.5 * geno[(i, 2)];
            let mu = log_mu.exp();
            y[i] = sample_nb(&mut rng, mu, r_true);
        }

        let mask = vec![true; n];
        let cis_snps: Vec<usize> = (0..p).collect();

        let config = MappingConfig {
            model: "gaussian".to_string(),
            likelihood: "nb".to_string(),
            num_samples: 10,
            num_iters: 300,
            prior_tau: 1.0,
            ..Default::default()
        };

        let result = map_gene_qtl("gene_nb", "T_cell", &cis_snps, &geno, &y, &mask, &config)?;

        assert_eq!(result.effect_sizes.len(), p);
        assert!(result.final_elbo.is_finite(), "ELBO should be finite");

        // The causal SNP (index 2) should have the largest absolute effect
        let max_idx = result
            .effect_sizes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2, "Causal SNP should have largest effect size");

        // Effect should be positive (matching the true positive effect)
        assert!(
            result.effect_sizes[2] > 0.0,
            "Causal SNP effect should be positive, got {}",
            result.effect_sizes[2]
        );

        Ok(())
    }

    #[test]
    fn test_map_gene_qtl_poisson() -> Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let n = 120;
        let p = 5;
        let normal = Normal::new(0.0f32, 1.0).unwrap();

        // Simulate genotypes (0/1/2)
        let mut geno = DMatrix::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                geno[(i, j)] = (normal.sample(&mut rng) > 0.0) as u8 as f32
                    + (normal.sample(&mut rng) > 0.0) as u8 as f32;
            }
        }

        // Generate Poisson counts: log(mu) = 2.0 + 0.8 * X[:,2]
        let mut y = DVector::<f32>::zeros(n);
        for i in 0..n {
            let log_mu = 2.0 + 0.8 * geno[(i, 2)];
            let lambda = (log_mu as f64).exp();
            let pois = Poisson::new(lambda).unwrap();
            y[i] = pois.sample(&mut rng) as f32;
        }

        let mask = vec![true; n];
        let cis_snps: Vec<usize> = (0..p).collect();

        let config = MappingConfig {
            model: "gaussian".to_string(),
            likelihood: "poisson".to_string(),
            num_samples: 10,
            num_iters: 300,
            prior_tau: 1.0,
            ..Default::default()
        };

        let result =
            map_gene_qtl("gene_pois", "T_cell", &cis_snps, &geno, &y, &mask, &config)?;

        assert_eq!(result.effect_sizes.len(), p);
        assert!(result.final_elbo.is_finite(), "ELBO should be finite");

        // The causal SNP (index 2) should have the largest absolute effect
        let max_idx = result
            .effect_sizes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2, "Causal SNP should have largest effect size");

        // Effect should be positive (matching the true positive effect)
        assert!(
            result.effect_sizes[2] > 0.0,
            "Causal SNP effect should be positive, got {}",
            result.effect_sizes[2]
        );

        Ok(())
    }
}
