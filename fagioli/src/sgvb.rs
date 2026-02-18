use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use candle_util::sgvb::variant_tree::VariantTree;
use candle_util::sgvb::{
    samples_local_reparam_loss, AnalyticalKL, BiSusieVar, FixedGaussianLikelihood,
    FixedGaussianPrior, LinearModelSGVB, LinearRegressionSGVB, LocalReparamSample,
    MultiLevelSusieVar, RssLikelihood, RssSvd, SGVBConfig, SusieVar, VariationalDistribution,
};
use log::info;
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;
use rand::prelude::*;
use std::collections::VecDeque;

/// Model type for SGVB fine-mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Susie,
    BiSusie,
    MultiLevelSusie,
}

impl std::str::FromStr for ModelType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "susie" => Ok(Self::Susie),
            "bisusie" => Ok(Self::BiSusie),
            "multilevel-susie" | "ml-susie" | "multilevel_susie" | "ml_susie" => {
                Ok(Self::MultiLevelSusie)
            }
            _ => anyhow::bail!("Unknown model type: {}", s),
        }
    }
}

/// Configuration for SGVB fine-mapping.
#[derive(Debug, Clone)]
pub struct FitConfig {
    pub model_type: ModelType,
    pub num_components: usize,
    pub num_sgvb_samples: usize,
    pub learning_rate: f64,
    pub num_iterations: usize,
    pub batch_size: usize,
    pub prior_vars: Vec<f32>,
    pub elbo_window: usize,
    pub seed: u64,
    /// Block size for MultiLevelSusieVar tree.
    pub ml_block_size: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Susie,
            num_components: 10,
            num_sgvb_samples: 20,
            learning_rate: 0.01,
            num_iterations: 500,
            batch_size: 1000,
            prior_vars: vec![0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
            elbo_window: 50,
            seed: 42,
            ml_block_size: 50,
        }
    }
}

/// Internal result tuple: (avg_elbo, pip, effect_mean, effect_std).
type PriorFitResult = (f32, DMatrix<f32>, DMatrix<f32>, DMatrix<f32>);

/// Result from fitting a single block.
#[derive(Debug)]
pub struct BlockFitResult {
    /// Per-(SNP, trait) posterior inclusion probabilities, shape (p, k).
    pub pip: DMatrix<f32>,
    /// Posterior mean effect sizes, shape (p, k).
    pub effect_mean: DMatrix<f32>,
    /// Posterior std of effect sizes, shape (p, k).
    pub effect_std: DMatrix<f32>,
    /// Model-averaged ELBO (log scale, for diagnostics).
    pub avg_elbo: f32,
}

/// Fit a fine-mapping model for a single LD block.
///
/// - `x_block`: Standardized genotypes (N × p_block).
/// - `y_block`: Response, e.g. PGS (N × T).
/// - `confounders`: Shared confounder matrix (N × K_conf), or None.
/// - `config`: Fit configuration.
///
/// Returns model-averaged PIPs and effect sizes across prior-var grid.
pub fn fit_block(
    x_block: &DMatrix<f32>,
    y_block: &DMatrix<f32>,
    confounders: Option<&DMatrix<f32>>,
    config: &FitConfig,
) -> Result<BlockFitResult> {
    let n = x_block.nrows();
    let p = x_block.ncols();
    let k = y_block.ncols();
    let device = Device::Cpu;

    let x_tensor = x_block.to_tensor(&device)?.contiguous()?;
    let y_tensor = y_block.to_tensor(&device)?.contiguous()?;
    let conf_tensor = confounders
        .map(|c| -> Result<Tensor> { Ok(c.to_tensor(&device)?.contiguous()?) })
        .transpose()?;

    let use_minibatch = n > config.batch_size;

    let tensors = BlockTensors {
        x: x_tensor,
        y: y_tensor,
        conf: conf_tensor,
        p,
        k,
        n,
        use_minibatch,
    };

    let mut results: Vec<PriorFitResult> = Vec::new();

    for &prior_var in &config.prior_vars {
        let (avg_elbo, pip, eff_mean, eff_std) =
            fit_single_prior(&tensors, prior_var, config, &device)?;
        info!("  prior_var={:.3}, avg_elbo={:.2}", prior_var, avg_elbo);
        results.push((avg_elbo, pip, eff_mean, eff_std));
    }

    // Model averaging via softmax over average ELBOs
    let elbos: Vec<f32> = results.iter().map(|(e, _, _, _)| *e).collect();
    let max_elbo = elbos.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = elbos.iter().map(|e| (e - max_elbo).exp()).collect();
    let sum_w: f32 = weights.iter().sum();
    let weights: Vec<f32> = weights.iter().map(|w| w / sum_w).collect();

    let mut pip_avg = DMatrix::<f32>::zeros(p, k);
    let mut eff_mean_avg = DMatrix::<f32>::zeros(p, k);
    let mut eff_std_avg = DMatrix::<f32>::zeros(p, k);

    for (w, (_elbo, pip, eff_m, eff_s)) in weights.iter().zip(results.iter()) {
        pip_avg += pip * *w;
        eff_mean_avg += eff_m * *w;
        eff_std_avg += eff_s * *w;
    }

    let weighted_elbo: f32 = weights.iter().zip(elbos.iter()).map(|(w, e)| w * e).sum();

    Ok(BlockFitResult {
        pip: pip_avg,
        effect_mean: eff_mean_avg,
        effect_std: eff_std_avg,
        avg_elbo: weighted_elbo,
    })
}

/// Precomputed tensors for block fitting.
struct BlockTensors {
    x: Tensor,
    y: Tensor,
    conf: Option<Tensor>,
    p: usize,
    k: usize,
    n: usize,
    use_minibatch: bool,
}

/// Train with a single prior_var. Returns (avg_elbo, pip, effect_mean, effect_std).
#[allow(clippy::too_many_arguments)]
fn fit_single_prior(
    tensors: &BlockTensors,
    prior_var: f32,
    config: &FitConfig,
    device: &Device,
) -> Result<PriorFitResult> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let prior = FixedGaussianPrior::new(prior_var);
    let sgvb_config = SGVBConfig {
        num_samples: config.num_sgvb_samples,
        kl_weight: 1.0,
    };

    // Build confounder model if needed
    let conf_model: Option<LinearRegressionSGVB<FixedGaussianPrior>> =
        if let Some(ref ct) = tensors.conf {
            let conf_prior = FixedGaussianPrior::new(1.0);
            Some(LinearRegressionSGVB::new(
                vb.pp("conf"),
                ct.clone(),
                tensors.k,
                conf_prior,
                sgvb_config.clone(),
            )?)
        } else {
            None
        };

    // Build genetic model — dispatch on model type using enum wrapper
    let genetic = GeneticModel::new(
        &vb,
        config.model_type,
        config.num_components,
        tensors.p,
        tensors.k,
        tensors.x.clone(),
        prior,
        sgvb_config,
        config.ml_block_size,
    )?;

    let mut optimizer = AdamW::new_lr(varmap.all_vars(), config.learning_rate)?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut elbo_buffer: VecDeque<f32> = VecDeque::with_capacity(config.elbo_window);

    for _iter in 0..config.num_iterations {
        // Optionally subsample rows with replacement
        let (x_batch, y_batch, conf_batch) = if tensors.use_minibatch {
            let indices: Vec<u32> = (0..config.batch_size)
                .map(|_| rng.random_range(0..tensors.n as u32))
                .collect();
            let idx_tensor = Tensor::from_vec(indices, (config.batch_size,), device)?;
            let xb = tensors.x.index_select(&idx_tensor, 0)?;
            let yb = tensors.y.index_select(&idx_tensor, 0)?;
            let cb = tensors
                .conf
                .as_ref()
                .map(|ct| ct.index_select(&idx_tensor, 0))
                .transpose()?;
            (xb, yb, cb)
        } else {
            (tensors.x.clone(), tensors.y.clone(), tensors.conf.clone())
        };

        let mut samples: Vec<LocalReparamSample> = Vec::new();

        // Genetic model sample (with custom design for minibatch)
        let gen_sample = genetic.local_reparam_sample(config.num_sgvb_samples, &x_batch)?;
        samples.push(gen_sample);

        // Confounder sample
        if let Some(ref cm) = conf_model {
            let conf_sample = if tensors.use_minibatch {
                local_reparam_with_design(
                    cm,
                    config.num_sgvb_samples,
                    conf_batch.as_ref().unwrap(),
                )?
            } else {
                cm.local_reparam_sample(config.num_sgvb_samples)?
            };
            samples.push(conf_sample);
        }

        let likelihood = FixedGaussianLikelihood::new(y_batch, 1.0);
        let loss = samples_local_reparam_loss(&samples, &likelihood, 1.0)?;

        optimizer.backward_step(&loss)?;

        let elbo_val = -loss.to_scalar::<f32>()?;
        if elbo_buffer.len() == config.elbo_window {
            elbo_buffer.pop_front();
        }
        elbo_buffer.push_back(elbo_val);
    }

    // Average ELBO over the last window
    let avg_elbo: f32 = elbo_buffer.iter().sum::<f32>() / elbo_buffer.len() as f32;

    // Extract results
    let (pip_tensor, eff_mean_tensor, eff_std_tensor) = genetic.extract_results()?;

    let pip: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&pip_tensor)?;
    let eff_mean: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&eff_mean_tensor)?;
    let eff_std: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&eff_std_tensor)?;

    Ok((avg_elbo, pip, eff_mean, eff_std))
}

// ---------------------------------------------------------------------------
// Enum-based dispatch for genetic model types
// ---------------------------------------------------------------------------

enum GeneticModel {
    Susie(LinearModelSGVB<SusieVar, FixedGaussianPrior>),
    BiSusie(LinearModelSGVB<BiSusieVar, FixedGaussianPrior>),
    MultiLevelSusie(LinearModelSGVB<MultiLevelSusieVar, FixedGaussianPrior>),
}

impl GeneticModel {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: &VarBuilder,
        model_type: ModelType,
        num_components: usize,
        p: usize,
        k: usize,
        x_design: Tensor,
        prior: FixedGaussianPrior,
        config: SGVBConfig,
        ml_block_size: usize,
    ) -> Result<Self> {
        Ok(match model_type {
            ModelType::Susie => {
                let var_dist = SusieVar::new(vb.pp("susie"), num_components, p, k)?;
                Self::Susie(LinearModelSGVB::from_variational(
                    var_dist, x_design, prior, config,
                ))
            }
            ModelType::BiSusie => {
                let var_dist = BiSusieVar::new(vb.pp("bisusie"), num_components, p, k)?;
                Self::BiSusie(LinearModelSGVB::from_variational(
                    var_dist, x_design, prior, config,
                ))
            }
            ModelType::MultiLevelSusie => {
                let tree = VariantTree::regular(p, ml_block_size);
                let var_dist =
                    MultiLevelSusieVar::new(vb.pp("mlsusie"), tree, num_components, k, 1.0)?;
                Self::MultiLevelSusie(LinearModelSGVB::from_variational(
                    var_dist, x_design, prior, config,
                ))
            }
        })
    }

    fn local_reparam_sample(
        &self,
        num_samples: usize,
        x_batch: &Tensor,
    ) -> Result<LocalReparamSample> {
        match self {
            Self::Susie(m) => local_reparam_with_design(m, num_samples, x_batch),
            Self::BiSusie(m) => local_reparam_with_design(m, num_samples, x_batch),
            Self::MultiLevelSusie(m) => local_reparam_with_design(m, num_samples, x_batch),
        }
    }

    fn extract_results(&self) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Susie(m) => {
                let pip = m.variational.pip()?;
                let eff_mean = m.variational.theta_mean()?;
                let eff_std = m.variational.var()?.sqrt()?;
                Ok((pip, eff_mean, eff_std))
            }
            Self::BiSusie(m) => {
                let pip = m.variational.pip()?;
                let eff_mean = m.variational.theta_mean()?;
                let eff_std = m.variational.var()?.sqrt()?;
                Ok((pip, eff_mean, eff_std))
            }
            Self::MultiLevelSusie(m) => {
                let pip = m.variational.pip()?;
                let eff_mean = m.variational.theta_mean()?;
                let eff_std = m.variational.var()?.sqrt()?;
                Ok((pip, eff_mean, eff_std))
            }
        }
    }
}

/// Fit a fine-mapping model for a single LD block using RSS likelihood.
///
/// Uses the eigenspace projection approach (Zhu & Stephens 2017, YPARK/zqtl):
/// z-scores are projected through the SVD of X/√n into K-dimensional space,
/// avoiding explicit R = X'X/n and operating in the efficient K-space.
///
/// - `x_block`: Standardized genotypes (N × p_block).
/// - `z_block`: Z-scores for this block (p_block × T).
/// - `config`: Fit configuration.
/// - `max_rank`: Maximum rank for rSVD.
/// - `lambda`: Regularization for D̃ = √(D² + λ).
pub fn fit_block_rss(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
    config: &FitConfig,
    max_rank: usize,
    lambda: f64,
) -> Result<BlockFitResult> {
    let p = x_block.ncols();
    let k = z_block.ncols(); // number of traits T
    let device = Device::Cpu;

    // Compute SVD of block genotypes (done once, reused across prior_vars)
    let x_tensor = x_block.to_tensor(&device)?;
    let z_tensor = z_block.to_tensor(&device)?;

    let svd = RssSvd::from_genotypes(&x_tensor, max_rank, lambda, &device)?;
    let rss = RssLikelihood::new(&svd, &z_tensor)?;
    let x_design = svd.x_design().clone(); // (K, p) — fat design
    let kk = svd.effective_rank(); // eigenspace dimension

    info!(
        "  RSS block: p={}, K={}, T={}, λ={:.2e}",
        p, kk, k, svd.lambda(),
    );

    let mut results: Vec<PriorFitResult> = Vec::new();

    for &prior_var in &config.prior_vars {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let prior = FixedGaussianPrior::new(prior_var);
        let sgvb_config = SGVBConfig {
            num_samples: config.num_sgvb_samples,
            kl_weight: 1.0,
        };

        // Intercept model: identity design I_K in eigenspace, broad prior
        let intercept_design = Tensor::eye(kk, DType::F32, &device)?;
        let intercept_prior = FixedGaussianPrior::new(10.0);
        let intercept_model = LinearRegressionSGVB::new(
            vb.pp("intercept"),
            intercept_design.clone(),
            k,
            intercept_prior,
            sgvb_config.clone(),
        )?;

        // Build genetic model with X̃ = D̃V' (K × p) as design
        let genetic = GeneticModel::new(
            &vb,
            config.model_type,
            config.num_components,
            p,
            k,
            x_design.clone(),
            prior,
            sgvb_config,
            config.ml_block_size,
        )?;

        let mut optimizer = AdamW::new_lr(varmap.all_vars(), config.learning_rate)?;
        let mut elbo_buffer: VecDeque<f32> = VecDeque::with_capacity(config.elbo_window);

        for _iter in 0..config.num_iterations {
            // No minibatch needed: K << N, design is already compact
            let gen_sample =
                genetic.local_reparam_sample(config.num_sgvb_samples, &x_design)?;
            let intercept_sample = intercept_model.local_reparam_sample(config.num_sgvb_samples)?;
            let loss =
                samples_local_reparam_loss(&[gen_sample, intercept_sample], &rss, 1.0)?;

            optimizer.backward_step(&loss)?;

            let elbo_val = -loss.to_scalar::<f32>()?;
            if elbo_buffer.len() == config.elbo_window {
                elbo_buffer.pop_front();
            }
            elbo_buffer.push_back(elbo_val);
        }

        let avg_elbo: f32 = elbo_buffer.iter().sum::<f32>() / elbo_buffer.len() as f32;

        let (pip_tensor, eff_mean_tensor, eff_std_tensor) = genetic.extract_results()?;
        let pip: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&pip_tensor)?;
        let eff_mean: DMatrix<f32> =
            <DMatrix<f32> as ConvertMatOps>::from_tensor(&eff_mean_tensor)?;
        let eff_std: DMatrix<f32> =
            <DMatrix<f32> as ConvertMatOps>::from_tensor(&eff_std_tensor)?;

        info!("  prior_var={:.3}, avg_elbo={:.2}", prior_var, avg_elbo);
        results.push((avg_elbo, pip, eff_mean, eff_std));
    }

    // Model averaging via softmax over average ELBOs
    let elbos: Vec<f32> = results.iter().map(|(e, _, _, _)| *e).collect();
    let max_elbo = elbos.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = elbos.iter().map(|e| (e - max_elbo).exp()).collect();
    let sum_w: f32 = weights.iter().sum();
    let weights: Vec<f32> = weights.iter().map(|w| w / sum_w).collect();

    let mut pip_avg = DMatrix::<f32>::zeros(p, k);
    let mut eff_mean_avg = DMatrix::<f32>::zeros(p, k);
    let mut eff_std_avg = DMatrix::<f32>::zeros(p, k);

    for (w, (_elbo, pip, eff_m, eff_s)) in weights.iter().zip(results.iter()) {
        pip_avg += pip * *w;
        eff_mean_avg += eff_m * *w;
        eff_std_avg += eff_s * *w;
    }

    let weighted_elbo: f32 = weights.iter().zip(elbos.iter()).map(|(w, e)| w * e).sum();

    Ok(BlockFitResult {
        pip: pip_avg,
        effect_mean: eff_mean_avg,
        effect_std: eff_std_avg,
        avg_elbo: weighted_elbo,
    })
}

/// Compute local reparameterization sample with a custom design matrix (for minibatch).
fn local_reparam_with_design<V: VariationalDistribution, P: AnalyticalKL>(
    model: &LinearModelSGVB<V, P>,
    num_samples: usize,
    x_batch: &Tensor,
) -> Result<LocalReparamSample> {
    let theta_mean = model.variational.mean()?; // (p, k)
    let theta_var = model.variational.var()?; // (p, k)

    let eta_mean = x_batch.matmul(&theta_mean)?; // (n_batch, k)
    let x_sq = x_batch.sqr()?;
    let eta_var = x_sq.matmul(&theta_var)?; // (n_batch, k)

    let (nb, k) = eta_mean.dims2()?;
    let device = eta_mean.device();
    let dtype = eta_mean.dtype();

    // Antithetic sampling
    let half_s = num_samples / 2;
    let epsilon = if half_s > 0 {
        let eps_half = Tensor::randn(0f32, 1f32, (half_s, nb, k), device)?.to_dtype(dtype)?;
        let eps_neg = eps_half.neg()?;
        if num_samples % 2 == 1 {
            let eps_extra = Tensor::randn(0f32, 1f32, (1, nb, k), device)?.to_dtype(dtype)?;
            Tensor::cat(&[eps_half, eps_neg, eps_extra], 0)?
        } else {
            Tensor::cat(&[eps_half, eps_neg], 0)?
        }
    } else {
        Tensor::randn(0f32, 1f32, (1, nb, k), device)?.to_dtype(dtype)?
    };

    // η = E[η] + √V[η] ⊙ ε
    let eta_std = (eta_var + 1e-8)?.sqrt()?;
    let eta = eta_mean
        .unsqueeze(0)?
        .broadcast_add(&epsilon.broadcast_mul(&eta_std.unsqueeze(0)?)?)?;

    // Analytical KL from prior
    let kl = model.prior.kl_from_gaussian(&theta_mean, &theta_var)?;

    Ok(LocalReparamSample { eta, kl })
}
