use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use candle_util::sgvb::variant_tree::VariantTree;
use candle_util::sgvb::{
    samples_local_reparam_loss, AnalyticalKL, BiSusieVar, FixedGaussianLikelihood,
    FixedGaussianPrior, LinearModelSGVB, LinearRegressionSGVB, LocalReparamSample,
    MultiLevelSusieVar, RssLikelihood, RssSvd, SGVBConfig, SusieVar, VariationalDistribution,
    WeightedGaussianLikelihood,
};
use clap::ValueEnum;
use log::info;
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;
use rand::prelude::*;
use std::collections::VecDeque;

/// Compute device selection for SGVB fitting.
#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

impl ComputeDevice {
    /// Create a candle `Device` from this enum.
    pub fn to_device(&self, device_no: usize) -> Result<Device> {
        Ok(match self {
            ComputeDevice::Metal => Device::new_metal(device_no)?,
            ComputeDevice::Cuda => Device::new_cuda(device_no)?,
            ComputeDevice::Cpu => Device::Cpu,
        })
    }
}

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
    /// Infinitesimal prior variance σ²_inf for polygenic background.
    /// When > 0, a dense `LinearRegressionSGVB` with prior variance
    /// σ²_inf / p is fitted alongside the sparse SuSiE term.
    pub sigma2_inf: f32,
    /// Prior concentration for SuSiE alpha (PIP prior).
    /// Each SNP gets prior probability prior_alpha / p.
    /// Default 1.0 (uniform prior).
    pub prior_alpha: f64,
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
            prior_vars: vec![0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.5],
            elbo_window: 50,
            seed: 42,
            ml_block_size: 50,
            sigma2_inf: 0.0,
            prior_alpha: 1.0,
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

/// Result from fitting a single block with per-prior-var results.
#[derive(Debug)]
pub struct BlockFitResultDetailed {
    /// Per-prior-var average ELBOs.
    pub per_prior_elbos: Vec<f32>,
    /// Per-prior-var PIPs, shape (p, k) each.
    pub per_prior_pips: Vec<DMatrix<f32>>,
    /// Per-prior-var effect means, shape (p, k) each.
    pub per_prior_effects: Vec<DMatrix<f32>>,
    /// Per-prior-var effect stds, shape (p, k) each.
    pub per_prior_stds: Vec<DMatrix<f32>>,
}

impl BlockFitResultDetailed {
    /// Pick the best prior by local ELBO argmax.
    pub fn best_result(&self) -> BlockFitResult {
        select_best_prior(self)
    }
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
    device: &Device,
) -> Result<BlockFitResult> {
    let detailed = fit_block_inner(x_block, y_block, None, confounders, config, device)?;
    Ok(select_best_prior(&detailed))
}

/// Fit a fine-mapping model for a single block with per-observation variance.
///
/// Like `fit_block()` but accepts a per-observation variance tensor `(N, K)`.
/// Uses `WeightedGaussianLikelihood` instead of `FixedGaussianLikelihood`.
///
/// Returns detailed results with per-prior-var ELBOs for empirical Bayes.
pub fn fit_block_weighted(
    x_block: &DMatrix<f32>,
    y_block: &DMatrix<f32>,
    var_block: &DMatrix<f32>,
    confounders: Option<&DMatrix<f32>>,
    config: &FitConfig,
    device: &Device,
) -> Result<BlockFitResultDetailed> {
    fit_block_inner(
        x_block,
        y_block,
        Some(var_block),
        confounders,
        config,
        device,
    )
}

/// Shared implementation for `fit_block` and `fit_block_weighted`.
fn fit_block_inner(
    x_block: &DMatrix<f32>,
    y_block: &DMatrix<f32>,
    var_block: Option<&DMatrix<f32>>,
    confounders: Option<&DMatrix<f32>>,
    config: &FitConfig,
    device: &Device,
) -> Result<BlockFitResultDetailed> {
    let n = x_block.nrows();
    let p = x_block.ncols();
    let k = y_block.ncols();

    let x_tensor = x_block.to_tensor(device)?.contiguous()?;
    let y_tensor = y_block.to_tensor(device)?.contiguous()?;
    let var_tensor = var_block
        .map(|v| -> Result<Tensor> { Ok(v.to_tensor(device)?.contiguous()?) })
        .transpose()?;
    let conf_tensor = confounders
        .map(|c| -> Result<Tensor> { Ok(c.to_tensor(device)?.contiguous()?) })
        .transpose()?;

    let use_minibatch = n > config.batch_size;

    let tensors = BlockTensors {
        x: x_tensor,
        y: y_tensor,
        var: var_tensor,
        conf: conf_tensor,
        p,
        k,
        n,
        use_minibatch,
    };

    let mut elbos_vec: Vec<f32> = Vec::new();
    let mut pips_vec: Vec<DMatrix<f32>> = Vec::new();
    let mut effects_vec: Vec<DMatrix<f32>> = Vec::new();
    let mut stds_vec: Vec<DMatrix<f32>> = Vec::new();

    for &prior_var in &config.prior_vars {
        let (avg_elbo, pip, eff_mean, eff_std) =
            fit_single_prior(&tensors, prior_var, config, device)?;
        info!("  prior_var={:.3}, avg_elbo={:.2}", prior_var, avg_elbo);
        elbos_vec.push(avg_elbo);
        pips_vec.push(pip);
        effects_vec.push(eff_mean);
        stds_vec.push(eff_std);
    }

    Ok(BlockFitResultDetailed {
        per_prior_elbos: elbos_vec,
        per_prior_pips: pips_vec,
        per_prior_effects: effects_vec,
        per_prior_stds: stds_vec,
    })
}

/// Precomputed tensors for block fitting.
struct BlockTensors {
    x: Tensor,
    y: Tensor,
    /// Per-observation variance (n, k). None → unit variance (FixedGaussianLikelihood).
    var: Option<Tensor>,
    conf: Option<Tensor>,
    p: usize,
    k: usize,
    n: usize,
    use_minibatch: bool,
}

/// Train with a single prior_var. Returns (avg_elbo, pip, effect_mean, effect_std).
fn fit_single_prior(
    tensors: &BlockTensors,
    prior_var: f32,
    config: &FitConfig,
    device: &Device,
) -> Result<PriorFitResult> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let prior = FixedGaussianPrior::new(prior_var.sqrt());
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
        let (x_batch, y_batch, var_batch, conf_batch) = if tensors.use_minibatch {
            let indices: Vec<u32> = (0..config.batch_size)
                .map(|_| rng.random_range(0..tensors.n as u32))
                .collect();
            let idx_tensor = Tensor::from_vec(indices, (config.batch_size,), device)?;
            let xb = tensors.x.index_select(&idx_tensor, 0)?;
            let yb = tensors.y.index_select(&idx_tensor, 0)?;
            let vb = tensors
                .var
                .as_ref()
                .map(|v| v.index_select(&idx_tensor, 0))
                .transpose()?;
            let cb = tensors
                .conf
                .as_ref()
                .map(|ct| ct.index_select(&idx_tensor, 0))
                .transpose()?;
            (xb, yb, vb, cb)
        } else {
            (
                tensors.x.clone(),
                tensors.y.clone(),
                tensors.var.clone(),
                tensors.conf.clone(),
            )
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

        // Choose likelihood based on whether per-observation variance is provided
        let loss = if let Some(ref vb) = var_batch {
            let likelihood = WeightedGaussianLikelihood::new(y_batch, vb)?;
            samples_local_reparam_loss(&samples, &likelihood, 1.0)?
        } else {
            let likelihood = FixedGaussianLikelihood::new(y_batch, 1.0);
            samples_local_reparam_loss(&samples, &likelihood, 1.0)?
        };

        optimizer.backward_step(&loss)?;

        let elbo_val = -loss.to_scalar::<f32>()?;
        if elbo_buffer.len() == config.elbo_window {
            elbo_buffer.pop_front();
        }
        elbo_buffer.push_back(elbo_val);
    }

    let avg_elbo = if elbo_buffer.is_empty() {
        0.0
    } else {
        elbo_buffer.iter().sum::<f32>() / elbo_buffer.len() as f32
    };

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

    fn kl_categorical(&self, prior_alpha: f64, device: &Device) -> Result<Tensor> {
        match self {
            Self::Susie(m) => Ok(m.variational.kl_categorical(prior_alpha)?),
            _ => Ok(Tensor::zeros(
                (),
                candle_util::candle_core::DType::F32,
                device,
            )?),
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

/// Estimate per-trait LDSC h² (slope) for a single LD block.
///
/// Performs rSVD on the block genotypes and regresses (V'z)²_k on d²_k.
/// Since d²_k are eigenvalues of R = X'X/N and E[(V'z)²_k] = N·h²·d²_k + a,
/// the raw slope is N·h²_block. We divide by N to return h²_block per trait.
/// Returns zeros if K <= 2.
pub fn estimate_block_h2(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
    max_rank: usize,
    lambda: f64,
    device: &Device,
) -> Result<Vec<f32>> {
    let n = x_block.nrows() as f32;
    let k = z_block.ncols();
    let x_tensor = x_block.to_tensor(device)?;
    let z_tensor = z_block.to_tensor(device)?;

    let svd = RssSvd::from_genotypes(&x_tensor, max_rank, lambda, device)?;
    let kk = svd.effective_rank();

    if kk <= 2 {
        return Ok(vec![0.0; k]);
    }

    let vt = svd.v_mat().t()?;
    let vt_z = vt.matmul(&z_tensor)?;

    let d_vals: Vec<f32> = svd.singular_values().to_vec1()?;
    let d_sq: Vec<f32> = d_vals.iter().map(|&d| d * d).collect();

    let vt_z_data: Vec<f32> = vt_z.flatten_all()?.to_vec1()?;
    let y_raw: Vec<Vec<f32>> = (0..kk)
        .map(|kk_i| (0..k).map(|tt| vt_z_data[kk_i * k + tt]).collect())
        .collect();

    let (_intercepts, slopes) = RssSvd::estimate_ldsc_intercept(&d_sq, &y_raw, k);
    // Divide by N: raw slope = N * h²_block
    Ok(slopes.iter().map(|&s| (s / n).max(0.0)).collect())
}

/// Build an adaptive prior_var grid centered on `h2 / num_components`.
///
/// When `n` is provided (RSS mode), the grid is scaled by `n` to convert
/// from per-SD variance to z-score–scale variance, since the RSS eigenspace
/// model parameterises effects on the z-score scale (β_z ≈ √n · β_sd).
pub fn adaptive_prior_grid(h2_estimate: f32, num_components: usize, n: Option<u64>) -> Vec<f32> {
    let center = (h2_estimate / num_components as f32).max(0.01);
    let scale = n.unwrap_or(1) as f32;
    let multipliers = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0];
    let mut grid: Vec<f32> = multipliers.iter().map(|&m| center * scale * m).collect();
    // Clamp to reasonable range
    grid.iter_mut().for_each(|v| *v = v.clamp(0.001, 1e6));
    grid
}

/// Fit a fine-mapping model for a single LD block using RSS likelihood.
///
/// Uses the eigenspace projection approach (Zhu & Stephens 2017, YPARK/zqtl):
/// z-scores are projected through the SVD of X/√n into K-dimensional space,
/// avoiding explicit R = X'X/n and operating in the efficient K-space.
///
/// Model averaging is performed over the prior_var grid.
/// When `config.sigma2_inf > 0`, an additional intercept component is included.
pub fn fit_block_rss(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
    config: &FitConfig,
    max_rank: usize,
    lambda: f64,
    device: &Device,
    ldsc_intercept: bool,
) -> Result<BlockFitResultDetailed> {
    let p = x_block.ncols();
    let k = z_block.ncols(); // number of traits T

    // Compute SVD of block genotypes (done once, reused across grid)
    let x_tensor = x_block.to_tensor(device)?;
    let mut z_tensor = z_block.to_tensor(device)?;

    let svd = RssSvd::from_genotypes(&x_tensor, max_rank, lambda, device)?;
    let x_design = svd.x_design().clone(); // (K, p) — fat design
    let kk = svd.effective_rank(); // eigenspace dimension

    info!(
        "  RSS block: p={}, K={}, T={}, λ={:.2e}, σ²_inf={:.2e}",
        p,
        kk,
        k,
        svd.lambda(),
        config.sigma2_inf,
    );

    // ── Local LDSC intercept estimation ──────────────────────────────
    if ldsc_intercept && kk > 2 {
        // Compute V'z (raw projection without D̃⁻¹ scaling) for LDSC
        let vt = svd.v_mat().t()?;
        let vt_z = vt.matmul(&z_tensor)?; // (K, T)

        let d_vals: Vec<f32> = svd.singular_values().to_vec1()?;
        let d_sq: Vec<f32> = d_vals.iter().map(|&d| d * d).collect();

        let vt_z_data: Vec<f32> = vt_z.flatten_all()?.to_vec1()?;
        let y_raw: Vec<Vec<f32>> = (0..kk)
            .map(|kk_i| (0..k).map(|tt| vt_z_data[kk_i * k + tt]).collect())
            .collect();

        let (intercepts, slopes) = RssSvd::estimate_ldsc_intercept(&d_sq, &y_raw, k);

        // Log per-trait intercept and slope (heritability proxy)
        for tt in 0..k {
            if intercepts[tt] > 1.01 || slopes[tt].abs() > 0.01 {
                info!(
                    "    LDSC trait {}: intercept={:.3}, slope(h)={:.4}",
                    tt, intercepts[tt], slopes[tt],
                );
            }
        }

        // Rescale z-scores per trait where intercept > 1
        let any_inflated = intercepts.iter().any(|&a| a > 1.0 + 1e-6);
        if any_inflated {
            let scale: Vec<f32> = intercepts.iter().map(|&a| 1.0 / a.sqrt()).collect();
            let scale_tensor =
                Tensor::from_vec(scale, (1, k), z_tensor.device())?.to_dtype(z_tensor.dtype())?;
            z_tensor = z_tensor.broadcast_mul(&scale_tensor)?;
        }
    }

    let y_tilde = svd.project_zscores(&z_tensor)?; // (K, T)
    let rss = RssLikelihood::from_projected(y_tilde);
    let mut results: Vec<PriorFitResult> = Vec::new();

    // Intercept design: D̃⁻¹ V' 1, shape (K, 1)
    // Projects an all-ones vector through the same eigenspace transform as z-scores.
    let intercept_design: Option<Tensor> = if config.sigma2_inf > 0.0 {
        let ones_p = Tensor::ones((p, 1), x_design.dtype(), device)?;
        Some(svd.project_zscores(&ones_p)?)
    } else {
        None
    };

    // Loop over prior variance grid
    for &prior_var in &config.prior_vars {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let prior = FixedGaussianPrior::new(prior_var.sqrt());
        let sgvb_config = SGVBConfig {
            num_samples: config.num_sgvb_samples,
            kl_weight: 1.0,
        };

        // Build genetic model with X̃ as design
        let genetic = GeneticModel::new(
            &vb,
            config.model_type,
            config.num_components,
            p,
            k,
            x_design.clone(),
            prior,
            sgvb_config.clone(),
            config.ml_block_size,
        )?;

        // Optional intercept: 1 variational param per trait, design = D̃⁻¹V'1
        let intercept_model: Option<LinearRegressionSGVB<FixedGaussianPrior>> =
            if let Some(ref int_design) = intercept_design {
                let int_prior = FixedGaussianPrior::new(config.sigma2_inf);
                Some(LinearRegressionSGVB::new(
                    vb.pp("intercept"),
                    int_design.clone(),
                    k,
                    int_prior,
                    sgvb_config,
                )?)
            } else {
                None
            };

        let mut optimizer = AdamW::new_lr(varmap.all_vars(), config.learning_rate)?;
        let mut elbo_buffer: VecDeque<f32> = VecDeque::with_capacity(config.elbo_window);

        for _iter in 0..config.num_iterations {
            // No minibatch needed: K << N, design is already compact
            let gen_sample = genetic.local_reparam_sample(config.num_sgvb_samples, &x_design)?;

            let mut samples = vec![gen_sample];
            if let Some(ref im) = intercept_model {
                samples.push(im.local_reparam_sample(config.num_sgvb_samples)?);
            }

            let loss = samples_local_reparam_loss(&samples, &rss, 1.0)?;
            let kl_cat = genetic.kl_categorical(config.prior_alpha, device)?;
            let loss = (loss + kl_cat)?;

            optimizer.backward_step(&loss)?;

            let elbo_val = -loss.to_scalar::<f32>()?;
            if elbo_buffer.len() == config.elbo_window {
                elbo_buffer.pop_front();
            }
            elbo_buffer.push_back(elbo_val);
        }

        let avg_elbo = if elbo_buffer.is_empty() {
            0.0
        } else {
            elbo_buffer.iter().sum::<f32>() / elbo_buffer.len() as f32
        };

        let (pip_tensor, eff_mean_tensor, eff_std_tensor) = genetic.extract_results()?;
        let pip: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&pip_tensor)?;
        let eff_mean: DMatrix<f32> =
            <DMatrix<f32> as ConvertMatOps>::from_tensor(&eff_mean_tensor)?;
        let eff_std: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&eff_std_tensor)?;

        info!("  prior_var={:.3}, avg_elbo={:.2}", prior_var, avg_elbo);
        results.push((avg_elbo, pip, eff_mean, eff_std));
    }

    let elbos: Vec<f32> = results.iter().map(|(e, _, _, _)| *e).collect();
    let pips: Vec<DMatrix<f32>> = results.iter().map(|(_, p, _, _)| p.clone()).collect();
    let effects: Vec<DMatrix<f32>> = results.iter().map(|(_, _, e, _)| e.clone()).collect();
    let stds: Vec<DMatrix<f32>> = results.iter().map(|(_, _, _, s)| s.clone()).collect();

    Ok(BlockFitResultDetailed {
        per_prior_elbos: elbos,
        per_prior_pips: pips,
        per_prior_effects: effects,
        per_prior_stds: stds,
    })
}

/// Select the best prior by ELBO argmax from a `BlockFitResultDetailed`.
pub fn select_best_prior(detailed: &BlockFitResultDetailed) -> BlockFitResult {
    let best_idx = elbo_argmax(&detailed.per_prior_elbos);
    BlockFitResult {
        pip: detailed.per_prior_pips[best_idx].clone(),
        effect_mean: detailed.per_prior_effects[best_idx].clone(),
        effect_std: detailed.per_prior_stds[best_idx].clone(),
        avg_elbo: detailed.per_prior_elbos[best_idx],
    }
}

/// Return the index of the maximum value (argmax).
pub fn elbo_argmax(elbos: &[f32]) -> usize {
    elbos
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
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
