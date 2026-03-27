use anyhow::Result;
use candle_util::candle_core::Device;
use clap::ValueEnum;
use nalgebra::DMatrix;

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
}

impl std::str::FromStr for ModelType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "susie" => Ok(Self::Susie),
            "bisusie" => Ok(Self::BiSusie),
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
    /// Infinitesimal prior variance σ²_inf for polygenic background.
    /// When > 0, a dense `GaussianRegressionSGVB` with prior variance
    /// σ²_inf / p is fitted alongside the sparse SuSiE term.
    pub sigma2_inf: f32,
    /// Prior concentration for SuSiE alpha (PIP prior).
    /// Each SNP gets prior probability prior_alpha / p.
    /// Default 1.0 (uniform prior).
    pub prior_alpha: f64,
    /// Multilevel configuration. When set, uses `MultilevelSusieSGVB` with
    /// LD-aware hierarchical softmax. Only supported with `ModelType::Susie`.
    pub multilevel: Option<MultilevelConfig>,
}

/// Configuration for LD-aware multilevel SuSiE.
#[derive(Debug, Clone)]
pub struct MultilevelConfig {
    /// Minimum SNPs per LD sub-block in level-0 partition.
    pub min_block_snps: usize,
    /// Maximum SNPs per LD sub-block in level-0 partition.
    pub max_block_snps: usize,
    /// Minimum p to trigger multilevel (below this, fall back to flat SuSiE).
    pub min_p: usize,
}

impl Default for MultilevelConfig {
    fn default() -> Self {
        Self {
            min_block_snps: 20,
            max_block_snps: 500,
            min_p: 200,
        }
    }
}

/// SNP genomic coordinates for LD-aware multilevel partition estimation.
pub struct SnpCoordinates<'a> {
    pub positions: &'a [u64],
    pub chromosomes: &'a [Box<str>],
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
            sigma2_inf: 0.0,
            prior_alpha: 1.0,
            multilevel: None,
        }
    }
}

/// Internal result tuple: (avg_elbo, pip, effect_mean, effect_std).
pub(crate) type PriorFitResult = (f32, DMatrix<f32>, DMatrix<f32>, DMatrix<f32>);

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

/// RSS-specific parameters for `fit_block_rss`.
pub struct RssParams<'a> {
    pub max_rank: usize,
    pub lambda: f64,
    pub ldsc_intercept: bool,
    pub coords: Option<&'a SnpCoordinates<'a>>,
}
