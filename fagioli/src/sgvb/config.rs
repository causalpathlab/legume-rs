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

/// Prior type for effect sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorType {
    /// Fixed single-Gaussian prior; grid search over prior_vars, pick best ELBO.
    Single,
    /// Mixture-of-Gaussians (ash) prior; prior_vars grid becomes the τ² mixture
    /// components with learnable weights. Single fit, no grid search.
    Ash,
}

impl std::str::FromStr for PriorType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "single" | "fixed" => Ok(Self::Single),
            "ash" | "mixture" | "mix" => Ok(Self::Ash),
            _ => anyhow::bail!("Unknown prior type: {} (expected: single, ash)", s),
        }
    }
}

/// Model type for SGVB fine-mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Susie,
    BiSusie,
    SpikeSlab,
}

impl std::str::FromStr for ModelType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "susie" => Ok(Self::Susie),
            "bisusie" => Ok(Self::BiSusie),
            "spike-slab" | "spikeslab" | "ss" => Ok(Self::SpikeSlab),
            _ => anyhow::bail!("Unknown model type: {}", s),
        }
    }
}

/// Configuration for SGVB fine-mapping.
#[derive(Debug, Clone)]
pub struct FitConfig {
    pub model_type: ModelType,
    pub prior_type: PriorType,
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
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Susie,
            prior_type: PriorType::Single,
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
        }
    }
}

impl FitConfig {
    /// Number of result slots: prior_vars.len() for grid search, 1 for ash.
    pub fn num_prior_results(&self) -> usize {
        match self.prior_type {
            PriorType::Single => self.prior_vars.len(),
            PriorType::Ash => 1,
        }
    }
}

use crate::summary_stats::common::BlockFitResult;

/// Internal result tuple: (avg_elbo, pip, effect_mean, effect_std).
pub(crate) type PriorFitResult = (f32, DMatrix<f32>, DMatrix<f32>, DMatrix<f32>);

/// Result from fitting a single block with per-prior-var results (SGVB-specific).
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
fn select_best_prior(detailed: &BlockFitResultDetailed) -> BlockFitResult {
    let best_idx = detailed
        .per_prior_elbos
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    BlockFitResult {
        pip: detailed.per_prior_pips[best_idx].clone(),
        effect_mean: detailed.per_prior_effects[best_idx].clone(),
        effect_std: detailed.per_prior_stds[best_idx].clone(),
        avg_elbo: detailed.per_prior_elbos[best_idx],
    }
}
