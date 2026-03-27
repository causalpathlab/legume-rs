mod config;
mod models;
mod partition;
mod rss;
mod training;

pub use config::{elbo_argmax, select_best_prior};
pub use config::{
    BlockFitResult, BlockFitResultDetailed, ComputeDevice, FitConfig, ModelType, MultilevelConfig,
    RssParams, SnpCoordinates,
};
pub use rss::{adaptive_prior_grid, estimate_block_h2, fit_block_rss};

use anyhow::Result;
use candle_util::candle_core::Device;
use candle_util::sgvb::block_partition::BlockPartition;
use log::info;
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;

use partition::estimate_multilevel_partitions;
use training::{fit_single_prior, BlockTensors};

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
    let detailed = fit_block_inner(x_block, y_block, None, confounders, config, device, None)?;
    Ok(select_best_prior(&detailed))
}

/// Fit a fine-mapping model for a single block with per-observation variance.
///
/// Like `fit_block()` but accepts a per-observation variance tensor `(N, K)`.
/// Uses `WeightedGaussianLikelihood` instead of `FixedGaussianLikelihood`.
///
/// When `config.multilevel` is set, `coords` provides SNP positions and
/// chromosomes for LD sub-block estimation. Pass `None` to fall back to
/// regular fixed-size partitions.
///
/// Returns detailed results with per-prior-var ELBOs for empirical Bayes.
pub fn fit_block_weighted(
    x_block: &DMatrix<f32>,
    y_block: &DMatrix<f32>,
    var_block: &DMatrix<f32>,
    confounders: Option<&DMatrix<f32>>,
    config: &FitConfig,
    device: &Device,
    coords: Option<&SnpCoordinates>,
) -> Result<BlockFitResultDetailed> {
    let partitions = estimate_multilevel_partitions(x_block, config, coords)?;
    fit_block_inner(
        x_block,
        y_block,
        Some(var_block),
        confounders,
        config,
        device,
        partitions.as_deref(),
    )
}

fn fit_block_inner(
    x_block: &DMatrix<f32>,
    y_block: &DMatrix<f32>,
    var_block: Option<&DMatrix<f32>>,
    confounders: Option<&DMatrix<f32>>,
    config: &FitConfig,
    device: &Device,
    partitions: Option<&[BlockPartition]>,
) -> Result<BlockFitResultDetailed> {
    let n = x_block.nrows();
    let p = x_block.ncols();
    let k = y_block.ncols();

    let x_tensor = x_block.to_tensor(device)?.contiguous()?;
    let y_tensor = y_block.to_tensor(device)?.contiguous()?;
    let var_tensor = var_block
        .map(|v| -> Result<_> { Ok(v.to_tensor(device)?.contiguous()?) })
        .transpose()?;
    let conf_tensor = confounders
        .map(|c| -> Result<_> { Ok(c.to_tensor(device)?.contiguous()?) })
        .transpose()?;

    // Multilevel model uses internal x_design; minibatch not supported
    let use_minibatch = n > config.batch_size && partitions.is_none();

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
            fit_single_prior(&tensors, prior_var, config, device, partitions)?;
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
