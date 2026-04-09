pub mod cavi_fit;
mod config;
mod models;
pub mod refinement;
mod rss;
mod training;

pub use config::{BlockFitResultDetailed, ComputeDevice, FitConfig, ModelType, PriorType};
pub use rss::{estimate_block_h2, fit_block_rss};

// Re-export shared types that were moved to summary_stats::common
pub use crate::summary_stats::common::{adaptive_prior_grid, BlockFitResult, RssParams};

use anyhow::Result;
use candle_util::candle_core::Device;
use log::info;
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;

use candle_util::candle_nn::VarBuilder;
use candle_util::sgvb::{FixedGaussianPrior, MixtureGaussianPrior, PriorKind};

use training::{fit_with_prior, BlockTensors};

/// Fit a fine-mapping model for a single LD block.
pub fn fit_block(
    x_block: &DMatrix<f32>,
    y_block: &DMatrix<f32>,
    confounders: Option<&DMatrix<f32>>,
    config: &FitConfig,
    device: &Device,
) -> Result<BlockFitResult> {
    let detailed = fit_block_inner(x_block, y_block, None, confounders, config, device)?;
    Ok(detailed.best_result())
}

/// Fit a fine-mapping model for a single block with per-observation variance.
///
/// Uses `WeightedGaussianLikelihood` with per-observation variance `(N, K)`.
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

type PriorFactory = Box<dyn Fn(&VarBuilder) -> anyhow::Result<PriorKind>>;

/// Build prior-construction closures from config: one per grid point for Single,
/// one mixture for Ash.
pub(crate) fn make_priors_for_config(config: &FitConfig) -> Vec<(PriorFactory, String)> {
    match config.prior_type {
        PriorType::Single => config
            .prior_vars
            .iter()
            .map(|&pv| {
                let factory: PriorFactory = Box::new(move |_vb: &VarBuilder| {
                    Ok(PriorKind::Fixed(FixedGaussianPrior::new(pv.sqrt())))
                });
                (factory, format!("prior_var={:.3}", pv))
            })
            .collect(),
        PriorType::Ash => {
            let prior_vars = config.prior_vars.clone();
            let factory: PriorFactory = Box::new(move |vb: &VarBuilder| {
                let tau_sq: Vec<f64> = std::iter::once(1e-10)
                    .chain(prior_vars.iter().map(|&v| v as f64))
                    .collect();
                let mixture = MixtureGaussianPrior::from_grid(vb.pp("mix_prior"), tau_sq)?;
                Ok(PriorKind::Mixture(mixture))
            });
            vec![(factory, "ash prior".to_string())]
        }
    }
}

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
        .map(|v| -> Result<_> { Ok(v.to_tensor(device)?.contiguous()?) })
        .transpose()?;
    let conf_tensor = confounders
        .map(|c| -> Result<_> { Ok(c.to_tensor(device)?.contiguous()?) })
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

    for (make_prior, label) in make_priors_for_config(config) {
        let (avg_elbo, pip, eff_mean, eff_std) =
            fit_with_prior(&tensors, &make_prior, config, device)?;
        info!("  {}, avg_elbo={:.2}", label, avg_elbo);
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
