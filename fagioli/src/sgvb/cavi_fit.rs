//! CAVI (Coordinate Ascent Variational Inference) SuSiE fitting.
//!
//! Wraps `candle_util::sgvb::cavi_susie` to produce `BlockFitResult` output.

use anyhow::Result;
use candle_util::candle_core::Device;
use candle_util::sgvb::cavi_susie::{cavi_susie, CaviSusieParams};
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;

use crate::summary_stats::common::BlockFitResult;

/// Fit a single block using CAVI SuSiE on individual-level data.
///
/// Uses median of `prior_vars` grid (CAVI is deterministic, no grid search needed).
pub fn fit_block_cavi(
    x_block: &DMatrix<f32>,
    y_block: &DMatrix<f32>,
    num_components: usize,
    max_iter: usize,
    tol: f64,
    prior_vars: &[f32],
) -> Result<BlockFitResult> {
    let p = x_block.ncols();
    let t = y_block.ncols();
    let device = Device::Cpu;

    let x_tensor = x_block.to_tensor(&device)?;

    let prior_var = if prior_vars.is_empty() {
        0.2
    } else {
        let mut sorted = prior_vars.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2] as f64
    };

    let params = CaviSusieParams {
        num_components,
        max_iter,
        tol,
        prior_variance: prior_var,
        estimate_residual_variance: true,
        prior_weights: None,
    };

    let mut pip_mat = DMatrix::<f32>::zeros(p, t);
    let mut eff_mean_mat = DMatrix::<f32>::zeros(p, t);
    let mut eff_std_mat = DMatrix::<f32>::zeros(p, t);
    let mut total_elbo = 0.0f32;

    for tt in 0..t {
        let y_col = y_block.column(tt);
        let y_tensor = candle_util::candle_core::Tensor::from_slice(
            y_col.as_slice(),
            (y_col.len(),),
            &device,
        )?;

        let result = cavi_susie(&x_tensor, &y_tensor, &params)?;
        let beta = result.beta_mean();

        for j in 0..p {
            pip_mat[(j, tt)] = result.pip[j] as f32;
            eff_mean_mat[(j, tt)] = beta[j] as f32;
            let mut var_j = 0.0f64;
            for l in 0..result.alpha.len() {
                var_j += result.alpha[l][j] * (result.s2[l][j] + result.mu[l][j].powi(2));
            }
            var_j -= (beta[j]).powi(2);
            eff_std_mat[(j, tt)] = var_j.max(0.0).sqrt() as f32;
        }

        if let Some(&last_elbo) = result.elbo_trace.last() {
            total_elbo += last_elbo as f32;
        }
    }

    Ok(BlockFitResult {
        pip: pip_mat,
        effect_mean: eff_mean_mat,
        effect_std: eff_std_mat,
        avg_elbo: total_elbo / t as f32,
    })
}
