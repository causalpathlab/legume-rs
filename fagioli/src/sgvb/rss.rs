use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{AdamW, VarBuilder, VarMap};
use candle_util::sgvb::{
    samples_local_reparam_loss, FixedGaussianPrior, GaussianRegressionSGVB, RssLikelihood, RssSvd,
    SGVBConfig,
};
use log::info;
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;

use super::config::{BlockFitResultDetailed, FitConfig, PriorFitResult, RssParams};
use super::models::{GeneticModel, GeneticModelSpec};
use super::partition::estimate_multilevel_partitions;
use super::training::run_sgvb_loop;

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
    rss: &RssParams,
    device: &Device,
) -> Result<BlockFitResultDetailed> {
    let p = x_block.ncols();
    let k = z_block.ncols();

    let partitions = estimate_multilevel_partitions(x_block, config, rss.coords)?;

    let x_tensor = x_block.to_tensor(device)?;
    let mut z_tensor = z_block.to_tensor(device)?;

    let svd = RssSvd::from_genotypes(&x_tensor, rss.max_rank, rss.lambda, device)?;
    let x_design = svd.x_design().clone();
    let kk = svd.effective_rank();

    info!(
        "  RSS block: p={}, K={}, T={}, λ={:.2e}, σ²_inf={:.2e}",
        p,
        kk,
        k,
        svd.lambda(),
        config.sigma2_inf,
    );

    // Local LDSC intercept estimation and z-score rescaling
    if rss.ldsc_intercept && kk > 2 {
        let vt = svd.v_mat().t()?;
        let vt_z = vt.matmul(&z_tensor)?;

        let d_vals: Vec<f32> = svd.singular_values().to_vec1()?;
        let d_sq: Vec<f32> = d_vals.iter().map(|&d| d * d).collect();

        let vt_z_data: Vec<f32> = vt_z.flatten_all()?.to_vec1()?;
        let y_raw: Vec<Vec<f32>> = (0..kk)
            .map(|kk_i| (0..k).map(|tt| vt_z_data[kk_i * k + tt]).collect())
            .collect();

        let (intercepts, slopes) = RssSvd::estimate_ldsc_intercept(&d_sq, &y_raw, k);

        for tt in 0..k {
            if intercepts[tt] > 1.01 || slopes[tt].abs() > 0.01 {
                info!(
                    "    LDSC trait {}: intercept={:.3}, slope(h)={:.4}",
                    tt, intercepts[tt], slopes[tt],
                );
            }
        }

        let any_inflated = intercepts.iter().any(|&a| a > 1.0 + 1e-6);
        if any_inflated {
            let scale: Vec<f32> = intercepts.iter().map(|&a| 1.0 / a.sqrt()).collect();
            let scale_tensor =
                Tensor::from_vec(scale, (1, k), z_tensor.device())?.to_dtype(z_tensor.dtype())?;
            z_tensor = z_tensor.broadcast_mul(&scale_tensor)?;
        }
    }

    let y_tilde = svd.project_zscores(&z_tensor)?;
    let rss_lik = RssLikelihood::from_projected(y_tilde);
    let mut results: Vec<PriorFitResult> = Vec::new();

    let intercept_design: Option<Tensor> = if config.sigma2_inf > 0.0 {
        let ones_p = Tensor::ones((p, 1), x_design.dtype(), device)?;
        Some(svd.project_zscores(&ones_p)?)
    } else {
        None
    };

    for &prior_var in &config.prior_vars {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let prior = FixedGaussianPrior::new(prior_var.sqrt());
        let sgvb_config = SGVBConfig {
            num_samples: config.num_sgvb_samples,
            kl_weight: 1.0,
        };

        let genetic = GeneticModel::new(GeneticModelSpec {
            vb: &vb,
            x_design: x_design.clone(),
            prior,
            sgvb_config: sgvb_config.clone(),
            model_type: config.model_type,
            num_components: config.num_components,
            p,
            k,
            partitions: partitions.as_deref(),
        })?;

        let intercept_model: Option<GaussianRegressionSGVB<FixedGaussianPrior>> =
            if let Some(ref int_design) = intercept_design {
                let int_prior = FixedGaussianPrior::new(config.sigma2_inf);
                Some(GaussianRegressionSGVB::new(
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

        let avg_elbo = run_sgvb_loop(
            &mut optimizer,
            config.num_iterations,
            config.elbo_window,
            || {
                let gen_sample =
                    genetic.local_reparam_sample(config.num_sgvb_samples, &x_design)?;
                let mut samples = vec![gen_sample];
                if let Some(ref im) = intercept_model {
                    samples.push(im.forward(config.num_sgvb_samples)?);
                }

                let loss = samples_local_reparam_loss(&samples, &rss_lik, 1.0)?;
                let kl_cat = genetic.kl_categorical(config.prior_alpha, device)?;
                Ok((loss + kl_cat)?)
            },
        )?;

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
