use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use candle_util::sgvb::{
    samples_local_reparam_loss, AnalyticalKL, FixedGaussianLikelihood, FixedGaussianPrior,
    GaussianRegressionSGVB, LocalReparamSample, RegressionSGVB, SGVBConfig,
    VariationalDistribution, WeightedGaussianLikelihood,
};
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;
use rand::prelude::*;
use std::collections::VecDeque;

use super::config::{FitConfig, PriorFitResult};
use super::models::{GeneticModel, GeneticModelSpec};

/// Precomputed tensors for block fitting.
pub(crate) struct BlockTensors {
    pub x: Tensor,
    pub y: Tensor,
    /// Per-observation variance (n, k). None → unit variance (FixedGaussianLikelihood).
    pub var: Option<Tensor>,
    pub conf: Option<Tensor>,
    pub p: usize,
    pub k: usize,
    pub n: usize,
    pub use_minibatch: bool,
}

/// Run the SGVB training loop and return the average ELBO over the trailing window.
pub(crate) fn run_sgvb_loop(
    optimizer: &mut AdamW,
    num_iterations: usize,
    elbo_window: usize,
    mut step_fn: impl FnMut() -> Result<Tensor>,
) -> Result<f32> {
    let mut elbo_buffer: VecDeque<f32> = VecDeque::with_capacity(elbo_window);

    for _iter in 0..num_iterations {
        let loss = step_fn()?;
        optimizer.backward_step(&loss)?;

        let elbo_val = -loss.to_scalar::<f32>()?;
        if elbo_buffer.len() == elbo_window {
            elbo_buffer.pop_front();
        }
        elbo_buffer.push_back(elbo_val);
    }

    Ok(if elbo_buffer.is_empty() {
        0.0
    } else {
        elbo_buffer.iter().sum::<f32>() / elbo_buffer.len() as f32
    })
}

/// Train with a single prior_var. Returns (avg_elbo, pip, effect_mean, effect_std).
pub(crate) fn fit_single_prior(
    tensors: &BlockTensors,
    prior_var: f32,
    config: &FitConfig,
    device: &Device,
) -> Result<PriorFitResult> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let sgvb_config = SGVBConfig {
        num_samples: config.num_sgvb_samples,
        kl_weight: 1.0,
    };
    // Build confounder model if needed
    let conf_model: Option<GaussianRegressionSGVB<FixedGaussianPrior>> =
        if let Some(ref ct) = tensors.conf {
            let conf_prior = FixedGaussianPrior::new(1.0);
            Some(GaussianRegressionSGVB::new(
                vb.pp("conf"),
                ct.clone(),
                tensors.k,
                conf_prior,
                sgvb_config.clone(),
            )?)
        } else {
            None
        };

    let genetic = GeneticModel::new(GeneticModelSpec {
        vb: &vb,
        x_design: tensors.x.clone(),
        sgvb_config,
        model_type: config.model_type,
        num_components: config.num_components,
        p: tensors.p,
        k: tensors.k,
        init_prior_std: prior_var.sqrt(),
    })?;

    let mut optimizer = AdamW::new_lr(varmap.all_vars(), config.learning_rate)?;
    let mut rng = StdRng::seed_from_u64(config.seed);

    let avg_elbo = run_sgvb_loop(
        &mut optimizer,
        config.num_iterations,
        config.elbo_window,
        || {
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
            let gen_sample = genetic.local_reparam_sample(config.num_sgvb_samples, &x_batch)?;
            samples.push(gen_sample);

            if let Some(ref cm) = conf_model {
                let conf_sample = if tensors.use_minibatch {
                    local_reparam_with_design(
                        cm,
                        config.num_sgvb_samples,
                        conf_batch.as_ref().unwrap(),
                    )?
                } else {
                    cm.forward(config.num_sgvb_samples)?
                };
                samples.push(conf_sample);
            }

            let base_loss = if let Some(ref vb) = var_batch {
                let likelihood = WeightedGaussianLikelihood::new(y_batch, vb)?;
                samples_local_reparam_loss(&samples, &likelihood, 1.0)?
            } else {
                let likelihood = FixedGaussianLikelihood::new(y_batch, 1.0);
                samples_local_reparam_loss(&samples, &likelihood, 1.0)?
            };
            let kl_sel = genetic.kl_selection(config.prior_alpha)?;
            Ok((base_loss + kl_sel)?)
        },
    )?;

    let (pip_tensor, eff_mean_tensor, eff_std_tensor) = genetic.extract_results()?;
    let pip: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&pip_tensor)?;
    let eff_mean: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&eff_mean_tensor)?;
    let eff_std: DMatrix<f32> = <DMatrix<f32> as ConvertMatOps>::from_tensor(&eff_std_tensor)?;

    Ok((avg_elbo, pip, eff_mean, eff_std))
}

/// Compute local reparameterization sample with a custom design matrix (for minibatch).
pub(crate) fn local_reparam_with_design<V: VariationalDistribution, P: AnalyticalKL>(
    model: &RegressionSGVB<V, P>,
    num_samples: usize,
    x_batch: &Tensor,
) -> Result<LocalReparamSample> {
    let theta_mean = model.variational.mean()?;
    let theta_var = model.variational.var()?;

    let eta_mean = x_batch.matmul(&theta_mean)?;
    let x_sq = x_batch.sqr()?;
    let eta_var = x_sq.matmul(&theta_var)?;

    let (nb, k) = eta_mean.dims2()?;
    let device = eta_mean.device();
    let dtype = eta_mean.dtype();

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

    let eta_std = (eta_var + 1e-8)?.sqrt()?;
    let eta = eta_mean
        .unsqueeze(0)?
        .broadcast_add(&epsilon.broadcast_mul(&eta_std.unsqueeze(0)?)?)?;

    let kl = model.prior.kl_from_gaussian(&theta_mean, &theta_var)?;
    Ok(LocalReparamSample { eta, kl })
}
