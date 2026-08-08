//! Logistic noise-contrastive training.
//!
//! # The objective
//!
//! With `ν` negatives per positive and a known, normalised noise density, the
//! Gutmann-Hyvärinen logistic loss on the log Bayes factor `G` is
//!
//! ```text
//! L = − E_data[ log σ(G − log ν) ] − ν E_noise[ log(1 − σ(G − log ν)) ]
//! ```
//!
//! The unit of observation is one eigen-coordinate: a `T`-vector `ž_k` scored
//! against negatives drawn at the same coordinate. Scoring whole blocks instead
//! would give one positive per block per step and a correspondingly starved
//! gradient.
//!
//! # The offset is a diagnostic, not a parameter
//!
//! Classical NCE learns a normalising constant because the model is
//! unnormalised. Here both densities are proper, so at the truth the offset has
//! nothing to absorb and should sit near zero. What it actually detects is
//! **overfitting**, and the asymmetry is structural: the positives are a fixed
//! dataset while the negatives are redrawn every step, so a model with more
//! free parameters than data memorises the positives and the offset rises to
//! pay for it.
//!
//! Measured on planted data (`test_offset_tracks_samples_per_parameter`): with
//! 40 parameters per block against 30 eigen-coordinates the offset settles at
//! `+0.91`; with 12 against 240 it settles at `-0.03`. Read a large offset as
//! "`U` has more freedom than this block can support" — raise `prior_inclusion`
//! sparsity, lower `H`, or accept fewer blocks.
//!
//! A drifting offset *can* also mean the noise model and the data disagree on
//! scale, which is the same failure the whiteness statistic reports; the two are
//! distinguished by whether the whiteness deviation is also large.

use anyhow::Result;
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_util::grad_clip::clip_grad_global_norm;
use log::info;

use super::model::{EmbedConfig, EmbedModel};
use super::noise::NoiseModel;
use super::whiten::WhitenedBlock;

/// What a fit produced.
pub struct EmbedFit {
    /// `V̌`, shape (T, H) — still in whitened trait coordinates.
    pub v_check: nalgebra::DMatrix<f32>,
    /// `E[U_b]` per block, each (p_b, H).
    pub u_mean: Vec<nalgebra::DMatrix<f32>>,
    /// Posterior inclusion probabilities per block, each (p_b, H).
    pub u_pip: Vec<nalgebra::DMatrix<f32>>,
    /// `V̌ (U'U) V̌'`, ready for the geometry verdict.
    pub trait_geometry: nalgebra::DMatrix<f32>,
    /// Trailing-mean loss per iteration.
    pub loss_trace: Vec<f32>,
    /// Final NCE offset; near zero means the score is correctly normalised.
    pub offset: f32,
    /// Fitted polygenic variance per program, when the dense arm is on.
    pub dense_variance: Option<Vec<f32>>,
    /// Final `‖V̌'V̌ − I‖²_F`; large means the gauge did not hold and the dense
    /// arm's diagonal assumption is violated.
    pub gauge_residual: f32,
}

/// Numerically stable `log σ(x)` = `−softplus(−x)`.
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(softplus(&x.neg()?)?.neg()?)
}

/// Numerically stable `log(1 − σ(x))` = `−softplus(x)`.
fn log_one_minus_sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(softplus(x)?.neg()?)
}

/// `softplus(x) = max(x,0) + log(1 + exp(−|x|))`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let relu = x.maximum(&zeros)?;
    let soft = ((x.abs()?.neg()?.exp()? + 1.0)?).log()?;
    Ok((relu + soft)?)
}

/// Fit the embedding by logistic NCE.
pub fn train(
    blocks: &[WhitenedBlock],
    noise: &NoiseModel,
    config: &EmbedConfig,
    device: &Device,
) -> Result<EmbedFit> {
    anyhow::ensure!(!blocks.is_empty(), "no whitened blocks to fit");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_util::candle_core::DType::F32, device);
    let model = EmbedModel::new(&vb, blocks, noise, config.clone(), device)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        },
    )?;

    let nu = config.num_negatives.max(1);
    let log_nu = (nu as f64).ln();
    let num_traits = model.num_traits;

    info!(
        "Training NCE embedding: {} blocks, {} traits, H={}, ν={}, {} iterations",
        model.num_blocks(),
        num_traits,
        config.embedding_dim,
        nu,
        config.num_iterations,
    );

    let mut loss_trace = Vec::with_capacity(config.num_iterations);

    for iter in 0..config.num_iterations {
        let mut total = Tensor::zeros((), candle_util::candle_core::DType::F32, device)?;

        for b in 0..model.num_blocks() {
            let bt = &model.blocks[b];
            let mean = model.mean_structure(b)?;

            // Positives: the real whitened data at each eigen-coordinate.
            let s_pos = model.score(b, &bt.z_white, &mean)?;
            let pos_term = log_sigmoid(&(s_pos.affine(1.0, -log_nu))?)?.sum_all()?;

            // Negatives: ν draws from the exactly-known noise density.
            let mut neg_term = Tensor::zeros((), candle_util::candle_core::DType::F32, device)?;
            for r in 0..nu {
                let mut rng = NoiseModel::block_rng(config.seed, b, iter * nu + r);
                let z_neg = noise.sample_block(b, num_traits, &mut rng);
                let z_neg = {
                    use matrix_util::traits::ConvertMatOps;
                    z_neg.to_tensor(device)?.contiguous()?
                };
                let s_neg = model.score(b, &z_neg, &mean)?;
                neg_term =
                    (neg_term + log_one_minus_sigmoid(&(s_neg.affine(1.0, -log_nu))?)?.sum_all()?)?;
            }

            let samples = bt.rank * (1 + nu);
            // Negative log-likelihood, averaged over the block's samples, plus
            // the selection KL amortized over the same count.
            let nce = ((pos_term + neg_term)?.neg()? / samples as f64)?;
            let kl = (model.kl_selection(b)?.sum_all()? / samples as f64)?;
            total = ((total + nce)? + kl)?;
        }

        // The dense arm's score assumes an orthonormal V̌, but the gauge is
        // applied independently of it: it is also what identifies V̌ at all, so
        // "dense arm on" and "gauge on" must be separable conditions or any
        // comparison between them is confounded.
        let mut loss = (total / model.num_blocks() as f64)?;
        if model.config.gauge_weight > 0.0 {
            loss = (loss + (model.gauge_penalty()? * model.config.gauge_weight)?)?;
        }
        let mut grads = loss.backward()?;
        if let Some(max_norm) = config.grad_clip {
            clip_grad_global_norm(&mut grads, max_norm)?;
        }
        opt.step(&grads)?;

        let l = loss.to_scalar::<f32>()?;
        loss_trace.push(l);
        if iter % 50 == 0 || iter + 1 == config.num_iterations {
            info!(
                "  iter {:>5}: loss {:.5}, offset {:+.4}",
                iter,
                l,
                model.offset_value()?
            );
        }
    }

    let offset = model.offset_value()?;
    if offset.abs() > 0.5 {
        log::warn!(
            "NCE offset drifted to {:+.3}. The score is exactly normalised, so this usually \
             means U has more freedom than the data supports and the fixed positives are \
             being memorised — increase sparsity or lower H. If the whiteness deviation is \
             also large, the noise model is the culprit instead.",
            offset,
        );
    }

    let u_mean = (0..model.num_blocks())
        .map(|b| model.u_mean_matrix(b))
        .collect::<Result<Vec<_>>>()?;
    let u_pip = (0..model.num_blocks())
        .map(|b| model.u_pip_matrix(b))
        .collect::<Result<Vec<_>>>()?;

    let gauge_residual = model.gauge_penalty()?.to_scalar::<f32>()?;
    if model.config.dense_arm {
        log::info!(
            "Dense arm: σ²_d = {:?}, gauge residual {:.4}",
            model
                .dense_variance()?
                .map(|v| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>()),
            gauge_residual,
        );
    }

    Ok(EmbedFit {
        v_check: model.v_check_matrix()?,
        trait_geometry: model.trait_geometry()?,
        u_mean,
        u_pip,
        loss_trace,
        offset,
        dense_variance: model.dense_variance()?,
        gauge_residual,
    })
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod tests;
