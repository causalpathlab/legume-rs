//! Shared helpers for the unified `senna predict` subcommand.
//!
//! - `estimate_delta`: per-batch δ from new-data pseudobulk vs frozen β
//!   (lifted from the original `eval_topic.rs` so the indexed path can use it).
//! - `decoder_only_inference_*`: optimize θ logits against the frozen decoder
//!   from a uniform start, no encoder involvement.
//! - `predictive_llik_*`: wrap decoder forwards to return just the per-cell
//!   log-likelihood tensor.

use crate::embed_common::*;
use crate::topic::eval::GeneRemap;
use candle_core::{Device, Result as CandleResult, Tensor, Var};
use candle_nn::ops;
use candle_util::candle_indexed_model_traits::IndexedDecoderT;
use candle_util::candle_model_traits::DecoderModuleT;

/// Lower / upper bounds on per-(gene, batch) δ. Stops a single noisy batch
/// from blowing up the encoder's null input when one batch has near-zero
/// coverage of a gene that is highly expressed under the training mixture.
pub const DELTA_CLAMP_MIN: f32 = 0.01;
pub const DELTA_CLAMP_MAX: f32 = 100.0;
/// Floor on the predicted-proportion denominator in the obs/pred ratio,
/// avoiding division by zero for genes that no topic produces.
pub const DELTA_PRED_EPS: f32 = 1e-10;

/// Latent inference mode chosen by the caller from CLI flags.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LatentMode {
    /// Forward pass through the encoder only (default, fastest).
    Encoder,
    /// Encoder warm-start, then decoder-side gradient on θ anchored to encoder by L2.
    EncoderRefine,
    /// Skip encoder; init θ uniform `log(1/K)`, optimize against frozen decoder.
    DecoderOnly,
}

/// Decoder-only latent inference for the dense topic model.
///
/// Initializes θ logits to zeros (uniform distribution after log_softmax) and
/// runs `num_steps` of plain gradient descent against the frozen decoder's
/// likelihood. No regularization — the data should pin θ on its own.
pub fn decoder_only_inference_dense<Dec: DecoderModuleT>(
    decoder: &Dec,
    x_nd: &Tensor,
    n_topics: usize,
    learning_rate: f64,
    num_steps: usize,
    dev: &Device,
) -> CandleResult<Tensor> {
    let n = x_nd.dim(0)?;
    let z_init = Tensor::zeros((n, n_topics), candle_core::DType::F32, dev)?;
    let z_var = Var::from_tensor(&z_init)?;

    for _step in 0..num_steps {
        let log_z = ops::log_softmax(z_var.as_tensor(), 1)?;
        let (_recon, llik) = decoder.forward_with_llik(&log_z, x_nd, &|_, _| unreachable!())?;
        let loss = llik.mean_all()?.neg()?;
        let grad = loss.backward()?;
        let z_grad = grad.get(z_var.as_tensor()).unwrap();
        let updated = (z_var.as_tensor() - (z_grad * learning_rate)?)?;
        z_var.set(&updated)?;
    }

    ops::log_softmax(z_var.as_tensor(), 1)
}

/// Bundled inputs for the indexed decoder forward at a single block.
/// Used to keep the decoder-only inference and predictive-llik APIs slim.
pub struct IndexedDecoderInput<'a> {
    pub union_indices: &'a Tensor,
    pub indexed_x: &'a Tensor,
    pub log_q_s: &'a Tensor,
}

/// Decoder-only latent inference for the indexed topic model.
pub fn decoder_only_inference_indexed<Dec: IndexedDecoderT>(
    decoder: &Dec,
    input: &IndexedDecoderInput<'_>,
    n_topics: usize,
    learning_rate: f64,
    num_steps: usize,
    dev: &Device,
) -> CandleResult<Tensor> {
    let n = input.indexed_x.dim(0)?;
    let z_init = Tensor::zeros((n, n_topics), candle_core::DType::F32, dev)?;
    let z_var = Var::from_tensor(&z_init)?;

    for _step in 0..num_steps {
        let log_z = ops::log_softmax(z_var.as_tensor(), 1)?;
        let (_recon, llik) =
            decoder.forward_indexed(&log_z, input.union_indices, input.indexed_x, input.log_q_s)?;
        let loss = llik.mean_all()?.neg()?;
        let grad = loss.backward()?;
        let z_grad = grad.get(z_var.as_tensor()).unwrap();
        let updated = (z_var.as_tensor() - (z_grad * learning_rate)?)?;
        z_var.set(&updated)?;
    }

    ops::log_softmax(z_var.as_tensor(), 1)
}

/// Per-cell predictive log-likelihood for the dense topic model.
pub fn predictive_llik_dense<Dec: DecoderModuleT>(
    decoder: &Dec,
    log_z_nk: &Tensor,
    x_nd: &Tensor,
) -> CandleResult<Tensor> {
    let (_, llik) = decoder.forward_with_llik(log_z_nk, x_nd, &|_, _| unreachable!())?;
    Ok(llik)
}

/// Per-cell predictive log-likelihood for the indexed topic model.
pub fn predictive_llik_indexed<Dec: IndexedDecoderT>(
    decoder: &Dec,
    log_z_nk: &Tensor,
    union_indices: &Tensor,
    indexed_x: &Tensor,
    log_q_s: &Tensor,
) -> CandleResult<Tensor> {
    let (_, llik) = decoder.forward_indexed(log_z_nk, union_indices, indexed_x, log_q_s)?;
    Ok(llik)
}

/// Per-batch sums accumulated across blocks during one TMLE iteration.
/// `pb_obs[d,b]` and `pb_pred[d,b]` are NB-Fisher-weighted when `phi` is
/// supplied; otherwise unweighted (Poisson / multinomial).
pub struct DeltaSums {
    pub pb_obs: Mat,  // [D_train, B]
    pub pb_pred: Mat, // [D_train, B]
}

impl DeltaSums {
    pub fn zeros(d_train: usize, n_batches: usize) -> Self {
        Self {
            pb_obs: Mat::zeros(d_train, n_batches),
            pb_pred: Mat::zeros(d_train, n_batches),
        }
    }

    pub fn merge_into(&mut self, other: &Self) {
        self.pb_obs += &other.pb_obs;
        self.pb_pred += &other.pb_pred;
    }
}

/// Closed-form δ from per-batch obs and predicted sums.
///
/// `δ[d,b] = pb_obs[d,b] / max(pb_pred[d,b], eps)`, clamped to [0.01, 100].
/// When all batches share the same generative law, δ stays near 1.
pub fn solve_delta_from_sums(sums: &DeltaSums) -> Mat {
    let d = sums.pb_obs.nrows();
    let b = sums.pb_obs.ncols();
    let mut delta = Mat::zeros(d, b);
    for bi in 0..b {
        for di in 0..d {
            let num = sums.pb_obs[(di, bi)];
            let den = sums.pb_pred[(di, bi)].max(DELTA_PRED_EPS);
            delta[(di, bi)] = (num / den).clamp(DELTA_CLAMP_MIN, DELTA_CLAMP_MAX);
        }
    }
    delta
}

/// NB Fisher info per cell-gene at the working δ=1 hypothesis.
///
/// `w[d,j] = φ[d] / (μ[d,j] + φ[d])` where `μ[d,j] = lib[j] · pred[d,j]`
/// and `pred[d,j] = Σ_k θ̂[j,k] · exp(β[d,k])`. Returns `1.0` for all (d,j)
/// when `phi` is `None` (Poisson / multinomial decoder).
pub fn nb_fisher_weight(phi: Option<&[f32]>, mu: f32, d: usize) -> f32 {
    match phi {
        None => 1.0,
        Some(phi) => {
            let pd = phi[d].max(1e-6);
            pd / (mu + pd)
        }
    }
}

/// Estimate per-batch δ from new-data pseudobulk and the frozen training dictionary.
///
/// `delta[d,b] = (pb_test[d,b] / lib_b) / predicted[d]`, where
/// `predicted[d] = Σ_k θ̄_train[k] · exp(β[d,k])` is the gene-d proportion
/// implied by the training topic mixture. This makes δ a proper contrast
/// against training: held-out batch composition vs the training marginal.
///
/// `theta_mean = None` falls back to uniform 1/K weighting (legacy behavior;
/// only correct if training θ̄ was uniform).
pub fn estimate_delta(
    data_vec: &SparseIoVec,
    beta_dk: &Mat,
    theta_mean: Option<&[f32]>,
    gene_remap: Option<&GeneRemap>,
    block_size: Option<usize>,
) -> anyhow::Result<Option<Mat>> {
    let n_batches = data_vec.num_batches();
    if n_batches <= 1 {
        info!("Single batch or no batches — skipping delta estimation");
        return Ok(None);
    }

    let d_train = beta_dk.nrows();
    let k = beta_dk.ncols();

    let exp_beta = beta_dk.map(f32::exp);
    let weights: Vec<f32> = if let Some(tm) = theta_mean {
        anyhow::ensure!(
            tm.len() == k,
            "theta_mean length {} != n_topics {}",
            tm.len(),
            k,
        );
        let s: f32 = tm.iter().sum();
        if s > 0.0 {
            tm.iter().map(|x| x / s).collect()
        } else {
            vec![1.0 / k as f32; k]
        }
    } else {
        info!("estimate_delta: theta_mean missing — falling back to uniform 1/K");
        vec![1.0 / k as f32; k]
    };
    let mut predicted = nalgebra::DVector::<f32>::zeros(d_train);
    for d in 0..d_train {
        let mut acc = 0.0;
        for kk in 0..k {
            acc += weights[kk] * exp_beta[(d, kk)];
        }
        predicted[d] = acc;
    }
    let pred_sum: f32 = predicted.iter().sum();
    if pred_sum > 0.0 {
        predicted /= pred_sum;
    }

    let d_new = data_vec.num_rows();
    let ntot = data_vec.num_columns();
    let mut pb_new = Mat::zeros(d_new, n_batches);

    let block_size =
        block_size.unwrap_or_else(|| matrix_util::utils::default_block_size(data_vec.num_rows()));
    for lb in (0..ntot).step_by(block_size) {
        let ub = (lb + block_size).min(ntot);
        let csc = data_vec.read_columns_csc(lb..ub)?;
        let batch_ids = data_vec.get_batch_membership(lb..ub);
        for (local_j, &batch_id) in batch_ids.iter().enumerate() {
            let col = csc.col(local_j);
            for (&row, &val) in col.row_indices().iter().zip(col.values().iter()) {
                pb_new[(row, batch_id)] += val;
            }
        }
    }

    let mut pb_train = Mat::zeros(d_train, n_batches);
    if let Some(remap) = gene_remap {
        for (new_idx, opt_train) in remap.new_to_train.iter().enumerate() {
            if let Some(&train_idx) = opt_train.as_ref() {
                pb_train.row_mut(train_idx).copy_from(&pb_new.row(new_idx));
            }
        }
    } else {
        pb_train.copy_from(&pb_new);
    }

    let mut delta_db = Mat::zeros(d_train, n_batches);
    for b in 0..n_batches {
        let lib: f32 = pb_train.column(b).sum();
        if lib <= 0.0 {
            delta_db.column_mut(b).fill(1.0);
            continue;
        }
        for d in 0..d_train {
            let obs_prop = pb_train[(d, b)] / lib;
            let pred = predicted[d].max(DELTA_PRED_EPS);
            delta_db[(d, b)] = (obs_prop / pred).clamp(DELTA_CLAMP_MIN, DELTA_CLAMP_MAX);
        }
    }

    info!("Estimated delta: {d_train} genes × {n_batches} batches");
    Ok(Some(delta_db))
}
