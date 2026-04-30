use super::common::{expand_delta_for_block, process_blocks};
use crate::embed_common::*;

use candle_core::{Device, Tensor, Var};
use candle_nn::ops;
use candle_util::candle_indexed_data_loader::top_k_indices_weighted;
use candle_util::candle_indexed_model_traits::*;
use candle_util::candle_topic_refinement::TopicRefinementConfig;
use std::collections::{BTreeSet, HashMap};

/// Convert a dense [N, D] tensor to indexed form: `union_indices` [S] and `indexed_x` [N, S].
///
/// `shortlist_weights` is the same per-gene weight vector used at training time
/// (see `IndexedTrainConfig::shortlist_weights`) — required so inference picks
/// shortlists from the same distribution the encoder was trained on.
pub(crate) fn dense_to_indexed(
    x_nd: &Tensor,
    context_size: usize,
    shortlist_weights: &[f32],
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let rows: Vec<Vec<f32>> = x_nd.to_vec2()?;
    dense_rows_to_indexed(&rows, context_size, shortlist_weights, dev)
}

/// Convert a dense [N, D] tensor to two indexed forms with a single host copy.
///
/// Returns `((enc_union, enc_x), (dec_union, dec_x))` using one `to_vec2()` call.
pub(crate) fn dense_to_indexed_pair(
    x_nd: &Tensor,
    enc_context_size: usize,
    dec_context_size: usize,
    shortlist_weights: &[f32],
    dev: &Device,
) -> anyhow::Result<((Tensor, Tensor), (Tensor, Tensor))> {
    let rows: Vec<Vec<f32>> = x_nd.to_vec2()?;
    let enc = dense_rows_to_indexed(&rows, enc_context_size, shortlist_weights, dev)?;
    let dec = dense_rows_to_indexed(&rows, dec_context_size, shortlist_weights, dev)?;
    Ok((enc, dec))
}

/// Build indexed representation from pre-extracted rows (avoids repeated `to_vec2`).
fn dense_rows_to_indexed(
    rows: &[Vec<f32>],
    context_size: usize,
    shortlist_weights: &[f32],
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n_batch = rows.len();

    let mut union_set = BTreeSet::new();
    let mut all_top_k: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(n_batch);

    for row in rows {
        let (indices, values) = top_k_indices_weighted(row, shortlist_weights, context_size);
        for &idx in &indices {
            union_set.insert(idx);
        }
        all_top_k.push((indices, values));
    }

    let union_vec: Vec<u32> = union_set.into_iter().collect();
    let s = union_vec.len();

    let pos_map: HashMap<u32, usize> = union_vec
        .iter()
        .enumerate()
        .map(|(pos, &idx)| (idx, pos))
        .collect();

    let mut x_data = vec![0.0f32; n_batch * s];
    for (row, (indices, values)) in all_top_k.iter().enumerate() {
        for (k, &feat_idx) in indices.iter().enumerate() {
            let col = pos_map[&feat_idx];
            x_data[row * s + col] = values[k];
        }
    }

    let union_indices =
        Tensor::from_vec(union_vec, (s,), dev)?.to_dtype(candle_core::DType::U32)?;
    let indexed_x = Tensor::from_vec(x_data, (n_batch, s), dev)?;

    Ok((union_indices, indexed_x))
}

pub(crate) struct EvaluateLatentConfig<'a, Dec> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub enc_context_size: usize,
    pub dec_context_size: usize,
    pub decoder: &'a Dec,
    pub refine_config: Option<&'a TopicRefinementConfig>,
    /// Same shortlist weights used during training.
    pub shortlist_weights: &'a [f32],
}

pub(crate) fn evaluate_latent_by_indexed_encoder<Enc, Dec>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    collapsed: &CollapsedOut,
    config: &EvaluateLatentConfig<Dec>,
) -> anyhow::Result<Mat>
where
    Enc: IndexedEncoderT + Send + Sync,
    Dec: IndexedDecoderT + Send + Sync,
{
    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    // Delta at full D — encoder operates at D_full
    let delta = match config.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| x.posterior_mean().clone())
    .map(|delta_db| {
        delta_db
            .to_tensor(config.dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
            .contiguous()
            .expect("contiguous")
    });

    let block_config = EvaluateBlockConfig {
        dev: config.dev,
        adj_method: config.adj_method,
        delta: delta.as_ref(),
        enc_context_size: config.enc_context_size,
        dec_context_size: config.dec_context_size,
        decoder: config.decoder,
        refine_config: config.refine_config,
        shortlist_weights: config.shortlist_weights,
    };

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        evaluate_indexed_block(block, data_vec, encoder, &block_config)
    })
}

struct EvaluateBlockConfig<'a, Dec> {
    dev: &'a Device,
    adj_method: &'a AdjMethod,
    delta: Option<&'a Tensor>,
    enc_context_size: usize,
    dec_context_size: usize,
    decoder: &'a Dec,
    refine_config: Option<&'a TopicRefinementConfig>,
    shortlist_weights: &'a [f32],
}

fn evaluate_indexed_block<Enc, Dec>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &Enc,
    config: &EvaluateBlockConfig<Dec>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: IndexedEncoderT,
    Dec: IndexedDecoderT,
{
    let (lb, ub) = block;

    // Read sparse -> dense [D, N], all at full D (no feature coarsening)
    let x_dn = data_vec.read_columns_csc(lb..ub)?;
    let x_nd = x_dn.to_tensor(config.dev)?.transpose(0, 1)?;

    // Get batch/residual correction if available
    let x0_nd = config
        .delta
        .map(|delta_bm| {
            expand_delta_for_block(data_vec, delta_bm, config.adj_method, lb, ub, config.dev)
        })
        .transpose()?;

    // Convert dense to indexed: single host copy when refinement needs both
    let need_decoder = config.refine_config.is_some();
    let (enc_result, dec_result) = if need_decoder {
        let ((eu, ex), (du, dx)) = dense_to_indexed_pair(
            &x_nd,
            config.enc_context_size,
            config.dec_context_size,
            config.shortlist_weights,
            config.dev,
        )?;
        ((eu, ex), Some((du, dx)))
    } else {
        let enc = dense_to_indexed(
            &x_nd,
            config.enc_context_size,
            config.shortlist_weights,
            config.dev,
        )?;
        (enc, None)
    };
    let (enc_union, enc_indexed_x) = enc_result;

    // Scatter batch correction at encoder union positions if available
    let enc_indexed_x_null = if let Some(x0) = &x0_nd {
        let union_vec: Vec<u32> = enc_union.to_vec1()?;
        let s = union_vec.len();
        let n_batch = ub - lb;
        let x0_vec: Vec<Vec<f32>> = x0.to_vec2()?;
        let mut x0_data = vec![0.0f32; n_batch * s];
        for (row, x0_row) in x0_vec.iter().enumerate() {
            for (col, &feat_idx) in union_vec.iter().enumerate() {
                x0_data[row * s + col] = x0_row[feat_idx as usize];
            }
        }
        Some(Tensor::from_vec(x0_data, (n_batch, s), config.dev)?)
    } else {
        None
    };

    let (log_z_nk, _) = encoder.forward_indexed_t(
        &enc_union,
        &enc_indexed_x,
        enc_indexed_x_null.as_ref(),
        false,
    )?;

    // Decoder refinement (inference — uniform log_q_s)
    let log_z_nk = if let Some(cfg) = config.refine_config {
        let (dec_union, dec_indexed_x) = dec_result.unwrap();
        let s = dec_union.dim(0)?;
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, config.dev)?;
        refine_indexed_topic_proportions(
            &log_z_nk,
            &dec_union,
            &dec_indexed_x,
            &log_q_s,
            config.decoder,
            cfg,
        )?
    } else {
        log_z_nk
    };

    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

/// Refine per-cell topic proportions by gradient descent against the frozen indexed decoder.
pub(crate) fn refine_indexed_topic_proportions<Dec: IndexedDecoderT>(
    log_z_nk: &Tensor,
    union_indices: &Tensor,
    indexed_x: &Tensor,
    log_q_s: &Tensor,
    decoder: &Dec,
    config: &TopicRefinementConfig,
) -> candle_core::Result<Tensor> {
    let z_logits_init = log_z_nk.detach();
    let z_var = Var::from_tensor(&z_logits_init)?;

    for _step in 0..config.num_steps {
        let log_z = ops::log_softmax(z_var.as_tensor(), 1)?;
        let (_, llik) = decoder.forward_indexed(&log_z, union_indices, indexed_x, log_q_s)?;

        let diff = (z_var.as_tensor() - &z_logits_init)?;
        let reg = (&diff * &diff)?.sum_all()?;

        let loss = ((reg * config.regularization)? - llik.mean_all()?)?;
        let grad = loss.backward()?;
        let z_grad = grad.get(z_var.as_tensor()).unwrap();

        let updated = (z_var.as_tensor() - (z_grad * config.learning_rate)?)?;
        z_var.set(&updated)?;
    }

    ops::log_softmax(z_var.as_tensor(), 1)
}
