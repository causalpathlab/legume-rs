//! NCE loss assembly for one minibatch on a given axis.
//!
//! For each positive, the loss is
//!
//!     L_i = -log( exp(score+_i) / (exp(score+_i) + Σ_k exp(score_neg_{i,k})) )
//!
//! where negatives concatenate random / swap-gene-mode draws (when
//! applicable for the sub-batch's stratum). All negatives are scored at
//! the same RHS column (cell id or pb id) as the positive.

use super::common::{candle_core, candle_nn};
use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};

use super::model::{Axis, GemModel};
use super::sampling::{Minibatch, NegativeSlate, SubBatch};

/// Combine sub-batch losses (anchor + modifier) into one scalar tensor.
pub fn minibatch_loss(model: &GemModel, axis: Axis, mb: &Minibatch) -> Result<Tensor> {
    let mut losses = Vec::new();
    if let Some(sub) = &mb.anchor {
        losses.push(sub_batch_loss(model, axis, sub)?);
    }
    if let Some(sub) = &mb.modifier {
        losses.push(sub_batch_loss(model, axis, sub)?);
    }
    match losses.len() {
        0 => Ok(Tensor::zeros((), DType::F32, &model.dev)?),
        1 => Ok(losses.into_iter().next().unwrap()),
        _ => {
            let scale = 1.0 / losses.len() as f64;
            let mut sum = losses[0].affine(scale, 0.0)?;
            for l in &losses[1..] {
                sum = (sum + l.affine(scale, 0.0)?)?;
            }
            Ok(sum)
        }
    }
}

fn sub_batch_loss(model: &GemModel, axis: Axis, sub: &SubBatch) -> Result<Tensor> {
    let pos = &sub.positives;
    let b = pos.len();
    if b == 0 {
        return Ok(Tensor::zeros((), DType::F32, &model.dev)?);
    }

    ////////////////////////////////////////
    // Positive scores
    ////////////////////////////////////////
    let (e_pos, b_pos) = model.embed_and_bias_rows(
        &pos.gene_for_rho,
        &pos.gene_for_z,
        &pos.modality_for_q,
        &pos.region_for_delta,
        &pos.gene_for_bias,
        &pos.modality_for_bias,
        &pos.is_agg,
    )?;
    let (e_rhs, b_rhs) = model.rhs_rows(axis, &pos.axis_id)?;
    let s_pos = GemModel::score_diag(&e_pos, &e_rhs, &b_pos, &b_rhs)?;

    ////////////////////////////////////////
    // Negative-score blocks
    ////////////////////////////////////////
    let mut neg_blocks: Vec<Tensor> = Vec::new();
    if let Some(block) = score_negative_slate(model, axis, &pos.axis_id, &sub.rand)? {
        neg_blocks.push(block);
    }
    if let Some(slate) = sub.swap_gene_mode.as_ref() {
        if let Some(block) = score_negative_slate(model, axis, &pos.axis_id, slate)? {
            neg_blocks.push(block);
        }
    }
    if let Some(slate) = sub.swap_modality.as_ref() {
        if let Some(block) = score_negative_slate(model, axis, &pos.axis_id, slate)? {
            neg_blocks.push(block);
        }
    }

    ////////////////////////////////////////
    // Cat + log-softmax
    ////////////////////////////////////////
    let s_pos_col = s_pos.unsqueeze(1)?;
    let all_scores = if neg_blocks.is_empty() {
        s_pos_col
    } else {
        let mut blocks = Vec::with_capacity(1 + neg_blocks.len());
        blocks.push(s_pos_col);
        blocks.extend(neg_blocks);
        Tensor::cat(&blocks, 1)?
    };

    let log_sm = candle_nn::ops::log_softmax(&all_scores, 1)?;
    let log_p_pos = log_sm.i((.., 0))?;
    let neg = log_p_pos.affine(-1.0, 0.0)?; // [B] per-positive NCE loss

    // Plain mean over positives. Abundance balance is handled upstream by the
    // sampler's `count^τ` draw weighting, so there is no per-positive loss
    // weight.
    Ok(neg.mean_all()?)
}

fn score_negative_slate(
    model: &GemModel,
    axis: Axis,
    pos_axis_ids: &[u32],
    slate: &NegativeSlate,
) -> Result<Option<Tensor>> {
    if slate.k == 0 || slate.is_empty() {
        return Ok(None);
    }
    let b = pos_axis_ids.len();
    let k = slate.k;
    debug_assert_eq!(b * k, slate.len(), "slate length {} != B*K", slate.len());

    let mut axis_tiled = Vec::with_capacity(b * k);
    for &a in pos_axis_ids {
        for _ in 0..k {
            axis_tiled.push(a);
        }
    }

    let (e_neg, b_neg) = model.embed_and_bias_rows(
        &slate.gene_for_rho,
        &slate.gene_for_z,
        &slate.modality_for_q,
        &slate.region_for_delta,
        &slate.gene_for_bias,
        &slate.modality_for_bias,
        &slate.is_agg,
    )?;
    let (e_rhs_t, b_rhs_t) = model.rhs_rows(axis, &axis_tiled)?;
    let s_neg = GemModel::score_diag(&e_neg, &e_rhs_t, &b_neg, &b_rhs_t)?;
    let s_neg = s_neg.reshape((b, k))?;
    Ok(Some(s_neg))
}
