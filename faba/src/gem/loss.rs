//! NCE loss assembly for one minibatch on a given axis.
//!
//! For each positive, the loss is
//!
//!     L_i = -log( exp(score+_i) / (exp(score+_i) + Σ_k exp(score_neg_{i,k})) )
//!
//! where negatives concatenate random / swap-gene-mode draws (when
//! applicable for the sub-batch's stratum). All negatives are scored at
//! the same RHS column (cell id or pb id) as the positive.

use super::common::candle_core;
use anyhow::Result;
use candle_core::{DType, Tensor};

use super::model::{Axis, GemModel};
use super::sampling::{Minibatch, NegativeEdges, SubBatch};

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
    let (e_pos, b_pos) = model.embed_and_bias_rows(&pos.feature_rows())?;
    let (e_rhs, b_rhs) = model.rhs_rows(axis, &pos.axis_id)?;
    let s_pos = GemModel::score_diag(&e_pos, &e_rhs, &b_pos, &b_rhs)?;

    ////////////////////////////////////////
    // Negative-score blocks
    ////////////////////////////////////////
    // Marginal-noise negatives (the NEG noise distribution) + structured hard
    // negatives (swap-modality / swap-gene-mode) which preserve the multimodal
    // and program contrasts as extra logistic slates.
    let mut neg_blocks: Vec<Tensor> = Vec::new();
    if let Some(block) = score_negative_edges(model, axis, &pos.axis_id, &sub.marginal)? {
        neg_blocks.push(block);
    }
    if let Some(edges) = sub.swap_gene_mode.as_ref() {
        if let Some(block) = score_negative_edges(model, axis, &pos.axis_id, edges)? {
            neg_blocks.push(block);
        }
    }
    if let Some(edges) = sub.swap_modality.as_ref() {
        if let Some(block) = score_negative_edges(model, axis, &pos.axis_id, edges)? {
            neg_blocks.push(block);
        }
    }

    ////////////////////////////////////////
    // Logistic (NEG/SGNS) NCE
    ////////////////////////////////////////
    // ℓ_i = -( log σ(s_pos_i) + Σ_blocks Σ_k log σ(-s_neg_{i,k}) ). Each block is
    // [B, k]; `logistic_nce` sums them per positive. No partition function, so
    // the objective does not depend on which features exist — robust to feature
    // QC, unlike the sampled softmax. The per-feature biases (b_agg / b_comp)
    // absorb the NEG `log(k·q)` noise correction, SGNS-style.
    let per_pos = graph_embedding_util::loss::logistic_nce(&s_pos, &neg_blocks)?;

    // Plain mean over positives. Abundance balance is handled upstream by the
    // sampler's `count^τ` draw weighting, so there is no per-positive loss weight.
    Ok(per_pos.mean_all()?)
}

fn score_negative_edges(
    model: &GemModel,
    axis: Axis,
    pos_axis_ids: &[u32],
    edges: &NegativeEdges,
) -> Result<Option<Tensor>> {
    if edges.k == 0 || edges.is_empty() {
        return Ok(None);
    }
    let b = pos_axis_ids.len();
    let k = edges.k;
    debug_assert_eq!(
        b * k,
        edges.len(),
        "edge-batch length {} != B*K",
        edges.len()
    );

    let mut axis_tiled = Vec::with_capacity(b * k);
    for &a in pos_axis_ids {
        for _ in 0..k {
            axis_tiled.push(a);
        }
    }

    let (e_neg, b_neg) = model.embed_and_bias_rows(&edges.feature_rows())?;
    let (e_rhs_t, b_rhs_t) = model.rhs_rows(axis, &axis_tiled)?;
    let s_neg = GemModel::score_diag(&e_neg, &e_rhs_t, &b_neg, &b_rhs_t)?;
    let s_neg = s_neg.reshape((b, k))?;
    Ok(Some(s_neg))
}
