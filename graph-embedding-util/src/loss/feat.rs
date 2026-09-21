//! The per-batch stratified cell-axis sampler, the feature-row gather every
//! parameterization shares, and the identity-coarsening NCE loss the
//! `module_gradient_contract` integration test exercises directly (`fit::hier`
//! is the trained path's own engine and does not call through here).

use crate::loss::modality::ModalityPools;
use crate::loss::{logistic_nce, softmax_nce, NceObjective};
use crate::model::JointEmbedModel;
use legume_numeric::candle::candle_core::{Device, Result, Tensor};
use legume_numeric::candle::fast_index::gather_rows;
use rand_distr::weighted::WeightedIndex;

pub struct EdgeBatch {
    pub coarse_cells: Vec<u32>,
    pub fine_feats: Vec<u32>,
    /// `[B*K]` row-major: negatives for positive `b` are at `[b*K..(b+1)*K]`.
    pub neg_feats: Vec<u32>,
    pub n_negatives: usize,
}

///////////////////////////////////////////////////
// Per-batch stratified cell sampler (cell axis) //
///////////////////////////////////////////////////

/// Two-stage per-batch sampler for the cell axis. Stage 1 picks a cell
/// (within this batch) with `q(c) ∝ degree(c)^alpha_cell`; stage 2
/// picks a feature within that cell weighted by `count`.
/// One sampler exists per batch. Negatives are drawn
/// UNIFORMLY over this batch's expressed-feature pool (abundance-independent),
/// as in [`PerBatchSampler`]. With `alpha_cell = 1`, this is approximately
/// equivalent to the flat sampler; with `alpha_cell = 0`, every cell
/// in the batch gets uniform coverage regardless of sequencing depth.
#[derive(Clone)]
pub struct PerBatchStratifiedCellSampler {
    /// Local-index picker into `active_cells`. Weights = `q(c)`.
    pub cell_picker: WeightedIndex<f32>,
    /// Global cell ids with ≥ 1 expressed feature in this batch, in
    /// stable order.
    pub active_cells: Vec<u32>,
    /// Per-active-cell feature sampler; aligned with `active_cells`.
    pub per_cell: Vec<CellFeatureSampler>,
    /// Negative pool: features with any nonzero count in this batch.
    pub neg: WeightedIndex<f32>,
    pub feature_pool: Vec<u32>,
    /// Per-modality pools, on a multiome axis; see [`crate::loss::modality`].
    /// `None` for a single-modality axis. Shared across the clones a
    /// phase-1 subsample makes, hence the `Arc`.
    pub modality: Option<std::sync::Arc<ModalityPools>>,
}

#[derive(Clone)]
pub struct CellFeatureSampler {
    /// Global feature ids expressed in this cell.
    pub features: Vec<u32>,
    /// Raw counts aligned with `features` (the `count` per `(cell, feature)`
    /// edge). Used by the analytical phase-2 projection
    /// ([`crate::cell_projection`]); the sampler itself draws via `picker`.
    pub counts: Vec<f32>,
    /// `WeightedIndex` over `features`; weights = `count`.
    pub picker: WeightedIndex<f32>,
}

/// Gather the effective feature rows `[b, H]` for feature indices `idx` on the
/// LIVE parameters: the composed rows for an adapter / module model, and the
/// free `e_feat` Var otherwise. This is the single feature-gather point for the
/// bge + gem trainers, and it never reads `e_feat` on a model where that field
/// is a detached snapshot.
pub fn gather_feature_rows(model: &JointEmbedModel, idx: &Tensor) -> Result<Tensor> {
    match model.composed() {
        Some(c) => c.compose_rows(idx),
        None => gather_rows(&model.e_feat, idx),
    }
}

/// NCE loss for the identity-coarsening case (every "pb-sample" is its own
/// row): a single `index_select` directly off `model.e_cell` / `model.b_cell`
/// gathers the cell side, since every block has exactly one fine child and
/// `mean([x]) == x`.
pub fn nce_loss_identity(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    objective: NceObjective,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    if b == 0 {
        return Tensor::zeros((), legume_numeric::candle::candle_core::DType::F32, dev);
    }
    let cell_idx_t = Tensor::from_slice(&batch.coarse_cells, b, dev)?;
    let e_cell_pos = gather_rows(&model.e_cell, &cell_idx_t)?;
    let b_cell_pos = gather_rows(&model.b_cell, &cell_idx_t)?;

    let k = batch.n_negatives;

    // Gather the feature rows scored THIS step (see `gather_feature_rows`). Shared
    // by pos + neg.
    let gather_feat = |idx: &Tensor| gather_feature_rows(model, idx);

    let pos_feat_idx_t = Tensor::from_slice(&batch.fine_feats, b, dev)?;
    let e_feat_pos = gather_feat(&pos_feat_idx_t)?;
    let b_feat_pos = gather_rows(&model.b_feat, &pos_feat_idx_t)?;

    let neg_feat_idx_t = Tensor::from_slice(&batch.neg_feats, b * k, dev)?;
    let e_feat_neg_flat = gather_feat(&neg_feat_idx_t)?;
    let b_feat_neg_flat = gather_rows(&model.b_feat, &neg_feat_idx_t)?;
    let h = e_feat_neg_flat.dim(1)?;
    let e_feat_neg = e_feat_neg_flat.reshape((b, k, h))?;
    let b_feat_neg = b_feat_neg_flat.reshape((b, k))?;

    // Raw bilinear scoring: score = E_feat[f]·E_cell[c] + b_feat (+ b_cell,
    // dropped by the bge driver). The cell row is shared by the pos and neg
    // scores. (Cosine + learnable temperature was tried — commit 12d758a —
    // but regressed cell-type recovery on real data vs the raw dot, so it
    // was removed.)
    let pos_score =
        JointEmbedModel::score_diag(&e_feat_pos, &e_cell_pos, &b_feat_pos, &b_cell_pos)?;
    let neg_score =
        JointEmbedModel::score_negatives(&e_feat_neg, &e_cell_pos, &b_feat_neg, &b_cell_pos)?;

    let nce = |pos: &Tensor, negs: &[Tensor]| match objective {
        NceObjective::Logistic => logistic_nce(pos, negs),
        NceObjective::Softmax => softmax_nce(pos, negs),
    };
    let per_edge = nce(&pos_score, std::slice::from_ref(&neg_score))?;

    // Unweighted mean over the batch's positives (pure count-weighted training:
    // the count weighting lives in the sampler's positive draw, not the loss).
    per_edge.mean(0)
}
