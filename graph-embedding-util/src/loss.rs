//! Count-NCE loss: NEG-style binary logistic over count-weighted
//! positive (cell, feature) edges vs within-batch marginal^α negatives.
//!
//! Negatives are drawn from features observed *in the positive cell's
//! batch*, so the model can't earn signal by separating cells along
//! technical-batch confounders — features that distinguish batches are
//! also exactly the candidate negatives for cells in those batches.

use crate::data::Triplet;
use crate::feature_network::{select_feat_emb, FeatureNetworkSmoother};
use crate::model::JointEmbedModel;
use candle_util::candle_core::{Device, Result, Tensor};
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use rand::Rng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

pub struct EdgeBatch {
    pub coarse_cells: Vec<u32>,
    pub fine_feats: Vec<u32>,
    /// `[B*K]` row-major: negatives for positive `b` are at `[b*K..(b+1)*K]`.
    pub neg_feats: Vec<u32>,
    pub edge_weights: Vec<f32>,
    pub n_negatives: usize,
}

pub struct PerBatchSampler {
    pub pos: WeightedIndex<f32>,
    pub neg: WeightedIndex<f32>,
    /// Indices into the global `triplets` slice for this batch's positives.
    pub triplet_indices: Vec<u32>,
    /// Global feature ids that constitute this batch's negative pool
    /// (features observed in any cell of this batch).
    pub feature_pool: Vec<u32>,
}

pub struct EdgeBatchArgs<'a> {
    pub triplets: &'a [Triplet],
    pub batch_sampler: &'a PerBatchSampler,
    pub cell_coarsening: &'a FeatureCoarsening,
    pub fine_feature_weights: Option<&'a [f32]>,
    pub batch_size: usize,
    pub n_negatives: usize,
}

pub fn sample_edge_batch(args: EdgeBatchArgs, rng: &mut impl Rng) -> EdgeBatch {
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    let sampler = args.batch_sampler;

    for _ in 0..args.batch_size {
        let local_idx = sampler.pos.sample(rng);
        let global_idx = sampler.triplet_indices[local_idx] as usize;
        let t = &args.triplets[global_idx];
        let c_coarse = args.cell_coarsening.fine_to_coarse[t.cell as usize] as u32;
        coarse_cells.push(c_coarse);
        fine_feats.push(t.feature);
        let w = args
            .fine_feature_weights
            .map(|w| w[t.feature as usize])
            .unwrap_or(1.0);
        weights.push(w);
    }

    let mut neg_feats = Vec::with_capacity(args.batch_size * args.n_negatives);
    for _ in 0..(args.batch_size * args.n_negatives) {
        let local = sampler.neg.sample(rng);
        neg_feats.push(sampler.feature_pool[local]);
    }

    EdgeBatch {
        coarse_cells,
        fine_feats,
        neg_feats,
        edge_weights: weights,
        n_negatives: args.n_negatives,
    }
}

/// Build a per-batch sampler. Each batch contributes the positive triplets
/// whose cells belong to that batch, and a negative pool restricted to the
/// features observed in those cells. Batches with zero observed edges are
/// returned as `None` (caller filters).
pub fn build_per_batch_samplers(
    triplets: &[Triplet],
    batch_membership: &[u32],
    n_batches: usize,
    n_features: usize,
    fisher_weights: &[f32],
    alpha_neg: f32,
) -> Vec<Option<PerBatchSampler>> {
    let mut per_batch_indices: Vec<Vec<u32>> = vec![Vec::new(); n_batches];
    for (i, t) in triplets.iter().enumerate() {
        let b = batch_membership[t.cell as usize] as usize;
        per_batch_indices[b].push(i as u32);
    }

    per_batch_indices
        .into_par_iter()
        .map(|trip_indices| {
            if trip_indices.is_empty() {
                return None;
            }

            let pos_w: Vec<f32> = trip_indices
                .iter()
                .map(|&i| {
                    let t = &triplets[i as usize];
                    let w = fisher_weights[t.feature as usize];
                    (t.count * w).max(1e-8)
                })
                .collect();
            let pos = WeightedIndex::new(pos_w).expect("non-empty batch positives");

            // Per-batch feature marginal (count-weighted), then α-smoothed.
            // Dense scratch is cheaper than a HashMap at our feature counts.
            let mut feat_count = vec![0f32; n_features];
            for &i in &trip_indices {
                let t = &triplets[i as usize];
                feat_count[t.feature as usize] += t.count;
            }
            let feature_pool: Vec<u32> = (0..n_features as u32)
                .filter(|&f| feat_count[f as usize] > 0.0)
                .collect();
            let neg_w: Vec<f32> = feature_pool
                .iter()
                .map(|&f| feat_count[f as usize].powf(alpha_neg))
                .collect();
            let neg = WeightedIndex::new(neg_w).expect("non-empty batch feature pool");

            Some(PerBatchSampler {
                pos,
                neg,
                triplet_indices: trip_indices,
                feature_pool,
            })
        })
        .collect()
}

pub fn nce_loss(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    cell_coarse_to_fine: &[Vec<usize>],
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let k = batch.n_negatives;

    let (unique_cells, cell_pos_idx) = unique_with_index(&batch.coarse_cells);
    let (e_cell_u, b_cell_u) = model.pool_cells(&unique_cells, cell_coarse_to_fine, dev)?;

    let cell_idx_t = Tensor::from_vec(cell_pos_idx, b, dev)?;
    let e_cell_pos = e_cell_u.index_select(&cell_idx_t, 0)?;
    let b_cell_pos = b_cell_u.index_select(&cell_idx_t, 0)?;

    let pos_feat_idx_t = Tensor::from_vec(batch.fine_feats, b, dev)?;
    let e_feat_pos = select_feat_emb(smoother, &model.e_feat, &pos_feat_idx_t)?;
    let b_feat_pos = model.b_feat.index_select(&pos_feat_idx_t, 0)?;

    let neg_feat_idx_t = Tensor::from_vec(batch.neg_feats, b * k, dev)?;
    let e_feat_neg_flat = select_feat_emb(smoother, &model.e_feat, &neg_feat_idx_t)?;
    let b_feat_neg_flat = model.b_feat.index_select(&neg_feat_idx_t, 0)?;
    let h = e_feat_neg_flat.dim(1)?;
    let e_feat_neg = e_feat_neg_flat.reshape((b, k, h))?;
    let b_feat_neg = b_feat_neg_flat.reshape((b, k))?;

    let pos_score =
        JointEmbedModel::score_diag(&e_feat_pos, &e_cell_pos, &b_feat_pos, &b_cell_pos)?;
    let neg_score =
        JointEmbedModel::score_negatives(&e_feat_neg, &e_cell_pos, &b_feat_neg, &b_cell_pos)?;

    let per_edge = (log_sigmoid(&pos_score)? + log_sigmoid(&neg_score.neg()?)?.sum(1)?)?.neg()?;

    // Normalize by Σw, not B: when most positives are housekeeping and
    // get downweighted, dividing by B leaves an O(mean(w)) gradient
    // attenuation that stalls learning.
    let w_sum: f32 = batch.edge_weights.iter().sum::<f32>().max(1e-8);
    let w_t = Tensor::from_vec(batch.edge_weights, b, dev)?;
    let weighted = (per_edge * w_t)?;
    weighted.sum(0)? / (w_sum as f64)
}

fn unique_with_index(values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut seen: FxHashMap<u32, u32> = FxHashMap::default();
    let mut unique = Vec::new();
    let mut idx_map = Vec::with_capacity(values.len());
    for &v in values {
        let id = *seen.entry(v).or_insert_with(|| {
            let id = unique.len() as u32;
            unique.push(v);
            id
        });
        idx_map.push(id);
    }
    (unique, idx_map)
}

// Numerically stable: log σ(x) = x - log_sum_exp([0, x]).
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    let stacked = Tensor::stack(&[x.zeros_like()?, x.clone()], 0)?;
    let softplus = stacked.log_sum_exp(0)?;
    x - softplus
}
