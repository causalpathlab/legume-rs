//! Count-NCE loss: NEG-style binary logistic over count-weighted
//! positive (cell, feature) edges vs same-file marginal^α negatives.

use crate::embed::data::Triplet;
use crate::embed::model::JointEmbedModel;
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

pub struct PerFileSampler {
    pub pos: WeightedIndex<f32>,
    pub neg: WeightedIndex<f32>,
    pub triplet_start: usize,
    pub feature_start: u32,
}

pub struct EdgeBatchArgs<'a> {
    pub triplets: &'a [Triplet],
    pub file_sampler: &'a PerFileSampler,
    pub cell_coarsening: &'a FeatureCoarsening,
    pub fine_feature_weights: Option<&'a [f32]>,
    pub batch_size: usize,
    pub n_negatives: usize,
}

pub fn sample_edge_batch(args: EdgeBatchArgs, rng: &mut impl Rng) -> EdgeBatch {
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    let trip_off = args.file_sampler.triplet_start;
    let feat_off = args.file_sampler.feature_start;

    for _ in 0..args.batch_size {
        let local_idx = args.file_sampler.pos.sample(rng);
        let t = &args.triplets[trip_off + local_idx];
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
        let local = args.file_sampler.neg.sample(rng) as u32;
        neg_feats.push(local + feat_off);
    }

    EdgeBatch {
        coarse_cells,
        fine_feats,
        neg_feats,
        edge_weights: weights,
        n_negatives: args.n_negatives,
    }
}

pub fn build_per_file_samplers(
    triplets: &[Triplet],
    file_triplet_ranges: &[std::ops::Range<usize>],
    file_feature_ranges: &[std::ops::Range<u32>],
    fisher_weights: &[f32],
    alpha_neg: f32,
) -> Vec<PerFileSampler> {
    assert_eq!(file_triplet_ranges.len(), file_feature_ranges.len());

    file_triplet_ranges
        .par_iter()
        .zip(file_feature_ranges.par_iter())
        .map(|(trip_range, feat_range)| {
            let pos_w: Vec<f32> = triplets[trip_range.clone()]
                .iter()
                .map(|t| {
                    let w = fisher_weights[t.feature as usize];
                    (t.count * w).max(1e-8)
                })
                .collect();
            let pos = WeightedIndex::new(pos_w).expect("non-empty file triplet stream");

            let n_local = (feat_range.end - feat_range.start) as usize;
            let mut neg_w = vec![0f32; n_local];
            for t in &triplets[trip_range.clone()] {
                let local = (t.feature - feat_range.start) as usize;
                neg_w[local] += t.count;
            }
            for w in neg_w.iter_mut() {
                *w = w.max(1e-8).powf(alpha_neg);
            }
            let neg = WeightedIndex::new(neg_w).expect("non-empty file feature axis");

            PerFileSampler {
                pos,
                neg,
                triplet_start: trip_range.start,
                feature_start: feat_range.start,
            }
        })
        .collect()
}

pub fn nce_loss(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    cell_coarse_to_fine: &[Vec<usize>],
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
    let e_feat_pos = model.e_feat.index_select(&pos_feat_idx_t, 0)?;
    let b_feat_pos = model.b_feat.index_select(&pos_feat_idx_t, 0)?;

    let neg_feat_idx_t = Tensor::from_vec(batch.neg_feats, b * k, dev)?;
    let e_feat_neg_flat = model.e_feat.index_select(&neg_feat_idx_t, 0)?;
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
