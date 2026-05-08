//! Count-NCE loss: NEG-style binary logistic over positive count
//! triplets vs uniform-random negative features.
//!
//! Per minibatch:
//! 1. Sample `B` positive triplets `(c, f, x)` weighted by count `x`.
//! 2. Map fine indices to coarse via the chosen seed's `FeatureCoarsening`.
//! 3. Sample `K` negative coarse features per positive (uniform random).
//! 4. Pool unique coarse blocks once via `JointEmbedModel::pool_*`.
//! 5. Score positives `[B]` and negatives `[B, K]`.
//! 6. NEG loss: `−log σ(score_pos) − Σ log σ(−score_neg_k)` per edge,
//!    weighted by per-edge NB-Fisher weight.

use crate::embed::data::Triplet;
use crate::embed::model::JointEmbedModel;
use candle_util::candle_core::{Device, Result, Tensor};
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use rand::Rng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use rustc_hash::FxHashMap;

/// One minibatch of positive coarse `(cell, feature)` edges plus
/// per-positive negatives (also at coarse level).
pub struct EdgeBatch {
    /// `[B]` coarse cell index per positive.
    pub coarse_cells: Vec<u32>,
    /// `[B]` coarse feature index per positive.
    pub coarse_feats: Vec<u32>,
    /// `[B * K]` row-major: negatives for positive `b` are at
    /// `[b*K..(b+1)*K]`.
    pub neg_feats: Vec<u32>,
    /// `[B]` per-edge weight (NB-Fisher etc).
    pub edge_weights: Vec<f32>,
    /// `K` negatives per positive.
    pub n_negatives: usize,
}

/// Inputs for `sample_edge_batch`.
pub struct EdgeBatchArgs<'a> {
    pub triplets: &'a [Triplet],
    pub edge_weights: &'a WeightedIndex<f32>,
    pub cell_coarsening: &'a FeatureCoarsening,
    pub feat_coarsening: &'a FeatureCoarsening,
    /// Marginal-weighted negative sampler over coarse feature blocks
    /// (word2vec-style — keeps negatives on the same marginal as
    /// positives so bias terms can't absorb the marginal asymmetry).
    /// One per coarsening seed, since block cardinalities vary.
    pub neg_sampler: &'a WeightedIndex<f32>,
    pub fine_feature_weights: Option<&'a [f32]>,
    pub batch_size: usize,
    pub n_negatives: usize,
}

/// Sample a minibatch of positive edges from the triplet stream and a
/// marginal-weighted set of negative coarse features per positive.
pub fn sample_edge_batch(args: EdgeBatchArgs, rng: &mut impl Rng) -> EdgeBatch {
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut coarse_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let idx = args.edge_weights.sample(rng);
        let t = &args.triplets[idx];
        let c_coarse = args.cell_coarsening.fine_to_coarse[t.cell as usize] as u32;
        let f_coarse = args.feat_coarsening.fine_to_coarse[t.feature as usize] as u32;
        coarse_cells.push(c_coarse);
        coarse_feats.push(f_coarse);
        let w = args
            .fine_feature_weights
            .map(|w| w[t.feature as usize])
            .unwrap_or(1.0);
        weights.push(w);
    }

    let mut neg_feats = Vec::with_capacity(args.batch_size * args.n_negatives);
    for _ in 0..(args.batch_size * args.n_negatives) {
        neg_feats.push(args.neg_sampler.sample(rng) as u32);
    }

    EdgeBatch {
        coarse_cells,
        coarse_feats,
        neg_feats,
        edge_weights: weights,
        n_negatives: args.n_negatives,
    }
}

/// Build a negative-sampler `WeightedIndex` over coarse feature blocks.
/// Weights are coarse-block marginals raised to `α` (word2vec uses
/// 0.75 — flattens the distribution slightly).
pub fn build_negative_sampler(
    triplets: &[Triplet],
    coarsening: &FeatureCoarsening,
    alpha: f32,
) -> WeightedIndex<f32> {
    let mut weights = vec![0f32; coarsening.num_coarse];
    for t in triplets {
        let block = coarsening.fine_to_coarse[t.feature as usize];
        weights[block] += t.count;
    }
    for w in weights.iter_mut() {
        *w = w.max(1e-8).powf(alpha);
    }
    WeightedIndex::new(weights).expect("non-empty coarsening")
}

/// Compute the NEG (binary NCE) loss for the unified `(feature, cell)`
/// relation.
///
/// Returns the per-batch weighted loss as a scalar tensor.
pub fn nce_loss(
    model: &JointEmbedModel,
    batch: &EdgeBatch,
    cell_coarse_to_fine: &[Vec<usize>],
    feat_coarse_to_fine: &[Vec<usize>],
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let k = batch.n_negatives;

    // Unique coarse blocks (cells, features incl. negatives).
    let (unique_cells, cell_pos_idx) = unique_with_index(&batch.coarse_cells);

    // Combined feature blocks (positives first, then flat negatives) so
    // we pool once.
    let mut combined_feats: Vec<u32> =
        Vec::with_capacity(batch.coarse_feats.len() + batch.neg_feats.len());
    combined_feats.extend_from_slice(&batch.coarse_feats);
    combined_feats.extend_from_slice(&batch.neg_feats);
    let (unique_feats, feat_combined_idx) = unique_with_index(&combined_feats);

    // Pool unique blocks via the unified feature axis.
    let (e_cell_u, b_cell_u) = model.pool_cells(&unique_cells, cell_coarse_to_fine, dev)?;
    let (e_feat_u, b_feat_u) =
        model.pool_features(&unique_feats, feat_coarse_to_fine, dev)?;

    // Gather per-position pooled embeddings.
    let cell_idx_t = Tensor::from_vec(cell_pos_idx, b, dev)?;
    let e_cell_pos = e_cell_u.index_select(&cell_idx_t, 0)?;
    let b_cell_pos = b_cell_u.index_select(&cell_idx_t, 0)?;

    let pos_feat_idx_t = Tensor::from_vec(feat_combined_idx[..b].to_vec(), b, dev)?;
    let e_feat_pos = e_feat_u.index_select(&pos_feat_idx_t, 0)?;
    let b_feat_pos = b_feat_u.index_select(&pos_feat_idx_t, 0)?;

    let neg_feat_idx_t = Tensor::from_vec(feat_combined_idx[b..].to_vec(), b * k, dev)?;
    let e_feat_neg_flat = e_feat_u.index_select(&neg_feat_idx_t, 0)?;
    let b_feat_neg_flat = b_feat_u.index_select(&neg_feat_idx_t, 0)?;
    let h = e_feat_neg_flat.dim(1)?;
    let e_feat_neg = e_feat_neg_flat.reshape((b, k, h))?;
    let b_feat_neg = b_feat_neg_flat.reshape((b, k))?;

    let pos_score =
        JointEmbedModel::score_diag(&e_feat_pos, &e_cell_pos, &b_feat_pos, &b_cell_pos)?;
    let neg_score =
        JointEmbedModel::score_negatives(&e_feat_neg, &e_cell_pos, &b_feat_neg, &b_cell_pos)?;

    // NEG loss: -log σ(s_pos) - Σ_k log σ(-s_neg_k)
    let per_edge = (log_sigmoid(&pos_score)? + log_sigmoid(&neg_score.neg()?)?.sum(1)?)?.neg()?;

    // Per-edge weight (NB-Fisher housekeeping downweight). Normalize
    // by Σ w (not B) so the gradient magnitude is preserved when many
    // positives are downweighted housekeeping — otherwise effective
    // gradient is ~mean(w)× weaker and learning crawls.
    let w_sum: f32 = batch.edge_weights.iter().sum::<f32>().max(1e-8);
    let w_t = Tensor::from_vec(batch.edge_weights.clone(), b, dev)?;
    let weighted = (per_edge * w_t)?;
    weighted.sum(0)? / (w_sum as f64)
}

/// Return `(unique_values, index_map)` where `index_map[i]` is the
/// position of `values[i]` in `unique_values`.
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

/// Numerically stable log-sigmoid: `log σ(x) = x - softplus(x)`.
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = log(exp(0) + exp(x)) = log_sum_exp([0, x]) along dim 0.
    let stacked = Tensor::stack(&[x.zeros_like()?, x.clone()], 0)?;
    let softplus = stacked.log_sum_exp(0)?;
    x - softplus
}

/// Build a count-weighted `WeightedIndex` over a triplet stream once;
/// reused across batches for the same epoch.
///
/// If `fisher_weights` is provided (NB-Fisher per-fine-feature),
/// positive sampling probability is `count × fisher_weight`.
/// Housekeeping features (low fisher weight) are sampled less often
/// as positives — this complements the loss-side reweighting and
/// ensures HVG-driven gradient dominates wall-clock learning.
///
/// Currently unused at runtime (the runner builds per-modality
/// samplers via `build_modality_edge_sampler`), kept available for
/// future single-modality use.
#[allow(dead_code)]
pub fn build_edge_sampler(
    triplets: &[Triplet],
    fisher_weights: Option<&[f32]>,
) -> WeightedIndex<f32> {
    let weights: Vec<f32> = triplets
        .iter()
        .map(|t| {
            let w = fisher_weights.map(|fw| fw[t.feature as usize]).unwrap_or(1.0);
            (t.count * w).max(1e-8)
        })
        .collect();
    WeightedIndex::new(weights).expect("non-empty triplet stream")
}

/// Approximate per-cell library size: sum of triplet counts for that cell.
/// Available for diagnostics / opt-in bias init.
#[allow(dead_code)]
pub fn compute_log_libsize(triplets: &[Triplet], n_cells: usize) -> Vec<f32> {
    let mut sizes = vec![0f32; n_cells];
    for t in triplets {
        sizes[t.cell as usize] += t.count;
    }
    for s in sizes.iter_mut() {
        *s = (*s + 1.0).ln();
    }
    sizes
}

/// Per-feature log marginal: sum of triplet counts for that feature.
/// Available for diagnostics / opt-in bias init.
#[allow(dead_code)]
pub fn compute_log_marginal(triplets: &[Triplet], n_features: usize) -> Vec<f32> {
    let mut sums = vec![0f32; n_features];
    for t in triplets {
        sums[t.feature as usize] += t.count;
    }
    for s in sums.iter_mut() {
        *s = (*s + 1.0).ln();
    }
    sums
}
