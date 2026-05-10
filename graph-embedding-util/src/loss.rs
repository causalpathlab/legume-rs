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

////////////////////////////////
// Stratified positive sampler //
////////////////////////////////

/// Two-stage stratified sampler for pseudobulk axes. Stage 1 picks a
/// pb (stratum) by `q(p) ∝ pb_size(p)^alpha_pb`; stage 2 picks a
/// feature within that pb weighted by `μ_pf · fisher(f)`. Compared to
/// flat `WeightedIndex` over all super-edges, this guarantees every pb
/// gets training coverage proportional to `q(p)` (uniform when
/// `alpha_pb = 0`, count-proportional when `alpha_pb = 1`), instead of
/// being dominated by housekeeping-gene super-edges.
///
/// Negatives come from a single global pb-level feature marginal
/// (count^`alpha_neg`), since `pb_unified` collapses all pseudobulks
/// into one synthetic "all" batch.
pub struct StratifiedSampler {
    /// Picks a local pb index into `active_pbs`. Weights = `q(p)`.
    pub pb_picker: WeightedIndex<f32>,
    /// Global pb ids that have ≥ 1 expressed feature, in stable order.
    pub active_pbs: Vec<u32>,
    /// Per-active-pb feature sampler; aligned with `active_pbs`.
    pub per_pb: Vec<PbFeatureSampler>,
    /// Negative pool: features with any nonzero pb-level count.
    pub neg: WeightedIndex<f32>,
    pub feature_pool: Vec<u32>,
}

pub struct PbFeatureSampler {
    /// Global feature ids expressed in this pb.
    pub features: Vec<u32>,
    /// `WeightedIndex` over `features`; weights = `μ_pf · fisher(f)`.
    pub picker: WeightedIndex<f32>,
}

/// Build a stratified sampler for a pseudobulk axis. Returns `None`
/// when the axis has zero positives or fewer than two active pb's
/// (degenerate stratum).
pub fn build_stratified_sampler(
    triplets: &[Triplet],
    n_pb: usize,
    n_features: usize,
    fisher_weights: &[f32],
    alpha_pb: f32,
    alpha_neg: f32,
) -> Option<StratifiedSampler> {
    if triplets.is_empty() {
        return None;
    }

    // Bucket triplets by pb; accumulate per-pb size and global feature marginal.
    let mut per_pb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_pb];
    let mut pb_size = vec![0f32; n_pb];
    let mut feat_count = vec![0f32; n_features];
    for t in triplets {
        per_pb[t.cell as usize].push((t.feature, t.count));
        pb_size[t.cell as usize] += t.count;
        feat_count[t.feature as usize] += t.count;
    }

    let mut active_pbs: Vec<u32> = Vec::new();
    let mut per_pb_samplers: Vec<PbFeatureSampler> = Vec::new();
    let mut pb_q: Vec<f32> = Vec::new();
    for (p, edges) in per_pb.into_iter().enumerate() {
        if edges.is_empty() {
            continue;
        }
        let features: Vec<u32> = edges.iter().map(|&(f, _)| f).collect();
        let weights: Vec<f32> = edges
            .iter()
            .map(|&(f, c)| (c * fisher_weights[f as usize]).max(1e-8))
            .collect();
        let picker = WeightedIndex::new(weights).expect("non-empty pb feature weights");
        per_pb_samplers.push(PbFeatureSampler { features, picker });
        active_pbs.push(p as u32);
        pb_q.push(pb_size[p].max(1e-8).powf(alpha_pb));
    }

    if active_pbs.is_empty() {
        return None;
    }
    let pb_picker = WeightedIndex::new(pb_q).expect("non-empty pb weights");

    let feature_pool: Vec<u32> = (0..n_features as u32)
        .filter(|&f| feat_count[f as usize] > 0.0)
        .collect();
    if feature_pool.is_empty() {
        return None;
    }
    let neg_w: Vec<f32> = feature_pool
        .iter()
        .map(|&f| feat_count[f as usize].powf(alpha_neg))
        .collect();
    let neg = WeightedIndex::new(neg_w).expect("non-empty negative pool");

    Some(StratifiedSampler {
        pb_picker,
        active_pbs,
        per_pb: per_pb_samplers,
        neg,
        feature_pool,
    })
}

pub struct StratifiedEdgeBatchArgs<'a> {
    pub sampler: &'a StratifiedSampler,
    pub fine_feature_weights: &'a [f32],
    pub batch_size: usize,
    pub n_negatives: usize,
}

/// Two-stage draw: pick pb by `q(p)`, then feature within pb. Output
/// `EdgeBatch` is interchangeable with [`sample_edge_batch`]'s — the
/// downstream NCE loss doesn't care how the batch was sampled.
pub fn sample_stratified_edge_batch(
    args: StratifiedEdgeBatchArgs,
    rng: &mut impl Rng,
) -> EdgeBatch {
    let s = args.sampler;
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let local_pb = s.pb_picker.sample(rng);
        let p = s.active_pbs[local_pb];
        let pf = &s.per_pb[local_pb];
        let local_f = pf.picker.sample(rng);
        let f = pf.features[local_f];
        coarse_cells.push(p);
        fine_feats.push(f);
        weights.push(args.fine_feature_weights[f as usize]);
    }

    let mut neg_feats = Vec::with_capacity(args.batch_size * args.n_negatives);
    for _ in 0..(args.batch_size * args.n_negatives) {
        let local = s.neg.sample(rng);
        neg_feats.push(s.feature_pool[local]);
    }

    EdgeBatch {
        coarse_cells,
        fine_feats,
        neg_feats,
        edge_weights: weights,
        n_negatives: args.n_negatives,
    }
}

////////////////////////////////
// Nested chain sampler (MVP) //
////////////////////////////////

/// Bottom-up nested chain sampler. Each chain starts from a real
/// `(cell, feature)` super-edge drawn by the existing per-batch
/// sampler, then walks **up** the pb-tree via stored parent maps to
/// derive a coordinated `pb_path` across every coarser level. The
/// resulting chain produces:
///
/// - one positive `(leaf_cell, leaf_feature)` for the cell axis;
/// - one positive `(P_k, leaf_feature)` for each pb level k via
///   `P_k = ancestor of leaf_cell at level k`;
/// - shared feature negatives drawn from the cell axis's batch pool.
///
/// All axes therefore share the same positive feature (and negatives)
/// per chain. The chain's per-axis cell-side index is the cell itself
/// (cell axis) or the cell's pb at that level (pb axes). One backward
/// step over the summed loss updates `E_feat` once with coherent
/// gradient contributions from every level — that's the variance-
/// reduction win versus independently sampling each axis.
///
/// MVP scope: single-level supergene (no gene-tree yet); pb-tree only.
/// Hard negatives via siblings are a future extension.
pub struct ChainSampler<'a> {
    pub batch_sampler: &'a PerBatchSampler,
    /// `cell_to_pb_per_level[k][cell] = pb id at level k`. Finest-first
    /// (matches the gbe convention after `collapsed_levels.reverse()`),
    /// so `pb_path[level=0]` is coarsest and `pb_path.last()` is finest.
    pub cell_to_pb_per_level: &'a [Vec<usize>],
}

pub struct ChainBatch {
    /// Length B; one leaf cell per chain. Pb-tree ancestors at each
    /// level are derived by indexing `cell_to_pb_per_level[level][cell]`
    /// at the call site — not stored on the batch to avoid the per-step
    /// `Vec<Vec<u32>>` allocation.
    pub leaf_cells: Vec<u32>,
    /// Shared feature side: one positive + K negatives per chain.
    /// Lifted into its own struct so [`nce_loss_chain`] can take it by
    /// value while the caller still borrows `leaf_cells` for the
    /// cell-axis [`ChainAxis`] entry.
    pub feats: ChainFeatureSide,
}

pub struct ChainFeatureSide {
    /// Length B; one positive feature per chain (shared across all axes).
    pub fine_feats: Vec<u32>,
    /// Length B*K row-major; shared negatives across all axes.
    pub neg_feats: Vec<u32>,
    pub edge_weights: Vec<f32>,
    pub n_negatives: usize,
}

pub struct ChainBatchArgs<'a> {
    pub triplets: &'a [Triplet],
    pub sampler: &'a ChainSampler<'a>,
    pub fine_feature_weights: &'a [f32],
    pub batch_size: usize,
    pub n_negatives: usize,
}

pub fn sample_chain_batch(args: ChainBatchArgs, rng: &mut impl Rng) -> ChainBatch {
    let bs = args.sampler.batch_sampler;
    let mut leaf_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let local_idx = bs.pos.sample(rng);
        let global_idx = bs.triplet_indices[local_idx] as usize;
        let t = &args.triplets[global_idx];
        leaf_cells.push(t.cell);
        fine_feats.push(t.feature);
        weights.push(args.fine_feature_weights[t.feature as usize]);
    }

    let mut neg_feats = Vec::with_capacity(args.batch_size * args.n_negatives);
    for _ in 0..(args.batch_size * args.n_negatives) {
        let local = bs.neg.sample(rng);
        neg_feats.push(bs.feature_pool[local]);
    }

    ChainBatch {
        leaf_cells,
        feats: ChainFeatureSide {
            fine_feats,
            neg_feats,
            edge_weights: weights,
            n_negatives: args.n_negatives,
        },
    }
}

/// One axis's cell-side resolution for chain scoring. The chain loss
/// gathers `e_cell.index_select(&indices)` and scores against the
/// shared (already-gathered) feature side. `indices` is a pre-built
/// `[B] u32` index tensor — building it in the caller (once per axis
/// per step) avoids a second `Vec<u32> → Tensor` round-trip inside the
/// loss.
pub struct ChainAxis<'a> {
    pub e_cell: &'a Tensor,
    pub b_cell: &'a Tensor,
    pub indices: &'a Tensor,
    pub lambda: f32,
    /// Used in `nce_loss_chain`'s error diagnostics.
    pub label: &'a str,
}

/// Score one chain batch across every axis using a single shared
/// positive/negative feature gather. `e_feat` / `b_feat` are
/// **shared** across axes (same Var); each axis differs only in its
/// cell-side embeddings table and the per-chain indices into it.
///
/// Returns the λ-weighted sum across axes. The per-edge weight uses
/// the cell-axis NCE's `Σw` normalization (Fisher-info-aware) so the
/// gradient magnitude is consistent with the existing per-axis losses.
pub fn nce_loss_chain(
    e_feat: &Tensor,
    b_feat: &Tensor,
    feats: ChainFeatureSide,
    axes: &[ChainAxis],
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let b = feats.fine_feats.len();
    if b == 0 || axes.is_empty() {
        return Ok(Tensor::zeros(
            (),
            candle_util::candle_core::DType::F32,
            dev,
        )?);
    }
    let k = feats.n_negatives;

    // Feature side gathered once for the whole chain step.
    let pos_feat_idx = Tensor::from_vec(feats.fine_feats, b, dev)?;
    let e_feat_pos = select_feat_emb(smoother, e_feat, &pos_feat_idx)?;
    let b_feat_pos = b_feat.index_select(&pos_feat_idx, 0)?;

    let neg_feat_idx = Tensor::from_vec(feats.neg_feats, b * k, dev)?;
    let e_feat_neg_flat = select_feat_emb(smoother, e_feat, &neg_feat_idx)?;
    let b_feat_neg_flat = b_feat.index_select(&neg_feat_idx, 0)?;
    let h = e_feat_neg_flat.dim(1)?;
    let e_feat_neg = e_feat_neg_flat.reshape((b, k, h))?;
    let b_feat_neg = b_feat_neg_flat.reshape((b, k))?;

    let w_sum: f32 = feats.edge_weights.iter().sum::<f32>().max(1e-8);
    let w_t = Tensor::from_vec(feats.edge_weights, b, dev)?;

    let mut total: Option<Tensor> = None;
    for axis in axes {
        let n = axis.indices.dims1()?;
        anyhow::ensure!(
            n == b,
            "chain axis {} indices ({n}) != batch_size ({b})",
            axis.label,
        );
        let e_cell_pos = axis.e_cell.index_select(axis.indices, 0)?;
        let b_cell_pos = axis.b_cell.index_select(axis.indices, 0)?;

        let pos_score =
            JointEmbedModel::score_diag(&e_feat_pos, &e_cell_pos, &b_feat_pos, &b_cell_pos)?;
        let neg_score =
            JointEmbedModel::score_negatives(&e_feat_neg, &e_cell_pos, &b_feat_neg, &b_cell_pos)?;
        let per_edge =
            (log_sigmoid(&pos_score)? + log_sigmoid(&neg_score.neg()?)?.sum(1)?)?.neg()?;
        let weighted = (per_edge * w_t.clone())?;
        let axis_loss = (weighted.sum(0)? / (w_sum as f64))?;
        let scaled = (axis_loss * axis.lambda as f64)?;
        total = Some(match total {
            Some(prev) => (prev + scaled)?,
            None => scaled,
        });
    }
    Ok(total.unwrap())
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
    let (unique_cells, cell_pos_idx) = unique_with_index(&batch.coarse_cells);
    let (e_cell_u, b_cell_u) = model.pool_cells(&unique_cells, cell_coarse_to_fine, dev)?;

    let cell_idx_t = Tensor::from_vec(cell_pos_idx, b, dev)?;
    let e_cell_pos = e_cell_u.index_select(&cell_idx_t, 0)?;
    let b_cell_pos = b_cell_u.index_select(&cell_idx_t, 0)?;

    nce_loss_with_cell_side(model, batch, e_cell_pos, b_cell_pos, smoother, dev)
}

/// Fast path for the identity-coarsening case (every "super-cell" is
/// its own row). Skips `unique_with_index`, `pool_cells`, and the
/// scatter-add — a single `index_select` directly off `model.e_cell` /
/// `model.b_cell` is mathematically equivalent because each block has
/// exactly one fine child and `mean([x]) == x`. Composite training
/// hits this path on every axis (cell axis + every pseudobulk level
/// use [`crate::coarsen::identity_axis`]).
pub fn nce_loss_identity(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let cell_idx_t = Tensor::from_vec(batch.coarse_cells.clone(), b, dev)?;
    let e_cell_pos = model.e_cell.index_select(&cell_idx_t, 0)?;
    let b_cell_pos = model.b_cell.index_select(&cell_idx_t, 0)?;
    nce_loss_with_cell_side(model, batch, e_cell_pos, b_cell_pos, smoother, dev)
}

/// Shared tail of [`nce_loss`] / [`nce_loss_identity`]: feature-side
/// gathers, bilinear scoring, count-weighted log-σ aggregation. The
/// cell-side embeddings come pre-resolved (pooled or directly
/// gathered) so we don't pay the path-selection cost twice.
fn nce_loss_with_cell_side(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    e_cell_pos: Tensor,
    b_cell_pos: Tensor,
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    let k = batch.n_negatives;

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

// =================================================================
// Cell-cell NCE — symmetric bilinear loss on cell pairs from a
// caller-provided graph (e.g. spatial KNN). Negatives drawn within
// the positive pair's batch.
// =================================================================

pub struct CellEdgeBatch {
    pub left_cells: Vec<u32>,  // [B] fine cell ids
    pub right_cells: Vec<u32>, // [B]
    /// `[B*K]` row-major: negatives for positive `b` are at `[b*K..(b+1)*K]`.
    pub neg_cells: Vec<u32>,
    pub n_negatives: usize,
}

pub struct PerBatchCellSampler {
    pub pos: WeightedIndex<f32>,
    pub neg: WeightedIndex<f32>,
    /// Indices into the global cell-cell `edges` slice for this batch's
    /// retained (within-batch) positives.
    pub edge_indices: Vec<u32>,
    /// Global cell ids that constitute this batch's negative pool
    /// (every cell in this batch).
    pub cell_pool: Vec<u32>,
}

pub struct CellEdgeBatchArgs<'a> {
    pub edges: &'a [(u32, u32)],
    pub batch_sampler: &'a PerBatchCellSampler,
    pub batch_size: usize,
    pub n_negatives: usize,
}

pub fn sample_cell_edge_batch(args: CellEdgeBatchArgs, rng: &mut impl Rng) -> CellEdgeBatch {
    let mut left_cells = Vec::with_capacity(args.batch_size);
    let mut right_cells = Vec::with_capacity(args.batch_size);

    let s = args.batch_sampler;
    for _ in 0..args.batch_size {
        let local = s.pos.sample(rng);
        let global = s.edge_indices[local] as usize;
        let (i, j) = args.edges[global];
        left_cells.push(i);
        right_cells.push(j);
    }

    let mut neg_cells = Vec::with_capacity(args.batch_size * args.n_negatives);
    for _ in 0..(args.batch_size * args.n_negatives) {
        let local = s.neg.sample(rng);
        neg_cells.push(s.cell_pool[local]);
    }

    CellEdgeBatch {
        left_cells,
        right_cells,
        neg_cells,
        n_negatives: args.n_negatives,
    }
}

/// Build a per-batch cell-pair sampler. Cross-batch edges (endpoints in
/// different batches) are dropped — negatives drawn within-batch make
/// no sense for them. Returns one entry per batch; `None` for batches
/// that retained no within-batch edges.
///
/// `cell_degrees` is the per-cell within-batch positive degree (counts
/// retained edges where the cell appears as either endpoint), used to
/// build the count^α negative distribution analogous to the bipartite
/// sampler's count^α over feature marginals.
pub fn build_per_batch_cell_samplers(
    edges: &[(u32, u32)],
    batch_membership: &[u32],
    n_batches: usize,
    n_cells: usize,
    alpha_neg: f32,
) -> (Vec<Option<PerBatchCellSampler>>, usize) {
    // First pass: bucket retained edges by batch + accumulate per-cell degree
    // (within retained edges) for the negative weight.
    let mut per_batch_edge_indices: Vec<Vec<u32>> = vec![Vec::new(); n_batches];
    let mut degree: Vec<f32> = vec![0.0; n_cells];
    let mut cross_batch = 0usize;

    for (i, &(u, v)) in edges.iter().enumerate() {
        let bu = batch_membership[u as usize];
        let bv = batch_membership[v as usize];
        if bu != bv {
            cross_batch += 1;
            continue;
        }
        per_batch_edge_indices[bu as usize].push(i as u32);
        degree[u as usize] += 1.0;
        degree[v as usize] += 1.0;
    }

    let samplers = per_batch_edge_indices
        .into_par_iter()
        .enumerate()
        .map(|(b, edge_indices)| {
            if edge_indices.is_empty() {
                return None;
            }

            // Uniform positive sampling — every retained edge equally likely.
            // (Edge weights could be learned per-edge later; today they're 1.0.)
            let pos_w: Vec<f32> = vec![1.0; edge_indices.len()];
            let pos = WeightedIndex::new(pos_w).expect("non-empty edge list");

            // Negative pool = every cell in this batch, weighted by retained
            // within-batch degree^α (cells with no edges still get a small
            // weight via the .max(1e-8) floor so rare cells aren't excluded).
            let cell_pool: Vec<u32> = batch_membership
                .iter()
                .enumerate()
                .filter_map(|(c, &bb)| (bb as usize == b).then_some(c as u32))
                .collect();
            let neg_w: Vec<f32> = cell_pool
                .iter()
                .map(|&c| degree[c as usize].max(1e-8).powf(alpha_neg))
                .collect();
            let neg = WeightedIndex::new(neg_w).expect("non-empty cell pool");

            Some(PerBatchCellSampler {
                pos,
                neg,
                edge_indices,
                cell_pool,
            })
        })
        .collect();

    (samplers, cross_batch)
}

/// Symmetric bilinear NCE on cell pairs. Reuses `score_diag` /
/// `score_negatives` — those functions are algebraically type-agnostic;
/// here both arguments come from the cell embedding table. Scoring is
/// on **fine-resolution** embeddings (no `pool_cells`).
pub fn cell_cell_nce_loss(
    model: &JointEmbedModel,
    batch: CellEdgeBatch,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.left_cells.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let k = batch.n_negatives;

    let left_idx = Tensor::from_vec(batch.left_cells, b, dev)?;
    let right_idx = Tensor::from_vec(batch.right_cells, b, dev)?;
    let neg_idx = Tensor::from_vec(batch.neg_cells, b * k, dev)?;

    let e_left = model.e_cell.index_select(&left_idx, 0)?;
    let b_left = model.b_cell.index_select(&left_idx, 0)?;
    let e_right = model.e_cell.index_select(&right_idx, 0)?;
    let b_right = model.b_cell.index_select(&right_idx, 0)?;

    let e_neg_flat = model.e_cell.index_select(&neg_idx, 0)?;
    let b_neg_flat = model.b_cell.index_select(&neg_idx, 0)?;
    let h = e_neg_flat.dim(1)?;
    // For each positive `b`, the K negatives replace the *right* endpoint —
    // we score (left vs neg right) so the model learns to discriminate true
    // neighbours from random within-batch cells, not vice-versa. Symmetry of
    // the bilinear form means scoring the other side is redundant.
    let e_neg = e_neg_flat.reshape((b, k, h))?;
    let b_neg = b_neg_flat.reshape((b, k))?;

    let pos_score = JointEmbedModel::score_diag(&e_left, &e_right, &b_left, &b_right)?;
    let neg_score = JointEmbedModel::score_negatives(&e_neg, &e_left, &b_neg, &b_left)?;

    let per_edge = (log_sigmoid(&pos_score)? + log_sigmoid(&neg_score.neg()?)?.sum(1)?)?.neg()?;
    per_edge.mean(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_cell_sampler_skips_cross_batch_edges() {
        // 4 cells, 2 batches. Edges: (0,1) within batch 0, (2,3) within
        // batch 1, (1,2) cross-batch.
        let edges = vec![(0u32, 1), (2, 3), (1, 2)];
        let batch_membership = vec![0u32, 0, 1, 1];
        let (samplers, cross_batch) =
            build_per_batch_cell_samplers(&edges, &batch_membership, 2, 4, 0.75);

        assert_eq!(cross_batch, 1, "expected one cross-batch edge dropped");
        let s0 = samplers[0]
            .as_ref()
            .expect("batch 0 has within-batch edges");
        let s1 = samplers[1]
            .as_ref()
            .expect("batch 1 has within-batch edges");
        assert_eq!(s0.edge_indices, vec![0]);
        assert_eq!(s1.edge_indices, vec![1]);
        assert_eq!(s0.cell_pool, vec![0, 1]);
        assert_eq!(s1.cell_pool, vec![2, 3]);
    }

    #[test]
    fn cell_cell_sampler_empty_batch_returns_none() {
        let edges = vec![(0u32, 1)];
        let batch_membership = vec![0u32, 0, 1, 1];
        let (samplers, cross_batch) =
            build_per_batch_cell_samplers(&edges, &batch_membership, 2, 4, 0.75);
        assert_eq!(cross_batch, 0);
        assert!(samplers[0].is_some());
        assert!(samplers[1].is_none(), "batch 1 has no edges → None");
    }
}
