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
use crate::progress::new_progress_bar;
use candle_util::candle_core::{Device, Result, Tensor};
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use indicatif::ParallelProgressIterator;
use rand::{Rng, RngExt};
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

///////////////////////////////////////////////////
// Per-batch stratified cell sampler (cell axis) //
///////////////////////////////////////////////////

/// Two-stage per-batch sampler for the cell axis. Stage 1 picks a cell
/// (within this batch) with `q(c) ∝ degree(c)^alpha_cell`; stage 2
/// picks a feature within that cell weighted by `count · fisher(f)`.
/// Mirrors [`StratifiedSampler`] but the outer stratum is fine cells
/// (not pseudobulks), and one sampler exists per batch. Negatives use
/// the same per-batch feature marginal (`count^alpha_neg`) as
/// [`PerBatchSampler`]. With `alpha_cell = 1`, this is approximately
/// equivalent to the flat sampler; with `alpha_cell = 0`, every cell
/// in the batch gets uniform coverage regardless of sequencing depth.
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
}

pub struct CellFeatureSampler {
    /// Global feature ids expressed in this cell.
    pub features: Vec<u32>,
    /// `WeightedIndex` over `features`; weights = `count · fisher(f)`.
    pub picker: WeightedIndex<f32>,
}

/// Build one stratified cell sampler per batch. Returns one entry per
/// original batch id (`None` for batches with zero positives). Caller
/// filters to the active subset before wiring into `AxisSampler`.
pub fn build_per_batch_stratified_cell_samplers(
    triplets: &[Triplet],
    batch_membership: &[u32],
    n_batches: usize,
    n_features: usize,
    fisher_weights: &[f32],
    alpha_cell: f32,
    alpha_neg: f32,
) -> Vec<Option<PerBatchStratifiedCellSampler>> {
    let bucket_bar = new_progress_bar(triplets.len() as u64);
    bucket_bar.set_message("bucketing triplets by batch (strat-cell)");
    let per_batch_indices: Vec<Vec<u32>> = triplets
        .par_iter()
        .enumerate()
        .progress_with(bucket_bar.clone())
        .fold(
            || vec![Vec::new(); n_batches],
            |mut acc, (i, t)| {
                let b = batch_membership[t.cell as usize] as usize;
                acc[b].push(i as u32);
                acc
            },
        )
        .reduce(
            || vec![Vec::new(); n_batches],
            |mut a, b| {
                for (av, bv) in a.iter_mut().zip(b.into_iter()) {
                    av.extend(bv);
                }
                a
            },
        );
    bucket_bar.finish_and_clear();

    let build_bar = new_progress_bar(n_batches as u64);
    build_bar.set_message("per-batch strat-cell sampler build");
    let samplers = per_batch_indices
        .into_par_iter()
        .progress_with(build_bar.clone())
        .map(|trip_indices| {
            if trip_indices.is_empty() {
                return None;
            }

            // Bucket triplets in this batch by cell. FxHashMap keyed on
            // global cell id; values are (feature, count) lists. Dense
            // n_cells scratch is wasteful here because each batch only
            // touches a fraction of the global cell axis.
            let mut per_cell_map: FxHashMap<u32, Vec<(u32, f32)>> = FxHashMap::default();
            let mut cell_degree: FxHashMap<u32, f32> = FxHashMap::default();
            let mut feat_count = vec![0f32; n_features];
            for &i in &trip_indices {
                let t = &triplets[i as usize];
                per_cell_map
                    .entry(t.cell)
                    .or_default()
                    .push((t.feature, t.count));
                *cell_degree.entry(t.cell).or_insert(0.0) += t.count;
                feat_count[t.feature as usize] += t.count;
            }

            let mut active_cells: Vec<u32> = per_cell_map.keys().copied().collect();
            active_cells.sort_unstable();

            let mut per_cell: Vec<CellFeatureSampler> = Vec::with_capacity(active_cells.len());
            let mut cell_w: Vec<f32> = Vec::with_capacity(active_cells.len());
            for &c in &active_cells {
                let edges = &per_cell_map[&c];
                let features: Vec<u32> = edges.iter().map(|&(f, _)| f).collect();
                let weights: Vec<f32> = edges
                    .iter()
                    .map(|&(f, cnt)| (cnt * fisher_weights[f as usize]).max(1e-8))
                    .collect();
                let picker = WeightedIndex::new(weights).expect("non-empty cell-feature weights");
                per_cell.push(CellFeatureSampler { features, picker });
                cell_w.push(cell_degree[&c].max(1e-8).powf(alpha_cell));
            }
            let cell_picker = WeightedIndex::new(cell_w).expect("non-empty cell weights");

            let feature_pool: Vec<u32> = (0..n_features as u32)
                .filter(|&f| feat_count[f as usize] > 0.0)
                .collect();
            let neg_w: Vec<f32> = feature_pool
                .iter()
                .map(|&f| feat_count[f as usize].powf(alpha_neg))
                .collect();
            let neg = WeightedIndex::new(neg_w).expect("non-empty batch feature pool");

            Some(PerBatchStratifiedCellSampler {
                cell_picker,
                active_cells,
                per_cell,
                neg,
                feature_pool,
            })
        })
        .collect();
    build_bar.finish_and_clear();
    samplers
}

pub struct PerBatchStratifiedEdgeBatchArgs<'a> {
    pub sampler: &'a PerBatchStratifiedCellSampler,
    pub cell_coarsening: &'a FeatureCoarsening,
    pub fine_feature_weights: &'a [f32],
    pub batch_size: usize,
    pub n_negatives: usize,
}

/// Two-stage draw: pick cell by `degree^alpha_cell`, then feature
/// within cell by `count · fisher`. Output `EdgeBatch` shape matches
/// the flat and stratified pb samplers — downstream NCE doesn't care.
pub fn sample_per_batch_stratified_edge_batch(
    args: PerBatchStratifiedEdgeBatchArgs,
    rng: &mut impl Rng,
) -> EdgeBatch {
    let s = args.sampler;
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let lc = s.cell_picker.sample(rng);
        let c = s.active_cells[lc];
        let pf = &s.per_cell[lc];
        let lf = pf.picker.sample(rng);
        let f = pf.features[lf];
        let c_coarse = args.cell_coarsening.fine_to_coarse[c as usize] as u32;
        coarse_cells.push(c_coarse);
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

    // Parallel bucket triplets by pb; per-thread local accumulators
    // (per_pb / pb_size / feat_count) then reduce. Per-pb edge lists
    // concat across threads; pb_size and feat_count sum elementwise.
    let bucket_bar = new_progress_bar(triplets.len() as u64);
    bucket_bar.set_message("bucketing triplets by pb");
    struct Bucket {
        per_pb: Vec<Vec<(u32, f32)>>,
        pb_size: Vec<f32>,
        feat_count: Vec<f32>,
    }
    let Bucket {
        per_pb,
        pb_size,
        feat_count,
    } = triplets
        .par_iter()
        .progress_with(bucket_bar.clone())
        .fold(
            || Bucket {
                per_pb: vec![Vec::new(); n_pb],
                pb_size: vec![0f32; n_pb],
                feat_count: vec![0f32; n_features],
            },
            |mut acc, t| {
                acc.per_pb[t.cell as usize].push((t.feature, t.count));
                acc.pb_size[t.cell as usize] += t.count;
                acc.feat_count[t.feature as usize] += t.count;
                acc
            },
        )
        .reduce(
            || Bucket {
                per_pb: vec![Vec::new(); n_pb],
                pb_size: vec![0f32; n_pb],
                feat_count: vec![0f32; n_features],
            },
            |mut a, b| {
                for (av, bv) in a.per_pb.iter_mut().zip(b.per_pb.into_iter()) {
                    av.extend(bv);
                }
                for (av, bv) in a.pb_size.iter_mut().zip(b.pb_size.into_iter()) {
                    *av += bv;
                }
                for (av, bv) in a.feat_count.iter_mut().zip(b.feat_count.into_iter()) {
                    *av += bv;
                }
                a
            },
        );
    bucket_bar.finish_and_clear();

    // Per-pb sampler build is embarrassingly parallel.
    let active_idx: Vec<usize> = (0..n_pb).filter(|&p| !per_pb[p].is_empty()).collect();
    if active_idx.is_empty() {
        return None;
    }
    let build_bar = new_progress_bar(active_idx.len() as u64);
    build_bar.set_message("per-pb sampler build");
    let built: Vec<(u32, PbFeatureSampler, f32)> = active_idx
        .par_iter()
        .progress_with(build_bar.clone())
        .map(|&p| {
            let edges = &per_pb[p];
            let features: Vec<u32> = edges.iter().map(|&(f, _)| f).collect();
            let weights: Vec<f32> = edges
                .iter()
                .map(|&(f, c)| (c * fisher_weights[f as usize]).max(1e-8))
                .collect();
            let picker = WeightedIndex::new(weights).expect("non-empty pb feature weights");
            (
                p as u32,
                PbFeatureSampler { features, picker },
                pb_size[p].max(1e-8).powf(alpha_pb),
            )
        })
        .collect();
    build_bar.finish_and_clear();

    let mut active_pbs: Vec<u32> = Vec::with_capacity(built.len());
    let mut per_pb_samplers: Vec<PbFeatureSampler> = Vec::with_capacity(built.len());
    let mut pb_q: Vec<f32> = Vec::with_capacity(built.len());
    for (p, s, q) in built {
        active_pbs.push(p);
        per_pb_samplers.push(s);
        pb_q.push(q);
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
    // Parallel bucketing of triplet indices by batch. Each worker folds
    // its chunk into per-batch local Vecs, then reduce concats per batch.
    let bucket_bar = new_progress_bar(triplets.len() as u64);
    bucket_bar.set_message("bucketing triplets by batch");
    let per_batch_indices: Vec<Vec<u32>> = triplets
        .par_iter()
        .enumerate()
        .progress_with(bucket_bar.clone())
        .fold(
            || vec![Vec::new(); n_batches],
            |mut acc, (i, t)| {
                let b = batch_membership[t.cell as usize] as usize;
                acc[b].push(i as u32);
                acc
            },
        )
        .reduce(
            || vec![Vec::new(); n_batches],
            |mut a, b| {
                for (av, bv) in a.iter_mut().zip(b.into_iter()) {
                    av.extend(bv);
                }
                a
            },
        );
    bucket_bar.finish_and_clear();

    let build_bar = new_progress_bar(n_batches as u64);
    build_bar.set_message("per-batch sampler build");
    let samplers = per_batch_indices
        .into_par_iter()
        .progress_with(build_bar.clone())
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
        .collect();
    build_bar.finish_and_clear();
    samplers
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

/// Fast path for the identity-coarsening case (every "pb-sample" is
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
// Cell-cell NCE — multi-level chain on cell pairs from a caller-
// provided graph (e.g. spatial KNN). Positives are gated by pb
// co-membership at every chain level; per-level sibling negatives
// drive multi-resolution classification.
// =================================================================

pub struct PerBatchCellSampler {
    pub pos: WeightedIndex<f32>,
    pub neg: WeightedIndex<f32>,
    /// Indices into the global cell-cell `edges` slice for this batch's
    /// retained (within-batch, and — if pb-chain filtering was on —
    /// within-pb at every chain level) positives.
    pub edge_indices: Vec<u32>,
    /// Global cell ids that constitute this batch's negative pool
    /// (every cell in this batch).
    pub cell_pool: Vec<u32>,
    /// Per-chain-position sibling pool. Populated only when the
    /// sampler was built with a [`PbChainFilter`] (so the chain levels
    /// and cell-to-pb maps are known). Indexed by chain position (i.e.
    /// position in the resolved `levels` list, NOT raw pb-level id).
    /// Empty `Vec` when not in chain mode. Crate-private — only the
    /// chain sampler reads it.
    pub(crate) chain_pools: Vec<LevelSiblingPool>,
}

/// Sibling pool for a single chain position. Coarsest position (i=0)
/// has no parent in the chain → falls back to the global `cell_pool`
/// rejection. Finer positions group `cell_pool` by the cell's parent
/// pb (= pb at the *previous* chain level), so siblings of an anchor
/// `u` are the cells under the same parent. Parents whose cells all
/// share the same pb at this chain level (no real siblings exist) are
/// omitted from the map — the sampler then sees `by_parent.get(...)`
/// return `None` and falls back to global rejection. This makes the
/// per-anchor "do siblings exist?" check O(1) instead of a linear
/// scan over the pool on every batch step.
pub(crate) enum LevelSiblingPool {
    /// Coarsest chain position. No parent constraint — negatives come
    /// from `cell_pool` with same-pb-at-this-level rejection.
    Root,
    /// Finer chain position. `by_parent[parent_pb]` is the list of
    /// cells in this batch whose parent pb (at the previous chain
    /// level) equals `parent_pb`; only parents with ≥2 distinct child
    /// pbs at this level are present.
    ByParent(FxHashMap<u32, Vec<u32>>),
}

/// Optional pb-chain filter passed to [`build_per_batch_cell_samplers`].
/// When `Some`, edges whose two endpoints disagree on pb id at *any*
/// listed level are dropped during sampler construction; in addition,
/// per-chain-position sibling pools are precomputed for the
/// hard-negative draw at sample time. The same `cell_to_pb_per_level`
/// reference is read at sample time, so it must outlive the sampler.
pub struct PbChainFilter<'a> {
    pub cell_to_pb_per_level: &'a [Vec<usize>],
    pub levels: &'a [usize],
}

/// Diagnostics returned by [`build_per_batch_cell_samplers`].
#[derive(Default)]
pub struct CellCellSamplerStats {
    pub cross_batch_dropped: usize,
    pub pb_mismatch_dropped: usize,
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
    pb_filter: Option<PbChainFilter<'_>>,
) -> (Vec<Option<PerBatchCellSampler>>, CellCellSamplerStats) {
    // First pass: bucket retained edges by batch + accumulate per-cell degree
    // (within retained edges) for the negative weight.
    let mut per_batch_edge_indices: Vec<Vec<u32>> = vec![Vec::new(); n_batches];
    let mut degree: Vec<f32> = vec![0.0; n_cells];
    let mut stats = CellCellSamplerStats::default();

    for (i, &(u, v)) in edges.iter().enumerate() {
        let bu = batch_membership[u as usize];
        let bv = batch_membership[v as usize];
        if bu != bv {
            stats.cross_batch_dropped += 1;
            continue;
        }
        if let Some(filter) = pb_filter.as_ref() {
            let mut keep = true;
            for &lvl in filter.levels {
                let pb = &filter.cell_to_pb_per_level[lvl];
                if pb[u as usize] != pb[v as usize] {
                    keep = false;
                    break;
                }
            }
            if !keep {
                stats.pb_mismatch_dropped += 1;
                continue;
            }
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

            let chain_pools = pb_filter
                .as_ref()
                .map(|f| build_chain_pools(&cell_pool, f.cell_to_pb_per_level, f.levels))
                .unwrap_or_default();

            Some(PerBatchCellSampler {
                pos,
                neg,
                edge_indices,
                cell_pool,
                chain_pools,
            })
        })
        .collect();

    (samplers, stats)
}

/// Pre-compute sibling pools for each chain position in this batch.
/// Position 0 has no parent and uses [`LevelSiblingPool::Root`]; each
/// subsequent position groups the batch cells by their pb at the
/// *previous chain level* (so siblings are cells under the same parent
/// in the user's chain, not necessarily adjacent levels in the full
/// hierarchy). When the user passes a non-contiguous chain like
/// `--cell-cell-pb-levels 0,2,4`, "parent" still means "previous chain
/// entry", e.g. parent of L=2 is L=0.
///
/// Parents whose cells all share the same pb at the current chain level
/// (no real siblings to contrast against) are dropped at build time so
/// the sample-time check collapses to one `by_parent.get(...)` lookup
/// instead of a per-anchor linear scan of the pool.
fn build_chain_pools(
    cells_in_batch: &[u32],
    cell_to_pb_per_level: &[Vec<usize>],
    chain_levels: &[usize],
) -> Vec<LevelSiblingPool> {
    let mut pools: Vec<LevelSiblingPool> = Vec::with_capacity(chain_levels.len());
    for i in 0..chain_levels.len() {
        if i == 0 {
            pools.push(LevelSiblingPool::Root);
            continue;
        }
        let parent_pb = &cell_to_pb_per_level[chain_levels[i - 1]];
        let self_pb = &cell_to_pb_per_level[chain_levels[i]];
        let mut by_parent: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        // `seen_self` tracks the first-seen child pb per parent; once a
        // second distinct child appears we flip to `u32::MAX` to mark
        // the parent as sibling-bearing.
        let mut seen_self: FxHashMap<u32, u32> = FxHashMap::default();
        for &c in cells_in_batch {
            let parent = parent_pb[c as usize] as u32;
            let pb = self_pb[c as usize] as u32;
            by_parent.entry(parent).or_default().push(c);
            seen_self
                .entry(parent)
                .and_modify(|e| {
                    if *e != u32::MAX && *e != pb {
                        *e = u32::MAX;
                    }
                })
                .or_insert(pb);
        }
        by_parent.retain(|p, _| seen_self.get(p) == Some(&u32::MAX));
        pools.push(LevelSiblingPool::ByParent(by_parent));
    }
    pools
}

// =================================================================
// Cell-cell chain — same `(left, right)` positive pair scored against
// per-level negatives that share batch but differ in pb at that level.
// Sum across levels with per-level λ. Cell-side analogue of the
// feature-side `nce_loss_chain`: gives the cell embedding multi-
// resolution classification signal in one coherent step.
// =================================================================

pub struct CellChainBatch {
    pub left_cells: Vec<u32>,  // [B]
    pub right_cells: Vec<u32>, // [B]
    /// `per_level_neg[lvl_idx]` is `[B*K]` row-major: negatives for
    /// positive `b` at level `lvl_idx` live at `[b*K..(b+1)*K]`. Length
    /// equals the number of chain levels.
    pub per_level_neg: Vec<Vec<u32>>,
    pub n_negatives: usize,
}

pub struct CellChainBatchArgs<'a> {
    pub edges: &'a [(u32, u32)],
    pub batch_sampler: &'a PerBatchCellSampler,
    pub batch_size: usize,
    pub n_negatives: usize,
    /// Pb assignment per chain level (same order as
    /// `lambdas` in [`cell_cell_nce_loss_chain`]). Each slice is
    /// length `n_cells`. Drawn from
    /// `MultilevelCollapseOut::cell_to_pb_per_level` after the
    /// coarsest-first reverse in `fit()`.
    pub pb_maps: &'a [&'a [usize]],
}

pub struct CellChainBatchStats {
    /// Per chain position: count of `(positive, level)` cell-anchor entries
    /// that fell back from sibling-pool sampling to the global
    /// `cell_pool` rejection (because the anchor's parent pb had no
    /// siblings at this chain level in this batch). Same length as the
    /// chain. Position 0 is always 0 (Root has no fallback path).
    pub per_level_fallback: Vec<usize>,
}

/// Draw `batch_size` positive cell pairs from `batch_sampler`, then for
/// each chain position draw K *sibling* negatives per positive — cells
/// that share the anchor's pb at the previous chain level (siblings in
/// the pb tree) but differ at the current chain level. Coarsest
/// position has no parent, so it draws from `cell_pool` with
/// same-pb-at-this-level rejection (legacy behaviour). When an
/// anchor's parent pb has only one child at this chain level (no
/// siblings exist), the draw falls back to `cell_pool` rejection and
/// the event is counted in [`CellChainBatchStats::per_level_fallback`]
/// for the caller to log.
pub fn sample_cell_chain_batch(
    args: CellChainBatchArgs,
    rng: &mut impl Rng,
) -> (CellChainBatch, CellChainBatchStats) {
    let s = args.batch_sampler;
    let mut left_cells = Vec::with_capacity(args.batch_size);
    let mut right_cells = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let local = s.pos.sample(rng);
        let global = s.edge_indices[local] as usize;
        let (i, j) = args.edges[global];
        left_cells.push(i);
        right_cells.push(j);
    }

    let n_chain = args.pb_maps.len();
    let mut per_level_neg: Vec<Vec<u32>> = Vec::with_capacity(n_chain);
    let mut per_level_fallback: Vec<usize> = vec![0; n_chain];

    for (chain_pos, pb_self) in args.pb_maps.iter().enumerate() {
        let pool_for_pos = s.chain_pools.get(chain_pos);
        let mut neg = Vec::with_capacity(args.batch_size * args.n_negatives);

        for &u in &left_cells {
            let pivot_self = pb_self[u as usize];

            // Resolve the candidate pool for this anchor at this chain
            // position. Coarsest position OR no-siblings → global pool.
            // (`build_chain_pools` already dropped parents with no real
            // siblings, so a `Some` entry here is guaranteed to contain
            // at least one cell with a different `pb_self` from `u`.)
            let sibling_pool: Option<&[u32]> = match pool_for_pos {
                Some(LevelSiblingPool::ByParent(by_parent)) => {
                    let parent_pb = args.pb_maps[chain_pos - 1][u as usize] as u32;
                    by_parent.get(&parent_pb).map(|v| v.as_slice())
                }
                _ => None,
            };

            let used_fallback = sibling_pool.is_none();
            if used_fallback {
                per_level_fallback[chain_pos] += 1;
            }

            for _ in 0..args.n_negatives {
                let c = draw_one_negative(s, sibling_pool, pb_self, pivot_self, rng);
                neg.push(c);
            }
        }
        per_level_neg.push(neg);
    }

    (
        CellChainBatch {
            left_cells,
            right_cells,
            per_level_neg,
            n_negatives: args.n_negatives,
        },
        CellChainBatchStats { per_level_fallback },
    )
}

/// Sample one cell `w` with rejection on `pb_self[w] == pivot`. Pool
/// is either a precomputed sibling pool (uniform draw) or `None` (fall
/// back to the degree^α-weighted global `cell_pool`).
fn draw_one_negative(
    s: &PerBatchCellSampler,
    sibling_pool: Option<&[u32]>,
    pb_self: &[usize],
    pivot: usize,
    rng: &mut impl Rng,
) -> u32 {
    const MAX_REJECTION_TRIES: u32 = 16;
    for _ in 0..MAX_REJECTION_TRIES {
        let c = match sibling_pool {
            Some(pool) => pool[rng.random_range(0..pool.len())],
            None => s.cell_pool[s.neg.sample(rng)],
        };
        if pb_self[c as usize] != pivot {
            return c;
        }
    }
    // Last-ditch fallback: take whatever we drew last. Reached only when
    // the candidate pool is overwhelmingly one pb after `has_sibling`
    // already returned true (e.g., one-in-N siblings and unlucky draws).
    match sibling_pool {
        Some(pool) => pool[rng.random_range(0..pool.len())],
        None => s.cell_pool[s.neg.sample(rng)],
    }
}

/// Multi-level cell-cell NCE. Shares one positive `(left, right)` per
/// row across every level; each level scores those positives against
/// its own per-level negatives and contributes `λ_L · L_NCE` to the
/// total. Returns the λ-weighted sum (no division by `num_levels` —
/// caller's `CellCellConfig::lambda` handles total scaling).
pub fn cell_cell_nce_loss_chain(
    model: &JointEmbedModel,
    batch: CellChainBatch,
    lambdas: &[f32],
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.left_cells.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let k = batch.n_negatives;
    assert_eq!(
        batch.per_level_neg.len(),
        lambdas.len(),
        "per_level_neg ({}) and lambdas ({}) length mismatch",
        batch.per_level_neg.len(),
        lambdas.len(),
    );

    let left_idx = Tensor::from_vec(batch.left_cells, b, dev)?;
    let right_idx = Tensor::from_vec(batch.right_cells, b, dev)?;
    let e_left = model.e_cell.index_select(&left_idx, 0)?;
    let b_left = model.b_cell.index_select(&left_idx, 0)?;
    let e_right = model.e_cell.index_select(&right_idx, 0)?;
    let b_right = model.b_cell.index_select(&right_idx, 0)?;
    let pos_score = JointEmbedModel::score_diag(&e_left, &e_right, &b_left, &b_right)?;
    let pos_term = log_sigmoid(&pos_score)?;

    let mut total: Option<Tensor> = None;
    for (lvl_neg, &lam) in batch.per_level_neg.into_iter().zip(lambdas.iter()) {
        let neg_idx = Tensor::from_vec(lvl_neg, b * k, dev)?;
        let e_neg_flat = model.e_cell.index_select(&neg_idx, 0)?;
        let b_neg_flat = model.b_cell.index_select(&neg_idx, 0)?;
        let h = e_neg_flat.dim(1)?;
        let e_neg = e_neg_flat.reshape((b, k, h))?;
        let b_neg = b_neg_flat.reshape((b, k))?;
        let neg_score = JointEmbedModel::score_negatives(&e_neg, &e_left, &b_neg, &b_left)?;
        let per_edge = (pos_term.clone() + log_sigmoid(&neg_score.neg()?)?.sum(1)?)?.neg()?;
        let level_loss = (per_edge.mean(0)? * lam as f64)?;
        total = Some(match total {
            Some(prev) => (prev + level_loss)?,
            None => level_loss,
        });
    }
    Ok(total.expect("non-empty pb_maps enforced upstream by PbChainSpec resolution"))
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
        let (samplers, stats) =
            build_per_batch_cell_samplers(&edges, &batch_membership, 2, 4, 0.75, None);

        assert_eq!(
            stats.cross_batch_dropped, 1,
            "expected one cross-batch edge dropped"
        );
        assert_eq!(stats.pb_mismatch_dropped, 0);
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
        let (samplers, stats) =
            build_per_batch_cell_samplers(&edges, &batch_membership, 2, 4, 0.75, None);
        assert_eq!(stats.cross_batch_dropped, 0);
        assert_eq!(stats.pb_mismatch_dropped, 0);
        assert!(samplers[0].is_some());
        assert!(samplers[1].is_none(), "batch 1 has no edges → None");
    }

    #[test]
    fn chain_pools_prune_parents_without_siblings() {
        // 4 cells, one batch. Parent pb_0 = {0,0,1,1}; finer pb_1 = {0,0,1,1}
        // — every parent has exactly ONE child pb at level 1, so no anchor
        // has a true sibling. The map should drop both parents.
        let edges = vec![(0u32, 1), (2, 3)];
        let batch_membership = vec![0u32; 4];
        let cell_to_pb_per_level: Vec<Vec<usize>> = vec![
            vec![0, 0, 1, 1], // L=0 parent
            vec![0, 0, 1, 1], // L=1 self — same partition as parent
        ];
        let filter = PbChainFilter {
            cell_to_pb_per_level: &cell_to_pb_per_level,
            levels: &[0, 1],
        };
        let (samplers, _) =
            build_per_batch_cell_samplers(&edges, &batch_membership, 1, 4, 0.75, Some(filter));
        let s = samplers[0].as_ref().unwrap();
        let LevelSiblingPool::ByParent(by_parent) = &s.chain_pools[1] else {
            panic!("expected ByParent at chain position 1");
        };
        assert!(
            by_parent.is_empty(),
            "parents whose children are all the same pb at this level should be dropped"
        );
    }

    #[test]
    fn chain_pools_group_by_parent_pb() {
        // 8 cells, one batch. Two-level chain over level 0 (coarse,
        // 2 pbs) and level 1 (fine, 4 pbs, where cells [0,1] and [2,3]
        // share parent pb_0=0; [4,5] and [6,7] share parent pb_0=1).
        // Build a sampler with chain filter, then inspect chain_pools.
        let edges = vec![(0u32, 1), (2, 3), (4, 5), (6, 7)];
        let batch_membership = vec![0u32; 8];
        let cell_to_pb_per_level: Vec<Vec<usize>> = vec![
            vec![0, 0, 0, 0, 1, 1, 1, 1], // L=0 coarse: {0..3} ↦ 0; {4..7} ↦ 1
            vec![0, 0, 1, 1, 2, 2, 3, 3], // L=1 fine
        ];
        let filter = PbChainFilter {
            cell_to_pb_per_level: &cell_to_pb_per_level,
            levels: &[0, 1],
        };
        let (samplers, _stats) =
            build_per_batch_cell_samplers(&edges, &batch_membership, 1, 8, 0.75, Some(filter));
        let s = samplers[0]
            .as_ref()
            .expect("batch 0 has within-pb edges at every chain level");

        assert_eq!(s.chain_pools.len(), 2);
        // Chain position 0 (coarsest) is the Root — no by_parent pool.
        assert!(matches!(s.chain_pools[0], LevelSiblingPool::Root));
        // Chain position 1: by_parent groups by L=0 pb id.
        let LevelSiblingPool::ByParent(by_parent) = &s.chain_pools[1] else {
            panic!("expected ByParent at chain position 1");
        };
        let mut parent0 = by_parent.get(&0).cloned().expect("parent pb 0 present");
        parent0.sort();
        assert_eq!(parent0, vec![0, 1, 2, 3]);
        let mut parent1 = by_parent.get(&1).cloned().expect("parent pb 1 present");
        parent1.sort();
        assert_eq!(parent1, vec![4, 5, 6, 7]);
    }

    #[test]
    fn sibling_negative_draws_share_parent_differ_at_self() {
        // Same 8-cell setup; verify that sibling-pool draws at the fine
        // chain level always produce cells with same L=0 pb as the anchor
        // but different L=1 pb (i.e. real siblings in the pb tree).
        use rand::SeedableRng;
        let edges = vec![(0u32, 1), (4, 5)];
        let batch_membership = vec![0u32; 8];
        let cell_to_pb_per_level: Vec<Vec<usize>> =
            vec![vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 0, 1, 1, 2, 2, 3, 3]];
        let filter = PbChainFilter {
            cell_to_pb_per_level: &cell_to_pb_per_level,
            levels: &[0, 1],
        };
        let (samplers, _) =
            build_per_batch_cell_samplers(&edges, &batch_membership, 1, 8, 0.75, Some(filter));
        let s = samplers[0].as_ref().unwrap();

        let pb_l0: &[usize] = &cell_to_pb_per_level[0];
        let pb_l1: &[usize] = &cell_to_pb_per_level[1];
        let pb_maps: Vec<&[usize]> = vec![pb_l0, pb_l1];

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (batch, stats) = sample_cell_chain_batch(
            CellChainBatchArgs {
                edges: &edges,
                batch_sampler: s,
                batch_size: 128, // exercise both positives many times
                n_negatives: 4,
                pb_maps: &pb_maps,
            },
            &mut rng,
        );

        // No fallbacks expected: every anchor's parent pb has 2 children at L=1.
        assert_eq!(stats.per_level_fallback[1], 0);
        // Fine-level negatives must agree with anchor at L=0 (sibling) and
        // disagree at L=1.
        let k = 4;
        for b in 0..batch.left_cells.len() {
            let u = batch.left_cells[b];
            let pu_l0 = pb_l0[u as usize];
            let pu_l1 = pb_l1[u as usize];
            for kk in 0..k {
                let w = batch.per_level_neg[1][b * k + kk];
                assert_eq!(
                    pb_l0[w as usize], pu_l0,
                    "fine-level neg should share parent pb with anchor"
                );
                assert_ne!(
                    pb_l1[w as usize], pu_l1,
                    "fine-level neg should differ from anchor at this level"
                );
            }
        }
    }

    #[test]
    fn cell_cell_sampler_filters_pb_mismatched_edges() {
        // 4 cells in one batch. Edges: (0,1) same pb at L0, (2,3) same
        // pb at L0, (0,2) different pb at L0 — should drop the last.
        let edges = vec![(0u32, 1), (2, 3), (0, 2)];
        let batch_membership = vec![0u32; 4];
        // L0: {0,1} ↦ 0; {2,3} ↦ 1.
        let cell_to_pb_per_level: Vec<Vec<usize>> = vec![vec![0, 0, 1, 1]];
        let filter = PbChainFilter {
            cell_to_pb_per_level: &cell_to_pb_per_level,
            levels: &[0],
        };
        let (samplers, stats) =
            build_per_batch_cell_samplers(&edges, &batch_membership, 1, 4, 0.75, Some(filter));
        assert_eq!(stats.cross_batch_dropped, 0);
        assert_eq!(stats.pb_mismatch_dropped, 1);
        let s0 = samplers[0]
            .as_ref()
            .expect("batch 0 has within-batch within-pb edges");
        assert_eq!(s0.edge_indices, vec![0, 1]);
    }
}
