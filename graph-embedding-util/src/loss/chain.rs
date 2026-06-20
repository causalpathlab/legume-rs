//! Multi-level cell-cell chain NCE.
//!
//! Same `(left, right)` positive pair per row is scored against
//! per-level negatives that share the anchor's pb at the previous chain
//! level but differ at the current one (the sibling-rejection variant).
//! Cell-side analogue of the feature-side `nce_loss_chain` in
//! [`crate::loss::feat`]: gives the cell embedding multi-resolution
//! classification signal in one coherent step.
//!
//! The gene-modulated public functions live here:
//!
//! - [`cell_cell_nce_loss_per_level_gated`] — returns `[L]` per-level
//!   losses for one [`CellChainBatch`], scored per-gene.
//! - [`cell_cell_nce_loss_per_level_batched_gated`] — returns `[G, L]`
//!   for `G` independent chain batches in one forward pass; per-gene
//!   scoring modulates by `e_gene[gene_ids[g]]`. Used by `pinto cage`
//!   to collapse 18k tiny CUDA forwards per epoch into ~600 big ones.

use crate::feature_network::{select_feat_emb, FeatureNetworkSmoother};
use crate::loss::cell::{LevelSiblingPool, PerBatchCellSampler};
use crate::loss::logistic_nce;
use crate::model::JointEmbedModel;
use candle_util::candle_core::{Device, Result, Tensor};
use rand::{Rng, RngExt};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

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
    /// Pb assignment per chain level (coarsest-first, one entry per chain
    /// position). Each slice is length `n_cells`. Drawn from
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
    // Default positive distribution: the sampler's own per-edge weights
    // (currently uniform-over-retained-edges). Delegate to the
    // `_with_pos` helper so cage's gene-gated sampler can share every
    // negative-side branch.
    let pos: &WeightedIndex<f32> = &args.batch_sampler.pos;
    let map: &[u32] = &args.batch_sampler.edge_indices;
    sample_cell_chain_batch_with_pos(args, pos, map, rng)
}

/// Same as [`sample_cell_chain_batch`] except the caller supplies the
/// positive index distribution and the local→global edge-id map. The
/// rest of the chain logic — sibling pools, fallbacks, per-level
/// rejection — is unchanged.
///
/// Indices drawn from `pos_override` are interpreted as offsets into
/// `pos_to_global_edge`, which must point into `args.edges`. The base
/// sampler's `pos` / `edge_indices` are not consulted on the positive
/// side; they remain in use for the negative-side `cell_pool` /
/// `chain_pools` / `neg` machinery.
pub fn sample_cell_chain_batch_with_pos(
    args: CellChainBatchArgs,
    pos_override: &WeightedIndex<f32>,
    pos_to_global_edge: &[u32],
    rng: &mut impl Rng,
) -> (CellChainBatch, CellChainBatchStats) {
    let s = args.batch_sampler;
    let mut left_cells = Vec::with_capacity(args.batch_size);
    let mut right_cells = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let local = pos_override.sample(rng);
        let global = pos_to_global_edge[local] as usize;
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
                    by_parent.get(&parent_pb).map(std::vec::Vec::as_slice)
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

////////////////////////////////////////////////////////////////
//                                                            //
// Gene-modulated (v3) variants — gene identity enters score //
//                                                            //
////////////////////////////////////////////////////////////////

/// Per-level cell-cell NCE with a gene-modulated per-row score
/// `score(u, v) = (e_gene·e_cell[u])(e_gene·e_cell[v]) + b_cell[u] + b_cell[v]`.
/// All B positives in `batch` share the same gene `gene_id`, so the
/// gene-side gather is one row broadcast across B.
///
/// `e_gene` is read from `model.e_feat` (the model's `e_feat` slot
/// holds the gene embedding under cage semantics — cells and genes
/// share one D-dim space). `b_gene` is unused — gene identity only
/// enters via the embedding direction; the biases are cell-side only,
/// matching the original non-gated loss.
///
/// `smoother` optionally applies SGC graph smoothing to the `e_gene`
/// gather, matching senna's `select_feat_emb` pattern.
pub fn cell_cell_nce_loss_per_level_gated(
    model: &JointEmbedModel,
    batch: CellChainBatch,
    gene_id: u32,
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.left_cells.len();
    let l = batch.per_level_neg.len();
    assert!(l > 0, "non-empty pb_maps required");
    if b == 0 {
        return Tensor::zeros(l, candle_util::candle_core::DType::F32, dev);
    }
    let k = batch.n_negatives;

    let left_idx = Tensor::from_vec(batch.left_cells, b, dev)?;
    let right_idx = Tensor::from_vec(batch.right_cells, b, dev)?;
    let e_left = model.e_cell.index_select(&left_idx, 0)?;
    let b_left = model.b_cell.index_select(&left_idx, 0)?;
    let e_right = model.e_cell.index_select(&right_idx, 0)?;
    let b_right = model.b_cell.index_select(&right_idx, 0)?;

    // One gene → one row of e_gene, broadcast to [B, H].
    let gene_idx_one = Tensor::from_vec(vec![gene_id], 1, dev)?;
    let e_gene_row = select_feat_emb(smoother, &model.e_feat, &gene_idx_one)?; // [1, H]
    let h = e_gene_row.dim(1)?;
    let e_gene_b = e_gene_row.broadcast_as((b, h))?;

    let pos_score =
        JointEmbedModel::score_cellcell_gated(&e_gene_b, &e_left, &e_right, &b_left, &b_right)?;

    let mut per_level: Vec<Tensor> = Vec::with_capacity(l);
    for lvl_neg in batch.per_level_neg {
        let neg_idx = Tensor::from_vec(lvl_neg, b * k, dev)?;
        let e_neg_flat = model.e_cell.index_select(&neg_idx, 0)?;
        let b_neg_flat = model.b_cell.index_select(&neg_idx, 0)?;
        let e_neg = e_neg_flat.reshape((b, k, h))?;
        let b_neg = b_neg_flat.reshape((b, k))?;
        let neg_score =
            JointEmbedModel::score_cellcell_gated_neg(&e_gene_b, &e_left, &e_neg, &b_left, &b_neg)?;
        let per_edge = logistic_nce(&pos_score, std::slice::from_ref(&neg_score))?;
        per_level.push(per_edge.mean(0)?);
    }
    Tensor::stack(&per_level, 0)
}

/// Batched gene-modulated per-level loss for `G` independent
/// `CellChainBatch`es, one per gene. Returns `[G, L]` per-(gene, level)
/// losses. cage calls this once per chunk; `gene_ids[g]` is the gene
/// id corresponding to `batches[g]`.
///
/// Optional `dim_gates: [L, D]` (already passed through `softplus_floored`
/// or similar positive transform) modulates `e_gene` per chain level
/// via elementwise multiply with the gate row. Cage uses this to
/// learn which embedding dimensions matter at each coarsening scale.
/// `None` recovers the un-gated gene-modulated score.
///
/// All batches must share the same `B`, `L`, and `K`.
pub fn cell_cell_nce_loss_per_level_batched_gated(
    model: &JointEmbedModel,
    batches: Vec<CellChainBatch>,
    gene_ids: &[u32],
    dim_gates: Option<&Tensor>,
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> Result<Tensor> {
    let g = batches.len();
    assert!(g > 0, "non-empty batches required");
    assert_eq!(
        gene_ids.len(),
        g,
        "gene_ids ({}) and batches ({}) length mismatch",
        gene_ids.len(),
        g
    );
    let b = batches[0].left_cells.len();
    let l = batches[0].per_level_neg.len();
    let k = batches[0].n_negatives;
    for cb in &batches {
        assert_eq!(cb.left_cells.len(), b, "batched_gated: B mismatch");
        assert_eq!(cb.per_level_neg.len(), l, "batched_gated: L mismatch");
        assert_eq!(cb.n_negatives, k, "batched_gated: K mismatch");
    }
    if b == 0 {
        return Tensor::zeros((g, l), candle_util::candle_core::DType::F32, dev);
    }

    let total_b = g * b;
    let total_neg = total_b * k;

    let mut all_left: Vec<u32> = Vec::with_capacity(total_b);
    let mut all_right: Vec<u32> = Vec::with_capacity(total_b);
    let mut all_neg_per_level: Vec<Vec<u32>> =
        (0..l).map(|_| Vec::with_capacity(total_neg)).collect();
    // Replicate each gene id B times so that the gene-side gather
    // lines up row-for-row with the [G*B] cell-side gather.
    let mut gene_repeat: Vec<u32> = Vec::with_capacity(total_b);
    for (cb, &gid) in batches.into_iter().zip(gene_ids.iter()) {
        all_left.extend(cb.left_cells);
        all_right.extend(cb.right_cells);
        for _ in 0..b {
            gene_repeat.push(gid);
        }
        for (lvl_idx, lvl_neg) in cb.per_level_neg.into_iter().enumerate() {
            all_neg_per_level[lvl_idx].extend(lvl_neg);
        }
    }

    let left_idx = Tensor::from_vec(all_left, total_b, dev)?;
    let right_idx = Tensor::from_vec(all_right, total_b, dev)?;
    let gene_idx = Tensor::from_vec(gene_repeat, total_b, dev)?;
    let e_left = model.e_cell.index_select(&left_idx, 0)?;
    let b_left = model.b_cell.index_select(&left_idx, 0)?;
    let e_right = model.e_cell.index_select(&right_idx, 0)?;
    let b_right = model.b_cell.index_select(&right_idx, 0)?;
    let e_gene = select_feat_emb(smoother, &model.e_feat, &gene_idx)?; // [G*B, H]

    let mut per_gene_per_level: Vec<Tensor> = Vec::with_capacity(l);
    for (lvl_idx, lvl_neg) in all_neg_per_level.into_iter().enumerate() {
        // Optional per-level per-dim gating: modulate `e_gene` by
        // `dim_gates[ℓ, :]` (already a positive `[L, D]` tensor; row ℓ
        // broadcasts over `G*B`). With no gate the score reduces to
        // the original gene-modulated form.
        let e_gene_lvl = if let Some(g_t) = dim_gates {
            let row = g_t.narrow(0, lvl_idx, 1)?;
            let h = e_gene.dim(1)?;
            let bcast = row.broadcast_as((total_b, h))?;
            (e_gene.clone() * bcast)?
        } else {
            e_gene.clone()
        };
        let pos_score = JointEmbedModel::score_cellcell_gated(
            &e_gene_lvl,
            &e_left,
            &e_right,
            &b_left,
            &b_right,
        )?;
        let neg_idx = Tensor::from_vec(lvl_neg, total_neg, dev)?;
        let e_neg_flat = model.e_cell.index_select(&neg_idx, 0)?;
        let b_neg_flat = model.b_cell.index_select(&neg_idx, 0)?;
        let h = e_neg_flat.dim(1)?;
        let e_neg = e_neg_flat.reshape((total_b, k, h))?;
        let b_neg = b_neg_flat.reshape((total_b, k))?;
        let neg_score = JointEmbedModel::score_cellcell_gated_neg(
            &e_gene_lvl,
            &e_left,
            &e_neg,
            &b_left,
            &b_neg,
        )?;
        let per_edge = logistic_nce(&pos_score, std::slice::from_ref(&neg_score))?;
        let per_edge_gb = per_edge.reshape((g, b))?;
        let per_gene = per_edge_gb.mean(1)?; // [G]
        per_gene_per_level.push(per_gene);
    }
    Tensor::stack(&per_gene_per_level, 1)
}
