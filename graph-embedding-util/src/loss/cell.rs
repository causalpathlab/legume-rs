//! Cell-cell sampling primitives over a caller-supplied edge list
//! (e.g. spatial KNN).
//!
//! Positives are gated by pb co-membership at every chain level;
//! cross-batch edges are dropped. Per-chain-position sibling pools are
//! precomputed so the chain sampler in [`crate::loss::chain`] can do
//! O(1) "is there a real sibling?" lookups per anchor.

use rand_distr::weighted::WeightedIndex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

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
#[must_use]
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
/// hierarchy). When the chain is non-contiguous, e.g. levels `0,2,4`,
/// "parent" still means "previous chain entry", e.g. parent of L=2 is L=0.
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
