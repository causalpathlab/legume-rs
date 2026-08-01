//! Multi-level cell-cell chain NCE.
//!
//! Same `(left, right)` positive pair per row is scored against
//! per-level negatives that share the anchor's pb at the previous chain
//! level but differ at the current one (the sibling-rejection variant).
//! Gives the cell embedding multi-resolution classification signal in one coherent
//! step. It had a feature-side analogue in [`crate::loss::feat`]; that one was deleted
//! along with the chain composite mode, which nothing could reach.

use crate::loss::cell::{LevelSiblingPool, PerBatchCellSampler};
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
