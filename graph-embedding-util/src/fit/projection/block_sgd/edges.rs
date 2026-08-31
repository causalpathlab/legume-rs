//! The host-side edge table and the block sizing that go with it: every node's
//! counts restricted to one pass's feature partition, flattened once, plus the `Bc`
//! that keeps a block's `[Bc, F]` activations inside the budget. Both are pure host
//! bookkeeping the device loop reads but never rebuilds.
//!
//! The two have different lifetimes, which is why they are separate calls: `Bc`
//! belongs to the frozen dictionary and is held by [`super::PassDict`], while the
//! edge table belongs to the nodes and is rebuilt for every group of them.

use super::{BLOCK_ACTIVATION_BYTES, LIVE_BLOCK_TENSORS, MAX_BLOCK_CELLS};
use crate::fit::projection::{cell_edges, CellBatchDivisor};

////////////////////////////
// Edge table (host, once) //
////////////////////////////

/// Every cell's edges restricted to one pass's feature partition, remapped to that
/// pass's local feature ids, grouped by cell position and flattened.
///
/// Built once per pass. A block takes a *slice* of these — no per-block copy, and
/// no per-cell `Vec` churn of the kind `cell_edges` does on the Newton path.
pub(super) struct EdgeTable {
    /// `offsets[i]..offsets[i + 1]` is cell `i`'s slice of `feat`/`count`.
    offsets: Vec<usize>,
    /// Pass-local feature ids (index into the pass's live dictionary).
    feat: Vec<u32>,
    count: Vec<f32>,
}

impl EdgeTable {
    pub(super) fn build(
        cells: &[(u32, &[u32], &[f32])],
        rows: &[u32],
        n_features: usize,
        batch_divisor: Option<CellBatchDivisor>,
    ) -> Self {
        // Global feature id → pass-local id, or `u32::MAX` when the feature is not
        // in this pass's partition.
        let mut local = vec![u32::MAX; n_features];
        for (l, &g) in rows.iter().enumerate() {
            local[g as usize] = l as u32;
        }
        let mut offsets = Vec::with_capacity(cells.len() + 1);
        let mut feat = Vec::new();
        let mut count = Vec::new();
        offsets.push(0);
        for &(cell, feats, counts) in cells {
            // The `μ_residual` divide happens here, once, instead of per solve.
            for (f, c) in cell_edges(cell, feats, counts, batch_divisor) {
                let l = local[f as usize];
                if l != u32::MAX {
                    feat.push(l);
                    count.push(c);
                }
            }
            offsets.push(feat.len());
        }
        Self {
            offsets,
            feat,
            count,
        }
    }

    pub(super) fn cell_slice(&self, i: usize) -> (&[u32], &[f32]) {
        let (s, e) = (self.offsets[i], self.offsets[i + 1]);
        (&self.feat[s..e], &self.count[s..e])
    }
}

//////////////////
// Block sizing //
//////////////////

/// Cells per block for a pass over `f` features.
pub(super) fn block_cells(f: usize) -> usize {
    if f == 0 {
        return 1;
    }
    (BLOCK_ACTIVATION_BYTES / (f * 4 * LIVE_BLOCK_TENSORS)).clamp(1, MAX_BLOCK_CELLS)
}
