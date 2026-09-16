//! PBG's batcher over typed relations: every batch holds ONE relation,
//! chosen with probability proportional to that relation's remaining
//! edges, and takes the next `batch_size` of them in the epoch's shuffled
//! order. The batch is cut into chunks of `c` positives (`num_batch_negs`)
//! and the last chunk is zero-padded so the fused step sees a dense
//! `[k, c, ·]` block. Pad rows point at node 0, carry no loss weight, and
//! are masked out of the negatives.
//!
//! The one typed change: a chunk's uniform negatives are drawn inside the
//! relation's own lhs / rhs node-type ranges, and a row's weight is the
//! relation weight times the edge's own weight.

use super::graph::{NodeTypeTable, RelationTable, TypedEdgeList};
use rand::{Rng, RngExt};
use std::ops::Range;

/// One single-relation batch, padded to `k · c` rows.
pub(crate) struct PaddedBatch {
    /// Number of chunks.
    pub k: usize,
    /// Chunk size (`num_batch_negs`).
    pub c: usize,
    /// Uniform negatives per chunk (`num_uniform_negs`).
    pub u: usize,
    /// Real (unpadded) positives, `1..=batch_size`.
    pub n_real: usize,
    /// The relation every row belongs to.
    pub rel: usize,
    /// `[k·c]` lhs global ids (0 on pad rows).
    pub lhs: Vec<u32>,
    /// `[k·c]` rhs global ids (0 on pad rows).
    pub rhs: Vec<u32>,
    /// `[k·c]` per-row loss weight: relation × edge weight on real rows, 0 on pads.
    pub row_w: Vec<f32>,
    /// `[k·c]` 1 on real rows, 0 on pads (pads must not act as negatives).
    pub col_valid: Vec<f32>,
    /// `[k·u]` uniform lhs negatives, `u` per chunk, inside the lhs type.
    pub uni_lhs: Vec<u32>,
    /// `[k·u]` uniform rhs negatives, `u` per chunk, inside the rhs type.
    pub uni_rhs: Vec<u32>,
}

/// The queues of edge indices for one epoch: each entry is one pass over
/// one relation's contiguous (already shuffled) block. A relation that
/// repeats within an epoch contributes several entries.
pub(crate) struct EpochBatcher {
    /// `(relation, what is left of the block)`; `start` advances as
    /// batches are handed out.
    remaining: Vec<(usize, Range<usize>)>,
    batch_size: usize,
}

impl EpochBatcher {
    /// One block per relation, in relation order, each visited once. An
    /// empty block is a relation with nothing to hand out.
    pub fn new(blocks: &[Range<usize>], batch_size: usize) -> Self {
        let entries: Vec<(usize, Range<usize>)> = blocks.iter().cloned().enumerate().collect();
        Self::from_entries(entries, batch_size)
    }

    /// Explicit `(relation, block)` entries; a relation listed `k` times is
    /// drawn `k` times over the epoch. The caller gives each repeat its own
    /// shuffle by re-shuffling the block between passes, or accepts the
    /// same order twice.
    pub fn from_entries(entries: Vec<(usize, Range<usize>)>, batch_size: usize) -> Self {
        Self {
            remaining: entries,
            batch_size: batch_size.max(1),
        }
    }

    /// Edges not yet handed out this epoch (repeats counted each time).
    pub fn remaining(&self) -> usize {
        self.remaining.iter().map(|(_, q)| q.len()).sum()
    }

    /// The next single-relation batch, or `None` once the epoch is drained.
    pub fn next_batch<R: Rng>(
        &mut self,
        edges: &TypedEdgeList,
        types: &NodeTypeTable,
        rels: &RelationTable,
        c: usize,
        u: usize,
        rng: &mut R,
    ) -> Option<PaddedBatch> {
        let total = self.remaining();
        if total == 0 {
            return None;
        }
        let c = c.max(1);
        // Multinomial over the relations' remaining edge counts.
        let mut x = rng.random_range(0..total);
        let mut e = self.remaining.len() - 1;
        for (i, (_, q)) in self.remaining.iter().enumerate() {
            if x < q.len() {
                e = i;
                break;
            }
            x -= q.len();
        }
        let r = self.remaining[e].0;
        let n_real = self.remaining[e].1.len().min(self.batch_size);
        let first = self.remaining[e].1.start;
        self.remaining[e].1.start += n_real;

        let k = n_real.div_ceil(c);
        let p = k * c;
        let relation = rels.get(r);
        let weight = relation.weight;
        let mut lhs = vec![0u32; p];
        let mut rhs = vec![0u32; p];
        let mut row_w = vec![0f32; p];
        let mut col_valid = vec![0f32; p];
        for (j, e) in (first..first + n_real).enumerate() {
            lhs[j] = edges.lhs[e];
            rhs[j] = edges.rhs[e];
            row_w[j] = weight * edges.edge_weight(e);
            col_valid[j] = 1.0;
        }
        let lhs_range = types.range(relation.lhs_type as usize);
        let rhs_range = types.range(relation.rhs_type as usize);
        let uni_lhs = (0..k * u)
            .map(|_| rng.random_range(lhs_range.clone()))
            .collect();
        let uni_rhs = (0..k * u)
            .map(|_| rng.random_range(rhs_range.clone()))
            .collect();
        Some(PaddedBatch {
            k,
            c,
            u,
            n_real,
            rel: r,
            lhs,
            rhs,
            row_w,
            col_valid,
            uni_lhs,
            uni_rhs,
        })
    }
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod batch_tests;
