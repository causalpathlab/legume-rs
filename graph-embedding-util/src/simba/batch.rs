//! PBG's batcher (`batching.py::batch_edges_group_by_relation_type`): every
//! batch holds ONE relation, chosen with probability proportional to that
//! relation's remaining edges, and takes the next `batch_size` of them in the
//! epoch's shuffled order. The batch is then cut into chunks of `c` positives
//! — PBG's `num_batch_negs` — and the last chunk is zero-padded to `c` rows
//! so the fused step sees a dense `[k, c, ·]` block. Pad rows point at entity
//! 0, carry no loss weight, and are masked out of the negatives.

use super::graph::{EdgeList, RelationTable};
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
    /// `[k·c]` cell ids (0 on pad rows).
    pub lhs: Vec<u32>,
    /// `[k·c]` gene ids (0 on pad rows).
    pub rhs: Vec<u32>,
    /// `[k·c]` per-row loss weight: `weight` on real rows, 0 on pads.
    pub row_w: Vec<f32>,
    /// `[k·c]` 1 on real rows, 0 on pads (pads must not act as negatives).
    pub col_valid: Vec<f32>,
    /// `[k·u]` uniform cell negatives, `u` per chunk.
    pub uni_lhs: Vec<u32>,
    /// `[k·u]` uniform gene negatives, `u` per chunk.
    pub uni_rhs: Vec<u32>,
}

/// Per-relation queues of edge indices for one epoch.
pub(crate) struct EpochBatcher {
    queues: Vec<Vec<u32>>,
    next: Vec<usize>,
    batch_size: usize,
}

impl EpochBatcher {
    /// Group the edges in `range` (already shuffled) by relation, keeping
    /// their order within each relation.
    pub fn new(
        edges: &EdgeList,
        range: Range<usize>,
        rel: &RelationTable,
        batch_size: usize,
    ) -> Self {
        assert!(edges.len() <= u32::MAX as usize, "edge index overflow");
        let mut queues: Vec<Vec<u32>> = vec![Vec::new(); rel.len()];
        for i in range {
            queues[rel.rel(edges.level[i])].push(i as u32);
        }
        Self {
            next: vec![0; queues.len()],
            queues,
            batch_size: batch_size.max(1),
        }
    }

    /// Edges not yet handed out this epoch.
    pub fn remaining(&self) -> usize {
        self.queues
            .iter()
            .zip(&self.next)
            .map(|(q, &n)| q.len() - n)
            .sum()
    }

    /// The next single-relation batch, or `None` once the epoch is drained.
    pub fn next_batch<R: Rng>(
        &mut self,
        edges: &EdgeList,
        rel: &RelationTable,
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
        let mut r = self.queues.len() - 1;
        for (i, q) in self.queues.iter().enumerate() {
            let rem = q.len() - self.next[i];
            if x < rem {
                r = i;
                break;
            }
            x -= rem;
        }
        let rem = self.queues[r].len() - self.next[r];
        let n_real = rem.min(self.batch_size);
        let idx = &self.queues[r][self.next[r]..self.next[r] + n_real];
        self.next[r] += n_real;

        let k = n_real.div_ceil(c);
        let p = k * c;
        let weight = rel.weights[r];
        let mut lhs = vec![0u32; p];
        let mut rhs = vec![0u32; p];
        let mut row_w = vec![0f32; p];
        let mut col_valid = vec![0f32; p];
        for (j, &e) in idx.iter().enumerate() {
            let e = e as usize;
            lhs[j] = edges.cell[e];
            rhs[j] = edges.gene[e];
            row_w[j] = weight;
            col_valid[j] = 1.0;
        }
        let uni_lhs = (0..k * u)
            .map(|_| rng.random_range(0..edges.n_cells) as u32)
            .collect();
        let uni_rhs = (0..k * u)
            .map(|_| rng.random_range(0..edges.n_genes) as u32)
            .collect();
        Some(PaddedBatch {
            k,
            c,
            u,
            n_real,
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
