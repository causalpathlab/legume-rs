//! Build the frozen contrastive index from a real count backend + a fitted
//! (MAP) cell side, so the posterior samplers run on actual data — not just the
//! synthetic fixtures.
//!
//! For the feature/gate side, the anchor is a **gene** and the frozen other side
//! is the **cell** embedding. [`build_gene_index`] streams the count backend once
//! (the same slab / `for_each_triplet` path `fit`'s cell sampler uses), buckets
//! nonzeros by gene into per-gene `(cell, count)` edges, and draws one **frozen
//! negative slate** of cells shared across genes (Trap 1 — the slate must not move
//! between sweeps). `partition_scale = n_cells / |slate|` folds the sampled rate
//! sum back up to the full Poisson normalizer; pass `n_partition = 0` for the
//! exact all-cells partition (small data only).

use super::lnpdf::{FrozenSide, NodeTerm};
use crate::data::UnifiedData;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Owned frozen index: the fixed other side (cells) plus per-anchor (gene)
/// observed edges and the shared negative slate. Views into it are handed to the
/// samplers via [`Self::frozen_side`] / [`Self::node_terms`].
pub struct ContrastiveIndex {
    /// Frozen other-side embeddings `[n_other × h]` row-major (the MAP cell side).
    pub other_e: Vec<f32>,
    /// Frozen other-side biases `[n_other]`.
    pub other_b: Vec<f32>,
    pub h: usize,
    /// Per-anchor observed `(other-index, count)` edges (`pos[g]` = gene `g`'s cells).
    pub pos: Vec<Vec<(u32, f32)>>,
    /// Per-anchor fixed bias `b_g` (held at the MAP for the gate sweep).
    pub anchor_b: Vec<f32>,
    /// Frozen negative slate of other-indices (cells), shared across anchors.
    pub partition: Vec<u32>,
    /// `n_other / |partition|` — folds the sampled slate up to the full sum.
    pub partition_scale: f64,
}

impl ContrastiveIndex {
    /// The frozen other side as a borrowing [`FrozenSide`].
    #[must_use]
    pub fn frozen_side(&self) -> FrozenSide<'_> {
        FrozenSide {
            e: &self.other_e,
            b: &self.other_b,
            h: self.h,
        }
    }

    /// One [`NodeTerm`] per anchor (all sharing the frozen slate).
    #[must_use]
    pub fn node_terms(&self) -> Vec<NodeTerm<'_>> {
        self.pos
            .iter()
            .map(|pos| NodeTerm {
                pos,
                partition: &self.partition,
                partition_scale: self.partition_scale,
            })
            .collect()
    }

    /// Number of anchors (genes).
    #[must_use]
    pub fn n_anchors(&self) -> usize {
        self.pos.len()
    }
}

/// Stream the count backend and build the per-gene contrastive index against a
/// frozen cell side.
///
/// `e_cell` is `[n_cells × h]` row-major and `b_cell` `[n_cells]` — the MAP cell
/// embedding to hold fixed; `b_feat` `[n_features]` is the MAP feature bias. When
/// `n_partition > 0` a frozen slate of that many cells is drawn once (seeded by
/// `seed`) and its scale set to `n_cells / n_partition`; `n_partition == 0` uses
/// every cell exactly (`scale = 1`).
pub fn build_gene_index(
    unified: &UnifiedData,
    e_cell: &[f32],
    b_cell: &[f32],
    b_feat: &[f32],
    h: usize,
    n_partition: usize,
    seed: u64,
) -> anyhow::Result<ContrastiveIndex> {
    let data = unified.count_backend();
    let n_cells = data.num_columns();
    let n_features = unified.n_features();
    let backend_rows = data.num_rows();

    // backend row → unified feature id (u32::MAX ⇒ dropped by a subset).
    let mut backend_to_unified = vec![u32::MAX; backend_rows];
    for (uid, &brow) in unified.feature_to_backend_row.iter().enumerate() {
        if brow < backend_rows {
            backend_to_unified[brow] = uid as u32;
        }
    }

    // Slab width ~8M edges (mirrors `fit`'s cell sampler); fall back to a fixed
    // cell-count slab when nnz can't be reported.
    let chunk = match data.num_non_zeros() {
        Ok(nnz) if nnz > 0 => {
            let avg = (nnz / n_cells.max(1)).max(1);
            (8_000_000 / avg).clamp(1, n_cells.max(1))
        }
        _ => (1usize << 14).min(n_cells.max(1)),
    };

    // Bucket nonzeros by gene. Passing `0..n_cells` makes `out_col` the global
    // cell id directly.
    let mut pos: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_features];
    data.for_each_triplet(0..n_cells, chunk, |brow, out_col, v| {
        if v == 0.0 {
            return;
        }
        let uid = backend_to_unified[brow as usize];
        if uid == u32::MAX {
            return;
        }
        pos[uid as usize].push((out_col as u32, v));
    })?;

    // Frozen negative slate (Trap 1): drawn once, shared across genes.
    let (partition, partition_scale) = if n_partition == 0 || n_partition >= n_cells {
        ((0..n_cells as u32).collect(), 1.0)
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut all: Vec<u32> = (0..n_cells as u32).collect();
        all.partial_shuffle(&mut rng, n_partition);
        all.truncate(n_partition);
        (all, n_cells as f64 / n_partition as f64)
    };

    Ok(ContrastiveIndex {
        other_e: e_cell.to_vec(),
        other_b: b_cell.to_vec(),
        h,
        pos,
        anchor_b: b_feat.to_vec(),
        partition,
        partition_scale,
    })
}
