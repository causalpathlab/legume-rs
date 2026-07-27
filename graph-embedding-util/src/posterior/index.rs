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
use crate::cell_projection::SCORE_CLAMP;
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

    /// Re-fit each anchor's bias so the **null** model reproduces that anchor's
    /// observed total count, and report how far the fitted model had to move.
    ///
    /// The frozen `anchor_b` arrives from an **NCE**-trained fit, where a bias
    /// absorbs a feature's marginal frequency only up to the objective's own
    /// normalization — it is not a Poisson log-rate. The samplers here *are*
    /// Poisson, and they hold the bias fixed while sampling the loading, so an
    /// uncalibrated intercept does not stay a harmless offset: with the rate
    /// term too small, `Σ_pos n·s` dominates the likelihood and every anchor is
    /// pushed toward the same count-weighted mean direction of the frozen side —
    /// a posterior that collapses onto one or two dims regardless of the data.
    ///
    /// At `θ = 0` the rate sum is `exp(b_a) · scale · Σ_{o ∈ partition} exp(b_o)`,
    /// so matching it to `T_a = Σ_pos n` is closed-form:
    ///
    /// ```text
    ///   b_a*  =  ln(T_a)  −  ln( scale · Σ_o exp(b_o) )
    /// ```
    ///
    /// The right-hand term is shared by every anchor, so this is one pass over
    /// the frozen side plus `O(1)` per anchor. An anchor with no counts has no
    /// rate to match and is parked at `-SCORE_CLAMP` (rate ≈ 0), which is what a
    /// never-observed feature should predict.
    ///
    /// This makes the null model exact per anchor, which is also what the
    /// spike-and-slab comparison in [`super::hyper_ss`] needs: its `ℓ(0)` is the
    /// baseline every inclusion decision is measured against.
    pub fn calibrate_anchor_bias(&mut self) -> BiasCalibration {
        // f64 + max-shift: `exp(b_o)` over the whole slate is exactly the sum
        // that loses its low bits when the biases spread.
        let b_max = self
            .other_b
            .iter()
            .filter(|b| b.is_finite())
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let log_part = if b_max.is_finite() {
            let s: f64 = self
                .partition
                .iter()
                .map(|&o| f64::from(self.other_b[o as usize] - b_max).exp())
                .sum();
            f64::from(b_max) + (self.partition_scale * s).max(f64::MIN_POSITIVE).ln()
        } else {
            0.0
        };

        let mut shifts: Vec<f32> = Vec::with_capacity(self.pos.len());
        let mut n_empty = 0usize;
        for (a, pos) in self.pos.iter().enumerate() {
            let total: f64 = pos.iter().map(|&(_, n)| f64::from(n)).sum();
            let b_new = if total > 0.0 {
                (total.ln() - log_part).clamp(-SCORE_CLAMP, SCORE_CLAMP) as f32
            } else {
                n_empty += 1;
                -SCORE_CLAMP as f32
            };
            shifts.push(b_new - self.anchor_b[a]);
            self.anchor_b[a] = b_new;
        }

        let mut abs: Vec<f32> = shifts.iter().map(|s| s.abs()).collect();
        abs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        BiasCalibration {
            median_abs_shift: abs.get(abs.len() / 2).copied().unwrap_or(0.0),
            max_abs_shift: abs.last().copied().unwrap_or(0.0),
            n_empty,
        }
    }
}

/// How far [`ContrastiveIndex::calibrate_anchor_bias`] had to move the frozen
/// biases — i.e. how far the trained model was from being Poisson-calibrated.
/// A shift of a few nats is routine (different objective); tens of nats means
/// the frozen intercepts carried essentially no rate information.
#[derive(Clone, Copy, Debug)]
pub struct BiasCalibration {
    /// Median `|b* − b|` in nats over all anchors.
    pub median_abs_shift: f32,
    /// Largest `|b* − b|` over all anchors.
    pub max_abs_shift: f32,
    /// Anchors with no observed counts (parked at a ≈0 rate).
    pub n_empty: usize,
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

#[cfg(test)]
#[path = "index_tests.rs"]
mod index_tests;
