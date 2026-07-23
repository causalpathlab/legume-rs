//! Phase-2 node projection onto the frozen feature dictionary. The block
//! Poisson-MAP SGD engine ([`block_sgd`]) is shared by two callers, split by what
//! they project: [`cells`] (per-cell Phase 2 → `e_cell`) and [`pseudobulk`]
//! (per-pb velocity readout → `θ_pb`/`δ_pb` landmarks). This root holds only what
//! both share — the ridge, the batch divisor, and the per-cell edge fold the
//! engine calls back into.

use matrix_util::dmatrix_util::adjust_by_poisson_ratio;
use nalgebra::DMatrix;

mod block_sgd;
mod cells;
mod pseudobulk;

pub(crate) use cells::project_cells_phase2;
pub(crate) use pseudobulk::project_pbs_phase2;
pub use pseudobulk::PbLevelVelocity;

/// Ridge prior strength λ on `e_cell` in the phase-2 projection.
///
/// A **mild** Gaussian prior, not a load-bearing bound: the block SGD
/// ([`block_sgd`]) sums the log-partition over every feature, which is what
/// identifies `θ`. The held-out-gene solve in [`crate::cell_projection`] still
/// fits observed features only, and there this same λ *is* the only thing standing
/// in for the partition.
pub(crate) const PHASE2_RIDGE: f32 = 1.0;

/// Phase-2 batch correction, mirroring `senna svd`/`topic`: divide each cell's
/// counts by its finest-pseudobulk `μ_residual` fold-factor before the
/// Poisson-MAP projection, so `e_cell` fits the de-batched signal. Built only
/// when the collapse fit a `μ_residual` (>1 batch); a no-op otherwise.
#[derive(Clone, Copy)]
pub(crate) struct CellBatchDivisor<'a> {
    /// `[n_features × n_pb]` batch fold-factor on the **unified** feature axis,
    /// so a cell's feature id indexes a row directly (no remap).
    pub mu_residual: &'a DMatrix<f32>,
    /// Cell id → finest-pseudobulk id (the `μ_residual` column to divide by).
    pub cell_to_pb: &'a [usize],
}

/// Divide one cell's `(feature, count)` edges by its pseudobulk batch fold-factor,
/// reusing matrix-util's [`adjust_by_poisson_ratio`] — the same self-normalizing
/// divide (`λ = Σx/Σd`, depth preserved for `b_cell`) `senna svd`/`topic` apply via
/// the `CscMatrix` trait, here straight on the cell's counts (no per-cell CSC).
/// `feats` index `μ_residual` rows directly.
fn adjust_cell_edges(
    feats: &[u32],
    counts: &[f32],
    pb: usize,
    mu_residual: &DMatrix<f32>,
) -> Vec<(u32, f32)> {
    let mut vals = counts.to_vec();
    adjust_by_poisson_ratio(&mut vals, |k| mu_residual[(feats[k] as usize, pb)]);
    feats.iter().copied().zip(vals).collect()
}

/// One node's `(feature, count)` edges, batch-divided by its pseudobulk
/// fold-factor when correction is on, else the raw edges. Called back by the block
/// SGD ([`block_sgd`]) as it flattens each node's edges. (The pb readout passes no
/// divisor — its aggregates are already batch-corrected.)
pub(crate) fn cell_edges(
    cell: u32,
    feats: &[u32],
    counts: &[f32],
    batch_divisor: Option<CellBatchDivisor>,
) -> Vec<(u32, f32)> {
    match batch_divisor {
        Some(bd) => adjust_cell_edges(feats, counts, bd.cell_to_pb[cell as usize], bd.mu_residual),
        None => feats.iter().copied().zip(counts.iter().copied()).collect(),
    }
}
