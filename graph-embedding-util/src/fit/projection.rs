//! Phase-2 node projection onto the frozen feature dictionary. The block
//! Poisson-MAP SGD engine ([`block_sgd`]) is shared by two callers, split by what
//! they project: [`cells`] (per-cell Phase 2 → `e_cell`) and [`pseudobulk`]
//! (per-pb velocity readout → `θ_pb`/`δ_pb` landmarks). This root holds only what
//! both share — the ridge, the batch divisor, and the per-cell edge fold the
//! engine calls back into.

use candle_util::candle_core::Device;
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
pub const PHASE2_RIDGE: f32 = 1.0;

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

/////////////////////////////////////////
// Projecting onto a frozen dictionary //
/////////////////////////////////////////

/// Inputs for [`project_onto_frozen`].
pub struct FrozenProjectionArgs<'a> {
    /// Frozen dictionary, row-major `[n_features × h]`.
    pub feat: &'a [f32],
    /// Frozen per-feature bias, `[n_features]`.
    pub b_feat: &'a [f32],
    pub h: usize,
    /// Ridge on the node latent. Pass [`PHASE2_RIDGE`] to match training.
    pub lambda: f64,
    pub dev: &'a Device,
}

/// `(θ [n × h] row-major, b_node [n])`.
pub struct FrozenProjection {
    pub theta: Vec<f32>,
    pub b_node: Vec<f32>,
}

/// Project nodes onto a **frozen** feature side with the same block SGD phase 2
/// uses, for callers outside the fit — `senna predict` on a bge model.
///
/// # Why not the Newton solver in [`crate::cell_projection`]
///
/// A run's own `cell_embedding.parquet` comes from `project_cells_phase2`, i.e.
/// this solver: Adam over cell blocks against the **full** log-partition. The
/// Newton/IRLS path fits a node's observed features only and lets a ridge stand
/// in for the partition, which is why the per-cell phase 2 was moved off it —
/// `‖θ‖` runs away. Projecting held-out cells with the other solver would put
/// train and test latents in different spaces and quietly confound any
/// train/test comparison built on them. One estimator, both halves.
///
/// `gauge_fix` is **off** here, unlike training: the gauge shift is a
/// (θ, b_feat) pair, and `b_feat` is frozen at predict time — re-centring θ
/// alone would break its correspondence with the dictionary it is scored
/// against.
pub fn project_onto_frozen(
    args: &FrozenProjectionArgs<'_>,
    nodes: &[(u32, &[u32], &[f32])],
    n_nodes: usize,
) -> anyhow::Result<FrozenProjection> {
    let out = block_sgd::project_cells(
        &block_sgd::Phase2Input {
            feat: args.feat,
            b_feat: args.b_feat,
            h: args.h,
            n_cells: n_nodes,
            lambda: args.lambda,
            dev: args.dev,
            label: "Projection",
            gauge_fix: false,
            joint: false,
        },
        nodes,
        None,
        None,
    )?;
    Ok(FrozenProjection {
        theta: out.theta,
        b_node: out.b_cell,
    })
}
