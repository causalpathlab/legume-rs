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

/// Inputs for [`FrozenProjector::new`].
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

/// A frozen feature side with its projection design already built — the entry
/// point for projecting nodes that are not part of a fit (`senna predict` on a bge
/// model, and any future streaming caller).
///
/// # Why a projector and not a function
///
/// Everything the block SGD needs that depends only on `(feat, b_feat)` — the
/// augmented design in both orientations, the live-feature scan and gate fold, the
/// null normalizer, the auto-scaled learning rate — is built by [`Self::new`] and
/// reused by every [`Self::project`] call. A per-call function rebuilt all of it
/// per call, which pushed the cost of a small call onto the *caller*: it had to
/// hand over enough nodes to hide the setup, and the only way to know "enough" was
/// to guess against constants inside the engine it could not see. There is nothing
/// left to hide, so a caller sizes its groups for its own memory — and can ask
/// [`Self::group_nodes`] for a size derived from the block budget rather than
/// guessing at it.
///
/// # Why this solver
///
/// A run's own `cell_embedding.parquet` comes from `project_cells_phase2`, i.e.
/// this same block SGD: Adam over cell blocks against the **full** log-partition.
/// The Newton/IRLS path in [`crate::cell_projection`] fits a node's observed
/// features only and lets a ridge stand in for the partition, which is why the
/// per-cell phase 2 was moved off it — `‖θ‖` runs away. Projecting held-out cells
/// with the other solver would put train and test latents in different spaces and
/// quietly confound any train/test comparison built on them. One estimator, both
/// halves.
///
/// `gauge_fix` is **off** here, unlike training: the gauge shift is a
/// (θ, b_feat) pair, and `b_feat` is frozen at predict time — re-centring θ alone
/// would break its correspondence with the dictionary it is scored against.
pub struct FrozenProjector<'a> {
    feat: &'a [f32],
    b_feat: &'a [f32],
    h: usize,
    lambda: f64,
    dev: &'a Device,
    dict: block_sgd::PassDict,
}

impl<'a> FrozenProjector<'a> {
    /// Build the projection design for a frozen dictionary. Does the whole
    /// per-dictionary setup once; [`Self::project`] then costs only what its own
    /// nodes cost.
    pub fn new(args: &FrozenProjectionArgs<'a>) -> anyhow::Result<Self> {
        let n_features = args.b_feat.len();
        anyhow::ensure!(
            args.feat.len() == n_features * args.h,
            "frozen projection: the dictionary has {} entries, expected {n_features} × {}",
            args.feat.len(),
            args.h
        );
        // The exact normalizer: every feature of the frozen side is in the partition.
        let rows: Vec<u32> = (0..n_features as u32).collect();
        let dict = block_sgd::PassDict::build(
            &block_sgd::DictSpec {
                feat: args.feat,
                b_feat: args.b_feat,
                h: args.h,
                lambda: args.lambda,
                dev: args.dev,
                label: "Projection",
                pass: "nodes",
            },
            rows,
        )?;
        Ok(Self {
            feat: args.feat,
            b_feat: args.b_feat,
            h: args.h,
            lambda: args.lambda,
            dev: args.dev,
            dict,
        })
    }

    /// Nodes to hand one [`Self::project`] call.
    ///
    /// A whole number of solver blocks, derived from the activation budget this
    /// crate owns — so a streaming caller that cuts its groups here fills every
    /// block exactly instead of guessing a size in a currency the engine never
    /// sees. It bounds host memory too: a node carries at most `F` nonzeros, and
    /// `block_cells × F` is what the activation budget caps, so a group's nonzero
    /// count does not grow with the feature axis.
    pub fn group_nodes(&self) -> usize {
        self.dict.group_nodes()
    }

    /// Project one group of nodes, advancing `bar` by one tick per node as the
    /// blocks step.
    ///
    /// `nodes` is `(group-local id, feature ids, counts)` and `n_nodes` bounds that
    /// local axis; ids are positions in the returned `[n_nodes × h]` θ, so a caller
    /// streaming groups re-bases them per group and stitches the results itself.
    ///
    /// The bar is the **caller's**, not one per call: a streaming caller already has
    /// a bar over the whole query, and a second one underneath it would either nest
    /// or fight it for the terminal.
    pub fn project(
        &self,
        nodes: &[(u32, &[u32], &[f32])],
        n_nodes: usize,
        bar: &indicatif::ProgressBar,
    ) -> anyhow::Result<FrozenProjection> {
        let out = block_sgd::project_prepared(
            &block_sgd::Phase2Input {
                feat: self.feat,
                b_feat: self.b_feat,
                h: self.h,
                n_cells: n_nodes,
                lambda: self.lambda,
                dev: self.dev,
                label: "Projection",
                gauge_fix: false,
                joint: false,
            },
            &self.dict,
            nodes,
            bar,
        )?;
        Ok(FrozenProjection {
            theta: out.theta,
            b_node: out.b_cell,
        })
    }
}
