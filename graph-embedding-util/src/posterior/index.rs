//! The frozen contrastive index: one side held fixed, per-anchor observed edges,
//! and the shared negative slate the rate normalizer is summed over.
//!
//! This is a container, not a builder. The only builder is
//! [`super::pb_index::build_pb_index_pair`], which constructs BOTH orientations
//! over the **pseudobulk** matrix. An earlier cell-anchored builder — anchors =
//! genes, other side = the full cell embedding — was retired: MCMC in this crate
//! runs at the pseudobulk level, and keeping a second sampler that did not would
//! invite it back.
//!
//! `partition_scale = n_other / |slate|` folds a sampled slate back up to the full
//! Poisson normalizer; a slate covering the whole other side has scale `1`.

use super::lnpdf::FrozenSide;

/// Owned frozen index: the fixed other side plus per-anchor observed edges and
/// the shared negative slate. A view of the fixed side is handed to the samplers
/// via [`Self::frozen_side`].
pub struct ContrastiveIndex {
    /// Frozen other-side embeddings `[n_other × h]` row-major.
    pub other_e: Vec<f32>,
    /// Frozen other-side biases `[n_other]`.
    pub other_b: Vec<f32>,
    pub h: usize,
    /// Per-anchor observed `(other-index, count)` edges.
    pub pos: Vec<Vec<(u32, f32)>>,
    /// Per-anchor fixed bias, carried for a fixed-intercept consumer. The profile
    /// likelihood the samplers use maximizes it out, so it is unused there.
    pub anchor_b: Vec<f32>,
    /// Frozen negative slate of other-indices, shared across anchors. It must not
    /// move between sweeps.
    pub partition: Vec<u32>,
    /// `n_other / |partition|` — folds the sampled slate up to the full sum.
    pub partition_scale: f64,
    /// Optional `[n_anchors × h]` row-major frozen directions, one per anchor,
    /// handed to each [`super::lnpdf::NodeTerm`] as its `offset` (see that field).
    /// `None` for the plain case where the sampler explores an absolute loading.
    pub anchor_offset: Option<Vec<f32>>,
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

    /// Number of anchors.
    #[must_use]
    pub fn n_anchors(&self) -> usize {
        self.pos.len()
    }
}
