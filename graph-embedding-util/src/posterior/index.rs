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

use super::lnpdf::{FrozenSide, NodeTerm};

/// Owned frozen index: the fixed other side plus per-anchor observed edges and
/// the shared negative slate. Views into it are handed to the samplers via
/// [`Self::frozen_side`] / [`Self::node_terms`].
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
    /// handed to each [`NodeTerm`] as its `offset` (see that field). `None` for
    /// the plain case where the sampler explores an absolute loading.
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

    /// One [`NodeTerm`] per anchor (all sharing the frozen slate), each carrying
    /// its [`Self::anchor_offset`] row when one is set.
    #[must_use]
    pub fn node_terms(&self) -> Vec<NodeTerm<'_>> {
        self.pos
            .iter()
            .enumerate()
            .map(|(a, pos)| NodeTerm {
                pos,
                partition: &self.partition,
                partition_scale: self.partition_scale,
                offset: self
                    .anchor_offset
                    .as_ref()
                    .map(|off| &off[a * self.h..(a + 1) * self.h]),
                moment: None,
            })
            .collect()
    }

    /// Number of anchors.
    #[must_use]
    pub fn n_anchors(&self) -> usize {
        self.pos.len()
    }

    /// Anchors carrying at least one observed count, in anchor order.
    ///
    /// The complement is not a subsample. An anchor with no counts makes
    /// [`super::lnpdf::multinomial_ll`] return a constant, so its posterior IS its
    /// prior and sampling it redraws something already known.
    #[must_use]
    pub fn informative_anchors(&self) -> Vec<usize> {
        (0..self.n_anchors())
            .filter(|&a| !self.pos[a].is_empty())
            .collect()
    }

    #[must_use]
    pub fn n_empty_anchors(&self) -> usize {
        self.pos.iter().filter(|p| p.is_empty()).count()
    }
}
