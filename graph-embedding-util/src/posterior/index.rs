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

/// Owned frozen index: the fixed other side plus per-anchor observed edges and the
/// shared negative slate.
///
/// A plain container. The samplers do NOT read the fixed side through an accessor here
/// — each builds its own [`super::lnpdf::FrozenSide`] literal over whatever arrays it
/// holds, which for the pb sweeps is usually not a `ContrastiveIndex` at all. An
/// accessor existed for a while and never had a caller.
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
