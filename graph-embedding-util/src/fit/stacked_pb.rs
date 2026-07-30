//! The stacked pseudobulk view: every collapse level's counts and embedding
//! concatenated onto one axis, on the **count** scale.
//!
//! Lived in `fit/feature_projection` until that module was removed. It has nothing to do
//! with held-out projection — it is the shared pseudobulk frame, and its remaining
//! consumers are `fit::stacked_pb_view`, which builds it out of the trained Vars, and the
//! posterior sampler, which reads it as the other side of every block.

use nalgebra::DMatrix;

/// Every collapse level's `θ_pb` / `b_pb` concatenated into one frozen table, with the
/// matching count matrices kept on the **full backend** feature axis.
///
/// **Exposure — the part that is easy to get wrong.** The collapse emits Gamma-posterior
/// *rates* (per-cell means), not counts. A rate has `Var(n) = μ / size_p`, so a Poisson fit
/// to it is a quasi-Poisson with a per-column dispersion: its deviance scales with the
/// pseudobulk's cell count and with the gene's expression, and is therefore *not*
/// `χ²`-calibrated. Measured on real data that broke a null gate outright —
/// `Spearman(LRT, detection) = +0.60`, lower quantiles collapsed to zero, `σ̂² → 0`, and 59%
/// of dropped genes called live against an estimated `π̂₀ = 0.81`.
///
/// So this view converts to the count scale: an edge carries `rate · size_p`, and `bias`
/// carries `b_pb + log(size_p)` — the standard Poisson exposure offset. The modelled rate
/// `exp(⟨β_g, θ_p⟩ + b_p + b_g)` is unchanged, so the frame still matches training; only
/// the likelihood's scale is now correct.
///
/// That matters most to the spike-and-slab posterior, whose whole output is `σ₀h²` and
/// `π₀h` — precisely the quantities a mis-scaled likelihood collapses.
pub(crate) struct StackedPb<'a> {
    /// `[Σ n_pb × H]` row-major, levels concatenated in `counts` order.
    pub theta: Vec<f32>,
    /// `[Σ n_pb]`, same order. Already includes the `log(size_p)` exposure offset.
    pub bias: Vec<f32>,
    /// One `[backend_rows × n_pb^(l)]` rate matrix per level.
    pub counts: Vec<&'a DMatrix<f32>>,
    /// Cells per pseudobulk, per level — the exposure. Aligned with `counts`.
    pub sizes: Vec<Vec<f32>>,
    /// `offsets[l]` = global pb index of level `l`'s first column.
    pub offsets: Vec<usize>,
}

impl StackedPb<'_> {
    /// Total pseudobulk columns across every level — the length of the stacked `bias` axis,
    /// and the size of the other side a pb-anchored sampler sees.
    pub(crate) fn n_pb_total(&self) -> usize {
        self.bias.len()
    }
}
