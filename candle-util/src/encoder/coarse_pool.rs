//! Reading a cell through a FIXED coarse grouping instead of a per-gene table.
//!
//! The masked encoder's feature side is a `[D, H]` table: one trained row per
//! gene. That makes the encoder the one part of the model that cannot read a
//! gene it has never seen, which is why continuing a fit onto a new gene axis
//! has to invent rows for the gained genes.
//!
//! Here the encoder's table is `[C, H]`, one row per COARSE GROUP, and a gene
//! reaches it through the grouping the data already fixed
//! ([`CoarseningMap`]). Two consequences, and they are the whole point:
//!
//! - Nothing the encoder learns is indexed by gene, so a gained gene needs no
//!   new parameter. It contributes to whichever group its profile put it in.
//! - Every observed gene contributes, so the top-K context cap is unnecessary:
//!   a cell's read is a `[N, C]` profile, whatever its support.
//!
//! This is NOT a module: the membership is fixed by the data before training
//! and is never a `Var`. The learned part is the group table and nothing else.
//!
//! The decoder is deliberately left at gene resolution. The encoder is where a
//! gene-keyed weight blocks growth, because it must read genes it has not seen;
//! the decoder must still predict a specific gene, and its table is gathered by
//! name when the axis changes.

use crate::decoder::coarsening_map::CoarseningMap;
use crate::fast_index::scatter_add_cols;
use candle_core::{Result, Tensor};

/// A cell's per-group profile `[N, C]`: each visible slot's gated value added
/// into its gene's group. Masked and padding slots contribute nothing, so the
/// value being imputed cannot leak into the read.
///
/// The mask is applied to the GATE rather than to the scatter, so a hidden slot
/// adds an exact zero to its group instead of being routed somewhere harmless:
/// there is no "somewhere harmless" in a sum.
pub fn coarse_profile(
    indices: &Tensor,
    gate: &Tensor,
    visible_mask: &Tensor,
    map: &CoarseningMap,
) -> Result<Tensor> {
    let groups = map.groups_of(indices)?;
    let visible_gate = (gate * visible_mask)?;
    scatter_add_cols(&groups, &visible_gate, map.n_coarse())
}

/// Pool a `[N, C]` profile against the `[C, H]` group table.
///
/// One matmul. The attention pool the per-gene path uses has no counterpart
/// here: there are no slots left to attend over once every gene has been added
/// into its group. An empty read pools to exactly zero, which is what leaves a
/// fully masked cell to the downstream bias rather than to padding content.
pub fn pool_groups(profile: &Tensor, group_table: &Tensor) -> Result<Tensor> {
    profile.matmul(group_table)
}

#[cfg(test)]
#[path = "coarse_pool_tests.rs"]
mod coarse_pool_tests;
