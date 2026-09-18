//! Stratified multilevel collapse: expression grouping cannot cross CNV
//! clone boundaries.
//!
//! Thin wrapper: sets [`MultilevelParams::strata`] and forwards to
//! [`collapse_columns_multilevel_with_hierarchy`]. Stratum bits are crossed
//! into finest codes, BBKNN matches only within stratum, and unmatched
//! (clone-only) observed mass is excluded from the δ update so private CN
//! stays in `mu_adjusted`; `mu_residual` on unmatched clones tracks δ.

use super::*;

/// Collapse with a hard CNV stratum partition on cells.
///
/// `cell_to_stratum[c]` is the clone id of global column `c` (`0` = mixable
/// residual). Requires `MultilevelParams.refine = Some(..)` — the same
/// contract as [`collapse_columns_multilevel_with_hierarchy`].
pub fn collapse_columns_multilevel_with_strata<T>(
    data_vec: &mut SparseIoVec,
    proj_kn: &DMatrix<f32>,
    batch_membership: &[T],
    params: &MultilevelParams,
    cell_to_stratum: &[usize],
) -> anyhow::Result<MultilevelCollapseOut>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    let n = data_vec.num_columns();
    anyhow::ensure!(
        cell_to_stratum.len() == n,
        "cell_to_stratum has {} entries, data has {} columns",
        cell_to_stratum.len(),
        n
    );
    anyhow::ensure!(
        proj_kn.ncols() == n,
        "proj has {} columns, data has {}",
        proj_kn.ncols(),
        n
    );
    anyhow::ensure!(
        batch_membership.len() == n,
        "batch_membership has {} entries, data has {} columns",
        batch_membership.len(),
        n
    );
    anyhow::ensure!(
        params.refine.is_some(),
        "collapse_columns_multilevel_with_strata requires \
         MultilevelParams.refine = Some(..)"
    );

    let local = params.with_strata(cell_to_stratum.to_vec());
    collapse_columns_multilevel_with_hierarchy(data_vec, proj_kn, batch_membership, &local)
}
