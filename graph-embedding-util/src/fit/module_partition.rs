//! The module partition: features grouped by their counts over the finest
//! pseudobulks, built once after the pseudobulk tree and fixed for the whole
//! run. It is a single-level [`coarsen_features`] — the same grouping `senna
//! topic` coarsens its feature axis with, shared through `data-beans`: features
//! with no grouping evidence form one background module, the rest are grouped
//! by k-means on their residual profiles.
//!
//! A membership learned from a cold start is rich-get-richer — the exact module
//! term rewards putting every feature in the module that already scores well —
//! so the partition comes from the data and does not move during training.

use super::config::{ParentModulesOwned, TrackSpec};
use data_beans::alg::feature_coarsening::coarsen_features;
use legume_numeric::matrix::rand_util::mix_seed;
use log::info;
use nalgebra::DMatrix;

/// The base track's rows of a `[n_features × S]` profile, re-keyed by gene:
/// `[n_genes × S]`. The module partition is over GENES, so a multi-track
/// feature axis is reduced to its base track before clustering — the base track
/// is the one the model itself is; every other track is an offset from it. A
/// gene with no base row (never observed on the base track) keeps a zero
/// profile row, which the warm start already routes to its background module.
/// Identity on a one-track axis, where row IS gene.
#[must_use]
pub fn base_track_profile(profile: &DMatrix<f32>, tracks: &TrackSpec) -> DMatrix<f32> {
    debug_assert_eq!(
        profile.nrows(),
        tracks.track_of_row.len(),
        "the profile and the track spec describe the same feature axis"
    );
    let mut out = DMatrix::<f32>::zeros(tracks.n_genes(), profile.ncols());
    for (row, (&t, &g)) in tracks
        .track_of_row
        .iter()
        .zip(&tracks.gene_of_row)
        .enumerate()
    {
        if t == 0 {
            out.set_row(g as usize, &profile.row(row));
        }
    }
    out
}

/// The module partition: a single-level [`coarsen_features`] at `n_modules`,
/// one module id (`< n_modules`) per feature.
pub fn partition_modules(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    n_modules: usize,
    seed: u64,
) -> anyhow::Result<Vec<u32>> {
    let d = counts.nrows();
    if n_modules < 2 || d < 2 {
        return Ok(vec![0; d]);
    }
    let level = coarsen_features(counts, sizes, &[n_modules], seed)?
        .pop()
        .expect("one level requested");
    Ok(level.fine_to_coarse.iter().map(|&g| g as u32).collect())
}

/// Membership logits `[D × M]` for a fit warm-started from a parent: a matched
/// feature takes the parent's membership row verbatim (a simplex point, which
/// sparsemax reproduces exactly), an unmatched one is initialized through the
/// parent's modules from its nearest matched neighbours by profile, or the
/// parent's module-average membership below the similarity floor.
#[must_use]
pub fn parent_module_logits(parent: &ParentModulesOwned, profiles: &DMatrix<f32>) -> DMatrix<f32> {
    use crate::transfer::{align_gene_axis, AlignInputs, ModuleTables};
    let d = parent.row_to_parent.len();
    let m = parent.pi.ncols();
    let al = align_gene_axis(&AlignInputs {
        rho: &parent.rho,
        b_feat: None,
        modules: Some(ModuleTables {
            pi: &parent.pi,
            mu: &parent.mu,
        }),
        new_to_train: &parent.row_to_parent,
        profiles_new: Some(profiles),
        knobs: parent.knobs,
    });
    let membership = al
        .membership
        .as_ref()
        .expect("module tables given ⇒ membership present");
    let mut logits = DMatrix::<f32>::zeros(d, m);
    let (mut n_init, mut n_diffuse) = (0usize, 0usize);
    for g in 0..d {
        let union = al.new_to_union[g].expect("every feature is matched or initialized");
        logits.set_row(g, &membership.row(union));
        if !al.is_scored(union) {
            n_init += 1;
            if al.provenance[union].as_ref().is_some_and(|p| p.diffuse) {
                n_diffuse += 1;
            }
        }
    }
    info!(
        "module warm start from a parent: {} features carry the parent's membership, \
         {n_init} initialized through its modules ({n_diffuse} on the diffuse prior)",
        d - n_init
    );
    logits
}

/// Per modality, whether its features are **module-only**: at least
/// `min_rows` of them. `None` when no modality qualifies or `min_rows` is `0`.
/// `modality` holds one modality id per feature row.
#[must_use]
pub fn module_only_modalities(modality: &[u32], min_rows: usize) -> Option<Vec<bool>> {
    if min_rows == 0 || modality.is_empty() {
        return None;
    }
    let n = modality.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut rows = vec![0usize; n];
    for &m in modality {
        rows[m as usize] += 1;
    }
    let flags: Vec<bool> = rows.iter().map(|&r| r >= min_rows).collect();
    flags.iter().any(|&f| f).then_some(flags)
}

/// [`partition_modules`] confined to GROUPS of rows: group `k`'s rows are
/// partitioned on their own into at most `n_per[k]` modules, numbered after the
/// previous groups'. Returns the labels and the total module count. Groups are
/// modalities, so one modality can drop its residual while the others keep
/// theirs.
pub fn partition_modules_by_group(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    group_of_row: &[u32],
    n_per: &[usize],
    seed: u64,
) -> anyhow::Result<(Vec<u32>, usize)> {
    let mut labels = vec![0u32; group_of_row.len()];
    let mut offset = 0usize;
    for (k, &n_k) in n_per.iter().enumerate() {
        let rows: Vec<usize> = (0..group_of_row.len())
            .filter(|&i| group_of_row[i] as usize == k)
            .collect();
        if !rows.is_empty() {
            let sub =
                DMatrix::<f32>::from_fn(rows.len(), counts.ncols(), |i, j| counts[(rows[i], j)]);
            let local = partition_modules(&sub, sizes, n_k, mix_seed(seed, k as u64))?;
            for (&r, &m) in rows.iter().zip(&local) {
                labels[r] = offset as u32 + m;
            }
        }
        offset += n_k;
    }
    Ok((labels, offset))
}

#[cfg(test)]
mod tests;
