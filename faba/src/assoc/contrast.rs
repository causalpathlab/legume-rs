//! Pseudotime binning + per-branch rate profiles shared by the association outputs.
//!
//! The between-branch statistics themselves are Bayesian (see [`super::contrast_bayes`]);
//! this module holds the two small helpers they share: [`bin_pseudotime`] slices the common
//! pseudotime axis into equal-width strata (the matched grouping the contrast conditions on),
//! and [`site_profile`] pseudobulks a site's editing rate per (branch, bin) for plotting.

use super::io::Site;

/// Equal-width pseudotime bins → per-cell bin id, clamped to `0..n_bins`.
pub(crate) fn bin_pseudotime(pt: &[f32], n_bins: usize) -> Vec<u32> {
    let (lo, hi) = pt
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let span = (hi - lo).max(1e-9);
    pt.iter()
        .map(|&x| {
            let b = ((x - lo) / span * n_bins as f32).floor() as i64;
            b.clamp(0, n_bins as i64 - 1) as u32
        })
        .collect()
}

/// Per-(branch, bin) pseudobulk `(branch, bin, K, N)` for one site (non-empty cells only) —
/// the editing-rate profile along pseudotime, for `branch_profile.parquet` / plotting.
pub fn site_profile(
    site: &Site,
    bins: &[u32],
    branch: &[usize],
    n_bins: usize,
    n_branches: usize,
) -> Vec<(usize, usize, u64, u64)> {
    let mut aggk = vec![vec![0u64; n_branches]; n_bins];
    let mut aggn = vec![vec![0u64; n_branches]; n_bins];
    for c in 0..site.n.len() {
        if site.n[c] > 0 {
            let (b, l) = (bins[c] as usize, branch[c]);
            aggk[b][l] += site.k[c] as u64;
            aggn[b][l] += site.n[c] as u64;
        }
    }
    let mut out = Vec::new();
    for b in 0..n_bins {
        for l in 0..n_branches {
            if aggn[b][l] > 0 {
                out.push((l, b, aggk[b][l], aggn[b][l]));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
