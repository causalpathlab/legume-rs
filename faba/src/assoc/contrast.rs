//! Counterfactual between-branch contrast, stratified by pseudotime.
//!
//! Per site s and branch L (one-vs-rest), we test whether editing differs between
//! branch L and the other-fate cells **at matched pseudotime**. Binning pseudotime
//! and taking, per bin, the 2×2 table [edited/unedited × L/rest] gives a
//! Cochran–Mantel–Haenszel setup: the statistic
//!   T_{s,L} = Σ_bin ( K_{L,bin} − N_{L,bin}·p̂_bin )
//! is the pseudotime-adjusted excess editing in branch L (observed − expected under
//! "branch independent of editing within a bin"). In the shared trunk the branches
//! coincide → T≈0; divergence only appears after the branch point. Calibration
//! permutes branch labels **within pseudotime bins** (preserving pseudotime and
//! coverage), so the null cannot manufacture divergence from the confound.

use rand::rngs::SmallRng;
use rand::SeedableRng;

use enrichment::null::permute_indices;

use super::io::{Lineage, Site};

pub struct AssocConfig {
    pub n_bins: usize,
    pub num_perm: usize,
    pub min_total_coverage: u64,
    pub min_cells: usize,
    pub seed: u64,
}

pub struct BranchResult {
    pub site: usize,
    pub branch: usize,
    pub n_cells: usize,
    pub total_cov: u64,
    /// Signed CMH numerator (pseudotime-adjusted excess edited in the branch).
    pub stat: f32,
    /// Adjusted rate difference proxy = stat / branch coverage.
    pub effect: f32,
    pub p_perm: f32,
}

/// Equal-width pseudotime bins → per-cell bin id.
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

/// One-vs-rest CMH numerators for every branch of one site, in a single pass over
/// the site's covered cells. `num[L]` = Σ_bin (K_{L,bin} − expected).
fn cmh_nums(
    site: &Site,
    covered: &[usize],
    bins: &[u32],
    branch: &[usize],
    n_bins: usize,
    n_branches: usize,
) -> Vec<f32> {
    let mut aggk = vec![vec![0u64; n_branches]; n_bins];
    let mut aggn = vec![vec![0u64; n_branches]; n_bins];
    for &c in covered {
        let (b, l) = (bins[c] as usize, branch[c]);
        aggk[b][l] += site.k[c] as u64;
        aggn[b][l] += site.n[c] as u64;
    }
    let mut num = vec![0f32; n_branches];
    for b in 0..n_bins {
        let sumk: u64 = aggk[b].iter().sum();
        let sumn: u64 = aggn[b].iter().sum();
        if sumn == 0 {
            continue;
        }
        let p_hat = sumk as f32 / sumn as f32;
        for l in 0..n_branches {
            let n1 = aggn[b][l];
            let n0 = sumn - n1;
            if n1 == 0 || n0 == 0 {
                continue; // no contrast in this bin
            }
            num[l] += aggk[b][l] as f32 - n1 as f32 * p_hat;
        }
    }
    num
}

/// Run the counterfactual contrast for every (site, branch) that passes QC.
pub fn run_contrasts(sites: &[Site], lin: &Lineage, cfg: &AssocConfig) -> Vec<BranchResult> {
    let bins = bin_pseudotime(&lin.pseudotime, cfg.n_bins);
    let (nb, nbr, ncell) = (cfg.n_bins, lin.n_branches, lin.cell_names.len());

    // Covered cells per site (n>0) — reused across observed + all permutations.
    let covered: Vec<Vec<usize>> = sites
        .iter()
        .map(|s| (0..ncell).filter(|&c| s.n[c] > 0).collect())
        .collect();

    // Observed numerators.
    let obs: Vec<Vec<f32>> = sites
        .iter()
        .enumerate()
        .map(|(si, s)| cmh_nums(s, &covered[si], &bins, &lin.branch, nb, nbr))
        .collect();

    // Permutation null: shuffle branch labels within pseudotime bins.
    let mut ge = vec![vec![0u32; nbr]; sites.len()];
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    for _ in 0..cfg.num_perm {
        let pi = permute_indices(ncell, Some(&bins), &mut rng);
        let pb: Vec<usize> = (0..ncell).map(|c| lin.branch[pi[c]]).collect();
        for (si, s) in sites.iter().enumerate() {
            let nums = cmh_nums(s, &covered[si], &bins, &pb, nb, nbr);
            for l in 0..nbr {
                if nums[l].abs() >= obs[si][l].abs() {
                    ge[si][l] += 1;
                }
            }
        }
    }

    // Assemble QC-passing results.
    let mut out = Vec::new();
    for (si, s) in sites.iter().enumerate() {
        for l in 0..nbr {
            let (mut ncells, mut cov) = (0usize, 0u64);
            for &c in &covered[si] {
                if lin.branch[c] == l {
                    ncells += 1;
                    cov += s.n[c] as u64;
                }
            }
            if cov < cfg.min_total_coverage || ncells < cfg.min_cells {
                continue;
            }
            let num = obs[si][l];
            let effect = if cov > 0 { num / cov as f32 } else { 0.0 };
            let p_perm = (1 + ge[si][l] as usize) as f32 / (1 + cfg.num_perm) as f32;
            out.push(BranchResult {
                site: si,
                branch: l,
                n_cells: ncells,
                total_cov: cov,
                stat: num,
                effect,
                p_perm,
            });
        }
    }
    out
}

/// Per-(branch, bin) pseudobulk `(branch, bin, K, N)` for one site (non-empty only)
/// — the counterfactual rate profile along pseudotime, for plotting.
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
