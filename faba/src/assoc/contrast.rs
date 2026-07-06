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
use rayon::prelude::*;

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

/// Reusable per-(bin, branch) accumulators for the one-vs-rest CMH numerators — one
/// flat `n_bins × n_branches` buffer pair, cleared and refilled per call so a site can
/// reuse it across all permutations instead of allocating a fresh 2-D grid each time.
struct CmhScratch {
    aggk: Vec<u64>,
    aggn: Vec<u64>,
    n_bins: usize,
    n_branches: usize,
}

impl CmhScratch {
    fn new(n_bins: usize, n_branches: usize) -> Self {
        Self {
            aggk: vec![0; n_bins * n_branches],
            aggn: vec![0; n_bins * n_branches],
            n_bins,
            n_branches,
        }
    }

    /// One-vs-rest CMH numerators for every branch of one site, in a single pass over
    /// the covered cells. `num[L]` = Σ_bin (K_{L,bin} − expected).
    fn numerators(
        &mut self,
        site: &Site,
        covered: &[usize],
        bins: &[u32],
        branch: &[usize],
    ) -> Vec<f32> {
        let nbr = self.n_branches;
        self.aggk.fill(0);
        self.aggn.fill(0);
        for &c in covered {
            let idx = bins[c] as usize * nbr + branch[c];
            self.aggk[idx] += site.k[c] as u64;
            self.aggn[idx] += site.n[c] as u64;
        }
        let mut num = vec![0f32; nbr];
        for b in 0..self.n_bins {
            let (row_k, row_n) = (
                &self.aggk[b * nbr..(b + 1) * nbr],
                &self.aggn[b * nbr..(b + 1) * nbr],
            );
            let sumn: u64 = row_n.iter().sum();
            if sumn == 0 {
                continue;
            }
            let p_hat = row_k.iter().sum::<u64>() as f32 / sumn as f32;
            for l in 0..nbr {
                let n1 = row_n[l];
                if n1 == 0 || sumn - n1 == 0 {
                    continue; // no contrast in this bin
                }
                num[l] += row_k[l] as f32 - n1 as f32 * p_hat;
            }
        }
        num
    }
}

/// Run the counterfactual contrast for every (site, branch) that passes QC.
pub fn run_contrasts(sites: &[Site], lin: &Lineage, cfg: &AssocConfig) -> Vec<BranchResult> {
    let bins = bin_pseudotime(&lin.pseudotime, cfg.n_bins);
    let (nb, nbr, ncell) = (cfg.n_bins, lin.n_branches, lin.cell_names.len());

    // Covered cells per site (n>0) — reused across observed + all permutations.
    let covered: Vec<Vec<usize>> = sites
        .par_iter()
        .map(|s| (0..ncell).filter(|&c| s.n[c] > 0).collect())
        .collect();

    // Observed one-vs-rest CMH numerators, per site.
    let obs: Vec<Vec<f32>> = sites
        .par_iter()
        .enumerate()
        .map(|(si, s)| CmhScratch::new(nb, nbr).numerators(s, &covered[si], &bins, &lin.branch))
        .collect();

    // Precompute the within-bin branch-label permutations once, with a single
    // sequential RNG — so the null is reproducible and independent of thread count.
    // Only the per-site counting below runs in parallel; the perm set is shared.
    let perms: Vec<Vec<usize>> = {
        let mut rng = SmallRng::seed_from_u64(cfg.seed);
        (0..cfg.num_perm)
            .map(|_| {
                let pi = permute_indices(ncell, Some(&bins), &mut rng);
                (0..ncell).map(|c| lin.branch[pi[c]]).collect()
            })
            .collect()
    };

    // Permutation null: per site, count permutations whose |CMH| ≥ |observed|.
    // Sites are independent → parallel over sites.
    let ge: Vec<Vec<u32>> = sites
        .par_iter()
        .enumerate()
        .map(|(si, s)| {
            let mut scr = CmhScratch::new(nb, nbr);
            let mut cnt = vec![0u32; nbr];
            for pb in &perms {
                let nums = scr.numerators(s, &covered[si], &bins, pb);
                for l in 0..nbr {
                    if nums[l].abs() >= obs[si][l].abs() {
                        cnt[l] += 1;
                    }
                }
            }
            cnt
        })
        .collect();

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
