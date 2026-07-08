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
    /// Raw per-test permutation p-value `(1 + #{|T^(b)| ≥ |T^obs|}) / (1 + B)`.
    pub p_perm: f32,
    /// Westfall–Young step-down min-P FWER-adjusted p-value (strong control across the
    /// whole (site, branch) family, accounting for their dependence via the shared
    /// permutation null). In `[1/(B+1), 1]`, monotone in the observed-p order.
    pub p_fwer: f32,
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

    // QC mask on the OBSERVED data: the (site, branch) family is fixed before
    // permutation. Records each passing test's metadata and its per-site branch id.
    struct Passing {
        si: usize,
        l: usize,
        n_cells: usize,
        cov: u64,
    }
    let mut passing: Vec<Passing> = Vec::new();
    for (si, s) in sites.iter().enumerate() {
        for l in 0..nbr {
            let (mut ncells, mut cov) = (0usize, 0u64);
            for &c in &covered[si] {
                if lin.branch[c] == l {
                    ncells += 1;
                    cov += s.n[c] as u64;
                }
            }
            if cov >= cfg.min_total_coverage && ncells >= cfg.min_cells {
                passing.push(Passing {
                    si,
                    l,
                    n_cells: ncells,
                    cov,
                });
            }
        }
    }
    if passing.is_empty() {
        return Vec::new();
    }

    // Per site, the (global test index, branch) pairs it owns — so the parallel
    // permutation pass can fill each test's column of the shared statistic matrix.
    let mut per_site: Vec<Vec<(usize, usize)>> = vec![Vec::new(); sites.len()];
    for (t, p) in passing.iter().enumerate() {
        per_site[p.si].push((t, p.l));
    }

    // Shared permutation statistic matrix `perm_abs[t] = [|T_t^(b)|; B]`. All tests share
    // the SAME `perms` (in the same order), so column b is one common permutation across
    // every test — the prerequisite for a valid Westfall–Young min-P null. Parallel over
    // sites; each site writes only the columns for the tests it owns.
    let site_cols: Vec<Vec<(usize, Vec<f32>)>> = sites
        .par_iter()
        .enumerate()
        .map(|(si, s)| {
            let owned = &per_site[si];
            if owned.is_empty() {
                return Vec::new();
            }
            let mut scr = CmhScratch::new(nb, nbr);
            let mut cols: Vec<Vec<f32>> = owned
                .iter()
                .map(|_| Vec::with_capacity(perms.len()))
                .collect();
            for pb in &perms {
                let nums = scr.numerators(s, &covered[si], &bins, pb);
                for (k, &(_, l)) in owned.iter().enumerate() {
                    cols[k].push(nums[l].abs());
                }
            }
            owned.iter().map(|&(t, _)| t).zip(cols).collect()
        })
        .collect();

    let mut perm_abs: Vec<Vec<f32>> = vec![Vec::new(); passing.len()];
    for sc in site_cols {
        for (t, col) in sc {
            perm_abs[t] = col;
        }
    }
    let obs_abs: Vec<f32> = passing.iter().map(|p| obs[p.si][p.l].abs()).collect();

    // Raw per-test permutation p-value, then the Westfall–Young FWER calibration.
    let p_perm: Vec<f32> = perm_abs
        .iter()
        .zip(&obs_abs)
        .map(|(col, &o)| {
            let ge = col.iter().filter(|&&v| v >= o).count();
            (1 + ge) as f32 / (1 + cfg.num_perm) as f32
        })
        .collect();
    let p_fwer = westfall_young_step_down_minp(&perm_abs, &obs_abs, cfg.num_perm);

    passing
        .iter()
        .enumerate()
        .map(|(t, p)| {
            let num = obs[p.si][p.l];
            let effect = if p.cov > 0 { num / p.cov as f32 } else { 0.0 };
            BranchResult {
                site: p.si,
                branch: p.l,
                n_cells: p.n_cells,
                total_cov: p.cov,
                stat: num,
                effect,
                p_perm: p_perm[t],
                p_fwer: p_fwer[t],
            }
        })
        .collect()
}

/// Westfall–Young **step-down min-P** FWER-adjusted p-values from a shared permutation
/// set (Westfall & Young 1993; algorithm as in Ge, Dudoit & Speed 2003).
///
/// `perm_abs[t]` are the `B` permutation statistics `|T_t^(b)|` for test `t` — all tests
/// share the same `B` within-bin branch-label permutations (column `b` is one common
/// relabelling), so the minimum across tests within a permutation is a coherent draw from
/// the global null. `obs_abs[t] = |T_t^obs|`. min-P (not max-T): each test is first mapped
/// to its own permutation p-value, so tests of differing coverage/scale compare fairly.
///
/// Per test the permutation p-value is `p_t^(c) = #{b: |T_t^(b)| ≥ |T_t^(c)|} / B` and
/// `p_t^obs = #{b: |T_t^(b)| ≥ |T_t^obs|} / B`. Tests are ordered by `p_t^obs`; stepping
/// down from the least significant, `q_i^(c) = min_{k ≥ i} p_{r_k}^(c)` is the successive
/// minimum over the not-yet-rejected tests, and the adjusted p is
/// `(1 + #{c: q_i^(c) ≤ p_{r_i}^obs}) / (1 + B)`, made monotone in the step-down order.
/// Result is in `[1/(B+1), 1]`.
fn westfall_young_step_down_minp(
    perm_abs: &[Vec<f32>],
    obs_abs: &[f32],
    num_perm: usize,
) -> Vec<f32> {
    let m = perm_abs.len();
    if m == 0 || num_perm == 0 {
        return vec![1.0; m];
    }
    let bf = num_perm as f32;

    // Ascending-sorted copy of each test's B permutation stats, for O(log B) tail counts.
    let sorted: Vec<Vec<f32>> = perm_abs
        .iter()
        .map(|col| {
            let mut v = col.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v
        })
        .collect();
    // #{b: perm_abs[t][b] ≥ v} via partition_point on the ascending column.
    let tail_ge = |t: usize, v: f32| -> usize {
        let s = &sorted[t];
        s.len() - s.partition_point(|&x| x < v)
    };

    let p_obs: Vec<f32> = (0..m).map(|t| tail_ge(t, obs_abs[t]) as f32 / bf).collect();

    // Tests ordered by observed p ascending (most → least significant).
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| {
        p_obs[i]
            .partial_cmp(&p_obs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Successive minima across permutations, accumulated from least to most significant.
    let mut running_min = vec![f32::INFINITY; num_perm];
    let mut adj = vec![0f32; m];
    for &t in order.iter().rev() {
        for (c, rm) in running_min.iter_mut().enumerate() {
            let ptc = tail_ge(t, perm_abs[t][c]) as f32 / bf;
            *rm = rm.min(ptc);
        }
        let cnt = running_min.iter().filter(|&&q| q <= p_obs[t]).count();
        adj[t] = (1 + cnt) as f32 / (1 + num_perm) as f32;
    }

    // Step-down monotonicity: adjusted p is non-decreasing in the observed-p order.
    let mut prev = 0f32;
    for &t in &order {
        adj[t] = adj[t].max(prev);
        prev = adj[t];
    }
    adj
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
