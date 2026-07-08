//! Bayesian between-branch contrast — the posterior of a branch's pseudotime-adjusted excess
//! editing, via the same elliptical-slice binomial machinery as [`trend_bayes`](super::trend_bayes).
//!
//! For site `s` and branch `L` (one-vs-rest), the covered cells are pseudo-bulked per
//! pseudotime bin `b` into branch-`L` counts `(K_L, N_L)` and other-fate counts `(K_R, N_R)`.
//! The model is a binomial GLM
//!
//! ```text
//!   logit(p_{b,g}) = α_b + β · 1[g = L]
//! ```
//!
//! where the per-bin baseline `α_b` conditions out pseudotime (the same matched-null idea as
//! the CMH contrast) and **`β` is the pseudotime-adjusted log-odds excess of branch `L`** — the
//! quantity of interest. Vague prior on `α_b`, a shrinkage prior `β ~ N(0, τ²)`. ESS samples the
//! posterior with no tuning; the summary is the posterior mean of `β`, its 90% credible interval,
//! and `lfsr = min(P(β>0), P(β<0))` — a posterior sign-error rate that replaces the permutation
//! p-value and does **not** suffer the `1/(B+1)` discreteness floor, so no BH/FWER permutation
//! machinery is needed. The shrinkage prior also damps noisy calls, stabilising them across seeds.

use nalgebra::DVector;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use mcmc_util::engine::EssSampler;

use super::bayes_common::{derive_seed, softplus, summarize_posterior, INTERCEPT_SD};
use super::contrast::bin_pseudotime;
use super::io::{Lineage, Site};

pub struct BayesContrastConfig {
    pub n_bins: usize,
    pub min_total_coverage: u64,
    pub min_cells: usize,
    /// Prior sd `τ` for the branch effect `β` (smaller ⇒ stronger shrinkage toward no effect).
    pub prior_sd: f32,
    pub n_samples: usize,
    pub warmup: usize,
    pub seed: u64,
}

pub struct BayesContrastResult {
    pub site: usize,
    pub branch: usize,
    pub n_cells: usize,
    pub total_cov: u64,
    /// Posterior mean of `β` — the pseudotime-adjusted log-odds excess of the branch vs rest.
    pub effect: f32,
    pub effect_sd: f32,
    /// 5% / 95% posterior quantiles of `β` (90% credible interval).
    pub effect_lo: f32,
    pub effect_hi: f32,
    /// Local false sign rate `min(P(β>0), P(β<0))` — small when the branch's editing differs
    /// from the other fates in a consistent direction.
    pub lfsr: f32,
}

/// Fit the Bayesian between-branch contrast for every (site, branch) that passes QC. Parallel
/// over sites; each branch is one sequential ESS chain (no nested rayon).
pub fn run_contrasts_bayes(
    sites: &[Site],
    lin: &Lineage,
    cfg: &BayesContrastConfig,
) -> Vec<BayesContrastResult> {
    let bins = bin_pseudotime(&lin.pseudotime, cfg.n_bins);
    sites
        .par_iter()
        .enumerate()
        .flat_map_iter(|(si, s)| site_contrasts_bayes(si, s, lin, &bins, cfg).into_iter())
        .collect()
}

fn site_contrasts_bayes(
    si: usize,
    s: &Site,
    lin: &Lineage,
    bins: &[u32],
    cfg: &BayesContrastConfig,
) -> Vec<BayesContrastResult> {
    let (nb, nbr) = (cfg.n_bins, lin.n_branches);
    // One pass over the covered cells: pseudobulk into (bin, branch) counts AND tally the
    // per-branch (n_cells, total_cov) QC stats — same loop, no rescan per branch.
    let mut aggk = vec![0u64; nb * nbr];
    let mut aggn = vec![0u64; nb * nbr];
    let mut n_cells = vec![0usize; nbr];
    let mut br_cov = vec![0u64; nbr];
    for (c, &b) in bins.iter().enumerate() {
        if s.n[c] > 0 {
            let g = lin.branch[c];
            let n_c = s.n[c] as u64;
            aggk[b as usize * nbr + g] += s.k[c] as u64;
            aggn[b as usize * nbr + g] += n_c;
            n_cells[g] += 1;
            br_cov[g] += n_c;
        }
    }

    let mut out = Vec::with_capacity(nbr);
    for l in 0..nbr {
        if br_cov[l] < cfg.min_total_coverage || n_cells[l] < cfg.min_cells {
            continue;
        }
        let seed = derive_seed(cfg.seed, si, l);
        if let Some(post) = fit_contrast(nb, nbr, l, &aggk, &aggn, cfg, seed) {
            out.push(BayesContrastResult {
                site: si,
                branch: l,
                n_cells: n_cells[l],
                total_cov: br_cov[l],
                effect: post.effect,
                effect_sd: post.effect_sd,
                effect_lo: post.effect_lo,
                effect_hi: post.effect_hi,
                lfsr: post.lfsr,
            });
        }
    }
    out
}

/// ESS posterior of `β` (branch `l` vs rest, pseudotime-adjusted) from the per-(bin, branch)
/// counts. `None` when no bin has both the branch and the rest covered (β unidentified).
///
/// The design has one row per (active bin, group∈{L, rest}) and one column per active bin
/// (the baseline `α_b`) plus one column for `β`. Because each row's non-zeros are exactly
/// `α_{bin_col[i]}` and, for L rows, `β`, the log-likelihood computes `η_i = β[bin_col[i]] +
/// is_L[i]·β[beta_col]` as two lookups — no full DMatrix multiply is worth the overhead for
/// `m ≤ 2·n_bins` and `p ≤ n_bins+1`.
fn fit_contrast(
    nb: usize,
    nbr: usize,
    l: usize,
    aggk: &[u64],
    aggn: &[u64],
    cfg: &BayesContrastConfig,
    seed: u64,
) -> Option<super::bayes_common::Posterior> {
    // Only bins carrying both L and the pooled rest inform β. Allocate straight into the
    // final slim vectors (no throwaway `rows`/`bin_col` intermediates): each active bin
    // adds two rows, whose column indices are the active-bin id assigned in order.
    let max_rows = 2 * nb;
    let mut kf: Vec<f32> = Vec::with_capacity(max_rows);
    let mut nf: Vec<f32> = Vec::with_capacity(max_rows);
    let mut bin_col: Vec<usize> = Vec::with_capacity(max_rows);
    let mut is_l: Vec<f32> = Vec::with_capacity(max_rows); // 1.0 for L rows, 0.0 for rest rows
    let mut n_active = 0usize;
    for b in 0..nb {
        let (kl, nl) = (aggk[b * nbr + l], aggn[b * nbr + l]);
        let mut kr = 0u64;
        let mut nr = 0u64;
        for g in 0..nbr {
            if g != l {
                kr += aggk[b * nbr + g];
                nr += aggn[b * nbr + g];
            }
        }
        if nl == 0 || nr == 0 {
            continue; // one-sided bin: α_b would absorb it, no contrast
        }
        let bcol = n_active;
        n_active += 1;
        // L row.
        kf.push(kl as f32);
        nf.push(nl as f32);
        bin_col.push(bcol);
        is_l.push(1.0);
        // rest row.
        kf.push(kr as f32);
        nf.push(nr as f32);
        bin_col.push(bcol);
        is_l.push(0.0);
    }
    if n_active == 0 {
        return None;
    }
    let m = kf.len();
    let p = n_active + 1;
    let beta_col = p - 1;

    // Binomial log-likelihood. η_i = β[bin_col[i]] + is_L[i]·β[beta_col] — two lookups per
    // row; DMatrix·DVector overhead would dwarf this on such a sparse design.
    let lnpdf = |beta: &DVector<f32>| -> f32 {
        let mut ll = 0.0f32;
        let beta_l = beta[beta_col];
        for i in 0..m {
            let eta = beta[bin_col[i]] + is_l[i] * beta_l;
            ll += kf[i] * eta - nf[i] * softplus(eta);
        }
        ll
    };

    // Prior: vague on the per-bin baselines, shrinkage τ on β.
    let mut sd = vec![INTERCEPT_SD; p];
    sd[beta_col] = cfg.prior_sd;
    let prior_draw = |rng: &mut SmallRng| -> DVector<f32> {
        DVector::from_fn(p, |j, _| {
            let z: f64 = StandardNormal.sample(rng);
            sd[j] * z as f32
        })
    };

    let init = DVector::from_element(p, 0.0f32);
    let sampler = EssSampler {
        n_samples: cfg.n_samples,
        warmup: cfg.warmup,
        thin: 1,
        seed,
    };
    let chain = sampler.run(&lnpdf, &prior_draw, &init);
    let effects: Vec<f32> = chain.samples.iter().map(|b| b[beta_col]).collect();
    summarize_posterior(effects)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clear branch-0 hyper-editing signal (branch 0 ~80% edited, others ~20%, across bins)
    /// should give a large positive posterior effect with a small lfsr.
    #[test]
    fn recovers_positive_branch_effect() {
        let cfg = BayesContrastConfig {
            n_bins: 4,
            min_total_coverage: 1,
            min_cells: 1,
            prior_sd: 3.0,
            n_samples: 400,
            warmup: 200,
            seed: 1,
        };
        let (nb, nbr) = (4, 2);
        // aggk/aggn per (bin, branch): branch 0 edited 80/100, branch 1 edited 20/100.
        let mut aggk = vec![0u64; nb * nbr];
        let mut aggn = vec![0u64; nb * nbr];
        for b in 0..nb {
            aggk[b * nbr] = 80;
            aggn[b * nbr] = 100;
            aggk[b * nbr + 1] = 20;
            aggn[b * nbr + 1] = 100;
        }
        let post = fit_contrast(nb, nbr, 0, &aggk, &aggn, &cfg, 7).expect("posterior");
        assert!(
            post.effect > 1.0,
            "expected large +effect, got {}",
            post.effect
        );
        assert!(post.lfsr < 0.05, "expected small lfsr, got {}", post.lfsr);
        assert!(
            post.effect_lo > 0.0,
            "90% CI should exclude 0, lo={}",
            post.effect_lo
        );
    }

    /// No difference between branches → effect near 0 and lfsr near 0.5 (uninformative sign).
    #[test]
    fn null_gives_large_lfsr() {
        let cfg = BayesContrastConfig {
            n_bins: 4,
            min_total_coverage: 1,
            min_cells: 1,
            prior_sd: 3.0,
            n_samples: 400,
            warmup: 200,
            seed: 2,
        };
        let (nb, nbr) = (4, 2);
        let mut aggk = vec![0u64; nb * nbr];
        let mut aggn = vec![0u64; nb * nbr];
        for b in 0..nb {
            for g in 0..nbr {
                aggk[b * nbr + g] = 50;
                aggn[b * nbr + g] = 100;
            }
        }
        let post = fit_contrast(nb, nbr, 0, &aggk, &aggn, &cfg, 3).expect("posterior");
        assert!(
            post.effect.abs() < 0.5,
            "expected ~0 effect, got {}",
            post.effect
        );
        assert!(post.lfsr > 0.2, "expected large lfsr, got {}", post.lfsr);
    }
}
