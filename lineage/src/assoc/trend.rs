//! Within-branch trajectory association: does a site's editing rate change *along* a
//! branch? Per (site, branch) we fit a binomial/quasi-binomial spline GAM of
//! `logit(k/n)` on pseudotime over the branch's covered cells and test the smooth
//! against a constant rate (tradeSeq `associationTest`). This is the within-branch
//! complement to the between-branch counterfactual [`contrast`](super::contrast).

use rayon::prelude::*;

use super::gam::{association_test, GamArgs};
use super::io::{branch_buckets, Lineage, Site};

pub struct TrendConfig {
    pub n_knots: usize,
    pub min_total_coverage: u64,
    pub min_cells: usize,
    /// Quasi-binomial F-test (true) vs plain Binomial LRT (false).
    pub overdispersion: bool,
}

pub struct TrendResult {
    pub site: usize,
    pub branch: usize,
    pub n_cells: usize,
    pub total_cov: u64,
    /// GAM test statistic (F ratio, or deviance χ² when overdispersion is off).
    pub stat: f32,
    /// Net change in fitted log-odds from branch start → end (signed).
    pub effect: f32,
    /// Estimated dispersion `φ` (1.0 when overdispersion is off).
    pub dispersion: f32,
    pub p_value: f32,
}

/// Fit the within-branch association GAM for every (site, branch) that passes QC.
/// Sites are independent, so the outer loop is parallel.
pub fn run_trends(sites: &[Site], lin: &Lineage, cfg: &TrendConfig) -> Vec<TrendResult> {
    let gam = GamArgs {
        n_knots: cfg.n_knots,
        overdispersion: cfg.overdispersion,
        ..GamArgs::default()
    };
    sites
        .par_iter()
        .enumerate()
        .flat_map_iter(|(si, s)| site_trends(si, s, lin, cfg, &gam).into_iter())
        .collect()
}

/// All QC-passing branch trends for one site.
fn site_trends(
    si: usize,
    s: &Site,
    lin: &Lineage,
    cfg: &TrendConfig,
    gam: &GamArgs,
) -> Vec<TrendResult> {
    let mut out = Vec::new();
    for (l, bd) in branch_buckets(s, lin).into_iter().enumerate() {
        if bd.k.len() < cfg.min_cells || bd.cov < cfg.min_total_coverage {
            continue;
        }
        if let Some(fit) = association_test(&bd.k, &bd.n, &bd.x, gam) {
            out.push(TrendResult {
                site: si,
                branch: l,
                n_cells: fit.n_obs,
                total_cov: bd.cov,
                stat: fit.stat,
                effect: fit.effect,
                dispersion: fit.dispersion,
                p_value: fit.p_value,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests;
