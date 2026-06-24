//! Ontology-scale over-representation analysis (ORA) with a cross-cluster
//! contrast — the GO/GMT scorer.
//!
//! Continuous GSEA `es_std` does not discriminate clusters within a single
//! lineage: every cluster is (say) an immune cell, so enrichment-vs-random-genes
//! lights up immune/housekeeping terms in *all* of them (see `7d` in the plan).
//! ORA on a **cluster-specific foreground** plus a **competitive-across-clusters**
//! contrast fixes this:
//!
//! 1. **One-vs-rest foreground.** Rank genes per cluster by `X·frac = μ²/Σ_k μ`
//!    (expression × specificity); take the top-N. This down-weights low-expressed
//!    ultra-specific noise and abundance-only housekeeping, keeping the genes that
//!    are both expressed and cluster-specific.
//! 2. **Hypergeometric over-representation** of each term's members in that
//!    foreground vs the annotated background (`universe`). Sparse: one tail sum
//!    per (cluster, term), no dense `g × terms` matrix.
//! 3. **Cross-cluster contrast.** Subtract each term's cross-cluster median
//!    `−log10 p`; a term enriched in *every* cluster (genuinely common) cancels to
//!    ≈0, while a cluster-specific term survives. The contrasted value maps back
//!    to a relative p-value `10^−max(d,0) = p_k / median_k(p)` in (0, 1], fed to
//!    the ontology TreeBH as `OntologyScore::Pvalue`.

use crate::Mat;
use rayon::prelude::*;
use special::Gamma;

#[derive(Debug, Clone)]
pub struct OntologyOraConfig {
    /// Foreground size per cluster (top-N genes by `X·frac`).
    pub foreground_top_n: usize,
    /// Minimum foreground∩term overlap for a term to be tested in a cluster.
    pub min_overlap: usize,
}

impl Default for OntologyOraConfig {
    fn default() -> Self {
        Self {
            foreground_top_n: 150,
            min_overlap: 2,
        }
    }
}

pub struct OntologyOraOutputs {
    /// K × T cross-cluster contrasted relative p-value in (0, 1] — the TreeBH
    /// input. `1.0` = not more enriched than the median cluster (common/absent).
    pub pvalue_kt: Mat,
    /// K × T raw `−log10` hypergeometric p (pre-contrast), for inspection.
    pub neglog10p_kt: Mat,
    /// Term ids indexing the columns, in input order.
    pub term_ids: Vec<Box<str>>,
}

#[inline]
fn ln_gamma(x: f64) -> f64 {
    Gamma::ln_gamma(x).0
}

/// `ln C(n, k)`.
#[inline]
fn log_choose(n: usize, k: usize) -> f64 {
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// Upper tail `P(X ≥ a)` of a hypergeometric: drawing `n_draw` from a population
/// of `n_pop` with `k_succ` successes. Computed in log space over the support.
fn hypergeom_sf(a: usize, n_pop: usize, k_succ: usize, n_draw: usize) -> f64 {
    if k_succ == 0 || n_draw == 0 || k_succ > n_pop || n_draw > n_pop {
        return if a == 0 { 1.0 } else { 0.0 };
    }
    let lo = n_draw.saturating_sub(n_pop - k_succ); // min possible overlap
    let hi = k_succ.min(n_draw); // max possible overlap
    let a = a.max(lo);
    if a == lo {
        return 1.0; // whole support is ≥ a
    }
    if a > hi {
        return 0.0;
    }
    let ln_denom = log_choose(n_pop, n_draw);
    let mut s = 0.0f64;
    for i in a..=hi {
        let lp = log_choose(k_succ, i) + log_choose(n_pop - k_succ, n_draw - i) - ln_denom;
        s += lp.exp();
    }
    s.min(1.0)
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Score `terms` against `profile_gk` by cross-cluster-contrasted ORA.
///
/// `terms` is `(term id, member row indices)` (size-windowed); `universe` is the
/// annotated-background rows (every row that carries ≥1 annotation). All indices
/// must be `< profile_gk.nrows()`.
pub fn ontology_ora(
    profile_gk: &Mat,
    terms: &[(Box<str>, Vec<usize>)],
    universe: &[usize],
    config: &OntologyOraConfig,
) -> anyhow::Result<OntologyOraOutputs> {
    let g = profile_gk.nrows();
    let k = profile_gk.ncols();
    let n_terms = terms.len();
    anyhow::ensure!(g > 0 && k > 0, "empty profile");
    anyhow::ensure!(n_terms > 0, "no gene-sets to score");
    anyhow::ensure!(!universe.is_empty(), "empty background universe");

    let mut in_univ = vec![false; g];
    for &r in universe {
        anyhow::ensure!(r < g, "universe row ≥ {g} genes");
        in_univ[r] = true;
    }
    let n_pop = in_univ.iter().filter(|&&b| b).count();

    // X·frac = μ²/Σ_k μ, the one-vs-rest foreground statistic.
    let mut xfrac = Mat::zeros(g, k);
    for gi in 0..g {
        let s: f32 = (0..k)
            .map(|c| profile_gk[(gi, c)].max(0.0))
            .sum::<f32>()
            .max(1e-12);
        for c in 0..k {
            let x = profile_gk[(gi, c)].max(0.0);
            xfrac[(gi, c)] = x * x / s;
        }
    }

    // Per cluster: top-N foreground, then a hypergeometric tail per term.
    let cols: Vec<Vec<f64>> = (0..k)
        .into_par_iter()
        .map(|c| {
            let mut fg_rows: Vec<usize> = universe.to_vec();
            fg_rows.sort_by(|&a, &b| {
                xfrac[(b, c)]
                    .partial_cmp(&xfrac[(a, c)])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let n_draw = config.foreground_top_n.min(fg_rows.len());
            let mut fg = vec![false; g];
            for &r in &fg_rows[..n_draw] {
                fg[r] = true;
            }
            let mut col = vec![0.0f64; n_terms];
            for (ti, (_, rows)) in terms.iter().enumerate() {
                let k_succ = rows.iter().filter(|&&r| in_univ[r]).count();
                let overlap = rows.iter().filter(|&&r| fg[r]).count();
                if overlap < config.min_overlap.max(1) {
                    continue;
                }
                let p = hypergeom_sf(overlap, n_pop, k_succ, n_draw);
                col[ti] = -(p.max(1e-300)).log10();
            }
            col
        })
        .collect();

    // Cross-cluster contrast: subtract per-term median −log10p; map the positive
    // excess back to a relative p in (0, 1].
    let mut pvalue_kt = Mat::zeros(k, n_terms);
    let mut neglog10p_kt = Mat::zeros(k, n_terms);
    for ti in 0..n_terms {
        let vals: Vec<f64> = (0..k).map(|c| cols[c][ti]).collect();
        let med = median(&vals);
        for c in 0..k {
            neglog10p_kt[(c, ti)] = vals[c] as f32;
            let d = (vals[c] - med).max(0.0);
            pvalue_kt[(c, ti)] = 10f64.powf(-d) as f32;
        }
    }

    Ok(OntologyOraOutputs {
        pvalue_kt,
        neglog10p_kt,
        term_ids: terms.iter().map(|(id, _)| id.clone()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypergeom_sf_matches_hand_value() {
        // N=10, K=3, n=4: P(X>=2) = P(2)+P(3) = 63/210 + 7/210 = 0.33333…
        let p = hypergeom_sf(2, 10, 3, 4);
        assert!((p - 0.333_333).abs() < 1e-4, "got {p}");
        assert_eq!(hypergeom_sf(0, 10, 3, 4), 1.0);
        // P(X>=1) = 1 - P(0) = 1 - C(7,4)/C(10,4) = 1 - 35/210 = 0.83333…
        assert!((hypergeom_sf(1, 10, 3, 4) - 0.833_333).abs() < 1e-4);
    }

    #[test]
    fn ora_discriminates_and_suppresses_common() {
        // 200 genes, 2 clusters. A (0..20) is k0-specific, B (100..120) is
        // k1-specific, C (50..70) is high in BOTH (genuinely common).
        let g = 200;
        let k = 2;
        let mut profile = Mat::zeros(g, k);
        for gi in 0..g {
            let (mut a, mut b) = (0.01f32, 0.01f32);
            if gi < 20 {
                a = 10.0; // A → cluster 0
            } else if (50..70).contains(&gi) {
                a = 10.0;
                b = 10.0; // C → both
            } else if (100..120).contains(&gi) {
                b = 10.0; // B → cluster 1
            }
            profile[(gi, 0)] = a;
            profile[(gi, 1)] = b;
        }
        let terms: Vec<(Box<str>, Vec<usize>)> = vec![
            ("A".into(), (0..20).collect()),
            ("B".into(), (100..120).collect()),
            ("C".into(), (50..70).collect()),
        ];
        let universe: Vec<usize> = (0..g).collect();
        let cfg = OntologyOraConfig {
            foreground_top_n: 50,
            min_overlap: 2,
        };
        let out = ontology_ora(&profile, &terms, &universe, &cfg).unwrap();
        assert_eq!(out.term_ids, vec!["A".into(), "B".into(), "C".into()]);
        let p = &out.pvalue_kt; // 2 × 3, cols = A,B,C

        // A is k0-specific: tiny p in k0, =1 in k1.
        assert!(p[(0, 0)] < 1e-3, "A in k0 p={}", p[(0, 0)]);
        assert!(p[(1, 0)] >= 0.999, "A in k1 p={}", p[(1, 0)]);
        // B is k1-specific.
        assert!(p[(1, 1)] < 1e-3, "B in k1 p={}", p[(1, 1)]);
        assert!(p[(0, 1)] >= 0.999, "B in k0 p={}", p[(0, 1)]);
        // C is common (enriched in both) → contrast cancels → not significant.
        assert!(
            p[(0, 2)] > 0.5 && p[(1, 2)] > 0.5,
            "C should cancel: {:?}",
            (p[(0, 2)], p[(1, 2)])
        );
    }
}
