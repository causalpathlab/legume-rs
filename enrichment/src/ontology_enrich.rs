//! Ontology-scale gene-set enrichment — the GO/GMT scorer.
//!
//! The shipped scorer is [`ontology_module_score`]: per (cluster, term) a
//! **descriptive module score** — the mean of the term's genes minus the
//! background mean, on `log1p(CP10K)` of the cluster's expression, then a
//! cross-cluster contrast (`d = s − median_k s`). It's a *mean*, so size-robust
//! (a small sharply-expressed set scores as strongly as a large module), the
//! contrast cancels housekeeping (high inside and outside → ~0), and unexpressed
//! genes contribute ~0 — empirically the cleanest per-cluster signature. (A
//! permutation-calibrated variant was tried and rejected — `÷sd` rewards small
//! stable-null terms over the big lineage modules.)

use crate::Mat;

/// Descriptive **module-score signature** — the shipped GO/GMT scorer.
///
/// Per (cluster, term) `s = mean_in − mean_out` of `log1p(CP10K)` on the
/// cluster's expression profile, then a cross-cluster contrast `d = s − median_k
/// s`. No permutation, no foreground cutoff, no calibration: empirically this
/// plain effect-size ranking recovers cluster lineage (T-cell, cell-cycle, …)
/// *more cleanly* than the permutation-z + TreeBH machinery, whose ÷sd
/// reweighting and single leaf-most tip surface small-variance/deep-process
/// noise instead. The trade-off is honest: `effect_kt` is descriptive (no
/// calibrated p) and carries the post-clustering double-dipping caveat.
pub struct OntologyModuleScore {
    /// K × T cross-cluster-contrasted module-score effect (`d = s − median_k s`).
    pub effect_kt: Mat,
    /// Term ids indexing the columns, in input order.
    pub term_ids: Vec<Box<str>>,
}

/// Deduped background rows + per-term member positions within that background.
/// Used by [`ontology_module_score`].
fn build_term_background(
    g: usize,
    terms: &[(Box<str>, Vec<usize>)],
    universe: &[usize],
) -> anyhow::Result<(Vec<usize>, Vec<Vec<usize>>)> {
    let mut bg: Vec<usize> = universe.to_vec();
    bg.sort_unstable();
    bg.dedup();
    for &r in &bg {
        anyhow::ensure!(r < g, "universe row ≥ {g} genes");
    }
    let mut bg_pos = vec![usize::MAX; g];
    for (bi, &r) in bg.iter().enumerate() {
        bg_pos[r] = bi;
    }
    let term_bg: Vec<Vec<usize>> = terms
        .iter()
        .map(|(_, rows)| {
            rows.iter()
                .filter_map(|&r| (r < g && bg_pos[r] != usize::MAX).then_some(bg_pos[r]))
                .collect()
        })
        .collect();
    Ok((bg, term_bg))
}

/// Per-cluster `log1p(CP10K)` background vectors: aggregate PB gene sums by
/// `assignment` into cluster profiles, CP10K-scale per cluster, `log1p`. Returns
/// `[n_clusters][n_bg]`. Used by `module_scores`.
fn aggregate_log_cp10k(
    pb_gene: &Mat,
    assignment: &[usize],
    n_clusters: usize,
    bg: &[usize],
) -> Vec<Vec<f64>> {
    let n_bg = bg.len();
    let mut csum = vec![vec![0.0f64; n_bg]; n_clusters];
    for (pb, &k) in assignment.iter().enumerate() {
        let row = &mut csum[k];
        for (bi, &g) in bg.iter().enumerate() {
            row[bi] += pb_gene[(g, pb)] as f64;
        }
    }
    csum.into_iter()
        .map(|c| {
            let colsum: f64 = c.iter().sum::<f64>().max(1e-12);
            let scale = 1e4 / colsum;
            c.iter().map(|&v| (v * scale).ln_1p()).collect()
        })
        .collect()
}

/// Module score `s[k][t]` for one PB→cluster `assignment`: `mean_in − mean_out`
/// of `log1p(CP10K)` per term.
fn module_scores(
    pb_gene: &Mat,
    assignment: &[usize],
    n_clusters: usize,
    bg: &[usize],
    term_bg: &[Vec<usize>],
) -> Vec<Vec<f32>> {
    let n_bg = bg.len();
    let n_terms = term_bg.len();
    let lge = aggregate_log_cp10k(pb_gene, assignment, n_clusters, bg);
    let mut out = vec![vec![0.0f32; n_terms]; n_clusters];
    for k in 0..n_clusters {
        let total: f64 = lge[k].iter().sum();
        for (ti, members) in term_bg.iter().enumerate() {
            let m = members.len() as f64;
            if m < 1.0 || (n_bg as f64 - m) < 1.0 {
                continue;
            }
            let sum_in: f64 = members.iter().map(|&bi| lge[k][bi]).sum();
            let within = sum_in / m;
            let outside = (total - sum_in) / (n_bg as f64 - m);
            out[k][ti] = (within - outside) as f32;
        }
    }
    out
}

/// Descriptive module-score signature (no permutation, no TreeBH).
///
/// `profile_gk` is the `G × K` per-cluster expression profile (the NB-Fisher
/// `weighted_mean_profile`, library-normalized); `terms` is `(id, member rows)`
/// (size-windowed); `universe` the annotated-background rows. Computes the
/// per-cluster module score on each profile column, then the cross-cluster
/// contrast `d = s − median_k s`. Returns the `K × T` effect matrix — rank a
/// cluster's positive-effect terms to read its GO/GMT signature.
pub fn ontology_module_score(
    profile_gk: &Mat,
    terms: &[(Box<str>, Vec<usize>)],
    universe: &[usize],
) -> anyhow::Result<OntologyModuleScore> {
    let g = profile_gk.nrows();
    let n_clusters = profile_gk.ncols();
    anyhow::ensure!(g > 0 && n_clusters > 1, "need ≥2 clusters to contrast");
    anyhow::ensure!(!terms.is_empty(), "no gene-sets to score");
    anyhow::ensure!(!universe.is_empty(), "empty background universe");

    let (bg, term_bg) = build_term_background(g, terms, universe)?;

    // Each cluster column IS its own pseudobulk → identity PB→cluster map; this
    // reuses `module_scores` to get the observed `s[k][t] = mean_in − mean_out`.
    let assignment: Vec<usize> = (0..n_clusters).collect();
    let s = module_scores(profile_gk, &assignment, n_clusters, &bg, &term_bg);

    // Cross-cluster contrast: a term high in every cluster (common/housekeeping)
    // has a high median → d ≈ 0 and drops out; a cluster-specific one survives.
    let n_terms = terms.len();
    let mut effect_kt = Mat::zeros(n_clusters, n_terms);
    for ti in 0..n_terms {
        let col: Vec<f32> = (0..n_clusters).map(|k| s[k][ti]).collect();
        let med = matrix_util::utils::median(&col);
        for k in 0..n_clusters {
            effect_kt[(k, ti)] = s[k][ti] - med;
        }
    }
    Ok(OntologyModuleScore {
        effect_kt,
        term_ids: terms.iter().map(|(id, _)| id.clone()).collect(),
    })
}
