//! Top-K variant selection, ranked ONCE per gene.
//!
//! Two rules here are load-bearing, and both were measured on the
//! prototype before they became code:
//!
//! 1. **Rank per gene, never per (gene, cell type).** A per-cell-type
//!    ranking compares different variants across cell types, so the
//!    specificity contrast the model is asked to learn — same variant,
//!    same gene, different context — is destroyed before training starts.
//! 2. **Keep every gene each selected variant was tested against**, not
//!    only the gene it was selected for. Those other (variant, gene) pairs
//!    are the gene-swap negative pool. Restricting to the selected pairs
//!    empties it — 92% of positives had no gene-swap negative — which
//!    starves the variant embedding, since it cancels in same-variant
//!    contrasts and is trained only by gene swaps.

use log::info;
use rayon::prelude::*;

use super::reader::QtlData;

/// One observation of a (gene, variant) pair in one real cell type.
#[derive(Debug, Clone, Copy)]
pub struct PairObs {
    pub celltype: u32,
    pub beta: f32,
    pub se: f32,
}

/// One (gene, variant) pair with every real cell type it was tested in,
/// ordered by cell type.
#[derive(Debug, Clone)]
pub struct Pair {
    pub gene: u32,
    pub variant: u32,
    pub obs: Vec<PairObs>,
}

/// Retained pairs plus the counters that describe what selection did.
#[derive(Debug)]
pub struct Selection {
    pub pairs: Vec<Pair>,
    /// Distinct variants that were top-K for at least one gene.
    pub n_selected_variants: usize,
    /// Pairs dropped because the variant was top-K for no gene.
    pub n_pairs_dropped: usize,
    /// Observations retained across all pairs.
    pub n_rows: usize,
}

/// Precision-weighted mean chi-square of one (gene, variant) group.
///
/// `chi2 = (beta/se)^2`, combined with weights `1/se^2`, so a large
/// chi-square seen only in an imprecise cell type does not carry the rank.
fn ranking_score(obs: &[PairObs]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for o in obs {
        let se = o.se as f64;
        let w = 1.0 / (se * se);
        let chi2 = (o.beta as f64 / se).powi(2);
        num += w * chi2;
        den += w;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Group one gene's sorted entries into per-variant runs.
fn pair_groups(data: &QtlData, gene: usize) -> Vec<Pair> {
    let entries = &data.tables[gene].entries;
    let mut out: Vec<Pair> = Vec::new();
    let mut start = 0usize;
    while start < entries.len() {
        let variant = entries[start].variant;
        let mut end = start;
        while end < entries.len() && entries[end].variant == variant {
            end += 1;
        }
        out.push(Pair {
            gene: gene as u32,
            variant,
            obs: entries[start..end]
                .iter()
                .map(|e| PairObs {
                    celltype: e.celltype,
                    beta: e.beta,
                    se: e.se,
                })
                .collect(),
        });
        start = end;
    }
    out
}

/// Keep the `top_k` best-ranked variants of every gene, then keep every
/// pair those variants belong to.
pub fn select_top_variants(data: &QtlData, top_k: usize) -> Selection {
    // Grouping and ranking are independent per gene; only the union of the
    // picks is shared, and that is a cheap serial fold afterwards.
    let ranked: Vec<(Vec<Pair>, Vec<u32>)> = (0..data.tables.len())
        .into_par_iter()
        .map(|g| {
            let pairs = pair_groups(data, g);
            let mut order: Vec<(f64, u32)> = pairs
                .iter()
                .map(|p| (ranking_score(&p.obs), p.variant))
                .collect();
            // Descending score; the variant id breaks ties so the choice does
            // not depend on input order.
            order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            order.truncate(top_k);
            let picks = order.into_iter().map(|(_, variant)| variant).collect();
            (pairs, picks)
        })
        .collect();

    // ── Rank once per gene ───────────────────────────────────────────────
    let mut selected = vec![false; data.variants.len()];
    let mut groups: Vec<Vec<Pair>> = Vec::with_capacity(ranked.len());
    for (pairs, picks) in ranked {
        for variant in picks {
            selected[variant as usize] = true;
        }
        groups.push(pairs);
    }

    // ── Keep ALL genes each selected variant was tested against ──────────
    let mut kept: Vec<Pair> = Vec::new();
    let mut n_pairs_dropped = 0usize;
    for pairs in &mut groups {
        for pair in pairs.drain(..) {
            if selected[pair.variant as usize] {
                kept.push(pair);
            } else {
                n_pairs_dropped += 1;
            }
        }
    }

    let n_rows = kept.iter().map(|p| p.obs.len()).sum();
    let n_selected_variants = selected.iter().filter(|&&s| s).count();
    info!(
        "Top-{} per gene: {} of {} variants selected, {} pairs kept ({} rows), {} pairs dropped",
        top_k,
        n_selected_variants,
        data.variants.len(),
        kept.len(),
        n_rows,
        n_pairs_dropped,
    );

    Selection {
        pairs: kept,
        n_selected_variants,
        n_pairs_dropped,
        n_rows,
    }
}

#[cfg(test)]
#[path = "select_tests.rs"]
mod tests;
