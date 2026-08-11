//! The per-pair readout: what the fit says about one (variant, gene).
//!
//! This is the answer people actually ask the model for — is this variant's
//! effect on this gene ubiquitous or specific, and to which cell type — so it
//! is assembled here rather than inside the writer. The CLI only formats what
//! this returns.

use rayon::prelude::*;

use super::evidence::EvidenceTable;
use super::model::EqtlFit;
use super::ubiquity::UbiquityRow;

/// One (variant, gene) pair, scored in every real context.
#[derive(Debug, Clone)]
pub struct SpecificityRow {
    pub variant: Box<str>,
    pub gene: Box<str>,
    /// The context-free anchor `<u_j, v_g>`.
    pub anchor: f32,
    /// Highest-scoring real context, absent when there are no real contexts.
    pub best_context: Option<Box<str>>,
    pub best_score: Option<f32>,
    /// Fitted score per real context, in [`EqtlFit::real_contexts`] order.
    pub scores: Vec<f32>,
    /// Model-free counterpart: the fraction of POWERED contexts with an edge.
    pub ubiquity: Option<f32>,
    pub n_powered: usize,
    pub n_edge: usize,
    pub n_unknown: usize,
}

/// Score every pair the fit can place, in evidence order.
///
/// Pairs whose variant or gene never reached a fitted row are skipped: the
/// model has nothing to say about them.
pub fn specificity_rows(
    evidence: &EvidenceTable,
    ubiquity: &[UbiquityRow],
    fit: &EqtlFit,
) -> Vec<SpecificityRow> {
    let real = fit.real_contexts();
    evidence
        .pairs
        .par_iter()
        .zip(ubiquity.par_iter())
        .filter_map(|(pair, index)| {
            debug_assert_eq!((pair.variant, pair.gene), (index.variant, index.gene));
            let j = fit.variants.slot[pair.variant as usize]? as usize;
            let g = fit.genes.slot[pair.gene as usize]? as usize;

            // The gate is the only per-context factor, so `u_j * v_g` is
            // computed once and reused across every context and the anchor.
            let uv = fit.pair_product(j, g);
            let scores: Vec<f32> = real.iter().map(|&k| fit.gated(&uv, k)).collect();
            let best = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, &s)| (fit.contexts.names[real[i]].clone(), s));

            Some(SpecificityRow {
                variant: fit.variants.names[j].clone(),
                gene: fit.genes.names[g].clone(),
                anchor: uv.iter().sum(),
                best_context: best.as_ref().map(|(name, _)| name.clone()),
                best_score: best.map(|(_, s)| s),
                scores,
                ubiquity: index.ubiquity,
                n_powered: index.n_powered,
                n_edge: index.n_edge,
                n_unknown: index.n_unknown,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
