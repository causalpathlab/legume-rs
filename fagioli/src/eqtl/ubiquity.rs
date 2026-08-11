//! The ubiquitous context, and the model-free ubiquity index.
//!
//! The ubiquitous context is a leave-one-out meta-analysis: the cell types
//! of one (gene, variant) pair are combined by precision AFTER the single
//! strongest one is dropped. Dropping it is what makes the contrast an
//! edge/non-edge question rather than a strong/weak one — a plain meta of a
//! cell-type-specific effect is diluted but nonzero, so "ubiquitous" would
//! otherwise be handed a weak copy of every specific effect and the model
//! would only ever learn magnitude.
//!
//! The ubiquity index is the model-free counterpart written next to the
//! fitted scores: the fraction of POWERED contexts that carry an edge. Only
//! powered contexts enter the denominator, so cell-type abundance — which
//! drives how many contexts are powered at all — cancels by construction.

use rayon::prelude::*;

use super::evidence::{EvidenceTable, State};
use super::select::PairObs;

/// Leave-one-out precision-weighted meta-analysis over a pair's real cell
/// types: drop the single largest `|beta/se|`, then combine the rest.
///
/// Returns `None` for a pair with fewer than two observations (nothing is
/// left after the drop) or when the combination is not finite.
pub fn loo_meta(obs: &[PairObs]) -> Option<(f32, f32)> {
    if obs.len() < 2 {
        return None;
    }
    let mut drop = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, o) in obs.iter().enumerate() {
        let z = (o.beta as f64 / o.se as f64).abs();
        if z > best {
            best = z;
            drop = i;
        }
    }

    let mut sum_w = 0.0f64;
    let mut sum_wb = 0.0f64;
    for (i, o) in obs.iter().enumerate() {
        if i == drop {
            continue;
        }
        let w = 1.0 / (o.se as f64 * o.se as f64);
        sum_w += w;
        sum_wb += w * o.beta as f64;
    }
    if sum_w <= 0.0 || !sum_w.is_finite() {
        return None;
    }
    let beta = sum_wb / sum_w;
    let se = (1.0 / sum_w).sqrt();
    (beta.is_finite() && se.is_finite()).then_some((beta as f32, se as f32))
}

/// Per-pair breadth counted over real contexts only.
#[derive(Debug, Clone, Copy)]
pub struct UbiquityRow {
    pub variant: u32,
    pub gene: u32,
    /// Contexts that were either an edge or a certified absence.
    pub n_powered: usize,
    pub n_edge: usize,
    pub n_unknown: usize,
    /// `n_edge / n_powered`, or `None` when nothing was powered.
    pub ubiquity: Option<f32>,
}

/// One [`UbiquityRow`] per (variant, gene) pair of the evidence table, in
/// pair order.
pub fn ubiquity_index(evidence: &EvidenceTable) -> Vec<UbiquityRow> {
    evidence
        .pairs
        .par_iter()
        .map(|pair| {
            let rows = &evidence.rows[pair.start..pair.start + pair.len];
            let mut n_powered = 0usize;
            let mut n_edge = 0usize;
            let mut n_unknown = 0usize;
            for row in rows.iter().filter(|r| r.context != evidence.ubiquitous) {
                match row.state {
                    State::Edge => {
                        n_powered += 1;
                        n_edge += 1;
                    }
                    State::CertifiedAbsent => n_powered += 1,
                    State::Unknown => n_unknown += 1,
                }
            }
            UbiquityRow {
                variant: pair.variant,
                gene: pair.gene,
                n_powered,
                n_edge,
                n_unknown,
                ubiquity: (n_powered > 0).then(|| n_edge as f32 / n_powered as f32),
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "ubiquity_tests.rs"]
mod tests;
