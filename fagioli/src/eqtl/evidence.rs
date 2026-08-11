//! Three-state evidence per (variant, context, gene).
//!
//! - **edge** — `|beta|/se >= detect_z`.
//! - **certified absent** — not detected AND the context had the power to
//!   see the effect: its 80%-power minimum detectable effect
//!   `mde = (1.96 + 0.84) * se` sits below the pair's reference effect.
//! - **unknown** — everything else, and it is sampled in NEITHER class.
//!
//! That last line is the whole point. An unpowered cell is not evidence of
//! absence; letting it become a negative teaches the model statistical
//! power instead of regulatory specificity, and power tracks cell-type
//! abundance.
//!
//! The reference effect is the SECOND largest `|beta|` ranked by `|beta/se|`,
//! not the largest. An unshrunk maximum is inflated — a median 1.19x and a
//! 90th-percentile 1.79x over the runner-up on this data — and absence is
//! certified when `mde < b_ref`, so an inflated reference over-certifies
//! exactly the non-edges that train the specificity term.

use anyhow::{ensure, Result};
use log::info;
use rayon::prelude::*;

use super::select::{PairObs, Selection};
use super::ubiquity::loo_meta;

/// Name of the pseudo-context carrying the leave-one-out meta-analysis.
pub const UBIQUITOUS: &str = "ubiquitous";

/// 80%-power multiplier: `1.96 + 0.84`, i.e. 80% power at alpha = 0.05.
pub const MDE_MULT: f32 = 2.8;

/// Evidence class of one (variant, context, gene) cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Edge,
    CertifiedAbsent,
    Unknown,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Edge => "edge",
            State::CertifiedAbsent => "certified_absent",
            State::Unknown => "unknown",
        }
    }
}

/// One classified (variant, gene, context) cell.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub variant: u32,
    pub gene: u32,
    pub context: u32,
    pub beta: f32,
    pub se: f32,
    pub state: State,
}

/// Where one (variant, gene) pair's rows live in [`EvidenceTable::rows`].
#[derive(Debug, Clone, Copy)]
pub struct PairRange {
    pub variant: u32,
    pub gene: u32,
    pub start: usize,
    pub len: usize,
    /// Reference effect used to certify absence for this pair.
    pub b_ref: f32,
}

/// Every classified cell, contiguous by pair.
#[derive(Debug)]
pub struct EvidenceTable {
    pub rows: Vec<Observation>,
    pub pairs: Vec<PairRange>,
    /// Real cell types followed by [`UBIQUITOUS`].
    pub contexts: Vec<Box<str>>,
    /// Index of the ubiquitous pseudo-context in `contexts`.
    pub ubiquitous: u32,
    pub n_edge: usize,
    pub n_certified_absent: usize,
    pub n_unknown: usize,
}

/// Reference effect of one pair: the `|beta|` of the runner-up by `|beta/se|`,
/// falling back to the single observation when there is only one.
fn reference_effect(obs: &[PairObs]) -> f32 {
    // One pass for the top two by |z|; no allocation, no sort.
    let (mut best, mut second) = (usize::MAX, usize::MAX);
    let (mut z_best, mut z_second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, o) in obs.iter().enumerate() {
        let z = (o.beta / o.se).abs();
        if z > z_best {
            z_second = z_best;
            second = best;
            z_best = z;
            best = i;
        } else if z > z_second {
            z_second = z;
            second = i;
        }
    }
    let pick = if second != usize::MAX { second } else { best };
    obs[pick].beta.abs()
}

/// Classify every retained pair, appending the leave-one-out ubiquitous
/// context to each.
pub fn classify_states(
    selection: &Selection,
    celltypes: &[Box<str>],
    detect_z: f32,
) -> Result<EvidenceTable> {
    ensure!(
        celltypes.iter().all(|c| c.as_ref() != UBIQUITOUS),
        "a real cell type is named '{}', which collides with the pseudo-context; \
         rename it in the input",
        UBIQUITOUS
    );
    ensure!(detect_z > 0.0, "--detect-z must be positive");

    let ubiquitous = celltypes.len() as u32;
    let mut contexts: Vec<Box<str>> = celltypes.to_vec();
    contexts.push(Box::from(UBIQUITOUS));

    // Classification is independent per pair; only the flatten into one
    // contiguous table has to be serial.
    let classified: Vec<(PairRange, Vec<Observation>)> = selection
        .pairs
        .par_iter()
        .filter(|pair| !pair.obs.is_empty())
        .map(|pair| {
            let b_ref = reference_effect(&pair.obs);
            let mut cells: Vec<Observation> = pair
                .obs
                .iter()
                .map(|o| Observation {
                    variant: pair.variant,
                    gene: pair.gene,
                    context: o.celltype,
                    beta: o.beta,
                    se: o.se,
                    state: state_of(o.beta, o.se, b_ref, detect_z),
                })
                .collect();
            if let Some((beta, se)) = loo_meta(&pair.obs) {
                cells.push(Observation {
                    variant: pair.variant,
                    gene: pair.gene,
                    context: ubiquitous,
                    beta,
                    se,
                    state: state_of(beta, se, b_ref, detect_z),
                });
            }
            // `start` and `len` are filled in during the serial flatten.
            let range = PairRange {
                variant: pair.variant,
                gene: pair.gene,
                start: 0,
                len: cells.len(),
                b_ref,
            };
            (range, cells)
        })
        .collect();

    let mut rows: Vec<Observation> = Vec::with_capacity(selection.n_rows + selection.pairs.len());
    let mut pairs: Vec<PairRange> = Vec::with_capacity(classified.len());
    let (mut n_edge, mut n_certified_absent, mut n_unknown) = (0usize, 0usize, 0usize);
    for (mut range, cells) in classified {
        range.start = rows.len();
        for cell in cells {
            match cell.state {
                State::Edge => n_edge += 1,
                State::CertifiedAbsent => n_certified_absent += 1,
                State::Unknown => n_unknown += 1,
            }
            rows.push(cell);
        }
        pairs.push(range);
    }

    let table = EvidenceTable {
        n_edge,
        n_certified_absent,
        n_unknown,
        rows,
        pairs,
        contexts,
        ubiquitous,
    };
    info!(
        "Evidence over {} cells: {} edge, {} certified absent, {} unknown ({} pairs, {} contexts)",
        table.rows.len(),
        table.n_edge,
        table.n_certified_absent,
        table.n_unknown,
        table.pairs.len(),
        table.contexts.len(),
    );
    Ok(table)
}

/// The three-state rule for one cell.
fn state_of(beta: f32, se: f32, b_ref: f32, detect_z: f32) -> State {
    if (beta / se).abs() >= detect_z {
        State::Edge
    } else if MDE_MULT * se < b_ref {
        State::CertifiedAbsent
    } else {
        State::Unknown
    }
}

#[cfg(test)]
#[path = "evidence_tests.rs"]
mod tests;
