//! `faba assoc` — counterfactual between-branch modality contrast along the lineage.
//!
//! Downstream of `faba lineage` (mirrors `gem → annotate`): the trajectory is fit
//! once; `assoc` asks, per modality site, the **counterfactual** question — *if a
//! cell had gone down a different branch, would its m6a/apa/atoi rate differ?* —
//! by comparing branches at **matched pseudotime** (tradeSeq `patternTest`, cocoa
//! matched-null spirit). Coverage n = edited+unedited is the binomial denominator,
//! so detection bias is conditioned out; the branches come from gem θ + velocity
//! (modality held out), so the contrast is not double-dipping.

pub mod contrast;
pub mod io;

use clap::ValueEnum;

use faba::feature_name::{DISTAL, EDITED, METHYLATED, PROXIMAL, UNEDITED, UNMETHYLATED};

/// Modality whose per-site rate is contrasted between branches.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Modality {
    /// m6A methylation (methylated / unmethylated).
    M6a,
    /// A-to-I editing (edited / unedited).
    Atoi,
    /// Alternative polyadenylation (proximal / distal) — gene-level only.
    Apa,
}

impl Modality {
    /// Feature-row modality token (`{gene}/{token}/{subunit}/{channel}`).
    pub fn token(self) -> &'static str {
        match self {
            Modality::M6a => "m6a",
            Modality::Atoi => "atoi",
            Modality::Apa => "apa",
        }
    }

    /// `(positive, negative)` channel tokens; the positive is the edited/methylated
    /// numerator, the pair sums to coverage n.
    pub fn channels(self) -> (&'static str, &'static str) {
        match self {
            Modality::M6a => (METHYLATED, UNMETHYLATED),
            Modality::Atoi => (EDITED, UNEDITED),
            Modality::Apa => (PROXIMAL, DISTAL),
        }
    }
}
