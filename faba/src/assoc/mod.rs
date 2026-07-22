//! `faba assoc` — modality dynamics along the lineage.
//!
//! Downstream of `faba lineage` (mirrors `gem → annotate`): the trajectory is fit
//! once; `assoc` then asks two complementary questions per modality site, with
//! coverage n = edited+unedited as the binomial denominator (so detection bias is
//! conditioned out) and the branches taken from gem θ + velocity with the modality
//! held out (so neither test double-dips):
//!
//! - **Between branches** ([`contrast`]) — the counterfactual *if a cell had gone
//!   down a different branch, would its rate differ?*, comparing branches at matched
//!   pseudotime (tradeSeq `patternTest`, cocoa matched-null spirit).
//! - **Along a branch** ([`trend`]) — *does the rate change as the branch
//!   progresses?*, a binomial/quasi-binomial spline GAM of `logit(k/n)` on pseudotime
//!   (tradeSeq `associationTest`).

mod bayes_common;
pub mod contrast;
pub mod contrast_bayes;
/// The `faba assoc` run. Binary entry: [`run::run_assoc`].
pub mod run;

/// The row both Bayesian tests report — see [`bayes_common::BayesResult`]. Re-exported here
/// because it is the shape of their (identical) output schemas, which `run_assoc` writes.
pub use bayes_common::BayesResult;
pub mod gam;
pub mod io;
pub mod trend;
pub mod trend_bayes;

#[cfg(test)]
mod test_util;
#[cfg(test)]
mod tests;

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
