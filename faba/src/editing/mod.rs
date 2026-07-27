pub mod bandwidth;
pub mod bed_output;
pub mod cell_activity;
pub mod io;
pub mod mask;
pub mod mixture;
pub mod mixture_pipeline;
pub mod pipeline;
pub mod sifter;

use crate::data::dna::DnaBaseCount;

/// Why a putative site did or did not survive the test. A putative site is
/// defined by the sequencing pattern alone (RAC/GTY motif + observed WT C→U with
/// enough coverage); the effect-size and p-value checks are applied afterward,
/// and a failing site is *recorded* with the reason it missed rather than
/// dropped, so `*_unselected.parquet` explains every call. `Selected` is also
/// the value carried through discovery, before the test.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallReason {
    /// Passed every test (or not yet decided, during discovery).
    #[default]
    Selected,
    /// Too little control (MUT) coverage to confirm WT-specificity.
    LowControl,
    /// Absolute effect size `p_WT − p_MUT` below `--m6a-min-delta`.
    Delta,
    /// Missed the p-value cutoff.
    ///
    /// m6A reaches this only after clearing the control-coverage and
    /// effect-size guards. A-to-I is single-sample and has no guards above the
    /// coverage floors, so for it the p-value is the whole test.
    ///
    /// faba does NO multiplicity correction. BH needs independence or positive
    /// regression dependence, and neighbouring sites are covered by the same
    /// reads -- a read converted at one is evidence against its unconverted
    /// neighbour, so the dependence is not even reliably positive. Under
    /// arbitrary dependence the valid procedure is Benjamini-Yekutieli, whose
    /// ~ln(m) penalty is 10.6x at 28k sites. Claiming a guarantee whose
    /// assumption fails is worse than claiming none.
    Pvalue,
}

impl CallReason {
    /// Lower-case token written to the `reason` parquet column.
    pub fn label(&self) -> &'static str {
        match self {
            CallReason::Selected => "selected",
            CallReason::LowControl => "low_control",
            CallReason::Delta => "delta",
            CallReason::Pvalue => "pvalue",
        }
    }

    /// Whether this reason denotes a kept call.
    pub fn is_selected(&self) -> bool {
        matches!(self, CallReason::Selected)
    }
}

/// Unified site type for base conversion events (m6A and A-to-I)
#[derive(Clone, Debug)]
pub enum ConversionSite {
    /// DART-seq m6A site: RAC pattern on forward strand, GTY on reverse
    M6A {
        m6a_pos: i64,
        conversion_pos: i64,
        wt_freq: DnaBaseCount,
        /// MUT (catalytically-dead control) base counts at the conversion
        /// position. The m6A call is `WT conversion > MUT conversion`, so this
        /// is always populated for m6A sites.
        mut_freq: DnaBaseCount,
        /// Per-site WT-vs-MUT contrast p-value — the statistic the call is made on.
        pv: f32,
        /// Test outcome (set by the test pass; `Selected` until then).
        reason: CallReason,
    },
    /// A-to-I RNA editing site: A->G on forward strand, T->C on reverse
    AtoI {
        editing_pos: i64,
        wt_freq: DnaBaseCount,
        /// Unused for A-to-I (single-sample, reference-anchored): ADAR is active
        /// in the YTHmut control too, so there is no control to contrast against.
        /// Kept as an empty default for a uniform `ConversionSite` shape.
        mut_freq: DnaBaseCount,
        pv: f32,
        /// Test outcome (set by the test pass; `Selected` until then).
        reason: CallReason,
    },
}

impl ConversionSite {
    /// Primary genomic position for this site
    pub fn primary_pos(&self) -> i64 {
        match self {
            ConversionSite::M6A { m6a_pos, .. } => *m6a_pos,
            ConversionSite::AtoI { editing_pos, .. } => *editing_pos,
        }
    }

    /// Conversion position (same as primary_pos for AtoI)
    pub fn conversion_pos(&self) -> i64 {
        match self {
            ConversionSite::M6A { conversion_pos, .. } => *conversion_pos,
            ConversionSite::AtoI { editing_pos, .. } => *editing_pos,
        }
    }

    /// Wild-type base frequencies
    pub fn wt_freq(&self) -> &DnaBaseCount {
        match self {
            ConversionSite::M6A { wt_freq, .. } => wt_freq,
            ConversionSite::AtoI { wt_freq, .. } => wt_freq,
        }
    }

    /// MUT (control) base frequencies at the conversion position. Populated for
    /// m6A; an empty default for A-to-I (single-sample).
    pub fn mut_freq(&self) -> &DnaBaseCount {
        match self {
            ConversionSite::M6A { mut_freq, .. } => mut_freq,
            ConversionSite::AtoI { mut_freq, .. } => mut_freq,
        }
    }

    /// The site's own p-value: Fisher exact WT-vs-MUT for m6A, beta-binomial
    /// against the sequencing-error null for A-to-I.
    pub fn pv(&self) -> f32 {
        match self {
            ConversionSite::M6A { pv, .. } => *pv,
            ConversionSite::AtoI { pv, .. } => *pv,
        }
    }

    /// Test outcome (`Selected` unless the test pass recorded a rejection).
    pub fn reason(&self) -> CallReason {
        match self {
            ConversionSite::M6A { reason, .. } => *reason,
            ConversionSite::AtoI { reason, .. } => *reason,
        }
    }

    /// Set the test outcome (called by the test pass).
    pub fn set_reason(&mut self, r: CallReason) {
        match self {
            ConversionSite::M6A { reason, .. } => *reason = r,
            ConversionSite::AtoI { reason, .. } => *reason = r,
        }
    }

    /// Whether this is an m6A site
    #[cfg(test)]
    pub fn is_m6a(&self) -> bool {
        matches!(self, ConversionSite::M6A { .. })
    }

    /// Whether this is an A-to-I site
    #[cfg(test)]
    pub fn is_atoi(&self) -> bool {
        matches!(self, ConversionSite::AtoI { .. })
    }

    /// Site type label for output
    pub fn mod_type(&self) -> &'static str {
        match self {
            ConversionSite::M6A { .. } => "m6A",
            ConversionSite::AtoI { .. } => "A2I",
        }
    }
}

impl PartialEq for ConversionSite {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ConversionSite::M6A {
                    m6a_pos: a,
                    conversion_pos: ca,
                    ..
                },
                ConversionSite::M6A {
                    m6a_pos: b,
                    conversion_pos: cb,
                    ..
                },
            ) => a == b && ca == cb,
            (
                ConversionSite::AtoI { editing_pos: a, .. },
                ConversionSite::AtoI { editing_pos: b, .. },
            ) => a == b,
            _ => false,
        }
    }
}

impl Eq for ConversionSite {}

impl PartialOrd for ConversionSite {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ConversionSite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.primary_pos()
            .cmp(&other.primary_pos())
            .then_with(|| self.conversion_pos().cmp(&other.conversion_pos()))
    }
}
