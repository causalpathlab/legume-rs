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
use genomic_data::sam::Strand;

/// Why a putative site did or did not survive the test. A putative site is
/// defined by the sequencing pattern alone (RAC/GTY motif + observed WT C→U with
/// enough coverage); the odds-ratio and p-value checks are applied afterward,
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
    /// Log odds ratio `ln((a_w·u_m)/(u_w·a_m))` below `--m6a-min-log-odds`.
    ///
    /// The default floor is near zero, so in practice this means the WT arm did
    /// not out-convert the control at all. A genomic C/T variant converts equally
    /// in both arms, so its odds ratio is exactly 1 and it lands here at any
    /// depth — the one job this guard has. Replaced `Delta`, an absolute
    /// `p_WT − p_MUT` floor; see [`faba::hypothesis_tests::log_odds_ratio`] for
    /// why that was the wrong scale.
    OddsRatio,
    /// Missed the p-value cutoff.
    ///
    /// m6A reaches this only after clearing the control-coverage and
    /// odds-ratio guards. A-to-I is single-sample and has no guards above the
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
            CallReason::OddsRatio => "odds_ratio",
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

    /// The WT-vs-MUT contrast 2×2 as `(a_w, u_w, a_m, u_m)` — converted and
    /// unconverted counts for each arm — or `None` when this site has no
    /// contrast to take.
    ///
    /// ```text
    ///           converted   unconverted
    ///   WT      a_w         u_w
    ///   MUT     a_m         u_m
    /// ```
    ///
    /// `None` for A-to-I, and that is the point of the `Option`: A-to-I is
    /// single-sample (ADAR is active in the control too), so its `mut_freq` is an
    /// empty placeholder and any 2×2 built from it would read as a measured
    /// absence of effect rather than an absent measurement. Returning `None`
    /// makes the type answer that question once, instead of each caller
    /// remembering to match on the variant and decide what the empty arm means.
    ///
    /// Strand-resolved, because the deamination is read on the transcript:
    /// forward reads C→T (converted T, unconverted C), reverse reads G→A. This is
    /// the same table [`cell_activity::scan::channel_bases`] holds for the
    /// per-cell scan; the two are not yet shared because that one keys off a
    /// `ModificationType`, which a site does not carry.
    pub fn contrast_counts(&self, strand: Strand) -> Option<(u64, u64, u64, u64)> {
        let ConversionSite::M6A {
            wt_freq, mut_freq, ..
        } = self
        else {
            return None;
        };
        Some(match strand {
            Strand::Forward => (
                wt_freq.count_t() as u64,
                wt_freq.count_c() as u64,
                mut_freq.count_t() as u64,
                mut_freq.count_c() as u64,
            ),
            Strand::Backward => (
                wt_freq.count_a() as u64,
                wt_freq.count_g() as u64,
                mut_freq.count_a() as u64,
                mut_freq.count_g() as u64,
            ),
        })
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
