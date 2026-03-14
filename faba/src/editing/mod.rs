pub mod io;
pub mod mask;
pub mod mixture;
pub mod pipeline;
pub mod sifter;

use crate::data::dna::DnaBaseCount;

/// Unified site type for base conversion events (m6A and A-to-I)
#[derive(Clone, Debug)]
pub enum ConversionSite {
    /// DART-seq m6A site: RAC pattern on forward strand, GTY on reverse
    M6A {
        m6a_pos: i64,
        conversion_pos: i64,
        wt_freq: DnaBaseCount,
        mut_freq: DnaBaseCount,
        pv: f32,
    },
    /// A-to-I RNA editing site: A->G on forward strand, T->C on reverse
    AtoI {
        editing_pos: i64,
        wt_freq: DnaBaseCount,
        mut_freq: DnaBaseCount,
        pv: f32,
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

    /// Mutant base frequencies
    pub fn mut_freq(&self) -> &DnaBaseCount {
        match self {
            ConversionSite::M6A { mut_freq, .. } => mut_freq,
            ConversionSite::AtoI { mut_freq, .. } => mut_freq,
        }
    }

    /// P-value from binomial test
    pub fn pv(&self) -> f32 {
        match self {
            ConversionSite::M6A { pv, .. } => *pv,
            ConversionSite::AtoI { pv, .. } => *pv,
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
