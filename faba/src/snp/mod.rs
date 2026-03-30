pub mod genotyper;
pub mod io;
pub mod pipeline;

use crate::data::dna::DnaBaseCount;
use std::fmt;

/// Genotype call at a known SNP site
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnpGenotype {
    HomRef,
    Het,
    HomAlt,
    NoCall,
}

impl fmt::Display for SnpGenotype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SnpGenotype::HomRef => write!(f, "0/0"),
            SnpGenotype::Het => write!(f, "0/1"),
            SnpGenotype::HomAlt => write!(f, "1/1"),
            SnpGenotype::NoCall => write!(f, "./."),
        }
    }
}

/// A SNP site with pileup evidence and genotype call
#[derive(Clone, Debug)]
pub struct SnpSite {
    pub chr: Box<str>,
    pub pos: i64,
    pub ref_allele: u8,
    pub alt_allele: u8,
    pub rsid: Option<Box<str>>,
    pub counts: DnaBaseCount,
    pub genotype: SnpGenotype,
    /// Phred-scaled genotype quality
    pub gq: f32,
}

#[allow(dead_code)]
impl SnpSite {
    pub fn ref_count(&self) -> usize {
        self.counts
            .get(crate::data::dna::Dna::from_byte(self.ref_allele).as_ref())
    }

    pub fn alt_count(&self) -> usize {
        self.counts
            .get(crate::data::dna::Dna::from_byte(self.alt_allele).as_ref())
    }

    pub fn depth(&self) -> usize {
        self.counts.total()
    }
}
