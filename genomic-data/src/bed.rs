use crate::gff::GeneId;
use crate::sam::Strand;
use std::hash::Hash;

#[derive(Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BedWithGene {
    pub chr: Box<str>,
    pub start: i64,
    pub stop: i64,
    pub gene: GeneId,
    pub strand: Strand,
}

/// display methylation site name
impl std::fmt::Display for BedWithGene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}-{}@{}@{}",
            self.chr, self.start, self.stop, self.gene, self.strand
        )
    }
}

#[derive(Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bed {
    pub chr: Box<str>,
    pub start: i64,
    pub stop: i64,
}

/// display methylation site name
impl std::fmt::Display for Bed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}-{}", self.chr, self.start, self.stop)
    }
}
