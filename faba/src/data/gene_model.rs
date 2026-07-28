//! Transcript (spliced) coordinates for a gene.
//!
//! A position's offset "within a gene" has two meanings, and only one of them is
//! a transcript coordinate. Measured from the gene's start in **genomic** space,
//! an intron between the start and the position counts toward the offset — for a
//! typical human gene the introns dominate, so the number says more about the
//! gene's intron content than about where in the mRNA the position sits.
//!
//! This module supplies the other meaning: the offset along the gene's merged
//! exons, which is what a reader means by "position in the transcript". APA's
//! 3'UTR takes the same correction — see `docs/profiling-methods.md` section
//! 1.1. The metagene takes it along one elected transcript instead (§1.2).

use genomic_data::gff::{build_exon_intervals, GeneId, GffRecord};
use genomic_data::sam::Strand;
use rustc_hash::FxHashMap;

/////////////////////
// Spliced genes   //
/////////////////////

/// Merged exon intervals per gene, as 0-based half-open `[start, end)`.
///
/// Built from `FeatureType::Exon` records pooled over every isoform, so the
/// model is **gene-level, merged across isoforms** — a base exonic in any
/// isoform is exonic here. That is deliberate: a read at such a base is real
/// evidence, and electing one canonical transcript would discard it.
///
/// `faba metagene` is the one exception, and elects the longest transcript
/// (`genomic_data::transcript`). Its estimand is different: it reports *where
/// along a transcript* sites fall, for comparison against published MetaPlotR
/// profiles, and that comparison needs one transcript's disjoint regions. What
/// reads this type — `rel_pos`, the methylation mixture, APA's 3'UTR — is
/// asking whether a base is exonic or how a 3' end is used, where any-isoform
/// evidence is the right answer. See `docs/profiling-methods.md` §1.2.
#[derive(Default)]
pub struct SplicedGenes {
    exons: FxHashMap<GeneId, Vec<(i64, i64)>>,
}

impl SplicedGenes {
    /// Merged exons from records the caller has ALREADY parsed.
    ///
    /// Takes records rather than a path because every caller has just parsed
    /// the same GFF to build its `GffRecordMap`, and a second parse of GENCODE
    /// costs ~9s. `GffRecordMap::from_map` exists for the same reason; this is
    /// the other half of that pair.
    pub fn from_records(records: &[GffRecord]) -> Self {
        Self {
            exons: build_exon_intervals(records).into_iter().collect(),
        }
    }

    /// Strand-aware offset of `pos` (0-based genomic) along the gene's merged
    /// exons, counting 5'->3'.
    ///
    /// `None` when the gene has no exon model or the position is intronic —
    /// there is no transcript coordinate for a base that is not in the
    /// transcript, and inventing one (the nearest exon edge, say) would put a
    /// value in the column that no downstream reader could tell from a real one.
    pub fn rel_pos(&self, gene: &GeneId, pos: i64, strand: Strand) -> Option<i64> {
        let exons = self.exons.get(gene)?;
        let total = Self::total_len(exons);
        let mut before = 0i64;
        for &(start, end) in exons.iter() {
            if pos < start {
                return None; // fell in the gap before this exon
            }
            if pos < end {
                let forward = before + (pos - start);
                return Some(match strand {
                    Strand::Forward => forward,
                    Strand::Backward => total - 1 - forward,
                });
            }
            before += end - start;
        }
        None
    }

    /// The gene's spliced (mature transcript) length.
    ///
    /// The partner of [`Self::rel_pos`]: a position and the length that
    /// normalises it must be measured on the same axis. Pairing a spliced
    /// offset with a genomic span would squeeze every site into the leading few
    /// percent of the range, since introns are most of a typical human gene.
    pub fn spliced_len(&self, gene: &GeneId) -> Option<i64> {
        self.exons.get(gene).map(|e| Self::total_len(e))
    }

    fn total_len(exons: &[(i64, i64)]) -> i64 {
        exons.iter().map(|&(s, e)| e - s).sum()
    }
}

#[cfg(test)]
mod tests;
