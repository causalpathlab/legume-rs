//! Per-transcript 5'UTR / CDS / 3'UTR models.
//!
//! Everything else in this crate is keyed on `GeneId` and merges isoforms
//! together. That is the right model for counting reads over a gene, but it is
//! the wrong one for a metagene: a union 3'UTR runs to the most distal poly(A)
//! site any isoform uses, which is longer than the transcript most molecules
//! actually carry, so "fraction of the way along the 3'UTR" means different
//! things for different molecules of the same gene.
//!
//! This module keeps the two axes in separate namespaces on purpose. Mixing
//! them is how `gff::build_codon_map`'s cross-isoform behaviour — it returns
//! the most 3'-distal stop codon across every isoform, which need not pair with
//! the start codon it also returns — became easy to miss.
//!
//! The region split follows MetaPlotR, which is what the published m6A
//! metagenes we compare against were made with:
//!
//! Olarerin-George AO, Jaffrey SR. *MetaPlotR: a Perl/R pipeline for plotting
//! metagenes of nucleotide modifications and other transcriptomic sites.*
//! Bioinformatics 33, 1563–1564 (2017). <https://doi.org/10.1093/bioinformatics/btx002>

use crate::gff::{FeatureType, GeneId, GeneSymbol, GeneType, GffRecord, TranscriptId};
use crate::sam::Strand;
use rustc_hash::FxHashMap;

/// Sort and merge intervals that overlap or abut.
///
/// Adjacency (`s == last_stop + 1`) merges too: two records that meet
/// base-to-base are one uninterrupted stretch of transcript, and leaving them
/// split would only add a seam with no coordinate consequence.
pub fn merge_intervals(intervals: &mut Vec<(i64, i64)>) {
    intervals.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(intervals.len());
    for &(start, stop) in intervals.iter() {
        match merged.last_mut() {
            Some(last) if start <= last.1 + 1 => last.1 = last.1.max(stop),
            _ => merged.push((start, stop)),
        }
    }
    *intervals = merged;
}

/// One isoform's exon structure, split into the three transcript regions.
///
/// Intervals are 1-based inclusive, sorted and disjoint, in genomic order on
/// both strands — `strand` says which end reads 5'.
#[derive(Clone, Debug)]
pub struct TranscriptModel {
    pub gene_id: GeneId,
    pub gene_name: GeneSymbol,
    pub transcript_id: TranscriptId,
    pub seqname: Box<str>,
    pub strand: Strand,
    pub utr5: Vec<(i64, i64)>,
    pub cds: Vec<(i64, i64)>,
    pub utr3: Vec<(i64, i64)>,
    /// Spliced lengths — introns consume none of them.
    pub utr5_size: i64,
    pub cds_size: i64,
    pub utr3_size: i64,
}

impl TranscriptModel {
    /// Spliced length of the mature transcript, the quantity the longest-isoform
    /// election ranks on.
    pub fn trx_len(&self) -> i64 {
        self.utr5_size + self.cds_size + self.utr3_size
    }
}

/// Everything one transcript's records contribute, before the CDS split.
///
/// The identity fields are seeded from the first record and are never absent
/// afterwards, so they are stored plainly rather than as `Option`s that every
/// reader would have to unwrap. Only `cds` is genuinely optional — a transcript
/// with no CDS record is not a coding transcript and gets dropped.
struct Builder {
    gene_id: GeneId,
    gene_name: GeneSymbol,
    strand: Strand,
    exons: Vec<(i64, i64)>,
    /// Genomic extent of the coding sequence, widened over the stop codon.
    cds: Option<(i64, i64)>,
}

impl Builder {
    fn new(rec: &GffRecord) -> Self {
        Self {
            gene_id: rec.gene_id.clone(),
            gene_name: rec.gene_name.clone(),
            strand: rec.strand,
            exons: Vec::new(),
            cds: None,
        }
    }

    fn widen_cds(&mut self, start: i64, stop: i64) {
        self.cds = Some(match self.cds {
            Some((lo, hi)) => (lo.min(start), hi.max(stop)),
            None => (start, stop),
        });
    }
}

/// Build one model per coding transcript.
///
/// Only `exon`, `CDS` and `stop_codon` records are read. The coding extent is
/// the transcript's own CDS span widened over its own stop codon: GENCODE
/// excludes the stop codon from `CDS`, whereas the UCSC gene-prediction tables
/// MetaPlotR consumes fold it into `cdsEnd`. Without that widening three bases
/// per transcript migrate into the 3'UTR, and a site inside a stop codon lands
/// in the FIRST 3'UTR bin — exactly the region an m6A metagene is read for.
///
/// The generic `UTR` / `five_prime_UTR` / `three_prime_UTR` records are ignored
/// entirely: exons plus the coding extent already determine the split, so there
/// is no codon-distance heuristic to get wrong.
pub fn build_transcript_models(records: &[GffRecord]) -> Vec<TranscriptModel> {
    // Nested seqname -> transcript id, for two reasons.
    //
    // Identity: `parse_ensembl_id` drops the `_PAR_Y` suffix, so the X and Y
    // copies of a pseudoautosomal transcript share an id. Pooling on the id
    // alone would place chrY exons on chrX and inflate the spliced length.
    // GENCODE v46 carries no `_PAR_Y` transcripts; older releases do.
    //
    // Cost: both levels probe by `&str`, so the hot loop allocates only on
    // first sight of a transcript. Keying one map on an owned
    // `(TranscriptId, Box<str>)` makes `entry` clone BOTH halves for every one
    // of ~2.1M qualifying GENCODE records and discard all but the ~111k that
    // open a new builder.
    let mut by_chr: FxHashMap<Box<str>, FxHashMap<Box<str>, Builder>> = FxHashMap::default();

    for rec in records.iter() {
        if rec.gene_type != GeneType::CodingGene || rec.stop < rec.start {
            continue;
        }
        let TranscriptId::Ensembl(ref tx) = rec.transcript_id else {
            continue; // gene-level rows carry no transcript
        };
        // One dispatch on the feature type: everything else is never read.
        let is_exon = match rec.feature_type {
            FeatureType::Exon => true,
            FeatureType::CDS | FeatureType::StopCodon => false,
            _ => continue,
        };

        let by_tx = match by_chr.get_mut(&*rec.seqname) {
            Some(m) => m,
            None => by_chr.entry(rec.seqname.clone()).or_default(),
        };
        let b = match by_tx.get_mut(&**tx) {
            Some(b) => b,
            None => by_tx.entry(tx.clone()).or_insert_with(|| Builder::new(rec)),
        };

        if is_exon {
            b.exons.push((rec.start, rec.stop));
        } else {
            b.widen_cds(rec.start, rec.stop);
        }
    }

    by_chr
        .into_iter()
        .flat_map(|(seqname, by_tx)| {
            by_tx.into_iter().filter_map(move |(tx_id, mut b)| {
                let (cds_lo, cds_hi) = b.cds?;
                if b.exons.is_empty() {
                    return None;
                }
                merge_intervals(&mut b.exons);

                // Split the spliced exon set on the coding extent. `lower` and
                // `upper` are genomic, so which one is the 5' flank depends on
                // the strand; an exon straddling a boundary feeds both sides.
                let mut lower = Vec::new();
                let mut middle = Vec::with_capacity(b.exons.len());
                let mut upper = Vec::new();
                for &(s, e) in b.exons.iter() {
                    if s < cds_lo {
                        lower.push((s, e.min(cds_lo - 1)));
                    }
                    if e > cds_hi {
                        upper.push((s.max(cds_hi + 1), e));
                    }
                    let (ms, me) = (s.max(cds_lo), e.min(cds_hi));
                    if ms <= me {
                        middle.push((ms, me));
                    }
                }
                if middle.is_empty() {
                    return None; // no coding exon: not a coding transcript here
                }

                let (utr5, utr3) = match b.strand {
                    Strand::Forward => (lower, upper),
                    Strand::Backward => (upper, lower),
                };
                let spliced = |v: &Vec<(i64, i64)>| v.iter().map(|&(s, e)| e - s + 1).sum::<i64>();

                Some(TranscriptModel {
                    gene_id: b.gene_id,
                    gene_name: b.gene_name,
                    transcript_id: TranscriptId::Ensembl(tx_id),
                    seqname: seqname.clone(),
                    strand: b.strand,
                    utr5_size: spliced(&utr5),
                    cds_size: spliced(&middle),
                    utr3_size: spliced(&utr3),
                    utr5,
                    cds: middle,
                    utr3,
                })
            })
        })
        .collect()
}

/// Keep one transcript per gene: the longest spliced one.
///
/// This is MetaPlotR's stated procedure. Its published `visualize_metagenes.R`
/// does not implement it — `dist[duplicated(dist$gene_name), ]` keeps rows two
/// through N, which drops every single-isoform gene outright and keeps all but
/// the shortest isoform of the rest — and the README variant de-duplicates an
/// unsorted table, so it elects whichever transcript came first in the file.
/// We follow the intent, not either implementation, and say so.
///
/// Ties break on `transcript_id` rather than on file order: `read_gff_record_vec`
/// collects through `par_bridge`, so record order is not reproducible and a
/// file-order tie-break would make the elected set vary between runs.
pub fn elect_longest_isoform(models: Vec<TranscriptModel>) -> Vec<TranscriptModel> {
    // The rule as one lexicographic key: longest wins, smaller id breaks ties.
    let key = |m: &TranscriptModel| (m.trx_len(), std::cmp::Reverse(m.transcript_id.clone()));

    // Keyed on gene AND sequence name, for the same reason the builder above
    // nests its map: `parse_ensembl_id` drops the `_PAR_Y` suffix, so the chrX
    // and chrY copies of a pseudoautosomal gene share a GeneId. Electing on the
    // id alone would put them in competition and discard one chromosome's copy
    // — undoing the separation the builder went out of its way to keep.
    let mut best: FxHashMap<(GeneId, Box<str>), TranscriptModel> = FxHashMap::default();
    for m in models {
        let k = (m.gene_id.clone(), m.seqname.clone());
        let replace = best.get(&k).is_none_or(|cur| key(&m) > key(cur));
        if replace {
            best.insert(k, m);
        }
    }
    best.into_values().collect()
}

#[cfg(test)]
mod tests;
