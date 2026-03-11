#![allow(dead_code)]

use crate::data::bam_io;
use crate::data::cell_membership::CellMembership;
use crate::data::dna::*;
use genomic_data::bed::Bed;
use genomic_data::gff::GffRecord;
use genomic_data::sam::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions};

pub use fnv::FnvHashMap as HashMap;

pub enum CountMode<'a> {
    /// Marginal/bulk statistics only — no cell barcode tracking
    Marginal { counts: HashMap<i64, DnaBaseCount> },
    /// Per-cell statistics with optional membership filter
    PerCell {
        counts: HashMap<i64, HashMap<CellBarcode, DnaBaseCount>>,
        cell_barcode_tag: Vec<u8>,
        cell_membership: Option<&'a CellMembership>,
    },
    /// Aggregated marginal counts filtered to a specific cell type
    CellTypeFiltered {
        counts: HashMap<i64, DnaBaseCount>,
        cell_barcode_tag: Vec<u8>,
        cell_membership: &'a CellMembership,
        target_celltype: Box<str>,
    },
}

pub struct DnaBaseFreqMap<'a> {
    mode: CountMode<'a>,
    anchor: Option<(i64, Dna)>,
}

impl<'a> Default for DnaBaseFreqMap<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> DnaBaseFreqMap<'a> {
    /// Empty frequency map without keeping track of cell barcode
    /// (computes marginal/bulk statistics only)
    pub fn new() -> Self {
        Self {
            mode: CountMode::Marginal {
                counts: HashMap::default(),
            },
            anchor: None,
        }
    }

    /// Empty frequency map, keeping track of cell barcodes
    ///
    /// * `cell_barcode_tag` - tag word, e.g., "CB" in 10x
    /// * `cell_membership` - optional cell membership filter
    pub fn new_with_cell_barcode(
        cell_barcode_tag: &str,
        cell_membership: Option<&'a CellMembership>,
    ) -> Self {
        Self {
            mode: CountMode::PerCell {
                counts: HashMap::default(),
                cell_barcode_tag: cell_barcode_tag.as_bytes().to_vec(),
                cell_membership,
            },
            anchor: None,
        }
    }

    /// Frequency map for a specific cell type only.
    /// Only cells matching the target cell type will be included.
    ///
    /// * `cell_barcode_tag` - tag word, e.g., "CB" in 10x
    /// * `cell_membership` - cell membership for filtering
    /// * `target_celltype` - only include cells of this type
    pub fn new_for_celltype(
        cell_barcode_tag: &str,
        cell_membership: &'a CellMembership,
        target_celltype: &str,
    ) -> Self {
        Self {
            mode: CountMode::CellTypeFiltered {
                counts: HashMap::default(),
                cell_barcode_tag: cell_barcode_tag.as_bytes().to_vec(),
                cell_membership,
                target_celltype: target_celltype.into(),
            },
            anchor: None,
        }
    }

    /// Set the anchor location and base for m6A counting:
    /// - `pos` can be m6A genomic position
    /// - `base` can be the corresponding DNA base, such as `A`
    pub fn set_anchor_position(&mut self, loc: i64, base: Dna) {
        self.anchor = Some((loc, base));
    }

    /// Update from BAM records in a BED region
    pub fn update_from_region(&mut self, bam_file_path: &str, region: &Bed) -> anyhow::Result<()> {
        bam_io::for_each_record_in_region(bam_file_path, region, |_chr, rec| {
            self.add_bam_record(rec);
        })
    }

    /// Update from BAM records matching a gene barcode
    pub fn update_from_gene(
        &mut self,
        bam_file_path: &str,
        gff_record: &GffRecord,
        gene_barcode_tag: &str,
        include_missing_barcode: bool,
    ) -> anyhow::Result<()> {
        bam_io::for_each_record_in_gene(
            bam_file_path,
            gff_record,
            gene_barcode_tag,
            include_missing_barcode,
            |rec| self.add_bam_record(rec),
        )
    }

    /// Accumulate base frequencies from aligned pairs into a marginal count map.
    fn accumulate_marginal(
        seq: &[u8],
        bam_record: &bam::Record,
        counts: &mut HashMap<i64, DnaBaseCount>,
    ) {
        for [rpos, gpos] in bam_record.aligned_pairs() {
            let base = Dna::from_byte(seq[rpos as usize]);
            let freq = counts.entry(gpos).or_default();
            freq.add(base.as_ref(), 1);
        }
    }

    fn add_bam_record(&mut self, bam_record: bam::Record) {
        let seq = bam_record.seq().as_bytes();

        match &mut self.mode {
            CountMode::Marginal { counts } => {
                Self::accumulate_marginal(&seq, &bam_record, counts);
            }

            CountMode::PerCell {
                counts,
                cell_barcode_tag,
                cell_membership,
            } => {
                let cell_barcode = bam_io::extract_cell_barcode(&bam_record, cell_barcode_tag);

                let passes_membership = if let Some(membership) = cell_membership {
                    membership.matches_barcode(&cell_barcode).is_some()
                } else {
                    true
                };

                if !passes_membership {
                    return;
                }

                // Single pass: check anchor while accumulating counts.
                // If anchor is set and not matched by end, discard the accumulated data.
                let anchor = self.anchor;
                let mut anchor_matched = anchor.is_none();
                let mut pending: Vec<(i64, Option<Dna>)> = Vec::new();

                for [rpos, gpos] in bam_record.aligned_pairs() {
                    let base = Dna::from_byte(seq[rpos as usize]);
                    pending.push((gpos, base));
                    if let Some((anchor_pos, anchor_base)) = anchor {
                        if gpos == anchor_pos && base.is_some_and(|b| b == anchor_base) {
                            anchor_matched = true;
                        }
                    }
                }

                if anchor_matched {
                    for (gpos, base) in pending {
                        let freq_map: &mut HashMap<CellBarcode, DnaBaseCount> =
                            counts.entry(gpos).or_default();
                        let freq = freq_map.entry(cell_barcode.clone()).or_default();
                        freq.add(base.as_ref(), 1);
                    }
                }
            }

            CountMode::CellTypeFiltered {
                counts,
                cell_barcode_tag,
                cell_membership,
                target_celltype,
            } => {
                let cell_barcode = bam_io::extract_cell_barcode(&bam_record, cell_barcode_tag);

                if cell_membership.matches_celltype(&cell_barcode, target_celltype) {
                    Self::accumulate_marginal(&seq, &bam_record, counts);
                }
            }
        }
    }

    /// Output sorted genomic positions
    pub fn sorted_positions(&self) -> Vec<i64> {
        let mut ret: Vec<i64> = match &self.mode {
            CountMode::Marginal { counts } => counts.keys().copied().collect(),
            CountMode::PerCell { counts, .. } => counts.keys().copied().collect(),
            CountMode::CellTypeFiltered { counts, .. } => counts.keys().copied().collect(),
        };
        ret.sort();
        ret
    }

    /// Frequency stratified by cell barcodes on a genomic position `pos`
    pub fn stratified_frequency_at(&self, pos: i64) -> Option<&HashMap<CellBarcode, DnaBaseCount>> {
        match &self.mode {
            CountMode::PerCell { counts, .. } => counts.get(&pos),
            _ => None,
        }
    }

    /// Marginal frequency on a genomic position `pos`
    pub fn marginal_frequency_at(&self, pos: i64) -> Option<&DnaBaseCount> {
        match &self.mode {
            CountMode::Marginal { counts } | CountMode::CellTypeFiltered { counts, .. } => {
                counts.get(&pos)
            }
            _ => None,
        }
    }

    /// Marginal frequency map
    pub fn marginal_frequency_map(&self) -> Option<&HashMap<i64, DnaBaseCount>> {
        match &self.mode {
            CountMode::Marginal { counts } | CountMode::CellTypeFiltered { counts, .. } => {
                Some(counts)
            }
            _ => None,
        }
    }
}
