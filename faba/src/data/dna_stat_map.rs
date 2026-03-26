use crate::data::bam_io;
use crate::data::cell_membership::CellMembership;
use crate::data::dna::*;
use genomic_data::bed::Bed;
use genomic_data::gff::GffRecord;
use genomic_data::sam::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions};

pub use rustc_hash::FxHashMap as HashMap;

pub enum CountMode<'a> {
    /// Marginal/bulk statistics only — no cell barcode tracking
    Marginal { counts: HashMap<i64, DnaBaseCount> },
    /// Per-cell statistics with optional membership filter
    PerCell {
        counts: HashMap<i64, HashMap<CellBarcode, DnaBaseCount>>,
        cell_barcode_tag: Vec<u8>,
        cell_membership: Option<&'a CellMembership>,
    },
}

pub struct DnaBaseFreqMap<'a> {
    mode: CountMode<'a>,
    anchor: Option<(i64, Dna)>,
    position_filter: Option<rustc_hash::FxHashSet<i64>>,
    min_base_quality: u8,
    min_mapping_quality: u8,
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
            position_filter: None,
            min_base_quality: 20,
            min_mapping_quality: 20,
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
            position_filter: None,
            min_base_quality: 20,
            min_mapping_quality: 20,
        }
    }

    /// Set the anchor location and base for m6A counting:
    /// only count bases on reads that cover the anchor position with the expected base.
    #[allow(dead_code)]
    pub fn set_anchor_position(&mut self, loc: i64, base: Dna) {
        self.anchor = Some((loc, base));
    }

    /// Set position filter to only accumulate frequencies for specific positions.
    /// This dramatically reduces memory usage when processing many sites in a large gene.
    /// If None, all positions in the region are stored (default behavior).
    pub fn set_position_filter(&mut self, positions: rustc_hash::FxHashSet<i64>) {
        self.position_filter = Some(positions);
    }

    /// Set quality thresholds for filtering bases and reads.
    ///
    /// * `min_base_quality` - minimum Phred base quality score (default: 20)
    /// * `min_mapping_quality` - minimum MAPQ score (default: 20)
    pub fn set_quality_thresholds(&mut self, min_base_quality: u8, min_mapping_quality: u8) {
        self.min_base_quality = min_base_quality;
        self.min_mapping_quality = min_mapping_quality;
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
        position_filter: Option<&rustc_hash::FxHashSet<i64>>,
        min_base_quality: u8,
    ) {
        let quals = bam_record.qual();

        for [rpos, gpos] in bam_record.aligned_pairs() {
            // Skip positions not in filter (if filter is set)
            if let Some(filter) = position_filter {
                if !filter.contains(&gpos) {
                    continue;
                }
            }

            // Filter by base quality
            if quals[rpos as usize] < min_base_quality {
                continue;
            }

            let base = Dna::from_byte(seq[rpos as usize]);
            let freq = counts.entry(gpos).or_default();
            freq.add(base.as_ref(), 1);
        }
    }

    fn add_bam_record(&mut self, bam_record: bam::Record) {
        // Quality filters (consistent with gene counting)

        // 1. Mapping quality
        if bam_record.mapq() < self.min_mapping_quality {
            return;
        }

        // 2. Skip PCR duplicates
        if bam_record.is_duplicate() {
            return;
        }

        // 3. Skip secondary/supplementary alignments (multi-mappers)
        if bam_record.is_secondary() || bam_record.is_supplementary() {
            return;
        }

        // 4. For paired-end reads, skip if not properly paired
        if bam_record.is_paired() && !bam_record.is_proper_pair() {
            return;
        }

        let seq = bam_record.seq().as_bytes();

        match &mut self.mode {
            CountMode::Marginal { counts } => {
                Self::accumulate_marginal(
                    &seq,
                    &bam_record,
                    counts,
                    self.position_filter.as_ref(),
                    self.min_base_quality,
                );
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
                let quals = bam_record.qual();

                for [rpos, gpos] in bam_record.aligned_pairs() {
                    // Skip positions not in filter (if filter is set)
                    if let Some(filter) = &self.position_filter {
                        if !filter.contains(&gpos) {
                            continue;
                        }
                    }

                    // Filter by base quality
                    if quals[rpos as usize] < self.min_base_quality {
                        continue;
                    }

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
        }
    }

    /// Output sorted genomic positions
    pub fn sorted_positions(&self) -> Vec<i64> {
        let mut ret: Vec<i64> = match &self.mode {
            CountMode::Marginal { counts } => counts.keys().copied().collect(),
            CountMode::PerCell { counts, .. } => counts.keys().copied().collect(),
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

    /// Marginal frequency map
    pub fn marginal_frequency_map(&self) -> Option<&HashMap<i64, DnaBaseCount>> {
        match &self.mode {
            CountMode::Marginal { counts } => Some(counts),
            _ => None,
        }
    }
}
