use crate::data::bam_io;
use crate::data::cell_membership::CellMembership;
use crate::data::dna::*;
use genomic_data::bed::Bed;
use genomic_data::gff::GffRecord;
use genomic_data::sam::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};
use std::hash::{Hash, Hasher};

pub use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet;

/// Hash a UMI string to u64 for compact dedup sets.
fn hash_umi(umi: &str) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    umi.hash(&mut h);
    h.finish()
}

pub enum CountMode<'a> {
    Marginal {
        counts: HashMap<i64, DnaBaseCount>,
        umi_seen: FxHashSet<(i64, u64)>,
    },
    PerCell {
        counts: HashMap<i64, HashMap<CellBarcode, DnaBaseCount>>,
        cell_barcode_tag: Vec<u8>,
        cell_membership: Option<&'a CellMembership>,
        umi_seen: FxHashSet<(i64, u64, CellBarcode)>,
    },
}

pub struct DnaBaseFreqMap<'a> {
    mode: CountMode<'a>,
    anchor: Option<(i64, Dna)>,
    position_filter: Option<FxHashSet<i64>>,
    min_base_quality: u8,
    min_mapping_quality: u8,
    umi_tag: Option<Vec<u8>>,
    use_base_quality: bool,
    quality_map: HashMap<i64, DnaBaseQual>,
}

impl<'a> Default for DnaBaseFreqMap<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> DnaBaseFreqMap<'a> {
    pub fn new() -> Self {
        Self {
            mode: CountMode::Marginal {
                counts: HashMap::default(),
                umi_seen: FxHashSet::default(),
            },
            anchor: None,
            position_filter: None,
            min_base_quality: 20,
            min_mapping_quality: 20,
            umi_tag: None,
            use_base_quality: false,
            quality_map: HashMap::default(),
        }
    }

    /// Empty frequency map, keeping track of cell barcodes
    pub fn new_with_cell_barcode(
        cell_barcode_tag: &str,
        cell_membership: Option<&'a CellMembership>,
    ) -> Self {
        Self {
            mode: CountMode::PerCell {
                counts: HashMap::default(),
                cell_barcode_tag: cell_barcode_tag.as_bytes().to_vec(),
                cell_membership,
                umi_seen: FxHashSet::default(),
            },
            anchor: None,
            position_filter: None,
            min_base_quality: 20,
            min_mapping_quality: 20,
            umi_tag: None,
            use_base_quality: false,
            quality_map: HashMap::default(),
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
    pub fn set_position_filter(&mut self, positions: FxHashSet<i64>) {
        self.position_filter = Some(positions);
    }

    pub fn set_quality_thresholds(&mut self, min_base_quality: u8, min_mapping_quality: u8) {
        self.min_base_quality = min_base_quality;
        self.min_mapping_quality = min_mapping_quality;
    }

    /// Enable UMI-based deduplication. Reads sharing the same UMI at the same
    /// position (and same cell barcode in per-cell mode) are counted only once.
    pub fn set_umi_tag(&mut self, tag: &str) {
        self.umi_tag = Some(tag.as_bytes().to_vec());
    }

    /// Enable per-base quality accumulation for Li 2011 genotype likelihood model.
    pub fn set_use_base_quality(&mut self, enabled: bool) {
        self.use_base_quality = enabled;
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

    /// Extract UMI hash from a BAM record, if umi_tag is set.
    fn extract_umi_hash(&self, bam_record: &bam::Record) -> Option<u64> {
        let tag = self.umi_tag.as_ref()?;
        match bam_record.aux(tag) {
            Ok(Aux::String(s)) => Some(hash_umi(s)),
            _ => None,
        }
    }

    fn add_bam_record(&mut self, bam_record: bam::Record) {
        if bam_record.mapq() < self.min_mapping_quality {
            return;
        }
        if bam_record.is_duplicate() {
            return;
        }
        if bam_record.is_secondary() || bam_record.is_supplementary() {
            return;
        }
        if bam_record.is_paired() && !bam_record.is_proper_pair() {
            return;
        }

        let seq = bam_record.seq().as_bytes();
        let umi_hash = self.extract_umi_hash(&bam_record);

        match &mut self.mode {
            CountMode::Marginal { counts, umi_seen } => {
                let quals = bam_record.qual();

                for [rpos, gpos] in bam_record.aligned_pairs() {
                    if let Some(filter) = &self.position_filter {
                        if !filter.contains(&gpos) {
                            continue;
                        }
                    }

                    let phred = quals[rpos as usize];
                    if phred < self.min_base_quality {
                        continue;
                    }

                    if let Some(h) = umi_hash {
                        if !umi_seen.insert((gpos, h)) {
                            continue;
                        }
                    }

                    let base = Dna::from_byte(seq[rpos as usize]);
                    counts.entry(gpos).or_default().add(base.as_ref(), 1);

                    if self.use_base_quality {
                        self.quality_map
                            .entry(gpos)
                            .or_default()
                            .add(base.as_ref(), phred);
                    }
                }
            }

            CountMode::PerCell {
                counts,
                cell_barcode_tag,
                cell_membership,
                umi_seen,
            } => {
                let cell_barcode = bam_io::extract_cell_barcode(&bam_record, cell_barcode_tag);

                if let Some(membership) = cell_membership {
                    if membership.matches_barcode(&cell_barcode).is_none() {
                        return;
                    }
                }

                let anchor = self.anchor;
                let mut anchor_matched = anchor.is_none();
                let mut pending: Vec<(i64, Option<Dna>, u8)> = Vec::new();
                let quals = bam_record.qual();

                for [rpos, gpos] in bam_record.aligned_pairs() {
                    if let Some(filter) = &self.position_filter {
                        if !filter.contains(&gpos) {
                            continue;
                        }
                    }

                    let phred = quals[rpos as usize];
                    if phred < self.min_base_quality {
                        continue;
                    }

                    let base = Dna::from_byte(seq[rpos as usize]);
                    pending.push((gpos, base, phred));
                    if let Some((anchor_pos, anchor_base)) = anchor {
                        if gpos == anchor_pos && base.is_some_and(|b| b == anchor_base) {
                            anchor_matched = true;
                        }
                    }
                }

                if anchor_matched {
                    for (gpos, base, _phred) in pending {
                        if let Some(h) = umi_hash {
                            if !umi_seen.insert((gpos, h, cell_barcode.clone())) {
                                continue;
                            }
                        }

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
            CountMode::Marginal { counts, .. } => counts.keys().copied().collect(),
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
            CountMode::Marginal { counts, .. } => Some(counts),
            _ => None,
        }
    }

    /// Quality accumulator map (marginal mode only, requires set_use_base_quality(true))
    pub fn quality_map(&self) -> &HashMap<i64, DnaBaseQual> {
        &self.quality_map
    }
}
