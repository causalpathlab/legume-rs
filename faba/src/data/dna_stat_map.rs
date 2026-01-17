#![allow(dead_code)]

use crate::data::cell_membership::CellMembership;
use crate::data::dna::*;
use crate::data::dna_stat_traits::*;
use crate::data::sam::*;
use crate::data::visitors_htslib::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};

pub use fnv::FnvHashMap as HashMap;

pub struct DnaBaseFreqMap<'a> {
    position_to_count_with_cell: Option<HashMap<i64, HashMap<CellBarcode, DnaBaseCount>>>,
    position_to_count: Option<HashMap<i64, DnaBaseCount>>,
    cell_barcode_tag: Option<Vec<u8>>,
    anchor_position: Option<i64>,
    anchor_base: Option<Dna>,
    cell_membership: Option<&'a CellMembership>,
}

impl<'a> VisitWithBamOps for DnaBaseFreqMap<'a> {}

impl<'a> DnaStatMap for DnaBaseFreqMap<'a> {
    fn add_bam_record(&mut self, bam_record: bam::Record) {
        let seq = bam_record.seq().as_bytes();

        // Helper closure to update frequency maps
        let update_freq_map = |map: &mut HashMap<_, _>, cell_barcode: Option<CellBarcode>| {
            for [rpos, gpos] in bam_record.aligned_pairs() {
                let base = Dna::from_byte(seq[rpos as usize]);
                let genome_pos = gpos + 1;
                let freq_map: &mut HashMap<CellBarcode, DnaBaseCount> =
                    map.entry(genome_pos).or_default();
                let key = cell_barcode.clone().unwrap_or(CellBarcode::Missing);
                let freq = freq_map.entry(key).or_insert_with(DnaBaseCount::new);
                freq.add(base.as_ref(), 1);
            }
        };

        // Records with cell barcode information
        if let (Some(tag), Some(freq_map)) = (
            &self.cell_barcode_tag,
            &mut self.position_to_count_with_cell,
        ) {
            let cell_barcode = match bam_record.aux(tag) {
                Ok(Aux::String(barcode)) => Some(CellBarcode::Barcode(barcode.into())),
                _ => Some(CellBarcode::Missing),
            };

            let anchor_match = if let (Some(anchor_base), Some(anchor_pos)) =
                (&self.anchor_base, self.anchor_position)
            {
                bam_record.aligned_pairs().any(|[rpos, gpos]| {
                    Dna::from_byte(seq[rpos as usize])
                        .map_or(false, |b| anchor_pos == gpos + 1 && b == *anchor_base)
                })
            } else {
                true
            };

            // Check cell membership if filter is active
            let passes_membership = if let Some(membership) = &self.cell_membership {
                if let Some(ref cb) = cell_barcode {
                    membership.matches_barcode(cb).is_some()
                } else {
                    false
                }
            } else {
                true // No filter active
            };

            if anchor_match && passes_membership {
                update_freq_map(freq_map, cell_barcode);
            }
        }

        // just keeping the marginal frequencies
        if let Some(map) = &mut self.position_to_count {
            for [rpos, gpos] in bam_record.aligned_pairs() {
                let base = Dna::from_byte(seq[rpos as usize]);
                let genome_pos = gpos + 1;
                let freq = map.entry(genome_pos).or_insert_with(DnaBaseCount::new);
                freq.add(base.as_ref(), 1);
            }
        }
    }

    fn sorted_positions(&self) -> Vec<i64> {
        let mut ret = if let Some(map) = &self.position_to_count {
            map.keys().copied().collect::<Vec<_>>()
        } else if let Some(map) = &self.position_to_count_with_cell {
            map.keys().copied().collect::<Vec<_>>()
        } else {
            vec![]
        };
        ret.sort();
        ret.dedup();
        ret
    }
}

impl<'a> DnaBaseFreqMap<'a> {
    /// empty frequency map, keeping track of cell barcodes
    ///
    /// * `cell_barcode_tag` - tag word, e.g., "CB" in 10x
    /// * `cell_membership` - optional cell membership filter
    pub fn new_with_cell_barcode(
        cell_barcode_tag: &str,
        cell_membership: Option<&'a CellMembership>,
    ) -> Self {
        Self {
            position_to_count_with_cell: Some(HashMap::default()),
            position_to_count: None,
            cell_barcode_tag: Some(cell_barcode_tag.as_bytes().to_vec()),
            anchor_position: None,
            anchor_base: None,
            cell_membership,
        }
    }

    /// empty frequency map without keeping track of cell barcode
    ///
    /// * `cell_membership` - optional cell membership filter (only used if tracking cells)
    pub fn new(cell_membership: Option<&'a CellMembership>) -> Self {
        Self {
            position_to_count_with_cell: None,
            position_to_count: Some(HashMap::default()),
            cell_barcode_tag: None,
            anchor_position: None,
            anchor_base: None,
            cell_membership,
        }
    }

    /// Set the anchor location and base for m6A counting:
    /// - `pos` can be m6A genomic position
    /// - `base` can be the corresponding DNA base, such as `A`
    pub fn set_anchor_position(&mut self, loc: i64, base: Dna) {
        self.anchor_position = Some(loc);
        self.anchor_base = Some(base);
    }

    /// frequency stratified by cell barcodes on a genomic position `pos`
    pub fn stratified_frequency_at(&self, pos: i64) -> Option<&HashMap<CellBarcode, DnaBaseCount>> {
        self.position_to_count_with_cell
            .as_ref()
            .and_then(|map| map.get(&pos))
    }

    /// marginal frequency on a genomic position `pos`
    pub fn marginal_frequency_at(&self, pos: i64) -> Option<&DnaBaseCount> {
        self.position_to_count
            .as_ref()
            .and_then(|map| map.get(&pos))
    }

    /// marginal frequency
    pub fn marginal_frequency_map(&self) -> Option<&HashMap<i64, DnaBaseCount>> {
        self.position_to_count.as_ref()
    }
}
