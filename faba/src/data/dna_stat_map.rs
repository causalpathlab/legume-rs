#![allow(dead_code)]

use crate::data::dna::*;
use crate::data::dna_stat_traits::*;
use crate::data::sam::*;
use crate::data::visitors_htslib::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};

pub use fnv::FnvHashMap as HashMap;

pub struct DnaBaseFreqMap {
    position_to_count_with_cell: Option<HashMap<i64, HashMap<CellBarcode, DnaBaseCount>>>,
    position_to_count: Option<HashMap<i64, DnaBaseCount>>,
    cell_barcode_tag: Option<Vec<u8>>,
}

impl VisitWithBamOps for DnaBaseFreqMap {}

impl DnaStatMap for DnaBaseFreqMap {
    fn add_bam_record(&mut self, bam_record: bam::Record) {
        let seq = bam_record.seq().as_bytes();

        // keeping records with cell barcode information
        if let (Some(tag), Some(map)) = (
            &self.cell_barcode_tag,
            &mut self.position_to_count_with_cell,
        ) {
            let cell_barcode = match bam_record.aux(tag) {
                Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
                _ => CellBarcode::Missing,
            };

            for [rpos, gpos] in bam_record.aligned_pairs() {
                let base = Dna::from_byte(seq[rpos as usize]);
                let genome_pos = gpos + 1;

                let freq_map = map.entry(genome_pos).or_default();

                let freq = freq_map
                    .entry(cell_barcode.clone())
                    .or_insert_with(DnaBaseCount::new);
                freq.add(base.as_ref(), 1);
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

impl DnaBaseFreqMap {
    pub fn new_with_cell_barcode(cell_barcode_tag: &str) -> Self {
        Self {
            position_to_count_with_cell: Some(HashMap::default()),
            position_to_count: None,
            cell_barcode_tag: Some(cell_barcode_tag.as_bytes().to_vec()),
        }
    }

    pub fn new() -> Self {
        Self {
            position_to_count_with_cell: None,
            position_to_count: Some(HashMap::default()),
            cell_barcode_tag: None,
        }
    }

    pub fn stratified_frequency_at(&self, pos: i64) -> Option<&HashMap<CellBarcode, DnaBaseCount>> {
        self.position_to_count_with_cell
            .as_ref()
            .and_then(|map| map.get(&pos))
    }

    pub fn marginal_frequency_map(&self) -> Option<&HashMap<i64, DnaBaseCount>> {
        self.position_to_count.as_ref()
    }
}
