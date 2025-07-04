#![allow(dead_code)]

use crate::data::dna::*;
use crate::data::dna_stat_traits::*;
use crate::data::sam::*;
use crate::data::visitors_htslib::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};

pub use fnv::{FnvHashMap as HashMap, FnvHashSet as HashSet};

// pub use std::collections::{HashMap, HashSet};

pub struct DnaBaseFreqMap<'a> {
    position_to_count_with_cell: HashMap<i64, HashMap<CellBarcode, DnaBaseCount>>,
    cells: HashSet<CellBarcode>,
    positions: HashSet<i64>,
    cell_barcode_tag: &'a [u8],
}

impl VisitWithBamOps for DnaBaseFreqMap<'_> {}

impl DnaStatMap for DnaBaseFreqMap<'_> {
    fn add_bam_record(&mut self, bam_record: bam::Record) {
        let cell_barcode = match bam_record.aux(self.cell_barcode_tag) {
            Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
            _ => CellBarcode::Missing,
        };

        if !self.cells.contains(&cell_barcode) {
            self.cells.insert(cell_barcode.clone());
        }

        // From `https://samtools.github.io/hts-specs/SAMv1.pdf`
        //
        // All mapped segments in alignment lines are represented on the
        // forward genomic strand. For segments that have been mapped to the
        // reverse strand, the recorded SEQ is reverse complemented from the
        // original unmapped sequence and CIGAR, QUAL, and strand-sensitive
        // optional fields are reversed and thus recorded consistently with
        // the sequence bases as represented.

        let seq = bam_record.seq().as_bytes();

        // Note: these are zero-based positions
        for [rpos, gpos] in bam_record.aligned_pairs() {
            let base = Dna::from_byte(seq[rpos as usize]);
            let genome_pos = gpos + 1; // 1-based position

            if !self.positions.contains(&genome_pos) {
                self.positions.insert(genome_pos);
            }

            let freq_map = self
                .position_to_count_with_cell
                .entry(genome_pos)
                .or_default();

            let freq = freq_map
                .entry(cell_barcode.clone())
                .or_insert_with(DnaBaseCount::new);
            freq.add(base.as_ref(), 1);
        }
    }

    fn cells(&self) -> Vec<&CellBarcode> {
        self.cells.iter().collect()
    }

    fn sorted_positions(&self) -> Vec<i64> {
        let mut ret = self.positions.iter().copied().collect::<Vec<_>>();
        ret.sort();
        ret.dedup();
        ret
    }
}

impl<'a> DnaBaseFreqMap<'a> {
    pub fn new(cell_barcode_tag: &'a str) -> Self {
        Self {
            position_to_count_with_cell: HashMap::default(),
            cells: HashSet::default(),
            positions: HashSet::default(),
            cell_barcode_tag: cell_barcode_tag.as_bytes(),
        }
    }

    /// picking on base-pair frequency at a particular `pos`
    /// stratified by cell barcodes
    pub fn stratified_frequency_at(&self, pos: i64) -> Option<&HashMap<CellBarcode, DnaBaseCount>> {
        self.position_to_count_with_cell.get(&pos)
    }

    /// frequency map by position across all the cells
    pub fn marginal_frequency_map(&self) -> HashMap<i64, DnaBaseCount> {
        let mut ret: HashMap<i64, DnaBaseCount> = HashMap::default();

        for (&pos, freq_map) in self.position_to_count_with_cell.iter() {
            let accum = ret.entry(pos).or_insert_with(DnaBaseCount::new);
            for (_, freq) in freq_map.iter() {
                *accum += freq;
            }
        }
        ret
    }
}
