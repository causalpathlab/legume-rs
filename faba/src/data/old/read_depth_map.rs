// #![allow(dead_code)]

use crate::data::dna_stat_traits::*;
use crate::data::sam::*;
use crate::data::visitors_htslib::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};

use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;

pub struct ReadDepthMap<'a> {
    position_to_depth_with_cell: HashMap<i64, HashMap<CellBarcode, usize>>,
    cells: HashSet<CellBarcode>,
    positions: HashSet<i64>,
    cell_barcode_tag: &'a [u8],
}

impl VisitWithBamOps for ReadDepthMap<'_> {}

impl DnaStatMap for ReadDepthMap<'_> {
    fn add_bam_record(&mut self, bam_record: bam::Record) {
        let cell_barcode = match bam_record.aux(self.cell_barcode_tag) {
            Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
            _ => CellBarcode::Missing,
        };

        if !self.cells.contains(&cell_barcode) {
            self.cells.insert(cell_barcode.clone());
        }

        // Note: these are zero-based positions
        for [_rpos, gpos] in bam_record.aligned_pairs() {
            let genome_pos = gpos + 1; // 1-based position

            if !self.positions.contains(&genome_pos) {
                self.positions.insert(genome_pos);
            }

            let freq_map = self
                .position_to_depth_with_cell
                .entry(genome_pos)
                .or_default();

            let freq = freq_map.entry(cell_barcode.clone()).or_default();
            *freq += 1;
        }
    }

    fn sorted_positions(&self) -> Vec<i64> {
        let mut ret = self.positions.iter().copied().collect::<Vec<_>>();
        ret.sort();
        ret.dedup();
        ret
    }
}

impl<'a> ReadDepthMap<'a> {
    pub fn new_with_cell_barcode(cell_barcode_tag: &'a str) -> Self {
        Self {
            position_to_depth_with_cell: HashMap::default(),
            cells: HashSet::default(),
            positions: HashSet::default(),
            cell_barcode_tag: cell_barcode_tag.as_bytes(),
        }
    }

    /// picking on depth at a particular `pos`
    /// stratified by cell barcodes
    pub fn stratified_depth_at(&self, pos: i64) -> Option<&HashMap<CellBarcode, usize>> {
        self.position_to_depth_with_cell.get(&pos)
    }
}
