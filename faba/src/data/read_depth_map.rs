use crate::data::dna_stat_traits::*;
use crate::data::sam::*;
use crate::data::visitors_htslib::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};
use std::collections::{HashMap, HashSet};

pub struct ReadDepthMap<'a> {
    position_to_depth_with_cell: HashMap<i64, HashMap<CellBarcode, usize>>,
    cells: HashSet<CellBarcode>,
    positions: HashSet<i64>,
    cell_barcode_tag: &'a [u8],
}

impl<'a> VisitWithBamOps for ReadDepthMap<'a> {}

impl<'a> DnaStatMap for ReadDepthMap<'a> {
    fn add_bam_record(&mut self, bam_record: bam::Record, lb: i64, ub: i64) {
        let cell_barcode = match bam_record.aux(&self.cell_barcode_tag) {
            Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
            _ => CellBarcode::Missing,
        };

        if !self.cells.contains(&cell_barcode) {
            self.cells.insert(cell_barcode.clone());
        }

        // Note: these are zero-based positions
        for [_rpos, gpos] in bam_record.aligned_pairs() {
            let (g, v) = (gpos as usize, gpos - lb);

            if g < (lb as usize) || g >= (ub as usize) || v < 0 {
                continue;
            }

            let genome_pos = gpos + 1; // 1-based position

            if !self.positions.contains(&genome_pos) {
                self.positions.insert(genome_pos.clone());
            }

            let freq_map = self
                .position_to_depth_with_cell
                .entry(genome_pos)
                .or_default();

            let freq = freq_map.entry(cell_barcode.clone()).or_default();
            *freq += 1;
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

impl<'a> ReadDepthMap<'a> {
    pub fn new(cell_barcode_tag: &'a str) -> Self {
        Self {
            position_to_depth_with_cell: HashMap::new(),
            cells: HashSet::new(),
            positions: HashSet::new(),
            cell_barcode_tag: cell_barcode_tag.as_bytes(),
        }
    }

    /// picking on depth at a particular `pos`
    /// stratified by cell barcodes
    pub fn stratified_depth_at(&self, pos: i64) -> Option<&HashMap<CellBarcode, usize>> {
        self.position_to_depth_with_cell.get(&pos)
    }

    /// depth by position across all the cells
    pub fn marginal_depth_map(&self) -> HashMap<i64, usize> {
        let mut ret: HashMap<i64, usize> =
            HashMap::with_capacity(self.position_to_depth_with_cell.len());

        for (&pos, freq_map) in self.position_to_depth_with_cell.iter() {
            let accum = ret.entry(pos).or_default();
            for (_, freq) in freq_map.iter() {
                *accum += freq;
            }
        }
        ret
    }
}
