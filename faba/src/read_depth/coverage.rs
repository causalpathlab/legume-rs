use crate::data::bam_io;
use genomic_data::sam::CellBarcode;

use coitrees::{COITree, Interval, IntervalTree};
use rust_htslib::bam::{self, ext::BamRecordExtensions};
use rustc_hash::{FxHashMap as HashMap, FxHashSet};

/// Read coverage tracker that organizes genomic intervals by cell barcode and chromosome.
///
/// This structure accumulates read alignment intervals from BAM files, organizing them
/// hierarchically by cell barcode and chromosome, then converts them to interval trees
/// for efficient querying.
pub struct ReadCoverageCollector<'a> {
    cell_chr_to_intervals: HashMap<CellBarcode, HashMap<Box<str>, Vec<Interval<()>>>>,
    cell_barcode_tag: &'a str,
    /// Restrict counting to these cells; `None` keeps every observed barcode.
    ///
    /// Borrowed, not `Arc`-shared as in [`crate::data::dna_stat_map::DnaBaseFreqMap`]:
    /// a collector lives for ONE block while the called-cell set lives for the
    /// whole run, so borrowing spares a set clone per block job — and a genome at
    /// 10 kb bins is tens of thousands of blocks.
    keep_cells: Option<&'a FxHashSet<CellBarcode>>,
}

impl<'a> ReadCoverageCollector<'a> {
    /// Creates a new ReadCoverage instance.
    ///
    /// # Arguments
    /// * `cell_barcode_tag` - The BAM tag used to identify cell barcodes (e.g., "CB" for 10x data)
    pub fn new(cell_barcode_tag: &'a str) -> Self {
        Self {
            cell_chr_to_intervals: HashMap::default(),
            cell_barcode_tag,
            keep_cells: None,
        }
    }

    /// Count only reads from `cells`, or from every barcode when `None`.
    ///
    /// No tag argument (unlike [`crate::data::dna_stat_map::DnaBaseFreqMap::set_keep_cells`])
    /// because this collector already owns the one tag it keys on: taking a second
    /// one could only introduce a disagreement between the gate and the counts.
    ///
    /// Takes an `Option` so "no restriction" is something a caller must state
    /// rather than forget.
    pub fn set_keep_cells(&mut self, cells: Option<&'a FxHashSet<CellBarcode>>) {
        self.keep_cells = cells;
    }

    /// Converts accumulated intervals into COITrees for efficient querying.
    ///
    /// Returns a nested HashMap structure: CellBarcode -> Chromosome -> COITree
    /// Each COITree enables O(log n + k) interval overlap queries.
    pub fn to_coitrees(&self) -> HashMap<CellBarcode, HashMap<Box<str>, COITree<(), u32>>> {
        let mut trees = HashMap::default();

        for (cb, chr_to_intervals) in self.cell_chr_to_intervals.iter() {
            let cb_trees: &mut HashMap<Box<str>, COITree<(), u32>> =
                trees.entry(cb.clone()).or_default();

            for (chr, nodes) in chr_to_intervals.iter() {
                cb_trees.insert(chr.clone(), COITree::new(nodes));
            }
        }

        trees
    }

    /// Convenience method to collect coverage from a BAM file region.
    ///
    /// # Arguments
    /// * `bam_file_path` - Path to the BAM file
    /// * `bed` - Genomic region to process
    pub fn collect_from_bam(
        &mut self,
        bam_file_path: &str,
        bed: &genomic_data::bed::Bed,
    ) -> anyhow::Result<()> {
        bam_io::for_each_record_in_region(bam_file_path, bed, |chr, rec| {
            self.update(chr, rec);
        })
    }

    /// Updates the coverage with a new BAM record.
    ///
    /// Extracts the cell barcode from the BAM record and adds the read's genomic interval
    /// to the appropriate cell/chromosome bucket.
    ///
    /// # Arguments
    /// * `chr` - Chromosome/contig name
    /// * `bam_record` - The BAM alignment record to process
    pub fn update(&mut self, chr: &str, bam_record: &bam::Record) {
        let cell_barcode =
            bam_io::extract_cell_barcode(bam_record, self.cell_barcode_tag.as_bytes());

        // Gate here, before the CIGAR walk in `reference_end()` and before any
        // map entry: this collector keeps one interval PER READ, so a barcode
        // admitted at this line costs memory for the rest of the block, not just
        // a hash lookup. An unfiltered run therefore scales with every barcode in
        // the BAM -- ambient droplets included, which is most of them.
        if let Some(keep) = self.keep_cells {
            // A read with no CB tag is not a called cell, so it must not survive
            // as the `"."` column once a keep-set is in force.
            if cell_barcode == CellBarcode::Missing || !keep.contains(&cell_barcode) {
                return;
            }
        }

        // coitrees intervals are END-INCLUSIVE, but `reference_end()` is the
        // first base PAST the alignment. Storing it verbatim stretched every
        // read one base to the right, so a read finishing just before a segment
        // still overlapped it.
        let first = bam_record.pos() as i32;
        let last = (bam_record.reference_end() as i32 - 1).max(first);

        let chr_to_intervals = self.cell_chr_to_intervals.entry(cell_barcode).or_default();

        let intervals = chr_to_intervals.entry(chr.into()).or_default();
        intervals.push(Interval::new(first, last, ()));
    }
}

#[cfg(test)]
mod tests;
