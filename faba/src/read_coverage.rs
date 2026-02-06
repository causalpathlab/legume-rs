use genomic_data::sam::CellBarcode;
use crate::data::visitors_htslib::VisitWithBamOps;

use coitrees::{COITree, Interval, IntervalTree};
use fnv::FnvHashMap as HashMap;
use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};

/// Read coverage tracker that organizes genomic intervals by cell barcode and chromosome.
///
/// This structure accumulates read alignment intervals from BAM files, organizing them
/// hierarchically by cell barcode and chromosome, then converts them to interval trees
/// for efficient querying.
pub struct ReadCoverageCollector<'a> {
    cell_chr_to_intervals: HashMap<CellBarcode, HashMap<Box<str>, Vec<Interval<()>>>>,
    cell_barcode_tag: &'a str,
}

impl VisitWithBamOps for ReadCoverageCollector<'_> {}

impl<'a> ReadCoverageCollector<'a> {
    /// Creates a new ReadCoverage instance.
    ///
    /// # Arguments
    /// * `cell_barcode_tag` - The BAM tag used to identify cell barcodes (e.g., "CB" for 10x data)
    pub fn new(cell_barcode_tag: &'a str) -> Self {
        Self {
            cell_chr_to_intervals: HashMap::default(),
            cell_barcode_tag,
        }
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
    /// This wraps the generic `visit_bam_by_region` method with the standard update logic.
    ///
    /// # Arguments
    /// * `bam_file_path` - Path to the BAM file
    /// * `bed` - Genomic region to process
    pub fn collect_from_bam(
        &mut self,
        bam_file_path: &str,
        bed: &genomic_data::bed::Bed,
    ) -> anyhow::Result<()> {
        self.visit_bam_by_region(bam_file_path, bed, &Self::update)
    }

    /// Updates the coverage with a new BAM record.
    ///
    /// Extracts the cell barcode from the BAM record and adds the read's genomic interval
    /// to the appropriate cell/chromosome bucket.
    ///
    /// # Arguments
    /// * `chr` - Chromosome/contig name
    /// * `bam_record` - The BAM alignment record to process
    pub fn update(&mut self, chr: &str, bam_record: bam::Record) {
        let cell_barcode = match bam_record.aux(self.cell_barcode_tag.as_bytes()) {
            Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
            _ => CellBarcode::Missing,
        };
        let first = bam_record.pos() as i32;
        let last = bam_record.reference_end() as i32;

        let chr_to_intervals = self.cell_chr_to_intervals.entry(cell_barcode).or_default();

        let intervals = chr_to_intervals.entry(chr.into()).or_default();
        intervals.push(Interval::new(first, last, ()));
    }
}
