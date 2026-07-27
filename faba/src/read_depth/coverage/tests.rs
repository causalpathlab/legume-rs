use crate::read_depth::coverage::ReadCoverageCollector;

use genomic_data::sam::CellBarcode;
use rust_htslib::bam::record::{Aux, Record};
use rustc_hash::FxHashSet;

fn cb(s: &str) -> CellBarcode {
    CellBarcode::Barcode(s.into())
}

/// One 4-base alignment at `pos`, tagged `CB` unless the barcode is `None`
/// (an untagged read, which resolves to [`CellBarcode::Missing`]).
fn read(cell: Option<&str>, pos: i64) -> Record {
    let mut rec = Record::new();
    let cigar = rust_htslib::bam::record::CigarString::try_from("4M").unwrap();
    rec.set(b"r", Some(&cigar), b"ACGT", &[40u8; 4]);
    rec.set_tid(0);
    rec.set_pos(pos);
    rec.set_mapq(60);
    rec.unset_unmapped();
    if let Some(cell) = cell {
        rec.push_aux(b"CB", Aux::String(cell)).unwrap();
    }
    rec
}

/// The gate has to bite BEFORE the interval is stored, not at output time: this
/// collector keeps one interval per read, so an ambient droplet that reaches the
/// map costs memory for the whole block. Asserting on the interval store (not on
/// `to_coitrees`) is what pins that down.
#[test]
fn a_keep_set_stores_nothing_for_barcodes_it_rejects() {
    let mut keep = FxHashSet::default();
    keep.insert(cb("CELL_A"));

    let mut coverage = ReadCoverageCollector::new("CB");
    coverage.set_keep_cells(Some(&keep));

    coverage.update("chr1", &read(Some("CELL_A"), 100));
    coverage.update("chr1", &read(Some("CELL_A"), 200));
    coverage.update("chr1", &read(Some("AMBIENT"), 100));
    coverage.update("chr1", &read(None, 100));

    assert_eq!(
        coverage.cell_chr_to_intervals.len(),
        1,
        "a rejected barcode must not allocate a bucket at all"
    );
    let intervals = &coverage.cell_chr_to_intervals[&cb("CELL_A")]["chr1"];
    assert_eq!(intervals.len(), 2, "both kept reads are stored");

    let trees = coverage.to_coitrees();
    assert!(trees.contains_key(&cb("CELL_A")));
    assert!(!trees.contains_key(&cb("AMBIENT")));
    assert!(
        !trees.contains_key(&CellBarcode::Missing),
        "an untagged read is not a called cell, so it must not become the '.' column"
    );
}

/// Default behaviour is unchanged: no keep-set means every barcode observed in
/// the BAM gets a column, untagged reads included.
#[test]
fn without_a_keep_set_every_barcode_survives() {
    let mut coverage = ReadCoverageCollector::new("CB");

    coverage.update("chr1", &read(Some("CELL_A"), 100));
    coverage.update("chr1", &read(Some("AMBIENT"), 100));
    coverage.update("chr1", &read(None, 100));

    let trees = coverage.to_coitrees();
    assert_eq!(trees.len(), 3);
    assert!(trees.contains_key(&CellBarcode::Missing));
}

/// A keep-set is installed per BAM on a collector that a caller may reuse, so
/// clearing has to actually clear -- a stale set would filter the next library
/// against barcodes that mean nothing in it.
#[test]
fn clearing_the_keep_set_restores_the_unfiltered_default() {
    let keep = FxHashSet::default();

    let mut coverage = ReadCoverageCollector::new("CB");
    coverage.set_keep_cells(Some(&keep));
    coverage.update("chr1", &read(Some("CELL_A"), 100));
    assert!(coverage.cell_chr_to_intervals.is_empty());

    coverage.set_keep_cells(None);
    coverage.update("chr1", &read(Some("CELL_A"), 100));
    assert_eq!(coverage.cell_chr_to_intervals.len(), 1);
}
