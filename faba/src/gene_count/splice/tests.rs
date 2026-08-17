use super::*;
use rust_htslib::bam::record::{Aux, CigarString, Record};

/// One exon covering [100, 200) — 0-based half-open, the shape
/// `build_exon_intervals` produces.
const EXONS: [(i64, i64); 1] = [(100, 200)];

/// An exon-contained, mapped read on one cell and one molecule. Only `pos` and
/// `mapq` vary across these tests; the counters key on `(CB, UB)`, and every test
/// here either uses a single molecule or depends on two reads sharing one, so
/// fixing the tags in the helper is what makes that sharing guaranteed rather
/// than retyped at each call.
fn read(pos: i64, mapq: u8) -> Record {
    let mut rec = Record::new();
    let cigar = CigarString::try_from("10M").unwrap();
    let seq = vec![b'A'; 10];
    let qual = vec![40u8; 10];
    rec.set(b"read", Some(&cigar), &seq, &qual);
    rec.set_tid(0);
    rec.set_pos(pos);
    rec.set_mapq(mapq);
    // `Record::new` starts out unmapped, which would make `aligned_blocks` moot.
    rec.unset_unmapped();
    rec.push_aux(b"CB", Aux::String("CELL1")).unwrap();
    rec.push_aux(b"UB", Aux::String("UMI1")).unwrap();
    rec
}

/// Tag config with the MAPQ floor under test; everything else is fixed.
fn opts(min_mapping_quality: u8) -> CountReadOpts<'static> {
    CountReadOpts {
        cell_barcode_tag: "CB",
        gene_barcode_tag: "GX",
        umi_tag: Some(b"UB"),
        min_mapping_quality,
    }
}

fn total(counter: &ReadCounter) -> usize {
    counter.to_vec().into_iter().map(|(_, n)| n).sum()
}

/// `(spliced, unspliced)` totals across cells.
fn splice_totals(counter: &SpliceAwareReadCounter) -> (usize, usize) {
    (
        counter.spliced.values().sum(),
        counter.unspliced.values().sum(),
    )
}

/////////////////////
// MAPQ admission  //
/////////////////////

#[test]
fn reads_below_min_mapq_are_not_counted() {
    let mut counter = ReadCounter::new(opts(20));
    counter.count(&read(100, 19));
    assert_eq!(total(&counter), 0, "MAPQ 19 must not pass a floor of 20");

    let mut counter = ReadCounter::new(opts(20));
    counter.count(&read(100, 20));
    assert_eq!(total(&counter), 1, "MAPQ 20 is at the floor, so it passes");
}

#[test]
fn zero_min_mapq_counts_every_mapped_alignment() {
    // The documented escape hatch: `--min-mapping-quality 0` waives the MAPQ
    // floor. The flag checks below still apply.
    let mut counter = ReadCounter::new(opts(0));
    counter.count(&read(100, 0));
    assert_eq!(total(&counter), 1);
}

#[test]
fn splice_counter_applies_the_same_mapq_floor() {
    let mut counter = SpliceAwareReadCounter::new(opts(20), &EXONS);
    counter.classify_and_count(&read(100, 19));
    assert_eq!(splice_totals(&counter), (0, 0));
}

////////////////////
// Flag admission //
////////////////////

// MAPQ 255 throughout: a passing score, so only the flag can reject these.

#[test]
fn secondary_alignments_are_never_counted() {
    let mut rec = read(100, 255);
    rec.set_secondary();
    let mut counter = ReadCounter::new(opts(0));
    counter.count(&rec);
    assert_eq!(total(&counter), 0);
}

#[test]
fn supplementary_alignments_are_never_counted() {
    let mut rec = read(100, 255);
    rec.set_supplementary();
    let mut counter = ReadCounter::new(opts(0));
    counter.count(&rec);
    assert_eq!(total(&counter), 0);
}

#[test]
fn unmapped_records_are_never_counted_as_spliced() {
    // An unmapped record has no aligned blocks, and `classify` reads "no
    // intronic block" as spliced — so without the `is_unmapped` guard a zero
    // MAPQ floor would silently bank it as a spliced molecule.
    let mut rec = read(100, 0);
    rec.set_unmapped();
    let mut counter = SpliceAwareReadCounter::new(opts(0), &EXONS);
    counter.classify_and_count(&rec);
    assert_eq!(splice_totals(&counter), (0, 0));
}

#[test]
fn paired_reads_without_a_proper_pair_flag_are_still_counted() {
    // Unlike `bam_io::passes_alignment_filters`, gene counting must not require
    // `is_proper_pair` — `faba genes` quantifies bulk libraries too.
    let mut rec = read(100, 255);
    rec.set_paired();
    let mut counter = ReadCounter::new(opts(20));
    counter.count(&rec);
    assert_eq!(total(&counter), 1);
}

/////////////////////////////////////
// Admission runs before UMI dedup //
/////////////////////////////////////

#[test]
fn a_rejected_read_does_not_shadow_its_molecule() {
    // Same (cell, UMI) twice: a failing alignment first, a passing one second.
    // If admission ran after dedup, the first read would claim the molecule's
    // slot and the second would be dropped as a duplicate — undercounting.
    let mut counter = ReadCounter::new(opts(20));
    counter.count(&read(100, 0));
    counter.count(&read(100, 255));
    assert_eq!(
        total(&counter),
        1,
        "the passing read of the molecule must survive the rejected one"
    );
}

#[test]
fn a_rejected_read_does_not_fix_the_splice_class() {
    // Same invariant on the splice-aware counter, where a shadowed molecule
    // would also inherit the rejected alignment's spliced/unspliced class.
    // Read 1 (rejected) is intronic; read 2 (passing) is exonic.
    let mut counter = SpliceAwareReadCounter::new(opts(20), &EXONS);
    counter.classify_and_count(&read(300, 0)); // intronic, MAPQ 0
    counter.classify_and_count(&read(100, 255)); // exonic, MAPQ 255
    assert_eq!(
        splice_totals(&counter),
        (1, 0),
        "the exonic read defines the molecule, not the rejected intronic one"
    );
}
