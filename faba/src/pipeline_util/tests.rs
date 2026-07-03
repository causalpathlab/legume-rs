//! Tests for the mitochondrial-fraction elbow cutoff and stat summarizers.

use super::{mito_elbow_cutoff, summarize_stats_per_site, summarize_stats_two_channel};
use crate::data::conversion::{BedWithGene, ConversionData};
use genomic_data::gff::GeneId;
use genomic_data::sam::{CellBarcode, Strand};

/// Build an ascending-sorted MT-fraction vector: `n_low` cells at `low`,
/// then `n_high` cells at `high`.
fn dist(n_low: usize, low: f64, n_high: usize, high: f64) -> Vec<f64> {
    let mut v = vec![low; n_low];
    v.extend(std::iter::repeat_n(high, n_high));
    v // already ascending since low < high
}

#[test]
fn elbow_cuts_a_clear_minority_burst_tail() {
    // 900 low-MT cells + 100 high-MT "burst" cells.
    let fracs = dist(900, 0.02, 100, 0.6);
    let cut = mito_elbow_cutoff(&fracs).expect("a clear tail should yield a cutoff");
    // Cutoff sits at the top of the bulk, so the 100 burst cells are dropped and
    // the 900 bulk cells are kept.
    assert!(
        (0.02..0.6).contains(&cut),
        "cutoff {cut} should separate bulk (0.02) from tail (0.6)"
    );
    let dropped = fracs.iter().filter(|&&f| f > cut).count();
    assert_eq!(dropped, 100, "should drop exactly the 100 burst cells");
}

#[test]
fn flat_distribution_yields_no_cut() {
    // No mito genes / no spread → nothing to cut.
    let fracs = vec![0.0; 1000];
    assert!(mito_elbow_cutoff(&fracs).is_none());
    let uniform = vec![0.03; 500];
    assert!(mito_elbow_cutoff(&uniform).is_none());
}

#[test]
fn too_few_cells_yields_no_cut() {
    let fracs = dist(40, 0.02, 9, 0.6); // 49 < 50
    assert!(mito_elbow_cutoff(&fracs).is_none());
}

#[test]
fn majority_high_is_not_filtered() {
    // Over-filtering guard: if the "tail" is actually the majority (elbow in the
    // lower half), don't cut — there is no clear minority burst population.
    let fracs = dist(100, 0.02, 900, 0.6);
    assert!(
        mito_elbow_cutoff(&fracs).is_none(),
        "should not cut when the high-MT group is the majority"
    );
}

/// One converted read at (`chr`, `pos`) with `unconv` reference reads, in `cb`.
fn stat(
    cb: &str,
    chr: &str,
    pos: i64,
    conv: usize,
    unconv: usize,
) -> (CellBarcode, BedWithGene, ConversionData) {
    (
        CellBarcode::Barcode(cb.into()),
        BedWithGene {
            chr: chr.into(),
            start: pos,
            stop: pos + 1,
            gene: GeneId::Ensembl("G1".into()),
            strand: Strand::Forward,
        },
        ConversionData {
            converted: conv,
            unconverted: unconv,
            site_pos: pos,
        },
    )
}

#[test]
fn per_site_keeps_distinct_single_base_sites_apart() {
    // One cell, one gene, two distinct sites on chr1 (positions 100 and 250).
    let stats = vec![
        stat("AAA", "chr1", 100, 3, 1),
        stat("AAA", "chr1", 250, 2, 5),
    ];
    let out = summarize_stats_per_site(
        &stats,
        |b| format!("{}", b.gene).into(),
        "m6a",
        "methylated",
        "unmethylated",
        0,
    );

    // Two sites × two channels = four rows, each carrying the `chr:pos` subunit.
    let mut rows = out.rows.clone();
    rows.sort();
    assert_eq!(
        rows.as_slice(),
        [
            "G1/m6a/chr1:100/methylated".into(),
            "G1/m6a/chr1:100/unmethylated".into(),
            "G1/m6a/chr1:250/methylated".into(),
            "G1/m6a/chr1:250/unmethylated".into(),
        ]
    );
}

#[test]
fn gene_level_pools_sites_that_per_site_keeps_separate() {
    // Same two sites pool into a single gene-level (unit, channel) pair: the
    // methylated row sums 3+2=5, the unmethylated sums 1+5=6.
    let stats = vec![
        stat("AAA", "chr1", 100, 3, 1),
        stat("AAA", "chr1", 250, 2, 5),
    ];
    let out = summarize_stats_two_channel(
        &stats,
        |b| format!("{}", b.gene).into(),
        "m6a",
        "methylated",
        "unmethylated",
    );

    let mut rows = out.rows.clone();
    rows.sort();
    assert_eq!(
        rows.as_slice(),
        ["G1/m6a/methylated".into(), "G1/m6a/unmethylated".into()]
    );

    // Locate the methylated row and confirm its value is the pooled 5.
    let meth_idx = out
        .rows
        .iter()
        .position(|r| &**r == "G1/m6a/methylated")
        .unwrap() as u64;
    let meth_val: f32 = out
        .triplets
        .iter()
        .filter(|(r, _, _)| *r == meth_idx)
        .map(|(_, _, v)| *v)
        .sum();
    assert_eq!(meth_val, 5.0);
}

#[test]
fn per_site_min_cells_drops_rare_sites_unit_aware() {
    // Site chr1:100 is seen in 3 cells; chr1:250 in only 1. With min_cells=2,
    // the rare site is dropped entirely — BOTH channels, never half.
    let stats = vec![
        stat("AAA", "chr1", 100, 3, 1),
        stat("BBB", "chr1", 100, 2, 0),
        stat("CCC", "chr1", 100, 1, 4),
        stat("AAA", "chr1", 250, 9, 9), // lone cell at the rare site
    ];
    let out = summarize_stats_per_site(
        &stats,
        |b| format!("{}", b.gene).into(),
        "m6a",
        "methylated",
        "unmethylated",
        2,
    );

    let mut rows = out.rows.clone();
    rows.sort();
    // Only the 3-cell site survives, with both its channels intact.
    assert_eq!(
        rows.as_slice(),
        [
            "G1/m6a/chr1:100/methylated".into(),
            "G1/m6a/chr1:100/unmethylated".into(),
        ]
    );
}
