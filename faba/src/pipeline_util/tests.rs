//! Tests for the mitochondrial-fraction elbow cutoff, the shared per-batch QC
//! ([`qc_one_batch`]), and the stat summarizers.

use super::{
    mito_elbow_cutoff, qc_one_batch, summarize_stats_per_site, summarize_stats_two_channel,
    GeneGate, MitoQcParams,
};
use crate::cell_qc::{CellCallParams, CellFilter};
use crate::data::conversion::{BedWithGene, ConversionData};
use genomic_data::gff::GeneId;
use genomic_data::sam::{CellBarcode, Strand};
use rustc_hash::FxHashSet;

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

///////////////////////////////////////////////////////////////////////
// qc_one_batch — the single place a batch's passing cell set is set //
///////////////////////////////////////////////////////////////////////

/// A gate whose only mitochondrial gene is `MT-CO1`, with no biotype subset.
fn gate(mito: MitoQcParams) -> GeneGate {
    GeneGate::from_keys(mito, FxHashSet::from_iter([Box::from("MT-CO1")]), None)
}

/// One `(cell, gene/count/{track}, count)` triplet — the shape `count_read_per_gene*`
/// emits.
fn triplet(cb: &str, gene: &str, track: &str, val: f32) -> (CellBarcode, Box<str>, f32) {
    (
        CellBarcode::Barcode(cb.into()),
        format!("{gene}/count/{track}").into(),
        val,
    )
}

/// `n_bulk` cells at `mt_bulk`/20 MT fraction, then `n_burst` at `mt_burst`/20 —
/// all counts on the spliced track. Barcodes are prefixed by group so the two are
/// distinguishable in the result.
fn batch(
    n_bulk: usize,
    mt_bulk: f32,
    n_burst: usize,
    mt_burst: f32,
) -> Vec<(CellBarcode, Box<str>, f32)> {
    let mut out = Vec::new();
    for (prefix, n, mt) in [("BULK", n_bulk, mt_bulk), ("BURST", n_burst, mt_burst)] {
        for i in 0..n {
            let cb = format!("{prefix}{i}");
            out.push(triplet(&cb, "MT-CO1", "spliced", mt));
            out.push(triplet(&cb, "GENE1", "spliced", 20.0 - mt));
        }
    }
    out
}

/// Cell calling off (`Nnz` keeps every observed barcode), so these tests isolate
/// the mito filter from the OrdMag/EmptyDrops call.
fn no_cell_calling() -> CellCallParams {
    CellCallParams {
        filter: CellFilter::Nnz,
        ..CellCallParams::default()
    }
}

fn n_kept(cells: &FxHashSet<CellBarcode>, prefix: &str) -> usize {
    cells
        .iter()
        .filter(|cb| cb.to_string().starts_with(prefix))
        .count()
}

/// The regression this helper exists to prevent: with the DEFAULT mito policy —
/// what `dartseq`/`atoi`/`apa` now get, and silently did not before — the high-MT
/// (dying-cell) tail is dropped from the passing set.
#[test]
fn qc_one_batch_drops_the_high_mito_burst_tail() {
    // 200 bulk cells at 5% MT + 60 burst cells at 60% MT.
    let spliced = batch(200, 1.0, 60, 12.0);
    let bq = qc_one_batch(
        &spliced,
        &[],
        &gate(MitoQcParams::default()),
        0,
        0,
        &no_cell_calling(),
        None,
    )
    .expect("qc should succeed");

    assert_eq!(
        n_kept(&bq.passing_cells, "BULK"),
        200,
        "bulk cells retained"
    );
    assert_eq!(
        n_kept(&bq.passing_cells, "BURST"),
        0,
        "the 60 high-MT burst cells must be dropped"
    );
}

/// `--no-mito-cell-qc` reports the metric but drops nobody — the same 260 cells
/// the pre-fix modality path was (wrongly) keeping by default.
#[test]
fn qc_one_batch_keeps_every_cell_when_mito_qc_is_disabled() {
    let spliced = batch(200, 1.0, 60, 12.0);
    let mito = MitoQcParams {
        no_mito_cell_qc: true,
        ..MitoQcParams::default()
    };
    let bq = qc_one_batch(&spliced, &[], &gate(mito), 0, 0, &no_cell_calling(), None)
        .expect("qc should succeed");

    assert_eq!(bq.passing_cells.len(), 260, "report-only: no cells dropped");
}

/// Ordering invariant: the MT fraction is computed on the FULL pre-filter counts
/// (spliced **+ unspliced**), not the spliced track the cell call uses. A cell
/// that looks high-MT on spliced alone (6/10 = 0.6) but is not on the true total
/// (6/100 = 0.06) must survive a 0.5 cutoff — gate the counts first and this cell
/// is silently lost.
#[test]
fn mito_fraction_uses_full_prefilter_counts_not_the_spliced_track() {
    let mut spliced = Vec::new();
    let mut unspliced = Vec::new();
    for i in 0..10 {
        let cb = format!("INTRONIC{i}");
        spliced.push(triplet(&cb, "MT-CO1", "spliced", 6.0));
        spliced.push(triplet(&cb, "GENE1", "spliced", 4.0));
        unspliced.push(triplet(&cb, "GENE1", "unspliced", 90.0));
    }
    // A genuinely high-MT cell: 80% MT even on the full counts.
    for i in 0..10 {
        let cb = format!("DYING{i}");
        spliced.push(triplet(&cb, "MT-CO1", "spliced", 80.0));
        spliced.push(triplet(&cb, "GENE1", "spliced", 20.0));
    }

    let mito = MitoQcParams {
        max_mito_frac: 0.5, // fixed cutoff, so no elbow / minimum-cell-count guard
        ..MitoQcParams::default()
    };
    let bq = qc_one_batch(
        &spliced,
        &unspliced,
        &gate(mito),
        0,
        0,
        &no_cell_calling(),
        None,
    )
    .expect("qc should succeed");

    assert_eq!(
        n_kept(&bq.passing_cells, "INTRONIC"),
        10,
        "MT fraction must be measured against total (spliced + unspliced) UMIs"
    );
    assert_eq!(
        n_kept(&bq.passing_cells, "DYING"),
        0,
        "cells above the cutoff on the true total are still dropped"
    );
}

/// Invariant: QC stays mito-blind. The MT gene is still present in `gene_stats`,
/// because excluding it from what gets *quantified* is the caller's decision
/// ([`GeneGate::quantify`]) — not something the cell call is allowed to see.
#[test]
fn qc_one_batch_leaves_mito_genes_in_the_gene_stats() {
    let spliced = batch(200, 1.0, 60, 12.0);
    let g = gate(MitoQcParams::default());
    let bq = qc_one_batch(
        &spliced,
        &[],
        &g,
        1, // gene_min_counts > 0 → gene totals populated
        0,
        &no_cell_calling(),
        None,
    )
    .expect("qc should succeed");

    assert!(
        bq.gene_stats.contains_key("MT-CO1"),
        "MT genes must survive into gene_stats; the quantification gate is the caller's"
    );
    assert!(g.quantify("GENE1"), "a non-MT gene is always quantified");
    assert!(
        !g.quantify("MT-CO1"),
        "an MT gene is excluded from quantification by default"
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
