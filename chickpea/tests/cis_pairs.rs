//! Cis peak-to-gene candidates and the fixed ABC contact weights `W`.

use chickpea::p2g::cis::{build_cis_pairs, AbcKernel};
use genomic_data::coordinates::{GeneTss, PeakCoord};

fn gene(chr: &str, tss: i64) -> Option<GeneTss> {
    Some(GeneTss {
        chr: chr.into(),
        tss,
    })
}

fn peak(chr: &str, start: i64, end: i64) -> Option<PeakCoord> {
    Some(PeakCoord {
        chr: chr.into(),
        start,
        end,
    })
}

/// GENE1 and GENE2 on chr1 share a peak between them; GENE3 has no position;
/// GENE4 is on chr2, named without the `chr` prefix.
fn fixture() -> (Vec<Option<GeneTss>>, Vec<Option<PeakCoord>>) {
    let genes = vec![
        gene("chr1", 10_000),
        gene("chr1", 30_000),
        None,
        gene("2", 5_000),
    ];
    let peaks = vec![
        peak("chr1", 9_900, 10_100),    // at GENE1's TSS
        peak("chr1", 19_900, 20_100),   // 10 kb from both GENE1 and GENE2
        peak("chr1", 200_000, 200_200), // out of every window
        peak("chr2", 5_900, 6_100),     // 1 kb from GENE4
        None,                           // unparseable name
    ];
    (genes, peaks)
}

fn kernel() -> AbcKernel {
    AbcKernel {
        window: 15_000,
        max_per_gene: 200,
        ..AbcKernel::default()
    }
}

fn peaks_of(pairs: &chickpea::p2g::cis::CisPairs, g: usize) -> Vec<u32> {
    pairs.peak[pairs.gene(g)].to_vec()
}

#[test]
fn candidates_are_the_peaks_inside_each_genes_window() {
    let (genes, peaks) = fixture();
    let pairs = build_cis_pairs(&genes, &peaks, &kernel());
    assert_eq!(pairs.n_genes(), 4);
    assert_eq!(peaks_of(&pairs, 0), vec![0, 1]);
    assert_eq!(peaks_of(&pairs, 1), vec![1]);
    assert!(
        peaks_of(&pairs, 2).is_empty(),
        "a gene without a position has no candidates"
    );
    assert_eq!(
        peaks_of(&pairs, 3),
        vec![3],
        "chr1 and 1 name the same chromosome"
    );
    assert_eq!(pairs.n_pairs(), 4);
    assert_eq!(pairs.n_genes_placed, 3);
}

#[test]
fn a_peak_near_two_genes_is_a_candidate_for_both() {
    let (genes, peaks) = fixture();
    let pairs = build_cis_pairs(&genes, &peaks, &kernel());
    assert!(peaks_of(&pairs, 0).contains(&1));
    assert!(peaks_of(&pairs, 1).contains(&1));
}

#[test]
fn distances_are_from_the_peak_midpoint_to_the_tss() {
    let (genes, peaks) = fixture();
    let pairs = build_cis_pairs(&genes, &peaks, &kernel());
    let r = pairs.gene(0);
    assert_eq!(pairs.dist[r.clone()], [0, 10_000]);
    assert_eq!(pairs.dist[pairs.gene(3)], [1_000]);
}

#[test]
fn contact_weights_decay_with_distance_and_sum_to_one_per_gene() {
    let (genes, peaks) = fixture();
    let k = kernel();
    assert!(k.contact(0) > k.contact(1_000));
    assert!(k.contact(1_000) > k.contact(100_000));
    let pairs = build_cis_pairs(&genes, &peaks, &k);
    for g in [0usize, 1, 3] {
        let s: f32 = pairs.weight[pairs.gene(g)].iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "gene {g} weights sum to {s}");
    }
    let w = &pairs.weight[pairs.gene(0)];
    assert!(w[0] > w[1], "the nearer peak weighs more: {w:?}");
    let expect = k.contact(0) / (k.contact(0) + k.contact(10_000));
    assert!((w[0] - expect).abs() < 1e-6);
}

#[test]
fn peaks_that_reach_no_gene_are_counted() {
    let (genes, peaks) = fixture();
    let pairs = build_cis_pairs(&genes, &peaks, &kernel());
    assert_eq!(pairs.n_unparsed_peaks, 1);
    assert_eq!(
        pairs.n_unreached_peaks, 2,
        "the far peak and the unparsed one"
    );
}

#[test]
fn the_per_gene_cap_keeps_the_nearest_peaks() {
    let (genes, peaks) = fixture();
    let k = AbcKernel {
        max_per_gene: 1,
        ..kernel()
    };
    let pairs = build_cis_pairs(&genes, &peaks, &k);
    assert_eq!(peaks_of(&pairs, 0), vec![0]);
    assert_eq!(pairs.weight[pairs.gene(0)], [1.0]);
}
