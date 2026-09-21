//! Link scoring on synthetic pseudobulk profiles.

mod common;

use chickpea::common::Mat;
use chickpea::p2g::link_map::*;
use common::{mat, peak, sine_signal, tss};

/// Causal peak co-varies with its gene; a cis bystander does not.
/// The causal edge must outrank the bystander; a far peak gets no edge.
#[test]
fn causal_cis_peak_outranks_bystander() {
    let s = 40usize;
    let signal = sine_signal(s);
    let rna = mat(1, s, |_, j| signal[j] * 10.0);
    let atac = mat(3, s, |p, j| match p {
        0 => signal[j] * 8.0 + 0.01, // causal
        1 => ((j % 7) as f32) * 0.3, // cis bystander, uncorrelated
        _ => signal[j] * 9.0,        // same signal but far away
    });

    let gene_tss = vec![tss(100_000)];
    let peak_coords = vec![peak(100_000), peak(120_000), peak(5_000_000)];
    let params = LinkParams {
        cis_window: 500_000,
        max_cis: 50,
        min_weight: 0.0,
        ..LinkParams::default()
    };

    let edges = link_peaks_to_genes(&rna, &atac, &gene_tss, &peak_coords, &params).unwrap();
    assert!(!edges.is_empty(), "expected at least the causal cis edge");
    assert!(
        edges.iter().all(|e| e.gene == 0 && e.peak != 2),
        "far peak must not appear; got {edges:?}"
    );
    let w0 = edges
        .iter()
        .find(|e| e.peak == 0)
        .expect("causal peak edge")
        .weight;
    let w1 = edges.iter().find(|e| e.peak == 1).map(|e| e.weight);
    assert!(w0 > 0.5, "causal weight too small: {w0}");
    if let Some(w1) = w1 {
        assert!(w0 > w1, "causal {w0} should beat bystander {w1}");
    }
}

fn abc_params() -> LinkParams {
    LinkParams {
        score: LinkScore::Abc,
        cis_window: 500_000,
        max_cis: 0,
        min_weight: 0.0,
        ..LinkParams::default()
    }
}

fn weight_of(edges: &[PeakGeneEdge], p: usize) -> Option<f32> {
    edges.iter().find(|e| e.peak == p).map(|e| e.weight)
}

#[test]
fn contact_floors_at_min_distance_and_settles_on_the_pseudocount() {
    let p = abc_params();
    let near = contact(0, &p);
    let same = contact(2_000, &p);
    let mid = contact(4_000_000, &p);
    let far = contact(1_000_000_000, &p);
    let pseudo = (1_000_000f32).powf(-0.87);
    assert!((near - same).abs() < 1e-9, "floor: {near} vs {same}");
    assert!(near > mid && mid > far, "decays: {near} > {mid} > {far}");
    assert!(
        far > pseudo && far < 1.01 * pseudo,
        "far tail = pseudocount: {far} vs {pseudo}"
    );
}

#[test]
fn abc_ignores_rna_and_ranks_by_activity_times_contact() {
    // Equal distance, peak 1 four times as active; RNA is noise either way.
    let mut atac = Mat::zeros(2, 4);
    for j in 0..4 {
        atac[(0, j)] = 1.0;
        atac[(1, j)] = 4.0;
    }
    let rna = Mat::from_fn(1, 4, |_, j| ((j * 7) % 5) as f32);
    let peaks = [peak(0), peak(200_000 - 500)];
    let edges = link_peaks_to_genes(&rna, &atac, &[tss(100_000)], &peaks, &abc_params()).unwrap();
    let (w0, w1) = (weight_of(&edges, 0).unwrap(), weight_of(&edges, 1).unwrap());
    assert!(
        (w1 / w0 - 4.0).abs() < 1e-3,
        "share ratio follows activity: {}",
        w1 / w0
    );
    assert!(
        (w0 + w1 - 1.0).abs() < 1e-5,
        "shares sum to one over the window"
    );
}

#[test]
fn abc_prefers_the_nearer_of_two_equally_active_peaks() {
    let atac = Mat::from_element(2, 3, 2.0);
    let rna = Mat::zeros(1, 3);
    let edges = link_peaks_to_genes(
        &rna,
        &atac,
        &[tss(100_000)],
        &[peak(100_000), peak(300_000)],
        &abc_params(),
    )
    .unwrap();
    assert!(weight_of(&edges, 0).unwrap() > weight_of(&edges, 1).unwrap());
}

#[test]
fn abc_is_invariant_to_activity_scale_and_empty_when_nothing_is_open() {
    let peaks = [peak(90_000), peak(130_000)];
    let rna = Mat::zeros(1, 2);
    let a = link_peaks_to_genes(
        &rna,
        &Mat::from_row_slice(2, 2, &[2.0, 2.0, 6.0, 6.0]),
        &[tss(100_000)],
        &peaks,
        &abc_params(),
    )
    .unwrap();
    let b = link_peaks_to_genes(
        &rna,
        &Mat::from_row_slice(2, 2, &[20.0, 20.0, 60.0, 60.0]),
        &[tss(100_000)],
        &peaks,
        &abc_params(),
    )
    .unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!((x.weight - y.weight).abs() < 1e-6);
    }
    let none = link_peaks_to_genes(
        &rna,
        &Mat::zeros(2, 2),
        &[tss(100_000)],
        &peaks,
        &abc_params(),
    )
    .unwrap();
    assert!(none.is_empty());
}

#[test]
fn top_k_per_gene_keeps_the_k_best_after_the_floor() {
    let mut atac = Mat::zeros(4, 2);
    for (p, a) in [1.0f32, 10.0, 5.0, 0.0].iter().enumerate() {
        atac[(p, 0)] = *a;
        atac[(p, 1)] = *a;
    }
    let peaks = [peak(100_000), peak(101_000), peak(102_000), peak(103_000)];
    let rna = Mat::zeros(1, 2);
    let p = LinkParams {
        top_k_per_gene: 2,
        ..abc_params()
    };
    let edges = link_peaks_to_genes(&rna, &atac, &[tss(100_000)], &peaks, &p).unwrap();
    let kept: Vec<usize> = edges.iter().map(|e| e.peak).collect();
    assert_eq!(
        kept,
        vec![1, 2],
        "the two largest shares, in rank order: {edges:?}"
    );
    // Off (0) keeps every peak above the floor; the closed peak never enters.
    let all = link_peaks_to_genes(&rna, &atac, &[tss(100_000)], &peaks, &abc_params()).unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
fn top_k_applies_to_the_pearson_score_too() {
    let s = 40usize;
    let signal = sine_signal(s);
    let rna = mat(1, s, |_, j| signal[j] * 10.0);
    let atac = mat(3, s, |p, j| match p {
        0 => signal[j] * 8.0 + 0.01,
        1 => signal[j] * 4.0 + ((j % 3) as f32) * 0.5,
        _ => signal[j] * 2.0 + ((j % 7) as f32) * 0.9,
    });
    let peaks = [peak(100_000), peak(120_000), peak(140_000)];
    let p = LinkParams {
        score: LinkScore::Pearson,
        cis_window: 500_000,
        max_cis: 0,
        min_weight: 0.0,
        top_k_per_gene: 1,
        ..LinkParams::default()
    };
    let edges = link_peaks_to_genes(&rna, &atac, &[tss(100_000)], &peaks, &p).unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].peak, 0, "the most correlated peak: {edges:?}");
}
