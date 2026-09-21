//! ArchR-style gene activity from ATAC pseudobulks.

use chickpea::common::Mat;
use chickpea::p2g::gene_activity::*;
use genomic_data::coordinates::{GeneLoc, PeakCoord};
use genomic_data::sam::Strand;

fn gene(chr: &str, start: i64, end: i64, strand: Strand) -> GeneLoc {
    let tss = match strand {
        Strand::Forward => start,
        Strand::Backward => end,
    };
    GeneLoc {
        chr: chr.into(),
        start,
        end,
        tss,
        strand,
    }
}

#[test]
fn nearer_peak_outweighs_far_peak_at_equal_accessibility() {
    let s = 4usize;
    let atac = Mat::from_element(2, s, 1.0);
    let genes = vec![Some(gene("1", 100_000, 101_000, Strand::Forward))];
    let peak_coords = vec![
        Some(PeakCoord {
            chr: "1".into(),
            start: 100_200,
            end: 100_700,
        }), // in body
        Some(PeakCoord {
            chr: "1".into(),
            start: 150_000,
            end: 150_500,
        }), // ~49 kb from body
    ];
    let params = GeneActivityParams {
        use_gene_boundaries: false,
        ..GeneActivityParams::default()
    };
    let act = gene_activity_from_atac_pb(&atac, &peak_coords, &genes, &params).unwrap();
    let near_w = archr_distance_weight(0, params.decay);
    let far_w = archr_distance_weight(150_250 - 101_000, params.decay);
    assert!(near_w > far_w);
    for j in 0..s {
        let expect = near_w + far_w; // size weight = 1 with one gene
        assert!(
            (act[(0, j)] - expect).abs() < 1e-4,
            "sample {j}: got {} want {expect}",
            act[(0, j)]
        );
    }
}

#[test]
fn gene_boundary_blocks_neighbor_promoter() {
    let s = 2usize;
    let mut atac = Mat::zeros(1, s);
    atac[(0, 0)] = 10.0;
    atac[(0, 1)] = 10.0;
    // Peak sits in gene B's body; gene A should not claim it when boundaries on.
    let genes = vec![
        Some(gene("1", 100_000, 110_000, Strand::Forward)),
        Some(gene("1", 200_000, 210_000, Strand::Forward)),
    ];
    let peak_coords = vec![Some(PeakCoord {
        chr: "1".into(),
        start: 205_000,
        end: 205_500,
    })];
    let params = GeneActivityParams::default();
    let act = gene_activity_from_atac_pb(&atac, &peak_coords, &genes, &params).unwrap();
    assert!(act[(0, 0)] < 1e-6, "gene A should not get B's body peak");
    assert!(act[(1, 0)] > 1.0, "gene B should score its body peak");
}
