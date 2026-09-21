//! Leiden clustering with the min-cell gate.

use chickpea::p2g::cluster::*;
use nalgebra::DMatrix;

#[test]
fn min_cell_gate_drops_tiny_cluster() {
    // Two well-separated blobs (8+8) and one singleton far away.
    let mut e = DMatrix::<f32>::zeros(17, 2);
    for i in 0..8 {
        e[(i, 0)] = 1.0;
        e[(i, 1)] = 0.0;
    }
    for i in 8..16 {
        e[(i, 0)] = 0.0;
        e[(i, 1)] = 1.0;
    }
    e[(16, 0)] = -1.0;
    e[(16, 1)] = -1.0;

    let out = cluster_cells(&e, Some(3), 3).unwrap();
    assert!(
        out.n_clusters >= 2,
        "expected ≥2 kept clusters, got {}",
        out.n_clusters
    );
    assert!(
        out.label[16].is_none(),
        "singleton should be dropped by min_cells=3; labels={:?}",
        out.label
    );
    let kept = out.label.iter().filter(|l| l.is_some()).count();
    assert!(kept >= 16, "both main blobs should survive; kept={kept}");
    assert_eq!(out.sizes.iter().sum::<usize>(), kept);
}
