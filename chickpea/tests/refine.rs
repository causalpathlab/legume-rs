//! Within-cluster link recomputation.

mod common;

use chickpea::common::Mat;
use chickpea::p2g::link_map::LinkParams;
use chickpea::p2g::refine::*;
use common::{peak, tss};

#[test]
fn link_appears_only_in_active_cluster() {
    let s = 40usize;
    let mut rna = Mat::zeros(1, s);
    let mut atac = Mat::zeros(1, s);
    let mut sample_cluster = vec![Some(0usize); s];
    for j in 0..s {
        if j < 20 {
            let z = (j as f32) * 0.2;
            let signal = z.sin().abs() + 0.1;
            rna[(0, j)] = signal * 10.0;
            atac[(0, j)] = signal * 8.0;
            sample_cluster[j] = Some(0);
        } else {
            // Cluster 1: gene and peak fluctuate independently.
            rna[(0, j)] = ((j % 5) as f32) * 0.5 + 0.1;
            atac[(0, j)] = ((j % 7) as f32) * 0.4 + 0.1;
            sample_cluster[j] = Some(1);
        }
    }

    let gene_tss = vec![tss(100_000)];
    let peak_coords = vec![peak(100_000)];
    let params = LinkParams {
        cis_window: 500_000,
        max_cis: 10,
        min_weight: 0.3,
        ..LinkParams::default()
    };

    let links = refine_within_clusters(
        &rna,
        &atac,
        &gene_tss,
        &peak_coords,
        &sample_cluster,
        5,
        &params,
    )
    .unwrap();

    let in0 = links.iter().filter(|l| l.cluster == 0).count();
    let in1 = links.iter().filter(|l| l.cluster == 1).count();
    assert!(
        in0 >= 1,
        "active cluster should keep the cis link; got {links:?}"
    );
    assert_eq!(
        in1, 0,
        "inactive cluster should not pass min_weight; got {links:?}"
    );
}
