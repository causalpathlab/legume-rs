//! Standalone tests for the shared BHC core.
//!
//! Pinto's link-community-flavored tests live alongside `LinkCommunityStats`
//! in pinto/src/link_community/tests/bhc.rs and exercise the same code via
//! the wrapper.

use data_beans_alg::bhc::{bhc_cut, bhc_merge, BhcInput};

/// Build (gene_sum, size_sum, effective_size) from per-cluster gene rows.
/// Empty rows (all-zero) get effective_size=0 so the core skips them.
fn make_input(per_cluster: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    assert!(!per_cluster.is_empty());
    let m = per_cluster[0].len();
    let mut gene_sum = Vec::with_capacity(per_cluster.len() * m);
    let mut size_sum = Vec::with_capacity(per_cluster.len());
    let mut eff = Vec::with_capacity(per_cluster.len());
    for row in per_cluster {
        assert_eq!(row.len(), m);
        let s: f64 = row.iter().sum();
        gene_sum.extend_from_slice(row);
        size_sum.push(s);
        // Use ceil so any positive mass becomes a real cluster with eff >= 1.
        eff.push(s.ceil().max(0.0) as usize);
    }
    (gene_sum, size_sum, eff)
}

fn run(per_cluster: &[Vec<f64>], gamma: f64) -> Vec<data_beans_alg::bhc::BhcMerge> {
    let (gs, ss, es) = make_input(per_cluster);
    bhc_merge(
        BhcInput {
            k: per_cluster.len(),
            m: per_cluster[0].len(),
            gene_sum: &gs,
            size_sum: &ss,
            effective_size: &es,
        },
        gamma,
    )
}

#[test]
fn two_disjoint_clusters_negative_bf() {
    let merges = run(
        &[vec![10.0, 0.0, 0.0, 0.0], vec![0.0, 10.0, 0.0, 0.0]],
        1.0 / 4.0,
    );
    assert_eq!(merges.len(), 1);
    assert!(merges[0].log_bf < 0.0, "got {}", merges[0].log_bf);
}

#[test]
fn two_identical_clusters_positive_bf() {
    let merges = run(
        &[vec![2.5, 2.5, 2.5, 2.5], vec![2.5, 2.5, 2.5, 2.5]],
        1.0 / 4.0,
    );
    assert_eq!(merges.len(), 1);
    assert!(merges[0].log_bf > 0.0, "got {}", merges[0].log_bf);
}

#[test]
fn empty_cluster_skipped() {
    // Cluster 1 has effective_size=0 → never appears in any merge.
    let (gs, ss, mut es) = make_input(&[
        vec![1.0, 2.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 2.0, 1.0],
    ]);
    es[1] = 0;
    let merges = bhc_merge(
        BhcInput {
            k: 3,
            m: 4,
            gene_sum: &gs,
            size_sum: &ss,
            effective_size: &es,
        },
        1.0,
    );
    assert_eq!(merges.len(), 1);
    assert_eq!(merges[0].left, 0);
    assert_eq!(merges[0].right, 2);
    let labels = bhc_cut(&merges, 3, 0.0);
    assert_eq!(labels[1], -1, "empty cluster gets -1");
    assert!(labels[0] >= 0 && labels[2] >= 0);
}

#[test]
fn cut_at_zero_collapses_identical_groups_only() {
    // A ≈ B (both gene 0 heavy), C distant.
    let merges = run(
        &[
            vec![8.0, 1.0, 0.5, 0.5],
            vec![8.5, 1.0, 0.5, 0.0],
            vec![0.5, 0.5, 0.5, 8.0],
        ],
        1.0,
    );
    assert!(merges[0].log_bf > 0.0);
    assert!(merges[1].log_bf < 0.0);
    let labels = bhc_cut(&merges, 3, 0.0);
    assert_eq!(labels[0], labels[1]);
    assert_ne!(labels[0], labels[2]);
}

#[test]
fn analytical_zero_bf_for_m1_gamma1() {
    // M=1, γ=1 ⇒ all f-terms collapse and log_BF = 0 exactly.
    let merges = run(&[vec![5.0], vec![5.0]], 1.0);
    assert_eq!(merges.len(), 1);
    assert!(merges[0].log_bf.abs() < 1e-9, "got {}", merges[0].log_bf);
}
