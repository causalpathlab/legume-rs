use crate::link_community::bhc::*;
use crate::link_community::model::{LinkCommunityStats, LinkProfileStore};

/// Build a `LinkCommunityStats` directly from hand-specified per-community
/// aggregates. Uses `from_profiles` so caches stay in sync.
fn stats_from_aggregates(k: usize, m: usize, per_cluster: &[Vec<f32>]) -> LinkCommunityStats {
    assert_eq!(per_cluster.len(), k);
    for row in per_cluster {
        assert_eq!(row.len(), m);
    }
    // One edge per non-empty cluster, each edge carrying the cluster's full profile.
    let mut profiles: Vec<f32> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for (c, row) in per_cluster.iter().enumerate() {
        let s: f32 = row.iter().sum();
        if s <= 0.0 {
            continue; // empty cluster — no edge
        }
        profiles.extend_from_slice(row);
        labels.push(c);
    }
    let n_edges = labels.len();
    let store = LinkProfileStore::new(profiles, n_edges, m);
    LinkCommunityStats::from_profiles(&store, k, &labels)
}

#[test]
fn test_two_disjoint_clusters() {
    // A has all mass on gene 0, B on gene 1 — merging should be discouraged.
    let stats = stats_from_aggregates(
        2,
        4,
        &[vec![10.0, 0.0, 0.0, 0.0], vec![0.0, 10.0, 0.0, 0.0]],
    );
    let merges = bhc_merge(&stats, 1.0 / 4.0);
    assert_eq!(merges.len(), 1);
    assert!(
        merges[0].log_bf < 0.0,
        "disjoint clusters should have log_bf < 0, got {}",
        merges[0].log_bf
    );
}

#[test]
fn test_two_identical_clusters() {
    // Identical profiles — merging should be strongly favored.
    let stats = stats_from_aggregates(2, 4, &[vec![2.5, 2.5, 2.5, 2.5], vec![2.5, 2.5, 2.5, 2.5]]);
    let merges = bhc_merge(&stats, 1.0 / 4.0);
    assert_eq!(merges.len(), 1);
    assert!(
        merges[0].log_bf > 0.0,
        "identical clusters should have log_bf > 0, got {}",
        merges[0].log_bf
    );
}

#[test]
fn test_three_cluster_ordering() {
    // A ≈ B (both gene 0 heavy), C distant (gene 3 heavy).
    // First merge must be (A, B).
    let stats = stats_from_aggregates(
        3,
        4,
        &[
            vec![8.0, 1.0, 0.5, 0.5],
            vec![8.5, 1.0, 0.5, 0.0],
            vec![0.5, 0.5, 0.5, 8.0],
        ],
    );
    let merges = bhc_merge(&stats, 1.0 / 4.0);
    assert_eq!(merges.len(), 2);
    // First merge: ids 0 and 1 (A and B).
    let m0 = &merges[0];
    let (lo, hi) = if m0.left < m0.right {
        (m0.left, m0.right)
    } else {
        (m0.right, m0.left)
    };
    assert_eq!(
        (lo, hi),
        (0, 1),
        "first merge should join A(0) and B(1), got ({}, {})",
        lo,
        hi
    );
    // Second merge: the new node (id 3) with C (id 2).
    let m1 = &merges[1];
    let (lo2, hi2) = if m1.left < m1.right {
        (m1.left, m1.right)
    } else {
        (m1.right, m1.left)
    };
    assert_eq!((lo2, hi2), (2, 3));
    // And its n_edges should be the total.
    assert_eq!(m1.n_edges, 3);
}

#[test]
fn test_skips_empty() {
    // K=3, community 1 is empty (zero profile → no edge inserted by the test helper).
    let stats = stats_from_aggregates(
        3,
        4,
        &[
            vec![1.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0], // empty
            vec![0.0, 0.0, 2.0, 1.0],
        ],
    );
    let merges = bhc_merge(&stats, 1.0 / 4.0);
    assert_eq!(merges.len(), 1, "K_eff=2 should produce exactly 1 merge");
    // id 1 must never appear.
    assert_ne!(merges[0].left, 1);
    assert_ne!(merges[0].right, 1);
    // left < right, both in {0, 2}.
    assert_eq!(merges[0].left, 0);
    assert_eq!(merges[0].right, 2);
}

#[test]
fn test_cut_all_positive_collapses_to_one() {
    let stats = stats_from_aggregates(
        3,
        4,
        &[
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ],
    );
    let merges = bhc_merge(&stats, 1.0);
    for m in &merges {
        assert!(m.log_bf > 0.0, "identical clusters should give positive BF");
    }
    let labels = bhc_cut(&merges, 3, 0.0);
    assert_eq!(labels, vec![0, 0, 0]);
}

#[test]
fn test_cut_all_negative_preserves_originals() {
    let stats = stats_from_aggregates(
        3,
        4,
        &[
            vec![10.0, 0.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0],
        ],
    );
    let merges = bhc_merge(&stats, 1.0);
    for m in &merges {
        assert!(m.log_bf < 0.0, "disjoint clusters should give negative BF");
    }
    let labels = bhc_cut(&merges, 3, 0.0);
    let mut sorted = labels.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2], "three distinct consensus labels");
}

#[test]
fn test_cut_mixed_groups_ab_separately_from_c() {
    // A ≈ B merge positively, then (AB)+C merges negatively.
    let stats = stats_from_aggregates(
        3,
        4,
        &[
            vec![8.0, 1.0, 0.5, 0.5],
            vec![8.5, 1.0, 0.5, 0.0],
            vec![0.5, 0.5, 0.5, 8.0],
        ],
    );
    let merges = bhc_merge(&stats, 1.0);
    assert!(merges[0].log_bf > 0.0);
    assert!(merges[1].log_bf < 0.0);
    let labels = bhc_cut(&merges, 3, 0.0);
    assert_eq!(labels[0], labels[1], "A and B should share a label");
    assert_ne!(labels[0], labels[2], "C should have its own label");
}

#[test]
fn test_cut_empty_cluster_gets_minus_one() {
    let stats = stats_from_aggregates(
        3,
        4,
        &[
            vec![1.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0], // empty
            vec![0.0, 0.0, 2.0, 1.0],
        ],
    );
    let merges = bhc_merge(&stats, 1.0);
    let labels = bhc_cut(&merges, 3, 0.0);
    assert_eq!(labels[1], -1, "empty community gets label -1");
    assert!(labels[0] >= 0 && labels[2] >= 0);
}

#[test]
fn test_bayes_factor_analytical() {
    // γ=1, M=1, bg=1 (after flooring), so γ·bg = 1.
    // f(T_eff=1, S_eff=1) = lgamma(1) − lgamma(1+1) + lgamma(1+1) − lgamma(1) = 0.
    // Merged T_eff=2, S_eff=2: f = lgamma(1) − lgamma(3) + lgamma(3) − lgamma(1) = 0.
    // log_BF = 0 − 0 − 0 = 0 exactly.
    let stats = stats_from_aggregates(2, 1, &[vec![5.0], vec![5.0]]);
    let merges = bhc_merge(&stats, 1.0);
    assert_eq!(merges.len(), 1);
    assert!(
        merges[0].log_bf.abs() < 1e-9,
        "closed form: log_bf should be 0 for M=1, γ=1, identical sides, got {}",
        merges[0].log_bf
    );
}
