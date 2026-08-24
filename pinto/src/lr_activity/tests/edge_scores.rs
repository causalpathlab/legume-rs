//! Contract of the descriptive edge-score mode: instances are BOTH
//! orientations of every within-community spatial edge, grouped by batch;
//! bridging edges never realize a stratum, straddling edges (endpoint
//! batch labels differ) score in no batch, and the two score columns match
//! a hand enumeration of those instances.

use crate::lr_activity::edge_scores::{compute_edge_scores, EdgeScoresInput};
use crate::lr_activity::orientation::CommunityStrata;
use crate::util::common::*;

type Edge = (usize, usize, u32, Option<Box<str>>);

/// Cells 0-2 sit in community 0, cells 3-5 in community 1. One bridging
/// edge (2,3) sits out by construction.
fn fixture_edges() -> Vec<Edge> {
    let b = |s: &str| -> Option<Box<str>> { Some(s.into()) };
    vec![
        (0, 1, 0, b("a")),
        (1, 2, 0, b("b")),
        (3, 4, 1, b("a")),
        (4, 5, 1, b("a")),
        (2, 3, 2, b("a")), // bridges C0 and C1: no stratum
    ]
}

/// L is global gene 10 (row 0 of x_lr), R is global gene 11 (row 1).
fn fixture_genes() -> (
    Vec<(Box<str>, Box<str>, usize, usize)>,
    HashMap<usize, usize>,
    Mat,
) {
    let pairs = vec![("LIG1".into(), "REC1".into(), 10usize, 11usize)];
    let mut gene_to_local: HashMap<usize, usize> = HashMap::default();
    gene_to_local.insert(10, 0);
    gene_to_local.insert(11, 1);
    // Counts per cell: L row then R row (column = cell).
    let x_lr = Mat::from_row_slice(
        2,
        6,
        &[
            2.0, 5.0, 0.0, 1.0, 3.0, 4.0, // ligand
            1.0, 0.0, 7.0, 2.0, 6.0, 0.0, // receptor
        ],
    );
    (pairs, gene_to_local, x_lr)
}

fn run(edges: &[Edge]) -> (Vec<crate::lr_activity::edge_scores::EdgeScoreRow>, usize) {
    let (pairs, gene_to_local, x_lr) = fixture_genes();
    let strata = CommunityStrata::from_edge_modes(edges, edges, 6);
    let log_depth: Vec<f32> = (0..6).map(|i| (10 + i) as f32).collect();
    compute_edge_scores(&EdgeScoresInput {
        edges,
        strata: &strata,
        pairs: &pairs,
        gene_to_local: &gene_to_local,
        x_lr: &x_lr,
        log_depth: &log_depth,
    })
}

/// The (batch, community) groups are exactly the ones with edges; a
/// zero-edge combination gets no row at all.
#[test]
fn a_zero_edge_batch_community_gets_no_row() {
    let (rows, n_straddling) = run(&fixture_edges());
    assert_eq!(n_straddling, 0);
    let keys: Vec<(String, u32)> = rows
        .iter()
        .map(|r| (r.batch.to_string(), r.community))
        .collect();
    // Batch "b" has no community-1 edge, and the bridge realizes nothing.
    assert_eq!(
        keys,
        vec![
            ("a".to_string(), 0),
            ("b".to_string(), 0),
            ("a".to_string(), 1)
        ]
    );
}

/// Hand enumeration of the (a, C0) group: one edge (0,1), two orientations
/// (0,1) and (1,0). With l(k) = ln1p(x_L(k)), r(k) = ln1p(x_R(k)):
/// product  = (l0*r1 + l1*r0) / 2
/// coupling = product - ((l0+l1)/2) * ((r0+r1)/2)
#[test]
fn scores_match_a_hand_enumeration() {
    let (rows, _) = run(&fixture_edges());
    let row = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 0)
        .expect("(a, C0) row");

    let l = |v: f32| v.ln_1p();
    let (l0, l1) = (l(2.0), l(5.0));
    let (r0, r1) = (l(1.0), l(0.0));
    let product = (l0 * r1 + l1 * r0) / 2.0;
    let coupling = product - ((l0 + l1) / 2.0) * ((r0 + r1) / 2.0);

    assert_eq!(row.n_edges, 1, "one physical edge behind two orientations");
    assert!((row.product - product).abs() < 1e-6, "got {}", row.product);
    assert!(
        (row.coupling - coupling).abs() < 1e-6,
        "got {}",
        row.coupling
    );
    // Unique cells of the group are 0 and 1; depth fixture is 10 + cell.
    assert!((row.mean_log_depth - 10.5).abs() < 1e-6);

    // The 2-edge (a, C1) group, same enumeration across both edges.
    let row = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 1)
        .expect("(a, C1) row");
    let (l3, l4, l5) = (l(1.0), l(3.0), l(4.0));
    let (r3, r4, r5) = (l(2.0), l(6.0), l(0.0));
    // Edges (3,4) and (4,5), both ways each: instances (3,4),(4,3),(4,5),(5,4).
    let product = (l3 * r4 + l4 * r3 + l4 * r5 + l5 * r4) / 4.0;
    let mean_l = (l3 + l4 + l4 + l5) / 4.0;
    let mean_r = (r4 + r3 + r5 + r4) / 4.0;
    let coupling = product - mean_l * mean_r;
    assert_eq!(row.n_edges, 2);
    assert!((row.product - product).abs() < 1e-6, "got {}", row.product);
    assert!(
        (row.coupling - coupling).abs() < 1e-6,
        "got {}",
        row.coupling
    );
}

/// Both-orientation enumeration makes the score symmetric in the pair as a
/// structural fact; this guards the enumeration.
#[test]
fn swapping_ligand_and_receptor_changes_nothing() {
    let edges = fixture_edges();
    let (mut pairs, gene_to_local, x_lr) = fixture_genes();
    pairs.push(("REC1".into(), "LIG1".into(), 11, 10));
    let strata = CommunityStrata::from_edge_modes(&edges, &edges, 6);
    let log_depth = vec![0.0f32; 6];
    let (rows, _) = compute_edge_scores(&EdgeScoresInput {
        edges: &edges,
        strata: &strata,
        pairs: &pairs,
        gene_to_local: &gene_to_local,
        x_lr: &x_lr,
        log_depth: &log_depth,
    });
    for key in [("a", 0u32), ("b", 0), ("a", 1)] {
        let group: Vec<_> = rows
            .iter()
            .filter(|r| r.batch.as_ref() == key.0 && r.community == key.1)
            .collect();
        assert_eq!(group.len(), 2, "both pair orders scored");
        assert!(
            (group[0].product - group[1].product).abs() < 1e-6,
            "{key:?}: product must be symmetric"
        );
        assert!(
            (group[0].coupling - group[1].coupling).abs() < 1e-6,
            "{key:?}: coupling must be symmetric"
        );
    }
}

/// A straddling edge (labels exist, but this edge's endpoints disagree so
/// its joint label is None) belongs to no single batch: it is dropped from
/// every group and counted once.
#[test]
fn straddling_edges_are_dropped_and_counted() {
    let mut edges = fixture_edges();
    edges.push((0, 2, 0, None)); // C0-internal but straddling
    let (rows, n_straddling) = run(&edges);
    assert_eq!(n_straddling, 1, "counted once, not once per orientation");
    let row = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 0)
        .unwrap();
    assert_eq!(row.n_edges, 1, "the straddler joined no group");
}

/// With no labels on file at all the run is single-batch: every edge lands
/// in the `all` pseudo-batch and nothing counts as straddling.
#[test]
fn unlabeled_runs_score_one_all_batch() {
    let edges: Vec<Edge> = fixture_edges()
        .into_iter()
        .map(|(i, j, k, _)| (i, j, k, None))
        .collect();
    let (rows, n_straddling) = run(&edges);
    assert_eq!(n_straddling, 0);
    assert!(rows.iter().all(|r| r.batch.as_ref() == "all"));
    let n_edges: Vec<u32> = rows.iter().map(|r| r.n_edges).collect();
    assert_eq!(n_edges, vec![2, 2], "C0 and C1, batches pooled");
}
