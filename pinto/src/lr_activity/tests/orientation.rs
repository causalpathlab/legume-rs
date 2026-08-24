//! What a stratum is, and what it must not depend on.
//!
//! Strata are link communities: only edges whose endpoints share a dominant
//! community realize one, and every such edge is listed once per orientation,
//! which makes anything summed over a stratum symmetric in the pair. The
//! load-bearing test is `swapping_the_endpoint_columns_changes_nothing` —
//! the edge list is a `KnnGraph` serialization whose endpoint order is an
//! artifact of barcode load order, and nothing here may depend on it.

use crate::lr_activity::orientation::CommunityStrata;

type Edge = (usize, usize, u32, Option<Box<str>>);

/// Two blocks of cells, 0..3 and 3..6, wired so every cell's incident edges
/// are mostly within its own block. Cells 0-2 end up dominant in community 0,
/// cells 3-5 in community 1, and the two edges across the middle bridge them.
fn two_block_graph() -> Vec<Edge> {
    let e = |i: usize, j: usize, k: u32| -> Edge { (i, j, k, None) };
    vec![
        e(0, 1, 0),
        e(1, 2, 0),
        e(0, 2, 0),
        e(3, 4, 1),
        e(4, 5, 1),
        e(3, 5, 1),
        // bridging edges, labelled with their own edge community
        e(2, 3, 2),
        e(2, 4, 2),
    ]
}

fn swapped(edges: &[Edge]) -> Vec<Edge> {
    edges
        .iter()
        .map(|(i, j, k, b)| (*j, *i, *k, b.clone()))
        .collect()
}

/// Strata are the communities whose cells share edges; edges bridging two
/// communities realize nothing.
#[test]
fn strata_are_communities_and_bridging_edges_sit_out() {
    let edges = two_block_graph();
    let d = CommunityStrata::from_edge_modes(&edges, &edges, 6);
    let comms: Vec<u32> = (0..d.n_strata()).map(|s| d.community(s)).collect();
    assert_eq!(comms, vec![0, 1], "one stratum per community, sorted");
    assert_eq!(d.label(0), "C0");
    // 3 within-block edges each, both orientations listed.
    assert_eq!(d.edges_in(0), 6);
    assert_eq!(d.edges_in(1), 6);
    // The two bridging edges realized nothing: totals account for every
    // instance.
    let total: usize = (0..d.n_strata()).map(|s| d.edges_in(s)).sum();
    assert_eq!(total, 12, "2 x (3 + 3) instances, bridges excluded");
}

/// Every edge appears exactly once per orientation, so any per-instance sum
/// is symmetric in the pair by construction.
#[test]
fn both_orientations_of_every_edge_are_listed() {
    let edges = two_block_graph();
    let d = CommunityStrata::from_edge_modes(&edges, &edges, 6);
    for s in 0..d.n_strata() {
        let mut plain = Vec::new();
        let mut flipped = Vec::new();
        for &(e, f) in d.oriented(s) {
            if f {
                flipped.push(e);
            } else {
                plain.push(e);
            }
        }
        plain.sort_unstable();
        flipped.sort_unstable();
        assert_eq!(
            plain, flipped,
            "stratum {s}: each edge must appear once each way"
        );
        assert_eq!(d.edges_in(s), d.oriented(s).len());
    }
}

/// The membership matrix is what the pseudobulk collapse weights cells by;
/// with both orientations enumerated there is one role, and a cell's mass
/// concentrates in its own community's stratum.
#[test]
fn memberships_weight_cells_into_their_own_community() {
    let edges = two_block_graph();
    let d = CommunityStrata::from_edge_modes(&edges, &edges, 6);
    let p = d.memberships(&edges, 6);
    for cell in 0..3 {
        assert!(p[(cell, 0)] > 0.0, "block-0 cell {cell} in C0");
        assert_eq!(p[(cell, 1)], 0.0, "block-0 cell {cell} not in C1");
    }
    for cell in 3..6 {
        assert!(p[(cell, 1)] > 0.0, "block-1 cell {cell} in C1");
        assert_eq!(p[(cell, 0)], 0.0, "block-1 cell {cell} not in C0");
    }
    // Rows are normalized over the cell's instances.
    for cell in 0..6 {
        let row: f32 = (0..d.n_strata()).map(|s| p[(cell, s)]).sum();
        assert!((row - 1.0).abs() < 1e-6, "cell {cell} row sums to 1");
    }
}

#[test]
fn swapping_the_endpoint_columns_changes_nothing() {
    let edges = two_block_graph();
    let flipped = swapped(&edges);
    let a = CommunityStrata::from_edge_modes(&edges, &edges, 6);
    let b = CommunityStrata::from_edge_modes(&flipped, &flipped, 6);

    assert_eq!(a.n_strata(), b.n_strata());
    for s in 0..a.n_strata() {
        assert_eq!(a.community(s), b.community(s), "stratum {s}");
        assert_eq!(a.edges_in(s), b.edges_in(s), "stratum {s} instance count");
    }
    assert_eq!(
        a.memberships(&edges, 6),
        b.memberships(&flipped, 6),
        "memberships must be blind to endpoint order"
    );
}

#[test]
fn stratum_ids_are_sorted_so_they_are_reproducible() {
    let edges = two_block_graph();
    let d = CommunityStrata::from_edge_modes(&edges, &edges, 6);
    let comms: Vec<u32> = (0..d.n_strata()).map(|s| d.community(s)).collect();
    let mut sorted = comms.clone();
    sorted.sort_unstable();
    assert_eq!(comms, sorted, "ids must follow sorted community order");
}

/// Expression pairs join cells that are similar but NOT adjacent. The
/// co-activity estimand presupposes contact, so such a pair must never be
/// tested. It may still inform the anchor: which community a cell belongs to
/// is a statement about the partition, and more evidence there is strictly
/// better. The two edge lists have different jobs; this pins the difference.
#[test]
fn expression_pairs_inform_the_anchor_but_are_never_tested() {
    let spatial = two_block_graph();

    // Cell 5 sits in block 1 by adjacency, but gets three expression pairs
    // into community 0. Those are enough to move its dominant community, and
    // they are the only thing that can, since none of them is adjacent.
    let e = |i: usize, j: usize, k: u32| -> Edge { (i, j, k, None) };
    let mut anchor = spatial.clone();
    anchor.extend([e(5, 0, 0), e(5, 1, 0), e(5, 2, 0)]);

    let tested_only = CommunityStrata::from_edge_modes(&spatial, &spatial, 6);
    let with_anchor = CommunityStrata::from_edge_modes(&anchor, &spatial, 6);

    // Cell 5 moved to community 0, so its two spatial edges (into 3 and 4)
    // now BRIDGE communities and drop out of the tested instances entirely.
    let n_tested = |d: &CommunityStrata| (0..d.n_strata()).map(|s| d.edges_in(s)).sum::<usize>();
    assert!(
        n_tested(&with_anchor) < n_tested(&tested_only),
        "reanchoring cell 5 must move its edges out of the C1 stratum"
    );
    // And no stratum gained instances: expression pairs never enter the
    // tested set themselves.
    let spread = |d: &CommunityStrata| -> Vec<(u32, usize)> {
        (0..d.n_strata())
            .map(|s| (d.community(s), d.edges_in(s)))
            .collect()
    };
    for (c, n) in spread(&with_anchor) {
        let before = spread(&tested_only)
            .into_iter()
            .find(|&(c2, _)| c2 == c)
            .map_or(0, |(_, n2)| n2);
        assert!(
            n <= before,
            "community {c} gained tested instances from expression pairs"
        );
    }
}
