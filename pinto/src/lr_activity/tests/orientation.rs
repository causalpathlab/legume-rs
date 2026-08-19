//! What "sender" means, and what it must not depend on.
//!
//! The load-bearing test is `swapping_the_endpoint_columns_changes_nothing`.
//! The edge list this command reads is a `KnnGraph` serialization, whose
//! endpoint order is canonical `(min_index, max_index)` and therefore an
//! artifact of barcode load order. A test that is directional about biology
//! must be blind to it.

use crate::lr_activity::orientation::DirectedStrata;

type Edge = (usize, usize, u32, Option<Box<str>>);

/// Two blocks of cells, 0..3 and 3..6, wired so every cell's incident edges
/// are mostly within its own block. Cells 0-2 end up dominant in community 0,
/// cells 3-5 in community 1, and the two edges across the middle are the
/// interface.
fn two_block_graph() -> Vec<Edge> {
    let e = |i: usize, j: usize, k: u32| -> Edge { (i, j, k, None) };
    vec![
        e(0, 1, 0),
        e(1, 2, 0),
        e(0, 2, 0),
        e(3, 4, 1),
        e(4, 5, 1),
        e(3, 5, 1),
        // the interface, labelled with its own edge community
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

#[test]
fn a_cell_takes_the_community_of_its_incident_edges() {
    let edges = two_block_graph();
    let d = DirectedStrata::from_edge_modes(&edges, &edges, 6);
    // Cells 2, 3 and 4 also touch the interface, but their own block still
    // dominates, which is what makes the assignment robust to a thin border.
    let hetero: Vec<(u32, u32)> = (0..d.n_strata())
        .filter(|&s| !d.is_homotypic(s))
        .map(|s| d.pair(s))
        .collect();
    assert!(hetero.contains(&(0, 1)), "0->1 missing from {hetero:?}");
    assert!(hetero.contains(&(1, 0)), "1->0 missing from {hetero:?}");
    assert_eq!(
        hetero.len(),
        2,
        "an interface between two communities gives exactly two directions"
    );
    let self_strata: Vec<(u32, u32)> = (0..d.n_strata())
        .filter(|&s| d.is_homotypic(s))
        .map(|s| d.pair(s))
        .collect();
    assert_eq!(self_strata, vec![(0, 0), (1, 1)]);
}

#[test]
fn both_directions_of_an_interface_are_offered() {
    let edges = two_block_graph();
    let d = DirectedStrata::from_edge_modes(&edges, &edges, 6);
    let ab = (0..d.n_strata()).find(|&s| d.pair(s) == (0, 1)).unwrap();
    let ba = (0..d.n_strata()).find(|&s| d.pair(s) == (1, 0)).unwrap();
    assert_ne!(ab, ba, "the two directions must be separate strata");
    assert_eq!(d.edges_in(ab), d.edges_in(ba), "same edges, both ways");

    // In `0->1` the block-0 cell sends; in `1->0` the block-1 cell sends.
    let (p_send, p_recv) = d.role_memberships(&edges, 6);
    for cell in [0usize, 1, 2] {
        assert!(
            p_send[(cell, ab)] >= p_recv[(cell, ab)],
            "block-0 cell {cell} should send in 0->1"
        );
    }
    for cell in [3usize, 4, 5] {
        assert!(
            p_recv[(cell, ab)] >= p_send[(cell, ab)],
            "block-1 cell {cell} should receive in 0->1"
        );
    }
}

#[test]
fn a_homotypic_stratum_is_marked_and_symmetric() {
    let edges = two_block_graph();
    let d = DirectedStrata::from_edge_modes(&edges, &edges, 6);
    let self0 = (0..d.n_strata()).find(|&s| d.pair(s) == (0, 0)).unwrap();
    assert!(d.is_homotypic(self0));
    assert_eq!(d.label(self0), "C0");

    // Both endpoints take both roles, so no direction can be read off it.
    let (p_send, p_recv) = d.role_memberships(&edges, 6);
    for cell in 0..6 {
        assert_eq!(
            p_send[(cell, self0)],
            p_recv[(cell, self0)],
            "cell {cell} must be symmetric in a self stratum"
        );
    }
}

#[test]
fn swapping_the_endpoint_columns_changes_nothing() {
    let edges = two_block_graph();
    let flipped = swapped(&edges);
    let a = DirectedStrata::from_edge_modes(&edges, &edges, 6);
    let b = DirectedStrata::from_edge_modes(&flipped, &flipped, 6);

    // Same strata, in the same order: ids are assigned by sorting the realized
    // pairs, not by the order edges arrive, so they are joinable across runs.
    assert_eq!(a.n_strata(), b.n_strata());
    for s in 0..a.n_strata() {
        assert_eq!(a.pair(s), b.pair(s), "stratum {s}");
        assert_eq!(a.edges_in(s), b.edges_in(s), "stratum {s} edge count");
    }

    // And the role memberships the statistic is built from are identical.
    let (sa, ra) = a.role_memberships(&edges, 6);
    let (sb, rb) = b.role_memberships(&flipped, 6);
    assert_eq!(sa, sb, "send memberships");
    assert_eq!(ra, rb, "recv memberships");
}

/// `edges_in` is the sparsity filter's input, so it has to agree with what
/// `oriented` actually yields. A homotypic edge is listed both ways, and if
/// `resolve` counted it once the self strata would face a 2x stricter bar.
#[test]
fn the_edge_count_matches_what_the_oriented_listing_yields() {
    let edges = two_block_graph();
    let d = DirectedStrata::from_edge_modes(&edges, &edges, 6);
    for s in 0..d.n_strata() {
        assert_eq!(
            d.edges_in(s),
            d.oriented(s).len(),
            "stratum {} ({})",
            s,
            d.label(s)
        );
    }
    // And the self strata really are the ones that double up.
    let self0 = (0..d.n_strata()).find(|&s| d.pair(s) == (0, 0)).unwrap();
    assert_eq!(
        d.edges_in(self0),
        6,
        "3 within-block edges, listed both ways"
    );
}

#[test]
fn stratum_ids_are_sorted_so_they_are_reproducible() {
    let edges = two_block_graph();
    let d = DirectedStrata::from_edge_modes(&edges, &edges, 6);
    let pairs: Vec<(u32, u32)> = (0..d.n_strata()).map(|s| d.pair(s)).collect();
    let mut sorted = pairs.clone();
    sorted.sort_unstable();
    assert_eq!(pairs, sorted, "ids must follow sorted (sender, receiver)");
}

#[test]
fn an_oriented_listing_puts_the_sender_first() {
    let edges = two_block_graph();
    let d = DirectedStrata::from_edge_modes(&edges, &edges, 6);
    let ab = (0..d.n_strata()).find(|&s| d.pair(s) == (0, 1)).unwrap();
    for &(e, flipped) in d.oriented(ab) {
        let (i, j, _, _) = edges[e as usize];
        let sender = if flipped { j } else { i };
        assert!(
            (0..3).contains(&sender),
            "sender {sender} should sit in block 0 for stratum 0->1"
        );
    }
}

/// Expression pairs join cells that are similar but NOT adjacent. A
/// directional ligand-receptor test presupposes contact, so such a pair has no
/// estimand and must never be tested. It may still inform the anchor: which
/// community a cell belongs to is a statement about the partition, and more
/// evidence there is strictly better.
///
/// So the two edge lists have different jobs, and this pins the difference.
#[test]
fn expression_pairs_inform_the_anchor_but_are_never_tested() {
    let spatial = two_block_graph();

    // Cell 5 sits in block 1 by adjacency, but gets three expression pairs
    // into community 0. Those are enough to move its dominant community, and
    // they are the only thing that can, since none of them is adjacent.
    let e = |i: usize, j: usize, k: u32| -> Edge { (i, j, k, None) };
    let mut anchor = spatial.clone();
    anchor.extend([e(5, 0, 0), e(5, 1, 0), e(5, 2, 0)]);

    let tested_only = DirectedStrata::from_edge_modes(&spatial, &spatial, 6);
    let with_anchor = DirectedStrata::from_edge_modes(&anchor, &spatial, 6);

    // The tested set is the spatial list in both, so the edge count cannot move.
    let n_tested = |d: &DirectedStrata| (0..d.n_strata()).map(|s| d.edges_in(s)).sum::<usize>();
    assert_eq!(
        n_tested(&with_anchor),
        n_tested(&tested_only),
        "an expression pair must never enter the tested set"
    );

    // But the anchor did move. Cell 5 is adjacent only within block 1, so
    // nothing spatial can reassign it; the three expression pairs can, and
    // that reassignment shows up as its two spatial edges moving out of the
    // within-block stratum and into a between-block one. Compare the edge
    // counts per stratum, not the stratum labels: the same four strata exist
    // either way, so the labels alone would not notice.
    let spread = |d: &DirectedStrata| -> Vec<((u32, u32), usize)> {
        (0..d.n_strata()).map(|s| (d.pair(s), d.edges_in(s))).collect()
    };
    assert_ne!(
        spread(&with_anchor),
        spread(&tested_only),
        "expression pairs must reach the anchor"
    );
}
