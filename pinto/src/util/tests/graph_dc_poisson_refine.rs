use crate::test_support::make_test_graph;
use crate::util::graph_dc_poisson_refine::{ConnectivityGuard, GraphProposer};
use data_beans_alg::dc_poisson::{CandidateProposer, MoveGuard};

#[test]
fn test_graph_proposer_intersects_siblings_with_graph_neighbors() {
    // Cells 0,1 belong to entity 0; cells 2,3 to entity 1; cells 4,5 to entity 2.
    // Graph edges within and across entities:
    //   (0,1)  intra entity 0 (no neighbor contribution)
    //   (1,2)  entity 0 ↔ entity 1
    //   (3,4)  entity 1 ↔ entity 2
    let graph = make_test_graph(6, vec![(0, 1), (1, 2), (3, 4)]);
    let cell_to_entity = vec![0usize, 0, 1, 1, 2, 2];

    // Two sibling groups: {0, 1} and {2}. Entity labels at this level:
    //   entity 0 in group 0
    //   entity 1 in group 1
    //   entity 2 in group 2
    let entity_labels = vec![0usize, 1, 2];
    let siblings = vec![
        vec![0usize, 1], // entity 0 can move between groups 0 and 1
        vec![0usize, 1], // entity 1 can too
        vec![2usize],    // entity 2 is alone in its sibling set
    ];

    let prop = GraphProposer::new(&graph, &cell_to_entity, &entity_labels, siblings);
    let cand = prop.propose(&entity_labels);

    // Entity 0's graph neighbor is entity 1 (group 1); siblings = {0,1}; intersect = {1};
    // current label 0 is appended → {0, 1}.
    assert_eq!(cand[0], vec![0, 1]);

    // Entity 1's graph neighbors are entity 0 (group 0) and entity 2 (group 2);
    // siblings = {0, 1}; intersect = {0}; current label 1 is appended → {0, 1}.
    assert_eq!(cand[1], vec![0, 1]);

    // Entity 2 is a singleton sibling — candidates are exactly its siblings.
    assert_eq!(cand[2], vec![2]);
}

#[test]
fn test_graph_proposer_falls_back_when_no_neighbor_in_siblings() {
    // Entities 0 and 1 have each other as sole graph neighbors, but live in
    // different sibling sets. Neither should have any valid intersection;
    // fallback is the full sibling list.
    let graph = make_test_graph(2, vec![(0, 1)]);
    let cell_to_entity = vec![0usize, 1];
    let entity_labels = vec![0usize, 1];
    let siblings = vec![vec![0usize, 2], vec![1usize, 3]];

    let prop = GraphProposer::new(&graph, &cell_to_entity, &entity_labels, siblings.clone());
    let cand = prop.propose(&entity_labels);
    assert_eq!(cand[0], siblings[0]);
    assert_eq!(cand[1], siblings[1]);
}

#[test]
fn test_connectivity_guard_rejects_cut_vertex_move() {
    // Four entities on a path: 0 — 1 — 2 — 3. One cell per entity.
    // Entities {0, 1, 2} are in cluster 0; entity 3 is alone in cluster 1.
    // Removing entity 1 from cluster 0 would split {0, 2} into disjoint
    // components. Moving entity 0 or entity 2 is fine (they're leaves in
    // the induced subgraph).
    let graph = make_test_graph(4, vec![(0, 1), (1, 2), (2, 3)]);
    let cell_to_entity = vec![0usize, 1, 2, 3];
    let labels = vec![0usize, 0, 0, 1];

    let guard = ConnectivityGuard::new(&graph, &cell_to_entity, 4);

    // Moving the cut vertex (entity 1) out of cluster 0 must be rejected.
    assert!(
        !guard.accept_move(1, 0, &labels),
        "moving cut vertex should be vetoed"
    );

    // Moving an articulation-safe vertex (entity 0, a leaf) should be OK.
    assert!(
        guard.accept_move(0, 0, &labels),
        "moving a leaf should be accepted"
    );

    // Moving entity 2 (also a leaf in the {0,1,2} subgraph since 3 is in
    // cluster 1) should be OK.
    assert!(
        guard.accept_move(2, 0, &labels),
        "moving the other leaf should be accepted"
    );
}

#[test]
fn test_connectivity_guard_trivial_cases() {
    // A cluster of size 1 (only the moving entity) trivially stays
    // connected — nothing to disconnect.
    let graph = make_test_graph(2, vec![(0, 1)]);
    let cell_to_entity = vec![0usize, 1];
    let labels = vec![0usize, 1];

    let guard = ConnectivityGuard::new(&graph, &cell_to_entity, 2);

    // Entity 0 is alone in cluster 0; it has no in-cluster neighbors.
    // Moving it cannot disconnect anything.
    assert!(guard.accept_move(0, 0, &labels));

    // Entity 0 in a cluster with only one other entity (1 in-cluster
    // neighbor): removal leaves a single-node subgraph, trivially connected.
    let labels_two = vec![0usize, 0];
    assert!(guard.accept_move(0, 0, &labels_two));
}
