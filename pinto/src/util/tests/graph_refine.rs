use crate::test_support::make_test_graph;
use crate::util::common::*;
use crate::util::graph_refine::*;

#[test]
fn test_would_disconnect_articulation() {
    // Path 0-1-2-3-4, all in one cluster.
    // Removing 2 (middle) disconnects {0,1} from {3,4}.
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
    let graph = make_test_graph(5, edges);
    let labels = vec![0, 0, 0, 0, 0];
    let mut s = ArticulationScratch::default();

    // 2 is an articulation point — must report disconnect.
    assert!(would_disconnect_cluster(2, 0, &graph, &labels, &mut s));
    // Endpoints (degree 1) are leaves — never disconnect.
    assert!(!would_disconnect_cluster(0, 0, &graph, &labels, &mut s));
    assert!(!would_disconnect_cluster(4, 0, &graph, &labels, &mut s));
}

#[test]
fn test_would_disconnect_triangle_safe() {
    // Triangle: every node has two neighbours that are themselves connected,
    // so removing any node leaves the others connected.
    let edges = vec![(0, 1), (1, 2), (0, 2)];
    let graph = make_test_graph(3, edges);
    let labels = vec![0, 0, 0];
    let mut s = ArticulationScratch::default();
    for i in 0..3 {
        assert!(!would_disconnect_cluster(i, 0, &graph, &labels, &mut s));
    }
}

#[test]
fn test_refine_labels_basic_move() {
    // 4 nodes: two well-separated feature clusters but the *initial* labels
    // are wrong — node 2 starts in cluster 0 while its features match cluster 1.
    // Edges: 0-1-2-3 (so node 2 is graph-adjacent to cluster 1 via node 3).
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let graph = make_test_graph(4, edges);

    // Features: nodes 0,1 → e0; nodes 2,3 → e1.
    let mut features = Mat::zeros(2, 4);
    features[(0, 0)] = 1.0;
    features[(0, 1)] = 1.0;
    features[(1, 2)] = 1.0;
    features[(1, 3)] = 1.0;

    // Initial wrong labels: 2 is mis-assigned to cluster 0.
    let mut labels = vec![0, 0, 0, 1];

    let moves = refine_labels(&features, &graph, &mut labels, 10, 42);
    assert!(moves > 0, "should move at least one node");
    // Node 2 should join cluster 1 (where its feature lives).
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
}

#[test]
fn test_refine_labels_respects_connectivity() {
    // Path 0-1-2-3-4 all in cluster 0; node 5 in cluster 1, connected to 2.
    // Node 2 has features matching cluster 1, so naive refinement would move it,
    // disconnecting {0,1} from {3,4}. The connectivity check must prevent this.
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)];
    let graph = make_test_graph(6, edges);

    // Cluster 0 features (e0); cluster 1 features (e1). Node 2's feature is e1.
    let mut features = Mat::zeros(2, 6);
    features[(0, 0)] = 1.0;
    features[(0, 1)] = 1.0;
    features[(0, 3)] = 1.0;
    features[(0, 4)] = 1.0;
    features[(1, 2)] = 1.0;
    features[(1, 5)] = 1.0;

    let mut labels = vec![0, 0, 0, 0, 0, 1];
    let _moves = refine_labels(&features, &graph, &mut labels, 10, 7);

    // Node 2 must remain in cluster 0 because moving it disconnects {0,1} from {3,4}.
    assert_eq!(
        labels[2], 0,
        "node 2 should be blocked by connectivity check"
    );
}

#[test]
fn test_refine_labels_converges_idempotent() {
    // Two cliques connected by a single bridge — initial labels already optimal.
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)];
    let graph = make_test_graph(6, edges);

    let mut features = Mat::zeros(4, 6);
    for i in 0..3 {
        features[(0, i)] = 1.0;
    }
    for i in 3..6 {
        features[(1, i)] = 1.0;
    }
    let mut labels = vec![0, 0, 0, 1, 1, 1];

    let moves = refine_labels(&features, &graph, &mut labels, 10, 11);
    assert_eq!(moves, 0, "already-optimal labels should produce no moves");
    assert_eq!(labels, vec![0, 0, 0, 1, 1, 1]);
}
