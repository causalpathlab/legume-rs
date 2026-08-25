use crate::cell_activity_graph_embedding::gene_gating::{
    fold_active_edges_to_super, GeneActiveEdges,
};

/// Weights of fine edges landing on the same super edge must SUM; the
/// intra-PB fine edge must vanish; ids come back sorted.
#[test]
fn fold_sums_weights_and_drops_intra_pb_edges() {
    // 4 fine edges; edges 0 and 2 map to super edge 1, edge 1 to super
    // edge 0, edge 3 is intra-PB.
    let fine_to_super = vec![Some(1usize), Some(0), Some(1), None];
    let activities = GeneActiveEdges {
        gene_active_edges: vec![vec![0u32, 1, 2, 3], vec![1, 3]],
        gene_active_edge_weights: vec![vec![0.5f32, 2.0, 0.25, 9.0], vec![4.0, 9.0]],
    };
    let folded = fold_active_edges_to_super(activities, &fine_to_super);
    assert_eq!(folded.gene_active_edges[0], vec![0u32, 1]);
    assert_eq!(folded.gene_active_edge_weights[0], vec![2.0f32, 0.75]);
    // Gene 1: only edge 1 survives (edge 3 is intra-PB).
    assert_eq!(folded.gene_active_edges[1], vec![0u32]);
    assert_eq!(folded.gene_active_edge_weights[1], vec![4.0f32]);
}

/// A gene whose every active fine edge is intra-PB ends up with an
/// EMPTY super-edge list, which the cache builder must treat as "this
/// (gene, batch) never samples" rather than erroring.
#[test]
fn fully_internal_gene_folds_to_empty() {
    let fine_to_super = vec![None, None];
    let activities = GeneActiveEdges {
        gene_active_edges: vec![vec![0u32, 1]],
        gene_active_edge_weights: vec![vec![1.0f32, 1.0]],
    };
    let folded = fold_active_edges_to_super(activities, &fine_to_super);
    assert!(folded.gene_active_edges[0].is_empty());
    assert!(folded.gene_active_edge_weights[0].is_empty());
}

use crate::cell_activity_graph_embedding::pb_frame::build_pb_frame;
use crate::util::graph_coarsen::MultiLevelCoarsenResult;
use matrix_util::knn_graph::KnnGraph;

/// 6 cells, nested 2-level coarsening, 5 fine edges of which 2 are
/// intra-PB. The frame must emit exactly the 3 PB-PB super edges, map
/// intra-PB fine edges to None, derive nested parent maps (finest =
/// identity), and assign pure per-PB batches.
fn toy_inputs() -> (MultiLevelCoarsenResult, KnnGraph, Vec<u32>) {
    // cells:   c0 c1 | c2 | c3 c4 | c5
    // finest:   0  0    1    2  2    3
    // coarse:   0  0    0    1  1    1
    let ml = MultiLevelCoarsenResult {
        all_pair_to_sample: vec![],
        all_num_samples: vec![],
        all_cell_labels: vec![vec![0, 0, 0, 1, 1, 1], vec![0, 0, 1, 2, 2, 3]],
    };
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let graph = KnnGraph {
        adjacency: nalgebra_sparse::CscMatrix::zeros(6, 6),
        distances: vec![1.0; edges.len()],
        n_nodes: 6,
        edges,
    };
    let batches = vec![0u32, 0, 0, 1, 1, 1];
    (ml, graph, batches)
}

#[test]
fn frame_folds_edges_and_nests_parents() {
    let (ml, graph, batches) = toy_inputs();
    let (frame, fine_to_super) = build_pb_frame(&ml, &graph, &batches, 2).unwrap();
    assert_eq!(frame.n_pb, 4);
    // Super edges (canonical min,max): pb0-pb1 and pb2-pb3. The fine
    // edge (c2, c3) crosses the batch boundary, so pb1-pb2 must NOT
    // exist: cross-batch fine edges neither form nor feed super edges.
    let mut se = frame.super_edges.clone();
    se.sort();
    assert_eq!(se, vec![(0u32, 1u32), (2, 3)]);
    // Intra-PB fine edges (0,1) and (3,4), and the cross-batch edge
    // (2,3), all fold into no super edge.
    assert!(fine_to_super[0].is_none());
    assert!(fine_to_super[2].is_none());
    assert!(fine_to_super[3].is_none());
    for e in [1usize, 4] {
        let s = fine_to_super[e].unwrap();
        assert!(s < frame.super_edges.len());
    }
    // Parent maps cover only levels COARSER than the finest: the coarse
    // level groups pbs {0,1} and {2,3}; the finest (identity) is omitted.
    assert_eq!(frame.pb_parent_maps.len(), 1);
    assert_eq!(frame.pb_parent_maps[0], vec![0, 0, 1, 1]);
    // Pure batches: pbs 0,1 in batch 0; pbs 2,3 in batch 1.
    assert_eq!(frame.pb_exp_batch, vec![0u32, 0, 1, 1]);
}

#[test]
fn frame_rejects_non_nested_levels() {
    let (mut ml, graph, batches) = toy_inputs();
    // Break nesting: cells 0 and 1 share finest pb 0 but disagree coarsely.
    ml.all_cell_labels[0] = vec![0, 1, 0, 1, 1, 1];
    let err = build_pb_frame(&ml, &graph, &batches, 2).unwrap_err();
    assert!(err.to_string().contains("do not nest"), "{err}");
}
