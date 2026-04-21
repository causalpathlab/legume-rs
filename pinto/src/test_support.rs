use crate::util::knn_graph::KnnGraph;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// Build a simple undirected `KnnGraph` from an edge list, with unit distances.
/// Used by tests across modules.
pub(crate) fn make_test_graph(n_nodes: usize, edges: Vec<(usize, usize)>) -> KnnGraph {
    let distances = vec![1.0; edges.len()];

    let mut coo = CooMatrix::new(n_nodes, n_nodes);
    for &(i, j) in &edges {
        coo.push(i, j, 1.0f32);
        coo.push(j, i, 1.0f32);
    }
    let adjacency = CscMatrix::from(&coo);

    KnnGraph {
        adjacency,
        edges,
        distances,
        n_nodes,
    }
}
