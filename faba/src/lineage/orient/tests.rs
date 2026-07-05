use super::*;
use nalgebra::DMatrix;

/// A linear chain 0—1—2—3 with velocity pointing 0→3 should pick node 0 as the
/// root and orient every edge forward.
#[test]
fn linear_chain_roots_at_source() {
    let k = 4;
    let mut centroids = DMatrix::<f32>::zeros(k, 1);
    for i in 0..k {
        centroids[(i, 0)] = i as f32;
    }
    // Uniform velocity in +x at every node.
    let node_velocity = DMatrix::<f32>::from_element(k, 1, 1.0);
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let flux = edge_velocity_flux(&centroids, &node_velocity, &edges);
    assert!(flux.iter().all(|&f| f > 0.0), "all edges flow forward");

    let root = pick_velocity_root(&edges, &flux, k);
    assert_eq!(root, 0, "source node is the root");

    let dir = directed_edges(&edges, &flux);
    assert_eq!(dir, vec![(0, 1), (1, 2), (2, 3)]);
}

/// Reversing the velocity field flips the root to the other end and reverses
/// every directed edge.
#[test]
fn reversed_velocity_flips_root() {
    let k = 4;
    let mut centroids = DMatrix::<f32>::zeros(k, 1);
    for i in 0..k {
        centroids[(i, 0)] = i as f32;
    }
    let node_velocity = DMatrix::<f32>::from_element(k, 1, -1.0);
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let flux = edge_velocity_flux(&centroids, &node_velocity, &edges);
    let root = pick_velocity_root(&edges, &flux, k);
    assert_eq!(root, 3, "reversed source is the far node");
    let dir = directed_edges(&edges, &flux);
    assert_eq!(dir, vec![(1, 0), (2, 1), (3, 2)]);
}

#[test]
fn node_velocity_is_cluster_mean() {
    // 2 nodes, 4 cells: cells 0,1 → node 0; cells 2,3 → node 1.
    let mut vel = DMatrix::<f32>::zeros(4, 1);
    vel[(0, 0)] = 1.0;
    vel[(1, 0)] = 3.0;
    vel[(2, 0)] = -2.0;
    vel[(3, 0)] = -4.0;
    let cluster = vec![0, 0, 1, 1];
    let v = aggregate_node_velocity(&vel, &cluster, 2);
    assert_eq!(v[(0, 0)], 2.0);
    assert_eq!(v[(1, 0)], -3.0);
}
