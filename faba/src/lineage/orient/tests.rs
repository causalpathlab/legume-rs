use super::*;
use nalgebra::DMatrix;

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

/// Build a chain of `k` nodes at x = 0,1,…, with `per` cells each carrying velocity `vel`.
fn chain_data(k: usize, per: usize, vel: f32) -> (DMatrix<f32>, DMatrix<f32>, Vec<usize>) {
    let mut centroids = DMatrix::<f32>::zeros(k, 1);
    for i in 0..k {
        centroids[(i, 0)] = i as f32;
    }
    let n = k * per;
    let mut velocity = DMatrix::<f32>::zeros(n, 1);
    let mut labels = vec![0usize; n];
    for i in 0..n {
        labels[i] = i / per;
        velocity[(i, 0)] = vel;
    }
    (centroids, velocity, labels)
}

fn cfg() -> EdgeDirectionConfig {
    EdgeDirectionConfig {
        n_boot: 200,
        n_perm: 500,
        alpha: 0.05,
        min_cells: 2,
        seed: 42,
    }
}

#[test]
fn candidate_edges_extend_the_mst() {
    let (centroids, _, _) = chain_data(4, 1, 1.0);
    let mst = vec![(0, 1), (1, 2), (2, 3)];
    let cand = candidate_edges(&centroids, &mst, 2);
    for e in &mst {
        assert!(cand.contains(e), "MST edge {e:?} must be a candidate");
    }
    // k_cand=2 pulls in the second-nearest neighbour too, e.g. (0,2).
    assert!(
        cand.len() > mst.len(),
        "candidate set extends beyond the MST"
    );
    assert!(cand.contains(&(0, 2)));
}

#[test]
fn strong_uniform_flow_calls_every_edge_forward() {
    let (centroids, velocity, labels) = chain_data(4, 30, 1.0);
    let mst = vec![(0, 1), (1, 2), (2, 3)];
    let cand = candidate_edges(&centroids, &mst, 1);
    let dirs = edge_directionality(&centroids, &velocity, &labels, &cand, &mst, &cfg());
    for d in &dirs {
        assert_eq!(
            d.call,
            EdgeCall::Forward,
            "edge {:?} should flow forward (flux {})",
            d.edge,
            d.flux
        );
        assert!(d.q < 0.05 && d.confidence > 0.9);
    }
}

#[test]
fn zero_mean_velocity_abstains() {
    // Alternating ±1 velocity → mean projection ≈ 0 → sign-flip null cannot reject.
    let (centroids, mut velocity, labels) = chain_data(4, 30, 1.0);
    for i in 0..velocity.nrows() {
        velocity[(i, 0)] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let mst = vec![(0, 1), (1, 2), (2, 3)];
    let cand = candidate_edges(&centroids, &mst, 1);
    let dirs = edge_directionality(&centroids, &velocity, &labels, &cand, &mst, &cfg());
    for d in &dirs {
        assert_eq!(
            d.call,
            EdgeCall::Abstain,
            "edge {:?} should abstain",
            d.edge
        );
    }
}

#[test]
fn too_few_cells_auto_abstains_with_nan_stats() {
    let (centroids, velocity, labels) = chain_data(3, 1, 1.0); // 1 cell/node
    let mst = vec![(0, 1), (1, 2)];
    let cand = candidate_edges(&centroids, &mst, 1);
    let dirs = edge_directionality(&centroids, &velocity, &labels, &cand, &mst, &cfg());
    // Each edge sees 2 cells (one per endpoint) — that meets min_cells=2, so lower it.
    let strict = EdgeDirectionConfig {
        min_cells: 3,
        ..cfg()
    };
    let dirs2 = edge_directionality(&centroids, &velocity, &labels, &cand, &mst, &strict);
    assert!(dirs2
        .iter()
        .all(|d| d.call == EdgeCall::Abstain && d.q.is_nan()));
    let _ = dirs;
}
