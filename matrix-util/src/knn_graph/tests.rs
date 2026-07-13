use super::*;

/// Two tight clusters of 5 points each in 2D, well separated
fn two_cluster_matrix() -> DMatrix<f32> {
    DMatrix::from_row_slice(
        10,
        2,
        &[
            // Cluster A near origin
            0.0, 0.0, //
            0.1, 0.0, //
            0.0, 0.1, //
            0.1, 0.1, //
            0.05, 0.05, //
            // Cluster B far away
            10.0, 10.0, //
            10.1, 10.0, //
            10.0, 10.1, //
            10.1, 10.1, //
            10.05, 10.05, //
        ],
    )
}

#[test]
fn test_knn_graph_construction() {
    let data = two_cluster_matrix();
    let graph = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 4,
            block_size: 100,
            reciprocal: true,
        },
    )
    .unwrap();

    // Basic properties
    assert_eq!(graph.num_nodes(), 10);
    assert!(graph.num_edges() > 0);
    assert_eq!(graph.edges.len(), graph.distances.len());

    // All edges should be (i < j)
    for &(i, j) in &graph.edges {
        assert!(i < j, "Edge ({}, {}) not canonical", i, j);
    }

    // All distances should be non-negative
    for &d in &graph.distances {
        assert!(d >= 0.0);
    }

    // Adjacency matrix dimensions
    assert_eq!(graph.adjacency.nrows(), 10);
    assert_eq!(graph.adjacency.ncols(), 10);

    // With k=4 and well-separated clusters, no edges should cross clusters
    for &(i, j) in &graph.edges {
        let same_cluster = (i < 5 && j < 5) || (i >= 5 && j >= 5);
        assert!(same_cluster, "Cross-cluster edge ({}, {}) found", i, j);
    }

    // Adjacency should be symmetric: if i is neighbor of j, j is neighbor of i
    for node in 0..graph.num_nodes() {
        for &neighbor in graph.neighbors(node) {
            let reverse_neighbors = graph.neighbors(neighbor);
            assert!(
                reverse_neighbors.contains(&node),
                "Node {} has neighbor {} but not vice versa",
                node,
                neighbor
            );
        }
    }
}

#[test]
fn test_from_columns_equivalent_to_from_rows() {
    let data = two_cluster_matrix();
    let transposed = data.transpose();

    let g_rows = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 3,
            block_size: 100,
            reciprocal: true,
        },
    )
    .unwrap();

    let g_cols = KnnGraph::from_columns(
        &transposed,
        KnnGraphArgs {
            knn: 3,
            block_size: 100,
            reciprocal: true,
        },
    )
    .unwrap();

    assert_eq!(g_rows.num_nodes(), g_cols.num_nodes());
    let diff = (g_rows.num_edges() as i64 - g_cols.num_edges() as i64).unsigned_abs();
    assert!(
        diff <= 2,
        "Edge counts differ: {} vs {}",
        g_rows.num_edges(),
        g_cols.num_edges()
    );
}

#[test]
fn test_exp_kernel_weights() {
    let data = two_cluster_matrix();
    let graph = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 4,
            block_size: 100,
            reciprocal: true,
        },
    )
    .unwrap();

    let weights = graph.exp_kernel_weights();
    assert_eq!(weights.len(), graph.num_edges());

    // All weights should be in (0, 1]
    for &w in &weights {
        assert!(w > 0.0, "Weight {} should be > 0", w);
        assert!(w <= 1.0, "Weight {} should be <= 1", w);
    }

    // Median edge gets exp(-1) ≈ 0.37; closer edges get higher weights
    let mean_w: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
    assert!(
        mean_w > 0.2 && mean_w < 0.9,
        "Mean weight {} should be in a reasonable range",
        mean_w
    );
}

#[test]
fn test_fuzzy_kernel_weights() {
    let data = two_cluster_matrix();
    let graph = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 4,
            block_size: 100,
            reciprocal: false, // union, like scanpy default
        },
    )
    .unwrap();

    let weights = graph.fuzzy_kernel_weights();
    assert_eq!(weights.len(), graph.num_edges());

    // All weights should be in (0, 1]
    for &w in &weights {
        assert!(w > 0.0, "Weight {} should be > 0", w);
        assert!(w <= 1.0, "Weight {} should be <= 1", w);
    }

    // With UMAP weights, no edge should be near zero (local sigma adapts)
    let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        min_w > 0.01,
        "Min fuzzy weight {} is too small; local sigma should prevent near-zero weights",
        min_w
    );
}

#[test]
fn test_smooth_knn_sigma() {
    // 5 distances, rho = 0.1 (nearest neighbor)
    let dists = [0.1, 0.2, 0.3, 0.5, 1.0];
    let rho = 0.1;
    let target = (5.0f32).log2(); // log2(k)

    let sigma = super::smooth_knn_sigma(&dists, rho, target);
    assert!(sigma > 0.0, "sigma should be positive");

    // Verify the sigma achieves the target
    let psum: f32 = dists
        .iter()
        .map(|&d| {
            let gap = d - rho;
            if gap > 0.0 {
                (-gap / sigma).exp()
            } else {
                1.0
            }
        })
        .sum();

    assert!(
        (psum - target).abs() < 0.1,
        "psum {:.3} should be close to target {:.3}",
        psum,
        target
    );
}

#[test]
fn test_median() {
    assert_eq!(crate::utils::median(&[1.0, 3.0, 2.0]), 2.0);
    assert_eq!(crate::utils::median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    assert_eq!(crate::utils::median(&[5.0]), 5.0);
}

#[test]
fn test_create_jobs_helper() {
    let jobs = create_jobs(10, 3);
    assert_eq!(jobs, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);

    let jobs = create_jobs(6, 3);
    assert_eq!(jobs, vec![(0, 3), (3, 6)]);

    let jobs = create_jobs(1, 100);
    assert_eq!(jobs, vec![(0, 1)]);

    // block_size=0 should fall back to DEFAULT_BLOCK_SIZE
    let jobs = create_jobs(5, 0);
    assert_eq!(jobs, vec![(0, 5)]);
}
