use super::*;
use nalgebra::DMatrix;

/// Three well-separated Gaussian-ish blobs in 2-D (deterministic point layout).
fn three_blobs() -> DMatrix<f32> {
    let centers = [(0.0f32, 0.0f32), (10.0, 0.0), (5.0, 10.0)];
    let per = 40usize;
    let mut z = DMatrix::<f32>::zeros(centers.len() * per, 2);
    let mut r = 0usize;
    for (cx, cy) in centers {
        for i in 0..per {
            // Small deterministic jitter so points are distinct but blobs stay separate.
            z[(r, 0)] = cx + 0.1 * (i as f32 * 0.7).sin();
            z[(r, 1)] = cy + 0.1 * (i as f32 * 1.3).cos();
            r += 1;
        }
    }
    z
}

#[test]
fn kmeans_seeded_is_reproducible() {
    let z = three_blobs();
    let (c1, l1) = kmeans_centroids_seeded(&z, 3, 50, 42);
    let (c2, l2) = kmeans_centroids_seeded(&z, 3, 50, 42);
    // Same seed → byte-identical centroids and labels.
    assert_eq!(l1, l2, "labels must be identical for the same seed");
    assert_eq!(c1.nrows(), 3);
    for i in 0..c1.nrows() {
        for j in 0..c1.ncols() {
            assert_eq!(c1[(i, j)], c2[(i, j)], "centroid ({i},{j}) differs at same seed");
        }
    }
}

#[test]
fn kmeans_seeded_recovers_separated_blobs() {
    let z = three_blobs();
    let (_c, labels) = kmeans_centroids_seeded(&z, 3, 50, 7);
    // Each 40-point blimp block should be internally consistent (one label per blob).
    for blob in 0..3 {
        let block = &labels[blob * 40..(blob + 1) * 40];
        let first = block[0];
        assert!(
            block.iter().all(|&l| l == first),
            "blob {blob} split across clusters: {block:?}"
        );
    }
    // …and the three blobs must land in three distinct clusters.
    let mut used: Vec<usize> = vec![labels[0], labels[40], labels[80]];
    used.sort_unstable();
    used.dedup();
    assert_eq!(used.len(), 3, "the three blobs must map to three distinct clusters");
}

#[test]
fn kmeans_seeded_handles_degenerate_k() {
    let z = three_blobs();
    // k = 1 → a single centroid (the column mean) and all-zero labels.
    let (c, labels) = kmeans_centroids_seeded(&z, 1, 50, 1);
    assert_eq!(c.nrows(), 1);
    assert!(labels.iter().all(|&l| l == 0));
}

#[test]
fn fits_a_line() {
    // 200 cells on a noisy 1-D line in 3-D space.
    let n = 200;
    let mut z = DMatrix::<f32>::zeros(n, 3);
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        z[(i, 0)] = t * 10.0;
        z[(i, 1)] = 0.05 * (i as f32 * 0.3).sin();
        z[(i, 2)] = 0.05 * (i as f32 * 0.7).cos();
    }
    let args = PrincipalGraphArgs {
        n_centroids: 20,
        gamma: 5.0,
        sigma: -1.0,
        max_iter: 30,
        tol: 1e-5,
        kmeans_max_iter: 100,
    };
    let g = fit_principal_graph(&z, &args).unwrap();
    assert_eq!(g.n_edges(), 19, "MST on 20 nodes must have 19 edges");

    let projs = project_cells_to_graph(&z, &g);
    // Use the node closest to the first cell as root.
    let root = closest_node_to_row(&z, 0, &g);
    let pt = pseudotime_from_root(&g, &projs, root);
    // pseudotime should be monotonic (within tolerance) along the line.
    let mut violations = 0;
    for i in 1..n {
        if pt[i] + 0.5 < pt[i - 1] {
            violations += 1;
        }
    }
    assert!(
        violations < n / 20,
        "too many monotonicity violations: {violations}"
    );
}
