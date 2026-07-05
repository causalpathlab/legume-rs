use super::*;
use nalgebra::DMatrix;

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
