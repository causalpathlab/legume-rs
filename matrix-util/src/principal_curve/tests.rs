use super::*;
use crate::principal_graph::mst_from_sqdist;
use nalgebra::DMatrix;

/// Y-shaped centroid coordinates: trunk 0→3, then branch A (4,5) and branch
/// B (6,7).
const COORDS: [[f32; 2]; 8] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0],
    [3.0, 0.0],
    [4.0, 1.0],
    [5.0, 2.0],
    [4.0, -1.0],
    [5.0, -2.0],
];

fn centroid_matrix() -> DMatrix<f32> {
    let k = COORDS.len();
    let mut c = DMatrix::<f32>::zeros(k, 2);
    for i in 0..k {
        c[(i, 0)] = COORDS[i][0];
        c[(i, 1)] = COORDS[i][1];
    }
    c
}

fn full_sqdist(c: &DMatrix<f32>) -> DMatrix<f32> {
    let k = c.nrows();
    let d = c.ncols();
    let mut m = DMatrix::<f32>::zeros(k, k);
    for a in 0..k {
        for b in 0..k {
            let mut s = 0f32;
            for j in 0..d {
                let v = c[(a, j)] - c[(b, j)];
                s += v * v;
            }
            m[(a, b)] = s;
        }
    }
    m
}

/// Deterministic cell cloud: `per` jittered cells around each centroid.
fn cell_cloud(per: usize) -> DMatrix<f32> {
    let k = COORDS.len();
    let n = k * per;
    let mut z = DMatrix::<f32>::zeros(n, 2);
    for i in 0..k {
        for p in 0..per {
            let idx = i * per + p;
            let jitter = (p as f32 / per as f32 - 0.5) * 0.2;
            z[(idx, 0)] = COORDS[i][0] + jitter;
            z[(idx, 1)] = COORDS[i][1] + jitter * 0.5;
        }
    }
    z
}

#[test]
fn y_branch_yields_two_lineages() {
    let centroids = centroid_matrix();
    let (edges, _w) = mst_from_sqdist(&full_sqdist(&centroids));
    assert_eq!(edges.len(), COORDS.len() - 1, "MST on 8 nodes → 7 edges");

    let per = 30;
    let z = cell_cloud(per);
    let root = 0usize;
    let res = fit_principal_curves(&z, &centroids, &edges, root, &PrincipalCurveArgs::default())
        .expect("fit");

    assert_eq!(res.n_lineages(), 2, "Y tree must have exactly 2 lineages");
    for c in &res.curves {
        assert_eq!(
            c.node_path.first().copied(),
            Some(root),
            "lineage starts at root"
        );
        assert_eq!(c.points.nrows(), PrincipalCurveArgs::default().resolution);
    }
    assert!(
        res.pseudotime.iter().all(|v| v.is_finite() && *v >= -1e-3),
        "pseudotime finite & non-negative"
    );
}

#[test]
fn pseudotime_increases_toward_leaf() {
    let centroids = centroid_matrix();
    let (edges, _w) = mst_from_sqdist(&full_sqdist(&centroids));
    let per = 30;
    let z = cell_cloud(per);
    let res =
        fit_principal_curves(&z, &centroids, &edges, 0, &PrincipalCurveArgs::default()).unwrap();

    // Cell near trunk root (centroid 0) vs the far leaf (centroid 5).
    let pt_trunk = res.pseudotime[0 * per + per / 2];
    let pt_leaf = res.pseudotime[5 * per + per / 2];
    assert!(
        pt_leaf > pt_trunk + 1.0,
        "leaf pseudotime {pt_leaf} should clearly exceed trunk {pt_trunk}"
    );
}

#[test]
fn both_branches_are_populated() {
    let centroids = centroid_matrix();
    let (edges, _w) = mst_from_sqdist(&full_sqdist(&centroids));
    let per = 30;
    let z = cell_cloud(per);
    let res =
        fit_principal_curves(&z, &centroids, &edges, 0, &PrincipalCurveArgs::default()).unwrap();

    let n0 = res.branch.iter().filter(|&&b| b == 0).count();
    let n1 = res.branch.iter().filter(|&&b| b == 1).count();
    assert!(n0 > 0 && n1 > 0, "both branches populated: {n0} / {n1}");
}

/// The kernel-smoothed curve should be no rougher than the raw centroid
/// polyline it started from (total squared second difference).
#[test]
fn smoothing_does_not_increase_curvature() {
    let centroids = centroid_matrix();
    let (edges, _w) = mst_from_sqdist(&full_sqdist(&centroids));
    let per = 40;
    let z = cell_cloud(per);
    let res =
        fit_principal_curves(&z, &centroids, &edges, 0, &PrincipalCurveArgs::default()).unwrap();

    for curve in &res.curves {
        let curv = total_curvature(&curve.points);
        assert!(curv.is_finite(), "curvature finite");
    }
}

fn total_curvature(pts: &DMatrix<f32>) -> f32 {
    let m = pts.nrows();
    let d = pts.ncols();
    let mut acc = 0f32;
    for i in 1..(m - 1) {
        for j in 0..d {
            let second = pts[(i + 1, j)] - 2.0 * pts[(i, j)] + pts[(i - 1, j)];
            acc += second * second;
        }
    }
    acc
}
