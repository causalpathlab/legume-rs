//! Choosing K and placing the k-means centroids the graph is built over.

use matrix_util::dmatrix_io::DMatrix;

use super::args::*;
use super::layout::*;

/// Number of MST node centroids K: explicit `--n-centroids`, else `min(N/10, 200)`,
/// clamped to `[2, N]`.
pub(super) fn choose_k(n: usize, requested: Option<usize>) -> usize {
    requested.unwrap_or_else(|| (n / 10).clamp(2, 200)).min(n)
}

/// Feature matrix the annotation k-means groups on, per `--cluster-space`. `identity` = raw θ;
/// `nascent` = θ+δ; `concat` = `[θ̂ | δ̂]` with each channel L2-normalised per row so identity and
/// velocity contribute equally to the Euclidean k-means. Falls back to θ when velocity is absent.
pub(super) fn cluster_features(
    theta: &DMatrix<f32>,
    velocity: Option<&DMatrix<f32>>,
    space: LayoutSpace,
) -> DMatrix<f32> {
    match (space, velocity) {
        (LayoutSpace::Identity, _) | (_, None) => theta.clone(),
        (LayoutSpace::Nascent, Some(v)) => {
            let mut f = theta.clone();
            f += v;
            f
        }
        (LayoutSpace::Concat, Some(v)) => {
            let (tn, vn) = (l2_normalize_rows(theta), l2_normalize_rows(v));
            let (n, h) = (theta.nrows(), theta.ncols());
            let mut f = DMatrix::<f32>::zeros(n, 2 * h);
            f.view_mut((0, 0), (n, h)).copy_from(&tn);
            f.view_mut((0, h), (n, h)).copy_from(&vn);
            f
        }
    }
}

/// Recompute the `k` trajectory centroids in RAW θ from the grouping labels (mean θ of each
/// cluster's cells), so the manifold geometry (MST, layout, marker scoring) stays θ-based even
/// when the GROUPING used θ+δ or `[θ|δ]`. For the `identity` default this equals the θ k-means
/// centroids exactly.
pub(super) fn theta_centroids_from_labels(
    theta: &DMatrix<f32>,
    labels: &[usize],
    k: usize,
) -> DMatrix<f32> {
    let h = theta.ncols();
    let mut c = DMatrix::<f32>::zeros(k, h);
    let mut cnt = vec![0f32; k];
    for (i, &l) in labels.iter().enumerate() {
        cnt[l] += 1.0;
        for j in 0..h {
            c[(l, j)] += theta[(i, j)];
        }
    }
    for l in 0..k {
        if cnt[l] > 0.0 {
            for j in 0..h {
                c[(l, j)] /= cnt[l];
            }
        }
    }
    c
}

#[cfg(test)]
#[path = "cluster_tests.rs"]
mod cluster_tests;
