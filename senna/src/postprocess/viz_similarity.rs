//! PB-PB similarity construction and the small family of pre-layout
//! transforms that operate on it (thresholding, local scaling, diagonal
//! regularization).

use crate::embed_common::*;

/// Cosine similarity between columns of `x_dp`. Columns are L2-normalized
/// (with a tiny epsilon floor on zero-norm columns) and then the gram
/// matrix is formed: `S = X̂ᵀ X̂`.
pub(super) fn compute_cosine_similarity(x_dp: &Mat) -> Mat {
    let n = x_dp.ncols();
    let mut x_norm = x_dp.clone();
    for j in 0..n {
        let norm = x_norm.column(j).norm();
        if norm > 1e-10 {
            x_norm.column_mut(j).scale_mut(1.0 / norm);
        }
    }
    x_norm.transpose() * &x_norm
}

/// Zero out entries below `threshold`.
pub(super) fn threshold_similarity(s: &Mat, threshold: f32) -> Mat {
    let mut result = s.clone();
    for val in result.iter_mut() {
        if *val < threshold {
            *val = 0.0;
        }
    }
    result
}

/// Zelnik-Manor & Perona (2004) local scaling: `S_scaled(i,j) = S(i,j) /
/// √(σ_i σ_j)` where `σ_i = 1 − S(i, k-th most similar neighbor)`. Compresses
/// dense regions and spreads sparse ones.
pub(super) fn local_scale_similarity(s: &Mat, k: usize) -> Mat {
    let n = s.nrows();
    let k = k.min(n - 1).max(1);

    let mut sigma = vec![0.0f32; n];
    for i in 0..n {
        let mut sims: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| s[(i, j)]).collect();
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let kth_sim = if k <= sims.len() {
            sims[k - 1]
        } else {
            sims.last().copied().unwrap_or(0.0)
        };
        sigma[i] = (1.0 - kth_sim).max(1e-6);
    }

    let mut result = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                result[(i, j)] = 1.0;
            } else {
                let scale = (sigma[i] * sigma[j]).sqrt();
                result[(i, j)] = s[(i, j)] / scale;
            }
        }
    }
    result
}

/// Add `eps` to the diagonal so isolated nodes can't collapse downstream
/// Laplacian / MDS computations. Logs a warning if any node still has very
/// low total similarity after regularization.
pub(super) fn regularize_similarity(similarity: &Mat, eps: f32) -> Mat {
    let n = similarity.nrows();
    let mut sim_reg = similarity.clone();
    for i in 0..n {
        sim_reg[(i, i)] += eps;
    }

    let low_degree_count = (0..n)
        .filter(|&i| sim_reg.row(i).iter().sum::<f32>() < eps * 2.0)
        .count();
    if low_degree_count > 0 {
        info!(
            "Warning: {} PB samples have very low similarity to others.",
            low_degree_count
        );
    }
    sim_reg
}
