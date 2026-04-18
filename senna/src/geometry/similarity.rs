//! PB-PB similarity construction and the small family of pre-layout
//! transforms that operate on it (thresholding, local scaling, diagonal
//! regularization).

use crate::embed_common::*;
use rayon::prelude::*;

/// f32 dot product on two equal-length slices. `zip().map().sum()` is the
/// form LLVM consistently lowers to SSE/AVX f32 SIMD reductions.
#[inline]
fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine similarity between columns of `x_dp`. Columns are L2-normalized
/// (with a tiny epsilon floor on zero-norm columns) and then the gram
/// matrix is formed: `S = X̂ᵀ X̂`. The O(n² · D) gram is built
/// column-parallel for large D, with SIMD-friendly per-column slices.
pub(crate) fn compute_cosine_similarity(x_dp: &Mat) -> Mat {
    let n = x_dp.ncols();

    let norms: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|j| {
            let col = x_dp.column(j);
            let norm = col.norm();
            let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
            col.iter().map(|v| v * inv).collect()
        })
        .collect();

    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = &norms[i];
            (0..n).map(|j| dot_slice(xi, &norms[j])).collect()
        })
        .collect();

    let mut s = Mat::zeros(n, n);
    for (i, col) in cols.iter().enumerate() {
        s.column_mut(i).copy_from_slice(col);
    }
    s
}

/// Zero out entries below `threshold`.
pub(crate) fn threshold_similarity(s: &Mat, threshold: f32) -> Mat {
    let mut result = s.clone();
    result.as_mut_slice().par_iter_mut().for_each(|v| {
        if *v < threshold {
            *v = 0.0;
        }
    });
    result
}

/// Zelnik-Manor & Perona (2004) local scaling: `S_scaled(i,j) = S(i,j) /
/// √(σ_i σ_j)` where `σ_i = 1 − S(i, k-th most similar neighbor)`. Compresses
/// dense regions and spreads sparse ones.
pub(crate) fn local_scale_similarity(s: &Mat, k: usize) -> Mat {
    let n = s.nrows();
    let k = k.min(n - 1).max(1);

    // σ_i = 1 − (k-th highest similarity to i). `select_nth_unstable_by`
    // partitions in O(n) without a full sort.
    let sigma: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut sims: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| s[(i, j)]).collect();
            let idx = (k - 1).min(sims.len() - 1);
            sims.select_nth_unstable_by(idx, |a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            (1.0 - sims[idx]).max(1e-6)
        })
        .collect();

    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|j| {
            (0..n)
                .map(|i| {
                    if i == j {
                        1.0
                    } else {
                        let scale = (sigma[i] * sigma[j]).sqrt();
                        s[(i, j)] / scale
                    }
                })
                .collect()
        })
        .collect();

    let mut out = Mat::zeros(n, n);
    for (j, col) in cols.iter().enumerate() {
        out.column_mut(j).copy_from_slice(col);
    }
    out
}

/// Add `eps` to the diagonal so isolated nodes can't collapse downstream
/// Laplacian / MDS computations. Logs a warning if any node still has very
/// low total similarity after regularization.
pub(crate) fn regularize_similarity(similarity: &Mat, eps: f32) -> Mat {
    let n = similarity.nrows();
    let mut sim_reg = similarity.clone();
    for i in 0..n {
        sim_reg[(i, i)] += eps;
    }

    let low_degree_count = (0..n)
        .filter(|&i| sim_reg.row(i).iter().sum::<f32>() < eps * 2.0)
        .count();
    if low_degree_count > 0 {
        info!("Warning: {low_degree_count} PB samples have very low similarity to others.");
    }
    sim_reg
}
