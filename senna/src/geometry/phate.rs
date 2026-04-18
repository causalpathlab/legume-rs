//! PHATE — Potential of Heat-diffusion for Affinity-based Transition Embedding.
//! Moon et al., Nat Biotechnol 2019 (PMC7073148).
//!
//! Given an n×K matrix of features `pb_z` (we use the whitened reconstruction-
//! space representation), PHATE produces an n×2 embedding that preserves both
//! local neighborhoods and global trajectory/branching structure:
//!
//! 1. Kernel affinity K with adaptive bandwidth σᵢ = distance to the `knn`-th
//!    neighbor, and alpha-decay exponent α (sharper than a Gaussian):
//!    `K[i,j] = ½ · (exp(-(d_ij/σᵢ)^α) + exp(-(d_ij/σⱼ)^α))`.
//! 2. Diffusion operator P = row-normalize(K).
//! 3. Diffusion in time: M = P^t (smooths noise, preserves global geometry).
//! 4. Potential distance: `U[i,j] = ‖-log M[i,:] − -log M[j,:]‖₂`.
//! 5. Classical MDS on U → 2D coordinates (eigendecomposition of the
//!    double-centered -½U² gram matrix, top-2 eigenvectors scaled by √λ).
//!
//! The whole thing is O(n²K + n² log t + n³) for the MDS, which is fast for
//! the n ≲ 10³ PB counts we actually see.

use crate::embed_common::Mat;
use crate::logging::new_progress_bar;
use log::info;
use rayon::prelude::*;

/// Build an `n × n` matrix by computing each column in parallel, then
/// copying the collected column vectors into a column-major `Mat` (one
/// `column_mut().copy_from_slice` per column — no inner strided writes).
fn build_nn_matrix<F>(n: usize, build_col: F) -> Mat
where
    F: Fn(usize) -> Vec<f32> + Sync + Send,
{
    let cols: Vec<Vec<f32>> = (0..n).into_par_iter().map(build_col).collect();
    let mut m = Mat::zeros(n, n);
    for (j, col) in cols.iter().enumerate() {
        m.column_mut(j).copy_from_slice(col);
    }
    m
}

/// Materialize each row of a column-major `(n × d)` matrix into its own
/// contiguous `Vec<f32>`. Rows of column-major storage are strided by
/// `n`, so iterating them directly misses SIMD; copying once lets every
/// downstream inner loop consume a contiguous slice.
fn rows_as_contiguous(m: &Mat) -> Vec<Vec<f32>> {
    let n = m.nrows();
    let d = m.ncols();
    (0..n)
        .into_par_iter()
        .map(|i| (0..d).map(|j| m[(i, j)]).collect())
        .collect()
}

/// SIMD-friendly squared L2 distance between two equal-length slices.
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

pub struct PhateArgs {
    pub t: usize,
    pub knn: usize,
    pub alpha: f32,
    /// Max SMACOF (metric MDS) iterations.
    pub mds_iter: usize,
    /// Relative stress-change tolerance for early SMACOF exit.
    pub mds_tol: f32,
}

impl Default for PhateArgs {
    fn default() -> Self {
        Self {
            t: 20,
            knn: 5,
            alpha: 10.0,
            mds_iter: 300,
            mds_tol: 1e-4,
        }
    }
}

/// Compute a 2D PHATE embedding of `pb_z` (n × K).
pub fn phate_layout_2d(pb_z: &Mat, args: &PhateArgs) -> Mat {
    let n = pb_z.nrows();
    if n < 3 {
        return Mat::zeros(n, 2);
    }
    let knn = args.knn.clamp(1, n - 1);
    info!(
        "PHATE start: n={} PBs, features={}, t={}, knn={}, α={}",
        n,
        pb_z.ncols(),
        args.t,
        knn,
        args.alpha
    );

    info!("PHATE 1/6: pairwise distances");
    // Materialize rows once so every inner distance reads a contiguous
    // slice; column-major `row()` views are strided and miss SIMD.
    let rows = rows_as_contiguous(pb_z);
    let dist = build_nn_matrix(n, |j| {
        let rj = &rows[j];
        (0..n)
            .map(|i| {
                if i == j {
                    0.0
                } else {
                    sq_dist(&rows[i], rj).sqrt()
                }
            })
            .collect()
    });

    info!("PHATE 2/6: adaptive bandwidth σᵢ (knn={knn})");
    let sigma: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| dist[(i, j)]).collect();
            row.select_nth_unstable_by(knn - 1, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            row[knn - 1].max(1e-6)
        })
        .collect();

    info!("PHATE 3/6: alpha-decay kernel + row-normalize to diffusion operator");
    let k_mat = build_nn_matrix(n, |j| {
        let mut col = vec![0.0f32; n];
        col[j] = 1.0;
        for i in 0..n {
            if i == j {
                continue;
            }
            let d = dist[(i, j)];
            let e1 = (-(d / sigma[i]).powf(args.alpha)).exp();
            let e2 = (-(d / sigma[j]).powf(args.alpha)).exp();
            col[i] = 0.5 * (e1 + e2);
        }
        col
    });
    let mut p_mat = k_mat;
    for i in 0..n {
        let s: f32 = p_mat.row(i).iter().sum::<f32>().max(1e-12);
        for j in 0..n {
            p_mat[(i, j)] /= s;
        }
    }

    info!("PHATE 4/6: diffusion M = P^{}", args.t.max(1));
    let m_mat = matrix_power(&p_mat, args.t.max(1));

    info!("PHATE 5/6: potential distance + classical MDS init");
    let log_m = build_nn_matrix(n, |j| {
        (0..n).map(|i| -(m_mat[(i, j)].max(1e-12)).ln()).collect()
    });
    let log_m_rows = rows_as_contiguous(&log_m);
    let pot_d2 = build_nn_matrix(n, |j| {
        let rj = &log_m_rows[j];
        (0..n)
            .map(|i| {
                if i == j {
                    0.0
                } else {
                    sq_dist(&log_m_rows[i], rj)
                }
            })
            .collect()
    });
    let y_init = classical_mds_2d(&pot_d2);

    let delta = build_nn_matrix(n, |j| (0..n).map(|i| pot_d2[(i, j)].sqrt()).collect());
    info!(
        "PHATE 6/6: SMACOF metric MDS (max_iter={}, tol={:.0e})",
        args.mds_iter, args.mds_tol
    );
    smacof_2d(&delta, &y_init, args.mds_iter, args.mds_tol)
}

/// Metric MDS via SMACOF (Scaling by `MAjorizing` a `COmplicated` Function).
///
/// Minimizes the stress
///     σ(Y) = Σ_{i<j} (`δ_ij` − `d_ij(Y))²`
/// over Y ∈ ℝ^{n × 2} via the Guttman transform:
///     Y_{k+1} = (1/n) · `B(Y_k)` · `Y_k`,
/// where B[i,j] = −δ[i,j] / `d_ij(Y_k)` for i ≠ j (and zero if `d_ij(Y_k)` is
/// numerically zero), B[i,i] = −Σ_{j ≠ i} B[i,j]. Each iteration strictly
/// decreases stress (majorization), so no learning rate or line search is
/// needed. Initialized from classical MDS so the basin is sensible.
fn smacof_2d(delta: &Mat, y_init: &Mat, max_iter: usize, tol: f32) -> Mat {
    let n = delta.nrows();
    let mut y = y_init.clone();
    let mut prev_stress = f32::INFINITY;
    let inv_n = 1.0 / n as f32;

    let pb = new_progress_bar(max_iter as u64);
    pb.set_message("SMACOF");
    for _ in 0..max_iter {
        pb.inc(1);
        // Pairwise distances in the current 2D configuration (column-parallel).
        let dy = build_nn_matrix(n, |j| {
            let (yj0, yj1) = (y[(j, 0)], y[(j, 1)]);
            (0..n)
                .map(|i| {
                    if i == j {
                        0.0
                    } else {
                        let dx = y[(i, 0)] - yj0;
                        let dz = y[(i, 1)] - yj1;
                        (dx * dx + dz * dz).sqrt()
                    }
                })
                .collect()
        });

        let stress: f32 = (0..n)
            .into_par_iter()
            .map(|i| {
                (i + 1..n)
                    .map(|j| {
                        let diff = delta[(i, j)] - dy[(i, j)];
                        diff * diff
                    })
                    .sum::<f32>()
            })
            .sum();

        let denom = prev_stress.max(1.0);
        if (prev_stress - stress).abs() / denom < tol {
            pb.set_message(format!("SMACOF converged (stress={stress:.3e})"));
            break;
        }
        prev_stress = stress;
        pb.set_message(format!("SMACOF stress={stress:.3e}"));

        // B(Y_k) column-parallel. Diagonal = -Σ off-diagonal in the column
        // (B is symmetric, so column and row sums match).
        let b_mat = build_nn_matrix(n, |j| {
            let mut col = vec![0.0f32; n];
            let mut diag = 0.0f32;
            for i in 0..n {
                if i == j {
                    continue;
                }
                let d_ij = dy[(i, j)];
                let b = if d_ij > 1e-10 {
                    -delta[(i, j)] / d_ij
                } else {
                    0.0
                };
                col[i] = b;
                diag -= b;
            }
            col[j] = diag;
            col
        });

        // Guttman update: Y ← (1/n) · B · Y.
        let mut y_new = &b_mat * &y;
        for v in y_new.iter_mut() {
            *v *= inv_n;
        }
        y = y_new;
    }
    pb.finish_and_clear();

    y
}

/// Repeated-squaring power of a square matrix (exponent in units of one
/// multiplication; result = M^exp).
fn matrix_power(m: &Mat, exp: usize) -> Mat {
    let n = m.nrows();
    let mut result: Mat = Mat::identity(n, n);
    let mut base = m.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = &result * &base;
        }
        e >>= 1;
        if e > 0 {
            base = &base * &base;
        }
    }
    result
}

/// Classical MDS from a squared-distance matrix D² to a 2D embedding.
fn classical_mds_2d(d2: &Mat) -> Mat {
    let n = d2.nrows();

    // Row / column / grand means of D².
    let row_means: Vec<f32> = (0..n)
        .map(|i| d2.row(i).iter().sum::<f32>() / n as f32)
        .collect();
    let col_means: Vec<f32> = (0..n)
        .map(|j| d2.column(j).iter().sum::<f32>() / n as f32)
        .collect();
    let grand_mean: f32 = row_means.iter().sum::<f32>() / n as f32;

    // Gram matrix B = -½ · (D² - row_mean - col_mean + grand_mean).
    let mut b = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] = -0.5 * (d2[(i, j)] - row_means[i] - col_means[j] + grand_mean);
        }
    }

    // Symmetrize (defensive against floating-point drift).
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (b[(i, j)] + b[(j, i)]);
            b[(i, j)] = v;
            b[(j, i)] = v;
        }
    }

    let eig = b.symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b_| {
        eig.eigenvalues[b_]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut coords = Mat::zeros(n, 2);
    for dim in 0..2 {
        let lam = eig.eigenvalues[order[dim]].max(0.0);
        let s = lam.sqrt();
        let ev = eig.eigenvectors.column(order[dim]);
        for i in 0..n {
            coords[(i, dim)] = s * ev[i];
        }
    }
    coords
}
