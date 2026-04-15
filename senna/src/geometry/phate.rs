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

    // 1. Pairwise squared distances in pb_z.
    let mut dist = Mat::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = pb_z.row(i) - pb_z.row(j);
            let d2: f32 = diff.iter().map(|&x| x * x).sum();
            let d = d2.sqrt();
            dist[(i, j)] = d;
            dist[(j, i)] = d;
        }
    }

    // 2. Adaptive bandwidth: σᵢ = distance to knn-th nearest neighbor.
    let sigma: Vec<f32> = (0..n)
        .map(|i| {
            let mut row: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| dist[(i, j)]).collect();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row[knn - 1].max(1e-6)
        })
        .collect();

    // 3. Alpha-decay kernel (symmetrized by averaging).
    let mut k_mat = Mat::zeros(n, n);
    for i in 0..n {
        k_mat[(i, i)] = 1.0;
        for j in (i + 1)..n {
            let d = dist[(i, j)];
            let e1 = (-(d / sigma[i]).powf(args.alpha)).exp();
            let e2 = (-(d / sigma[j]).powf(args.alpha)).exp();
            let v = 0.5 * (e1 + e2);
            k_mat[(i, j)] = v;
            k_mat[(j, i)] = v;
        }
    }

    // 4. Row-normalize to the diffusion operator P.
    let mut p_mat = k_mat;
    for i in 0..n {
        let s: f32 = p_mat.row(i).iter().sum::<f32>().max(1e-12);
        for j in 0..n {
            p_mat[(i, j)] /= s;
        }
    }

    // 5. Diffusion in time: M = P^t (via repeated squaring for efficiency).
    let m_mat = matrix_power(&p_mat, args.t.max(1));

    // 6. Potential distance: U[i,j] = ‖-log M[i,:] - -log M[j,:]‖₂.
    let mut log_m = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            log_m[(i, j)] = -(m_mat[(i, j)].max(1e-12)).ln();
        }
    }
    let mut pot_d2 = Mat::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = log_m.row(i) - log_m.row(j);
            let v: f32 = diff.iter().map(|&x| x * x).sum();
            pot_d2[(i, j)] = v;
            pot_d2[(j, i)] = v;
        }
    }

    // 7a. Classical MDS for initialization (cheap, gets us to a good basin).
    let y_init = classical_mds_2d(&pot_d2);

    // 7b. Metric MDS via SMACOF on the (unsquared) potential distances.
    let mut delta = Mat::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pot_d2[(i, j)].sqrt();
            delta[(i, j)] = d;
            delta[(j, i)] = d;
        }
    }
    smacof_2d(&delta, &y_init, args.mds_iter, args.mds_tol)
}

/// Metric MDS via SMACOF (Scaling by MAjorizing a COmplicated Function).
///
/// Minimizes the stress
///     σ(Y) = Σ_{i<j} (δ_ij − d_ij(Y))²
/// over Y ∈ ℝ^{n × 2} via the Guttman transform:
///     Y_{k+1} = (1/n) · B(Y_k) · Y_k,
/// where B[i,j] = −δ[i,j] / d_ij(Y_k) for i ≠ j (and zero if d_ij(Y_k) is
/// numerically zero), B[i,i] = −Σ_{j ≠ i} B[i,j]. Each iteration strictly
/// decreases stress (majorization), so no learning rate or line search is
/// needed. Initialized from classical MDS so the basin is sensible.
fn smacof_2d(delta: &Mat, y_init: &Mat, max_iter: usize, tol: f32) -> Mat {
    let n = delta.nrows();
    let mut y = y_init.clone();
    let mut prev_stress = f32::INFINITY;

    // Reusable scratch matrices. SMACOF allocates n×n twice per iteration
    // in the naive form; hoisting saves ~600 MB / run at n = 500, t = 300.
    let mut dy = Mat::zeros(n, n);
    let mut b_mat = Mat::zeros(n, n);
    let inv_n = 1.0 / n as f32;

    for _ in 0..max_iter {
        // Pairwise distances in the current 2D configuration.
        for i in 0..n {
            dy[(i, i)] = 0.0;
            for j in (i + 1)..n {
                let dx = y[(i, 0)] - y[(j, 0)];
                let dz = y[(i, 1)] - y[(j, 1)];
                let d = (dx * dx + dz * dz).sqrt();
                dy[(i, j)] = d;
                dy[(j, i)] = d;
            }
        }

        // Stress σ(Y).
        let mut stress = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = delta[(i, j)] - dy[(i, j)];
                stress += diff * diff;
            }
        }

        // Relative-tolerance convergence on stress change.
        let denom = prev_stress.max(1.0);
        if (prev_stress - stress).abs() / denom < tol {
            break;
        }
        prev_stress = stress;

        // B(Y_k). Overwrite in place.
        for i in 0..n {
            let mut diag = 0.0f32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d_ij = dy[(i, j)];
                let b = if d_ij > 1e-10 {
                    -delta[(i, j)] / d_ij
                } else {
                    0.0
                };
                b_mat[(i, j)] = b;
                diag -= b;
            }
            b_mat[(i, i)] = diag;
        }

        // Guttman update: Y ← (1/n) · B · Y.
        let mut y_new = &b_mat * &y;
        for v in y_new.iter_mut() {
            *v *= inv_n;
        }
        y = y_new;
    }

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
