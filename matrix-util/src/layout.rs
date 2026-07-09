//! 2D layout primitives shared across the workspace: PHATE diffusion
//! embedding and a generic alpha-decay-kernel Nyström projection for placing
//! out-of-sample points (cells, genes, type anchors) onto an existing layout.
//!
//! These are pure functions on `nalgebra` matrices with no I/O and no CLI
//! dependencies, so they can be consumed by `senna` (PB-landmark layout),
//! `faba gem-annotate` (cell layout off the leiden kNN graph + feature
//! placement), and any other caller. Originally `pub(crate)` inside senna;
//! lifted here so faba can reuse them without duplication.

use crate::knn_match::{ColumnDict, VecPoint};
use crate::traits::RandomizedAlgs;
use log::info;
use nalgebra::DMatrix;
use rayon::prelude::*;

type Mat = DMatrix<f32>;

///////////
// PHATE //
///////////

// PHATE — Potential of Heat-diffusion for Affinity-based Transition Embedding.
// Moon et al., Nat Biotechnol 2019 (PMC7073148).
//
// Given an n×K matrix of features `data` (e.g. the whitened reconstruction-
// space representation, cell embeddings, or landmark/PB embeddings), PHATE
// produces an n×2 embedding that preserves both local neighborhoods and
// global trajectory/branching structure:
//
// 1. Kernel affinity K with adaptive bandwidth σᵢ = distance to the `knn`-th
//    neighbor, and alpha-decay exponent α (sharper than a Gaussian):
//    `K[i,j] = ½ · (exp(-(d_ij/σᵢ)^α) + exp(-(d_ij/σⱼ)^α))`.
// 2. Diffusion operator P = row-normalize(K).
// 3. Diffusion in time: M = P^t (smooths noise, preserves global geometry).
// 4. Potential distance: `U[i,j] = ‖-log M[i,:] − -log M[j,:]‖₂`.
// 5. Classical MDS on U → 2D coordinates, then SMACOF metric-MDS refinement.
//
// The whole thing is O(n²K + n² log t + n³) for the MDS, which is fast for
// the n ≲ 10³ point counts it is meant for. For many points, lay out
// landmarks with this and lift the rest via [`project_cells_nystrom`].

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
            alpha: 40.0,
            mds_iter: 300,
            mds_tol: 1e-4,
        }
    }
}

/// Compute a 2D PHATE embedding of `data` (n × K).
#[must_use]
pub fn phate_layout_2d(data: &Mat, args: &PhateArgs) -> Mat {
    let n = data.nrows();
    if n < 3 {
        return Mat::zeros(n, 2);
    }
    let knn = args.knn.clamp(1, n - 1);
    info!(
        "PHATE start: n={} points, features={}, t={}, knn={}, α={}",
        n,
        data.ncols(),
        args.t,
        knn,
        args.alpha
    );

    info!("PHATE 1/6: pairwise distances");
    // Materialize rows once so every inner distance reads a contiguous
    // slice; column-major `row()` views are strided and miss SIMD.
    let rows = rows_as_contiguous(data);
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

    // Exact diffusion M = P^t by repeated squaring. P is row-stochastic and
    // NON-symmetric, so an SVD-based `U Σ^t V^T` power is mathematically wrong
    // (that identity needs a normal matrix) — it corrupted the diffusion and
    // collapsed the embedding. Exact power is O(n³ log t), fine at PHATE's n≲10³.
    info!(
        "PHATE 4/6: diffusion M = P^{} (repeated squaring)",
        args.t.max(1)
    );
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

    let prog_bar = crate::progress::new_progress_bar(max_iter as u64);
    prog_bar.set_message("SMACOF");
    for _ in 0..max_iter {
        prog_bar.inc(1);
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
            prog_bar.set_message(format!("SMACOF converged (stress={stress:.3e})"));
            break;
        }
        prev_stress = stress;
        prog_bar.set_message(format!("SMACOF stress={stress:.3e}"));

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
    prog_bar.finish_and_clear();

    y
}

/// Exact matrix power `M^exp` by repeated squaring.
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
/// Uses randomized SVD for efficiency (matches reference PHATE's randmds).
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

    // Classical MDS embeds along the TOP eigenvectors of the Gram matrix. For
    // PHATE these are the potential-distance axes (the trajectory itself), so we
    // must use the top two — NOT skip the first. (Skipping it, as an earlier
    // "library size" heuristic did, left the init near-collinear; SMACOF's
    // Guttman update preserves a zero coordinate, so the layout then collapsed
    // to ~1D. This is MDS on potential distances, where no library-size axis
    // exists.)
    let rank = n.min(10);
    if let Ok((u, s, _v)) = b.rsvd(rank) {
        let mut coords = Mat::zeros(n, 2);
        for dim in 0..2 {
            if dim >= rank {
                break;
            }
            let scale = s[dim].max(0.0).sqrt(); // sqrt because SVD gives sqrt(eigenvalue)
            for i in 0..n {
                coords[(i, dim)] = u[(i, dim)] * scale;
            }
        }
        coords
    } else {
        // Fallback to exact eigendecomposition if RSVD fails
        let eig = b.symmetric_eigen();
        let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
        order.sort_by(|&a, &b_| {
            eig.eigenvalues[b_]
                .partial_cmp(&eig.eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Top two eigenvectors (see note above — do not skip the first).
        let mut coords = Mat::zeros(n, 2);
        for dim in 0..2 {
            if dim >= order.len() {
                break;
            }
            let s = eig.eigenvalues[order[dim]].max(0.0).sqrt();
            let ev = eig.eigenvectors.column(order[dim]);
            for i in 0..n {
                coords[(i, dim)] = s * ev[i];
            }
        }
        coords
    }
}

////////////////////////
// Nyström projection //
////////////////////////

/// Alpha-decay-kernel Nyström projection: place each *query* point into an
/// existing 2D layout as a smooth weighted average of the *landmark* points'
/// 2D positions, with weights from the same alpha-decay diffusion kernel
/// PHATE uses internally.
///
/// This is the generic out-of-sample extension. Two uses in the workspace:
/// - **cells → landmark layout** (faba large-N PHATE): query = cells,
///   landmark = k-means centroids of the cell embedding, both in e_cell space.
/// - **features → cell layout** (faba part ii): query = genes/features,
///   landmark = cells, both in the H-dim embedding; the cell 2D layout is the
///   target.
///
/// `query_kn` and `landmark_kp` are `(k × ·)` column-major — each point is a
/// contiguous column slice, so the inner distance kernel vectorizes. They
/// must share the same feature dimension `k`. `landmark_coords` is
/// `(n_landmark × 2)`. Returns `(n_query × 2)`.
///
/// For each query q:
///   `σ_q` = distance from q to its `knn`-th nearest landmark in feature space
///   `K_qp` = exp(-(‖z_q − `z_p`‖ / `σ_q)^α`)
///   `w_qp` = `K_qp` / `Σ_p`' `K_qp`'
///   (`x_q`, `y_q`) = `Σ_p` `w_qp` · `landmark_coord`[p]
#[must_use]
pub fn project_cells_nystrom(
    query_kn: &Mat,
    landmark_kp: &Mat,
    landmark_coords: &Mat,
    knn: usize,
    alpha: f32,
) -> Mat {
    let n_query = query_kn.ncols();
    let n_land = landmark_kp.ncols();
    let mut coords = Mat::zeros(n_query, 2);
    if n_query == 0 || n_land == 0 {
        return coords;
    }
    let k = knn.clamp(1, n_land);
    let kth = (k - 1).min(n_land - 1);

    // Cache each landmark's contiguous feature slice once so every query's
    // task reuses it instead of re-fetching strided column views.
    let query_slice = query_kn.as_slice();
    let land_slice = landmark_kp.as_slice();
    let feat_dim = query_kn.nrows();
    let land_x: Vec<f32> = (0..n_land).map(|p| landmark_coords[(p, 0)]).collect();
    let land_y: Vec<f32> = (0..n_land).map(|p| landmark_coords[(p, 1)]).collect();

    let results: Vec<(f32, f32)> = (0..n_query)
        .into_par_iter()
        .map(|c| {
            let q = &query_slice[c * feat_dim..(c + 1) * feat_dim];

            let mut d_cp = vec![0.0f32; n_land];
            for p in 0..n_land {
                let lp = &land_slice[p * feat_dim..(p + 1) * feat_dim];
                d_cp[p] = sq_dist(q, lp).sqrt();
            }

            let mut sigma_buf = d_cp.clone();
            sigma_buf.select_nth_unstable_by(kth, |x, y| {
                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
            });
            let sigma = sigma_buf[kth].max(1e-6);

            let mut w: Vec<f32> = d_cp
                .iter()
                .map(|&d| (-(d / sigma).powf(alpha)).exp())
                .collect();
            let sum: f32 = w.iter().sum();
            if sum > 1e-12 {
                let inv = 1.0 / sum;
                for v in &mut w {
                    *v *= inv;
                }
            } else {
                let argmin = d_cp
                    .iter()
                    .enumerate()
                    .min_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i);
                w.fill(0.0);
                w[argmin] = 1.0;
            }

            let x: f32 = w.iter().zip(land_x.iter()).map(|(a, b)| a * b).sum();
            let y: f32 = w.iter().zip(land_y.iter()).map(|(a, b)| a * b).sum();
            (x, y)
        })
        .collect();

    for (i, (x, y)) in results.into_iter().enumerate() {
        coords[(i, 0)] = x;
        coords[(i, 1)] = y;
    }
    coords
}

/// Place each *query* point onto an existing 2D layout of *landmark* points
/// via **kNN in feature space** — the scalable out-of-sample extension when
/// there are too many landmarks to score every (query, landmark) pair
/// densely (as [`project_cells_nystrom`] does).
///
/// For each query, its `knn` nearest landmarks are found through a prebuilt
/// HNSW index (`landmark_dict`, built over the landmark feature columns), and
/// the query lands at the alpha-decay-kernel weighted mean of those
/// neighbors' 2D coords. The index uses L2 distance, so **L2-normalize both
/// the query columns and the landmark features beforehand** for cosine
/// semantics.
///
/// * `query_cols` — `H × n_query` column-major; each column a query point.
/// * `landmark_dict` — HNSW over the landmarks, names = landmark row index.
/// * `landmark_coords` — `n_landmark × 2` target layout.
///
/// Returns `n_query × 2`. Cost is O(n_query · knn · log n_landmark).
#[must_use]
pub fn project_via_knn(
    query_cols: &Mat,
    landmark_dict: &ColumnDict<usize>,
    landmark_coords: &Mat,
    knn: usize,
    alpha: f32,
) -> Mat {
    let n_query = query_cols.ncols();
    let h = query_cols.nrows();
    let mut coords = Mat::zeros(n_query, 2);
    if n_query == 0 || landmark_coords.nrows() == 0 {
        return coords;
    }
    let k = knn.max(1);
    let lx: Vec<f32> = (0..landmark_coords.nrows())
        .map(|p| landmark_coords[(p, 0)])
        .collect();
    let ly: Vec<f32> = (0..landmark_coords.nrows())
        .map(|p| landmark_coords[(p, 1)])
        .collect();

    let qslice = query_cols.as_slice();
    let results: Vec<(f32, f32)> = (0..n_query)
        .into_par_iter()
        .map(|q| {
            let vp = VecPoint {
                data: qslice[q * h..(q + 1) * h].to_vec(),
            };
            let Ok((idx, dist)) = landmark_dict.search_by_query_data(&vp, k) else {
                return (f32::NAN, f32::NAN);
            };
            if idx.is_empty() {
                return (f32::NAN, f32::NAN);
            }
            // σ = farthest of the kNN (so the nearest gets ~unit weight and
            // the kth ~exp(-1)); alpha-decay weights, like PHATE's kernel.
            let sigma = dist.iter().cloned().fold(0.0_f32, f32::max).max(1e-6);
            let mut wsum = 0.0_f32;
            let (mut x, mut y) = (0.0_f32, 0.0_f32);
            for (&p, &d) in idx.iter().zip(dist.iter()) {
                let w = (-(d / sigma).powf(alpha)).exp();
                wsum += w;
                x += w * lx[p];
                y += w * ly[p];
            }
            if wsum > 1e-12 {
                (x / wsum, y / wsum)
            } else {
                // Degenerate weights → fall back to the single nearest.
                (lx[idx[0]], ly[idx[0]])
            }
        })
        .collect();

    for (i, (x, y)) in results.into_iter().enumerate() {
        coords[(i, 0)] = x;
        coords[(i, 1)] = y;
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn_match::ColumnDict;

    /// PHATE must recover genuine 2D structure: three blobs whose centers form a
    /// *triangle* (non-collinear → intrinsically 2D) stay separated in the layout
    /// and use *both* output dimensions. This guards the two bugs that once
    /// collapsed the embedding to ~1D: classical-MDS skipping the first
    /// eigenvector, and an SVD-based `P^t` power (invalid for the non-symmetric
    /// diffusion operator).
    #[test]
    fn phate_recovers_2d_cluster_structure() {
        let centers = [[0.0_f32, 0.0], [10.0, 0.0], [5.0, 8.0]];
        let (blobs, per, d) = (3, 20, 5);
        let n = blobs * per;
        let mut data = Mat::zeros(n, d);
        for (b, center) in centers.iter().enumerate() {
            for p in 0..per {
                let i = b * per + p;
                let jit = |s: usize| ((i * 7 + s * 13) % 11) as f32 / 11.0 - 0.5;
                data[(i, 0)] = center[0] + jit(0) * 0.6;
                data[(i, 1)] = center[1] + jit(1) * 0.6;
                for j in 2..d {
                    data[(i, j)] = jit(j) * 0.3;
                }
            }
        }
        let y = phate_layout_2d(&data, &PhateArgs::default());
        assert_eq!((y.nrows(), y.ncols()), (n, 2));
        assert!(y.iter().all(|v| v.is_finite()), "coords must be finite");

        // Both output dimensions carry real variance (guards the 1D collapse).
        let var = |c: usize| {
            let m = y.column(c).mean();
            y.column(c).iter().map(|v| (v - m).powi(2)).sum::<f32>() / n as f32
        };
        assert!(
            var(0) > 1e-6 && var(1) > 1e-6,
            "2D structure collapsed: var=({:.2e}, {:.2e})",
            var(0),
            var(1)
        );

        // Blobs stay separated: max within-blob radius < min between-centroid gap.
        let centroid = |b: usize| {
            let (mut x, mut z) = (0.0f32, 0.0f32);
            for p in 0..per {
                let i = b * per + p;
                x += y[(i, 0)];
                z += y[(i, 1)];
            }
            (x / per as f32, z / per as f32)
        };
        let cens: Vec<(f32, f32)> = (0..blobs).map(centroid).collect();
        let mut max_within = 0f32;
        for (b, &c) in cens.iter().enumerate() {
            for p in 0..per {
                let i = b * per + p;
                let dd = ((y[(i, 0)] - c.0).powi(2) + (y[(i, 1)] - c.1).powi(2)).sqrt();
                max_within = max_within.max(dd);
            }
        }
        let mut min_between = f32::INFINITY;
        for a in 0..blobs {
            for b in (a + 1)..blobs {
                let dd = ((cens[a].0 - cens[b].0).powi(2) + (cens[a].1 - cens[b].1).powi(2)).sqrt();
                min_between = min_between.min(dd);
            }
        }
        assert!(
            min_between > max_within,
            "blobs not separated: within={max_within:.3} between={min_between:.3}"
        );
    }

    /// A query sitting exactly on a landmark, with `knn = 1`, lands on that
    /// landmark's 2D coordinate (kernel bandwidth → 0, mass concentrates).
    /// With larger `knn` the kernel is deliberately smooth (a weighted
    /// average), so concentration is only exact at `knn = 1`.
    #[test]
    fn nystrom_query_on_landmark() {
        // 3 landmarks in 2D feature space, far apart.
        let landmark_kp = Mat::from_column_slice(2, 3, &[0.0, 0.0, 10.0, 0.0, 0.0, 10.0]);
        // Their target 2D coords.
        let landmark_coords = Mat::from_row_slice(3, 2, &[1.0, 1.0, 5.0, 2.0, 2.0, 5.0]);
        // Query equals landmark 1.
        let query_kn = Mat::from_column_slice(2, 1, &[10.0, 0.0]);
        let out = project_cells_nystrom(&query_kn, &landmark_kp, &landmark_coords, 1, 40.0);
        assert_eq!(out.nrows(), 1);
        assert!((out[(0, 0)] - 5.0).abs() < 1e-3, "x={}", out[(0, 0)]);
        assert!((out[(0, 1)] - 2.0).abs() < 1e-3, "y={}", out[(0, 1)]);
    }

    /// kNN projector: a query near one landmark lands near that landmark's
    /// 2D coordinate; uses the HNSW dict path.
    #[test]
    fn project_via_knn_lands_near_nearest() {
        // 4 landmarks in 3D feature space (columns), unit-ish, far apart.
        let landmarks = Mat::from_column_slice(
            3,
            4,
            &[
                1.0, 0.0, 0.0, // l0
                0.0, 1.0, 0.0, // l1
                0.0, 0.0, 1.0, // l2
                -1.0, 0.0, 0.0, // l3
            ],
        );
        let coords = Mat::from_row_slice(4, 2, &[0.0, 0.0, 10.0, 0.0, 0.0, 10.0, -10.0, -10.0]);
        let dict = ColumnDict::<usize>::from_dmatrix(landmarks, (0..4).collect());
        // Query very close to l1 = (0,1,0) → coord (10, 0). With knn=1 the
        // single nearest neighbor is returned exactly (the multi-neighbor
        // kernel is a deliberate blend, covered by the Nyström tests).
        let query = Mat::from_column_slice(3, 1, &[0.02, 0.98, 0.0]);
        let out = project_via_knn(&query, &dict, &coords, 1, 20.0);
        assert_eq!(out.nrows(), 1);
        assert!((out[(0, 0)] - 10.0).abs() < 1e-3, "x={}", out[(0, 0)]);
        assert!(out[(0, 1)].abs() < 1e-3, "y={}", out[(0, 1)]);
    }

    /// A query at the midpoint between two equidistant landmarks (with the
    /// third far away) lands near the average of their 2D coords.
    #[test]
    fn nystrom_query_between_landmarks() {
        let landmark_kp = Mat::from_column_slice(2, 3, &[0.0, 0.0, 2.0, 0.0, 100.0, 100.0]);
        let landmark_coords = Mat::from_row_slice(3, 2, &[0.0, 0.0, 10.0, 0.0, -50.0, -50.0]);
        let query_kn = Mat::from_column_slice(2, 1, &[1.0, 0.0]); // midpoint of l0,l1
        let out = project_cells_nystrom(&query_kn, &landmark_kp, &landmark_coords, 2, 2.0);
        assert!((out[(0, 0)] - 5.0).abs() < 1.0, "x={}", out[(0, 0)]);
        assert!(out[(0, 1)].abs() < 1.0, "y={}", out[(0, 1)]);
    }
}
