//! Cell-level layout: Nyström projection of individual cells onto the PB
//! layout, followed by an optional per-PB parallel t-SNE fine-tune that
//! uses cells in the K 2D-nearest PBs as fixed context anchors.

use crate::embed_common::*;

/// Number of nearest PBs (in the 2D layout) whose cells are used as context
/// anchors during local refinement. Small but enough to cover local
/// structure without blowing up the O(m²) KL step.
const LOCAL_CONTEXT_KNN: usize = 5;

/// Nyström cell projection: each cell's 2D position is a smooth weighted
/// average of every PB's 2D position, where weights come from the same
/// alpha-decay diffusion kernel PHATE uses internally. Mixture cells smear
/// naturally along trajectories between PBs; pure cells cluster near their
/// dominant PB.
///
/// For each cell c:
///   σ_c = distance from c to its `knn`-th nearest PB in z-space
///   K_cp = exp(-(‖z_c − z_p‖ / σ_c)^α)
///   w_cp = K_cp / Σ_p' K_cp'
///   (x_c, y_c) = Σ_p w_cp · pb_coord[p]
pub(crate) fn project_cells_nystrom(
    cell_z: &Mat,
    pb_z: &Mat,
    pb_coords: &Mat,
    knn: usize,
    alpha: f32,
) -> Mat {
    use rayon::prelude::*;

    let n_cells = cell_z.nrows();
    let n_pb = pb_z.nrows();
    let mut coords = Mat::zeros(n_cells, 2);
    if n_cells == 0 || n_pb == 0 {
        return coords;
    }
    let k = knn.clamp(1, n_pb);
    let kth = (k - 1).min(n_pb - 1);

    let results: Vec<(f32, f32)> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let mut d_cp = vec![0.0f32; n_pb];
            for p in 0..n_pb {
                let mut s = 0.0f32;
                for j in 0..cell_z.ncols() {
                    let diff = cell_z[(c, j)] - pb_z[(p, j)];
                    s += diff * diff;
                }
                d_cp[p] = s.sqrt();
            }

            // Partial sort for σ in O(n_pb) rather than O(n_pb log n_pb).
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
                for v in &mut w {
                    *v /= sum;
                }
            } else {
                let argmin = d_cp
                    .iter()
                    .enumerate()
                    .min_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                w.fill(0.0);
                w[argmin] = 1.0;
            }

            let mut x = 0.0f32;
            let mut y = 0.0f32;
            for p in 0..n_pb {
                x += w[p] * pb_coords[(p, 0)];
                y += w[p] * pb_coords[(p, 1)];
            }
            (x, y)
        })
        .collect();

    for (i, (x, y)) in results.into_iter().enumerate() {
        coords[(i, 0)] = x;
        coords[(i, 1)] = y;
    }
    coords
}

/// Bundled inputs for the local cell-refinement step.
pub(crate) struct LocalRefineArgs<'a> {
    pub cell_z: &'a Mat,
    pub pb_coords: &'a Mat,
    pub pb_membership: &'a [usize],
    pub iters: usize,
    pub perplexity: f32,
    pub learning_rate: f32,
}

/// Local cell-level fine-tune: after the global PB layout and Nyström
/// initialization, refine each PB's cells against cells in its `K` 2D-nearest
/// PBs (fixed context anchors). PBs are processed independently in parallel
/// via rayon — each task writes only to its own focus cells (disjoint across
/// PBs), so there's no write contention. Only cells whose dominant PB is in
/// `pb_membership` (i.e. belongs to a kept PB after coverage filtering) are
/// refined; orphaned cells keep their Nyström positions.
pub(crate) fn refine_cells_local(coords: &mut Mat, a: &LocalRefineArgs) {
    use rayon::prelude::*;

    let n_pb = a.pb_coords.nrows();
    if n_pb < 2 || a.iters == 0 {
        return;
    }

    // Bucket cells by PB.
    let mut pb_cells: Vec<Vec<usize>> = vec![Vec::new(); n_pb];
    for (cell, &pb) in a.pb_membership.iter().enumerate() {
        if pb < n_pb {
            pb_cells[pb].push(cell);
        }
    }

    // Precompute K 2D-nearest PBs per PB.
    let k = LOCAL_CONTEXT_KNN.min(n_pb.saturating_sub(1));
    let pb_knn: Vec<Vec<usize>> = (0..n_pb)
        .map(|u| {
            let mut dists: Vec<(usize, f32)> = (0..n_pb)
                .filter(|&v| v != u)
                .map(|v| {
                    let dx = a.pb_coords[(u, 0)] - a.pb_coords[(v, 0)];
                    let dy = a.pb_coords[(u, 1)] - a.pb_coords[(v, 1)];
                    (v, dx * dx + dy * dy)
                })
                .collect();
            dists.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.into_iter().take(k).map(|(v, _)| v).collect()
        })
        .collect();

    // Par-iter over PBs. Each task reads `coords` / `cell_z` read-only and
    // returns focus-cell updates; writes are flushed after the parallel phase.
    let updates: Vec<(usize, f32, f32)> = (0..n_pb)
        .into_par_iter()
        .flat_map_iter(|u| {
            let focus = &pb_cells[u];
            if focus.len() < 2 {
                return Vec::new();
            }
            let mut context: Vec<usize> = Vec::new();
            for &v in &pb_knn[u] {
                context.extend(pb_cells[v].iter().copied());
            }
            // Cap context size to keep the O(m²) step fast.
            let max_context = (focus.len() * 3).max(32);
            if context.len() > max_context {
                context.truncate(max_context);
            }
            let batch: Vec<usize> = focus.iter().chain(context.iter()).copied().collect();
            let n_focus = focus.len();
            let refined = local_tsne_step(
                coords,
                a.cell_z,
                &batch,
                n_focus,
                a.iters,
                a.perplexity,
                a.learning_rate,
            );
            focus
                .iter()
                .enumerate()
                .map(|(i, &cell)| (cell, refined[i].0, refined[i].1))
                .collect()
        })
        .collect();

    for (cell, x, y) in updates {
        coords[(cell, 0)] = x;
        coords[(cell, 1)] = y;
    }
}

/// Run `iters` manual t-SNE gradient descent steps on `batch` cells.
/// Reads `coords` as a snapshot and returns the refined positions of the
/// first `n_focus` rows (context rows are fixed anchors). Safe to call
/// concurrently for disjoint focus sets.
fn local_tsne_step(
    coords: &Mat,
    cell_z: &Mat,
    batch: &[usize],
    n_focus: usize,
    iters: usize,
    perplexity: f32,
    learning_rate: f32,
) -> Vec<(f32, f32)> {
    let n = batch.len();
    if n < 2 || n_focus == 0 {
        return Vec::new();
    }

    // Pairwise squared distances on `cell_z` for the batch.
    let mut d2 = vec![0.0f32; n * n];
    for i in 0..n {
        let ci = batch[i];
        for j in (i + 1)..n {
            let cj = batch[j];
            let mut s = 0.0f32;
            for k in 0..cell_z.ncols() {
                let diff = cell_z[(ci, k)] - cell_z[(cj, k)];
                s += diff * diff;
            }
            d2[i * n + j] = s;
            d2[j * n + i] = s;
        }
    }

    // Joint P with adaptive Gaussian bandwidth σ_i (k ≈ perplexity).
    let knn = (perplexity as usize).clamp(2, n - 1);
    let mut p = vec![0.0f32; n * n];
    let mut row_buf: Vec<f32> = Vec::with_capacity(n.saturating_sub(1));
    for i in 0..n {
        row_buf.clear();
        row_buf.extend((0..n).filter(|&j| j != i).map(|j| d2[i * n + j]));
        let kth = (knn - 1).min(row_buf.len() - 1);
        row_buf.select_nth_unstable_by(kth, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let sigma2 = row_buf[kth].max(1e-10);
        let mut sum = 0.0f32;
        for j in 0..n {
            if i != j {
                let v = (-d2[i * n + j] / (2.0 * sigma2)).exp();
                p[i * n + j] = v;
                sum += v;
            }
        }
        if sum > 1e-12 {
            for j in 0..n {
                p[i * n + j] /= sum;
            }
        }
    }

    // Symmetrize and globally normalize.
    let mut p_sym = vec![0.0f32; n * n];
    let mut p_total = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (p[i * n + j] + p[j * n + i]);
            p_sym[i * n + j] = v;
            if i != j {
                p_total += v;
            }
        }
    }
    if p_total > 1e-12 {
        for v in &mut p_sym {
            *v /= p_total;
        }
    }

    // Current 2D positions for the batch (focus + context).
    let mut y: Vec<(f32, f32)> = batch
        .iter()
        .map(|&c| (coords[(c, 0)], coords[(c, 1)]))
        .collect();

    // Manual t-SNE gradient descent. Only focus rows move; context rows stay.
    // `q` and `grads` are reused across iterations to avoid allocation churn.
    //
    // Update rule: Jacobi, not Gauss-Seidel. All focus gradients are computed
    // against the *same* snapshot of y, then applied synchronously. The old
    // in-place loop updated y[i] mid-iteration so later i's computed their
    // gradient against already-moved anchors — a subtle correctness bug.
    let mut q = vec![0.0f32; n * n];
    let mut grads: Vec<(f32, f32)> = vec![(0.0, 0.0); n_focus];
    for _ in 0..iters {
        q.fill(0.0);
        let mut q_sum = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = y[i].0 - y[j].0;
                let dy_ = y[i].1 - y[j].1;
                let v = 1.0 / (1.0 + dx * dx + dy_ * dy_);
                q[i * n + j] = v;
                q[j * n + i] = v;
                q_sum += 2.0 * v;
            }
        }
        let q_norm = q_sum.max(1e-12);

        // Phase 1: compute gradients from the frozen y snapshot.
        // ∂KL/∂y_i = 4 Σ_j (P_ij − Q_ij) · q_unnorm_ij · (y_i − y_j)
        for i in 0..n_focus {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let p_ij = p_sym[i * n + j];
                let q_unnorm = q[i * n + j];
                let q_ij = q_unnorm / q_norm;
                let coef = 4.0 * (p_ij - q_ij) * q_unnorm;
                gx += coef * (y[i].0 - y[j].0);
                gy += coef * (y[i].1 - y[j].1);
            }
            grads[i] = (gx, gy);
        }

        // Phase 2: synchronous update.
        for i in 0..n_focus {
            y[i].0 -= learning_rate * grads[i].0;
            y[i].1 -= learning_rate * grads[i].1;
        }
    }

    y.truncate(n_focus);
    y
}
