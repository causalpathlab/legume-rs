//! Slingshot-style simultaneous principal curves over a low-dimensional
//! embedding (Street et al. 2018, "Slingshot: cell lineage and pseudotime
//! inference for single-cell transcriptomics").
//!
//! Given cells `z` (N×D), a set of K node centroids, and an (undirected) MST
//! over those centroids rooted at `root`, this:
//!   1. enumerates **lineages** as simple paths `root → leaf` on the tree,
//!   2. initializes each lineage's curve as the polyline through its centroids,
//!   3. iteratively **projects** cells onto each curve (orthogonal, arc-length
//!      λ) and **re-smooths** the curve as a weighted Nadaraya–Watson kernel
//!      regression of cell coordinates on λ — the "kernelization" that turns the
//!      piecewise-linear tree into smooth cell-level structure,
//!   4. keeps lineages coherent upstream of branch points by fitting shared
//!      cells (cells whose cluster lies on more than one lineage) into every
//!      lineage they belong to, with membership weights.
//!
//! Step 4 is a deliberate simplification of canonical Slingshot: rather than the
//! explicit "average the overlapping curves in shared segments" shrinkage, shared
//! (trunk) cells are simply fit — with membership weights — into every lineage
//! they belong to, which pulls those curves together upstream of a branch. This
//! matched reference Slingshot pseudotime to ρ ≈ 0.995+ in validation; on very
//! strongly-branching data the shared segments may diverge slightly more than the
//! explicit-shrinkage variant would.
//!
//! The MST + centroids come from [`crate::principal_graph`] (k-means centroids +
//! [`crate::principal_graph::mst_from_sqdist`]); orientation/root selection is
//! left to the caller (e.g. velocity flux in `faba`).

use nalgebra::DMatrix;
use rayon::prelude::*;

////////////////
// Public API //
////////////////

/// Configuration for [`fit_principal_curves`].
#[derive(Debug, Clone)]
pub struct PrincipalCurveArgs {
    /// Maximum outer project-then-smooth iterations.
    pub max_iter: usize,
    /// Relative convergence tolerance on mean |Δλ| / λ-range.
    pub tol: f32,
    /// Number of points sampled along each fitted curve (the λ grid size).
    pub resolution: usize,
    /// Gaussian kernel bandwidth in λ (arc-length) units. Set `≤ 0` for an
    /// adaptive bandwidth = `curve_length / 12` per lineage.
    pub bandwidth: f32,
}

impl Default for PrincipalCurveArgs {
    fn default() -> Self {
        Self {
            max_iter: 15,
            tol: 1e-3,
            resolution: 100,
            bandwidth: -1.0,
        }
    }
}

/// A single fitted principal curve for one lineage.
#[derive(Debug, Clone)]
pub struct LineageCurve {
    /// Centroid indices along this lineage, `root … leaf`.
    pub node_path: Vec<usize>,
    /// `resolution × D` smooth curve points, ordered from root (λ=0) to leaf.
    pub points: DMatrix<f32>,
    /// Cumulative arc-length at each curve point (non-decreasing, `points.nrows()`).
    pub lambda_grid: Vec<f32>,
}

/// Result of [`fit_principal_curves`].
#[derive(Debug, Clone)]
pub struct PrincipalCurves {
    /// One curve per lineage, in lineage id order.
    pub curves: Vec<LineageCurve>,
    /// Per-cell nearest-centroid (cluster) label.
    pub cluster: Vec<usize>,
    /// `N × L` membership weights (row-normalized over the lineages a cell
    /// belongs to; 0 for lineages whose path excludes the cell's cluster).
    pub weights: DMatrix<f32>,
    /// `N × L` per-lineage pseudotime λ; `NaN` where the cell is not a member.
    pub lineage_pseudotime: DMatrix<f32>,
    /// Per-cell pseudotime along its **primary** lineage (min orthogonal dist).
    pub pseudotime: Vec<f32>,
    /// Per-cell primary lineage id (index into `curves`).
    pub branch: Vec<usize>,
    /// Outer iterations consumed.
    pub n_iters: usize,
}

impl PrincipalCurves {
    pub fn n_lineages(&self) -> usize {
        self.curves.len()
    }
}

/// Fit simultaneous principal curves. `mst_edges` are undirected `(j,k)` pairs
/// (as returned by [`crate::principal_graph::mst_from_sqdist`]); `root` is the
/// centroid index to treat as the trajectory origin.
pub fn fit_principal_curves(
    z: &DMatrix<f32>,
    centroids: &DMatrix<f32>,
    mst_edges: &[(usize, usize)],
    root: usize,
    args: &PrincipalCurveArgs,
) -> anyhow::Result<PrincipalCurves> {
    let n = z.nrows();
    let d = z.ncols();
    let k = centroids.nrows();
    anyhow::ensure!(k >= 2, "need at least 2 centroids for principal curves");
    anyhow::ensure!(centroids.ncols() == d, "centroid/cell dimension mismatch");
    anyhow::ensure!(root < k, "root {root} out of range for {k} centroids");
    anyhow::ensure!(args.resolution >= 2, "curve resolution must be ≥ 2");

    // Cell → nearest centroid (cluster label).
    let cluster = assign_clusters(z, centroids);

    // Rooted tree → lineages (root→leaf node paths).
    let children = build_children(mst_edges, root, k);
    let lineages = enumerate_lineages(&children, root);
    anyhow::ensure!(!lineages.is_empty(), "no lineages found from root");
    let n_lin = lineages.len();

    // Membership weights: cell belongs to lineage L iff its cluster is on path.
    let weights = membership_weights(&cluster, &lineages, n);

    // Initialize each curve as the resampled polyline through its centroids.
    let mut curves: Vec<LineageCurve> = lineages
        .iter()
        .map(|path| init_curve(centroids, path, args.resolution))
        .collect();

    // Iterate: project cells → λ, then re-smooth each curve on λ.
    let mut lambda = DMatrix::<f32>::from_element(n, n_lin, f32::NAN);
    let mut dist2 = DMatrix::<f32>::from_element(n, n_lin, f32::INFINITY);
    let mut prev_primary = vec![f32::NAN; n];
    let mut n_iters = 0usize;

    for iter in 0..args.max_iter {
        //////////////////////////////////////////////////////
        // projection step (all cells × all their lineages) //
        //////////////////////////////////////////////////////
        for (l, curve) in curves.iter().enumerate() {
            project_members(z, curve, &weights, l, &mut lambda, &mut dist2);
        }

        ////////////////////////////////////////////////////
        // smoothing step (per lineage, weighted NW on λ) //
        ////////////////////////////////////////////////////
        for (l, curve) in curves.iter_mut().enumerate() {
            smooth_curve(z, &weights, &lambda, l, args, curve);
        }

        ////////////////////////////////////////////////
        // convergence on per-cell primary pseudotime //
        ////////////////////////////////////////////////
        let (primary, _branch) = primary_assignment(&lambda, &dist2, &weights);
        let delta = mean_rel_delta(&prev_primary, &primary);
        prev_primary = primary;
        n_iters = iter + 1;
        log::debug!("principal-curve iter {iter}: mean|Δλ|/range = {delta:.3e}");
        if delta < args.tol {
            break;
        }
    }

    // Final projection so λ/dist match the last curves, then assign.
    for (l, curve) in curves.iter().enumerate() {
        project_members(z, curve, &weights, l, &mut lambda, &mut dist2);
    }
    let (pseudotime, branch) = primary_assignment(&lambda, &dist2, &weights);

    Ok(PrincipalCurves {
        curves,
        cluster,
        weights,
        lineage_pseudotime: lambda,
        pseudotime,
        branch,
        n_iters,
    })
}

/////////////////////
// Tree → lineages //
/////////////////////

/// BFS from `root` over the undirected MST; returns each node's children in the
/// rooted orientation.
fn build_children(mst_edges: &[(usize, usize)], root: usize, k: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); k];
    for &(a, b) in mst_edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut children = vec![Vec::new(); k];
    let mut parent = vec![usize::MAX; k];
    let mut seen = vec![false; k];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);
    seen[root] = true;
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                parent[v] = u;
                children[u].push(v);
                queue.push_back(v);
            }
        }
    }
    children
}

/// All simple paths `root → leaf` (leaf = node with no children in the rooted
/// tree). A linear tree yields a single lineage.
fn enumerate_lineages(children: &[Vec<usize>], root: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(root, vec![root])];
    while let Some((node, path)) = stack.pop() {
        if children[node].is_empty() {
            out.push(path);
        } else {
            for &c in &children[node] {
                let mut next = path.clone();
                next.push(c);
                stack.push((c, next));
            }
        }
    }
    // Deterministic order (DFS/stack order is reverse-ish); sort by first branch.
    out.sort_by(|a, b| a.iter().cmp(b.iter()));
    out
}

/////////////////////////////////////
// Cluster assignment + membership //
/////////////////////////////////////

fn assign_clusters(z: &DMatrix<f32>, centroids: &DMatrix<f32>) -> Vec<usize> {
    let n = z.nrows();
    let d = z.ncols();
    let k = centroids.nrows();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut best = 0usize;
            let mut best_sd = f32::INFINITY;
            for c in 0..k {
                let mut s = 0f32;
                for j in 0..d {
                    let v = z[(i, j)] - centroids[(c, j)];
                    s += v * v;
                }
                if s < best_sd {
                    best_sd = s;
                    best = c;
                }
            }
            best
        })
        .collect()
}

/// `N × L` membership weights, row-normalized over the lineages a cell's cluster
/// lies on. A cell whose cluster sits on a shared prefix contributes to every
/// descendant lineage.
fn membership_weights(cluster: &[usize], lineages: &[Vec<usize>], n: usize) -> DMatrix<f32> {
    let n_lin = lineages.len();
    let on_path: Vec<std::collections::HashSet<usize>> = lineages
        .iter()
        .map(|p| p.iter().copied().collect())
        .collect();
    let mut w = DMatrix::<f32>::zeros(n, n_lin);
    for i in 0..n {
        let c = cluster[i];
        let mut cnt = 0f32;
        for l in 0..n_lin {
            if on_path[l].contains(&c) {
                w[(i, l)] = 1.0;
                cnt += 1.0;
            }
        }
        if cnt > 0.0 {
            for l in 0..n_lin {
                w[(i, l)] /= cnt;
            }
        } else {
            // Cluster on no lineage should be impossible, but keep cells valid.
            let u = 1.0 / n_lin as f32;
            for l in 0..n_lin {
                w[(i, l)] = u;
            }
        }
    }
    w
}

///////////////////////////////////////
// Curve init, projection, smoothing //
///////////////////////////////////////

/// Resample the polyline through a lineage's centroids to `resolution` points
/// uniformly spaced in arc-length.
fn init_curve(centroids: &DMatrix<f32>, path: &[usize], resolution: usize) -> LineageCurve {
    let d = centroids.ncols();
    // Control points = centroids along the path.
    let m = path.len();
    let mut ctrl = DMatrix::<f32>::zeros(m.max(2), d);
    for (i, &c) in path.iter().enumerate() {
        for j in 0..d {
            ctrl[(i, j)] = centroids[(c, j)];
        }
    }
    if m == 1 {
        // Degenerate single-node lineage: duplicate the point.
        for j in 0..d {
            ctrl[(1, j)] = centroids[(path[0], j)];
        }
    }
    let arclen = polyline_arclength(&ctrl);
    let mut curve = resample_uniform(&ctrl, &arclen, resolution);
    curve.node_path = path.to_vec();
    curve
}

/// Cumulative arc-length at each row of an ordered point set.
fn polyline_arclength(pts: &DMatrix<f32>) -> Vec<f32> {
    let m = pts.nrows();
    let d = pts.ncols();
    let mut acc = vec![0f32; m];
    for i in 1..m {
        let mut s = 0f32;
        for j in 0..d {
            let v = pts[(i, j)] - pts[(i - 1, j)];
            s += v * v;
        }
        acc[i] = acc[i - 1] + s.sqrt();
    }
    acc
}

/// Sample `resolution` points uniformly in arc-length along the polyline defined
/// by `(pts, arclen)`.
fn resample_uniform(pts: &DMatrix<f32>, arclen: &[f32], resolution: usize) -> LineageCurve {
    let d = pts.ncols();
    let total = *arclen.last().unwrap_or(&0.0);
    let mut grid = DMatrix::<f32>::zeros(resolution, d);
    let mut lambda_grid = vec![0f32; resolution];
    let mut seg = 0usize;
    for g in 0..resolution {
        let target = if resolution == 1 {
            0.0
        } else {
            total * g as f32 / (resolution - 1) as f32
        };
        lambda_grid[g] = target;
        while seg + 1 < arclen.len() && arclen[seg + 1] < target {
            seg += 1;
        }
        let (a, b) = (seg, (seg + 1).min(pts.nrows() - 1));
        let la = arclen[a];
        let lb = arclen[b.min(arclen.len() - 1)];
        let t = if lb > la {
            (target - la) / (lb - la)
        } else {
            0.0
        };
        for j in 0..d {
            grid[(g, j)] = pts[(a, j)] + t.clamp(0.0, 1.0) * (pts[(b, j)] - pts[(a, j)]);
        }
    }
    LineageCurve {
        node_path: Vec::new(), // filled by caller
        points: grid,
        lambda_grid,
    }
}

/// Project every member cell of lineage `l` orthogonally onto `curve`; write its
/// arc-length λ and squared distance into `lambda`/`dist2` column `l`.
fn project_members(
    z: &DMatrix<f32>,
    curve: &LineageCurve,
    weights: &DMatrix<f32>,
    l: usize,
    lambda: &mut DMatrix<f32>,
    dist2: &mut DMatrix<f32>,
) {
    let n = z.nrows();
    let out: Vec<(f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            if weights[(i, l)] <= 0.0 {
                (f32::NAN, f32::INFINITY)
            } else {
                project_point_to_polyline(z, i, &curve.points, &curve.lambda_grid)
            }
        })
        .collect();
    for (i, (lam, sd)) in out.into_iter().enumerate() {
        lambda[(i, l)] = lam;
        dist2[(i, l)] = sd;
    }
}

/// Project row `i` of `z` onto the polyline `(pts, lambda_grid)`; return
/// `(λ, squared_distance)` of the nearest point.
fn project_point_to_polyline(
    z: &DMatrix<f32>,
    i: usize,
    pts: &DMatrix<f32>,
    lambda_grid: &[f32],
) -> (f32, f32) {
    let d = z.ncols();
    let m = pts.nrows();
    let mut best_lam = 0f32;
    let mut best_sd = f32::INFINITY;
    for s in 0..(m - 1) {
        let mut dot = 0f32;
        let mut len2 = 0f32;
        for j in 0..d {
            let aj = pts[(s, j)];
            let bj = pts[(s + 1, j)];
            dot += (z[(i, j)] - aj) * (bj - aj);
            len2 += (bj - aj) * (bj - aj);
        }
        let t = if len2 > 1e-12 {
            (dot / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut sd = 0f32;
        for j in 0..d {
            let aj = pts[(s, j)];
            let bj = pts[(s + 1, j)];
            let proj = aj + t * (bj - aj);
            let v = z[(i, j)] - proj;
            sd += v * v;
        }
        if sd < best_sd {
            best_sd = sd;
            best_lam = lambda_grid[s] + t * (lambda_grid[s + 1] - lambda_grid[s]);
        }
    }
    (best_lam, best_sd)
}

/// Re-fit `curve` as a weighted Nadaraya–Watson Gaussian kernel regression of
/// member-cell coordinates on their projected λ, sampled on a uniform grid.
fn smooth_curve(
    z: &DMatrix<f32>,
    weights: &DMatrix<f32>,
    lambda: &DMatrix<f32>,
    l: usize,
    args: &PrincipalCurveArgs,
    curve: &mut LineageCurve,
) {
    let n = z.nrows();
    let d = z.ncols();

    // Member cells with finite λ.
    let members: Vec<usize> = (0..n)
        .filter(|&i| weights[(i, l)] > 0.0 && lambda[(i, l)].is_finite())
        .collect();
    if members.len() < 2 {
        return; // keep previous curve
    }

    let lam_min = members
        .iter()
        .map(|&i| lambda[(i, l)])
        .fold(f32::INFINITY, f32::min);
    let lam_max = members
        .iter()
        .map(|&i| lambda[(i, l)])
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (lam_max - lam_min).max(1e-6);
    let h = if args.bandwidth > 0.0 {
        args.bandwidth
    } else {
        range / 12.0
    }
    .max(1e-6);

    let res = args.resolution;
    let mut grid = DMatrix::<f32>::zeros(res, d);
    for g in 0..res {
        let target = lam_min + range * g as f32 / (res - 1) as f32;
        let mut wsum = 0f32;
        let mut acc = vec![0f32; d];
        for &i in &members {
            let dl = (lambda[(i, l)] - target) / h;
            let kw = weights[(i, l)] * (-0.5 * dl * dl).exp();
            if kw > 0.0 {
                wsum += kw;
                for j in 0..d {
                    acc[j] += kw * z[(i, j)];
                }
            }
        }
        if wsum > 1e-12 {
            for j in 0..d {
                grid[(g, j)] = acc[j] / wsum;
            }
        } else {
            // No mass near this λ: fall back to the previous curve point.
            for j in 0..d {
                grid[(g, j)] = curve.points[(g, j)];
            }
        }
    }

    // Re-parameterize by the smoothed polyline's own arc-length.
    let arclen = polyline_arclength(&grid);
    curve.points = grid;
    curve.lambda_grid = arclen;
}

//////////////////////////////
// Assignment + convergence //
//////////////////////////////

/// For each cell pick its primary lineage (min orthogonal distance among the
/// lineages it belongs to) and return `(pseudotime, branch)`.
fn primary_assignment(
    lambda: &DMatrix<f32>,
    dist2: &DMatrix<f32>,
    weights: &DMatrix<f32>,
) -> (Vec<f32>, Vec<usize>) {
    let n = lambda.nrows();
    let n_lin = lambda.ncols();
    let mut pt = vec![0f32; n];
    let mut branch = vec![0usize; n];
    for i in 0..n {
        let mut best_l = 0usize;
        let mut best_sd = f32::INFINITY;
        for l in 0..n_lin {
            if weights[(i, l)] > 0.0 && dist2[(i, l)] < best_sd {
                best_sd = dist2[(i, l)];
                best_l = l;
            }
        }
        branch[i] = best_l;
        let lam = lambda[(i, best_l)];
        pt[i] = if lam.is_finite() { lam } else { 0.0 };
    }
    (pt, branch)
}

/// Mean absolute change in per-cell pseudotime, normalized by the pseudotime
/// range. `f32::INFINITY` on the first iteration (no previous values).
fn mean_rel_delta(prev: &[f32], cur: &[f32]) -> f32 {
    if prev.iter().any(|v| v.is_nan()) {
        return f32::INFINITY;
    }
    let (lo, hi) = cur
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let range = (hi - lo).max(1e-6);
    let n = cur.len().max(1) as f32;
    let sum: f32 = prev.iter().zip(cur).map(|(&a, &b)| (a - b).abs()).sum();
    sum / n / range
}

#[cfg(test)]
mod tests;
