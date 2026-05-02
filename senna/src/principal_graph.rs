//! Principal graph fitting (SimplePPT) on a low-dimensional embedding.
//!
//! Faithful port of Mao et al. 2015, "SimplePPT: A Simple Principal Tree
//! Algorithm" — the same fitter Monocle 3 uses inside `learn_graph()` once
//! cells have been embedded in a low-dim space (UMAP for Monocle 3,
//! topic θ / SVD components for senna).
//!
//! The objective is
//!
//!   L(Y, R) = Σ_n Σ_k r_nk ‖z_n − y_k‖²
//!           + σ Σ_n Σ_k r_nk log r_nk
//!           + (γ/2) Σ_(j,k)∈E(Y) ‖y_j − y_k‖²
//!
//! and is solved by alternating soft-assignment, MST recomputation, and
//! a Laplacian-regularized linear solve `(diag(R^T 1) + γL) Y = R^T Z`.

use nalgebra::DMatrix;
use petgraph::algo::{dijkstra, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;

use matrix_util::clustering::{Kmeans, KmeansArgs};

/// Configuration for [`fit_principal_graph`].
#[derive(Debug, Clone)]
pub struct PrincipalGraphArgs {
    /// Number of centroids K (graph nodes). Monocle 3 default ≈ 200.
    pub n_centroids: usize,
    /// Tree-smoothing strength γ. Higher = stiffer / fewer wiggles.
    pub gamma: f32,
    /// Soft-assignment bandwidth σ (in the same units as ‖z‖²).
    /// Set ≤ 0 to use an adaptive σ = mean of per-cell nearest-centroid dist².
    pub sigma: f32,
    /// Maximum outer SimplePPT iterations.
    pub max_iter: usize,
    /// Relative objective change for early stop.
    pub tol: f32,
    /// k-means iterations for centroid initialization.
    pub kmeans_max_iter: usize,
}

impl Default for PrincipalGraphArgs {
    fn default() -> Self {
        Self {
            n_centroids: 200,
            gamma: 10.0,
            sigma: -1.0,
            max_iter: 25,
            tol: 1e-4,
            kmeans_max_iter: 100,
        }
    }
}

/// A fitted principal graph (tree) over the latent space.
#[derive(Debug, Clone)]
pub struct PrincipalGraph {
    /// K × D centroid coordinates in the latent space.
    pub nodes: DMatrix<f32>,
    /// MST edges as `(j, k)` with `j < k`. Length = K − 1 for a connected K-tree.
    pub edges: Vec<(usize, usize)>,
    /// Euclidean edge weights, parallel to `edges`.
    pub edge_weights: Vec<f32>,
    /// Outer iterations consumed.
    pub n_iters: usize,
    /// Final objective value.
    pub final_objective: f32,
}

impl PrincipalGraph {
    pub fn n_nodes(&self) -> usize {
        self.nodes.nrows()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Fit a principal tree to the rows of `z` (cells × D).
pub fn fit_principal_graph(
    z: &DMatrix<f32>,
    args: &PrincipalGraphArgs,
) -> anyhow::Result<PrincipalGraph> {
    anyhow::ensure!(args.n_centroids >= 2, "need at least 2 centroids");
    anyhow::ensure!(
        z.nrows() >= args.n_centroids,
        "fewer cells ({}) than requested centroids ({})",
        z.nrows(),
        args.n_centroids
    );
    anyhow::ensure!(args.gamma >= 0.0, "gamma must be ≥ 0");

    let k = args.n_centroids;
    let d = z.ncols();

    let mut y = init_centroids(z, k, args.kmeans_max_iter);

    let mut prev_obj = f32::INFINITY;
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut edge_weights: Vec<f32> = Vec::new();
    let mut n_iters = 0usize;

    for iter in 0..args.max_iter {
        let dist_nk = pairwise_sqdist_rows_to_rows(z, &y);

        let sigma = if args.sigma > 0.0 {
            args.sigma
        } else {
            adaptive_sigma(&dist_nk)
        };

        let r_nk = softmin_rows(&dist_nk, sigma);

        let dist_kk = pairwise_sqdist_rows_to_rows(&y, &y);
        let (mst_edges, mst_weights) = mst_from_sqdist(&dist_kk);

        let s = r_nk.row_sum();
        let mut sys = laplacian(k, &mst_edges) * args.gamma;
        for i in 0..k {
            sys[(i, i)] += s[i] + 1e-6;
        }
        let rhs = r_nk.transpose() * z;
        y = solve_spd(&sys, &rhs)?;

        let obj = objective(&dist_nk, &r_nk, &y, &mst_edges, sigma, args.gamma);
        n_iters = iter + 1;
        edges = mst_edges;
        edge_weights = mst_weights;

        let denom = prev_obj.abs().max(1.0);
        let rel = (prev_obj - obj).abs() / denom;
        log::debug!("SimplePPT iter {iter}: obj={obj:.4} (Δrel={rel:.2e}, σ={sigma:.4})");
        if rel < args.tol {
            prev_obj = obj;
            break;
        }
        prev_obj = obj;
    }

    debug_assert_eq!(y.nrows(), k);
    debug_assert_eq!(y.ncols(), d);

    Ok(PrincipalGraph {
        nodes: y,
        edges,
        edge_weights,
        n_iters,
        final_objective: prev_obj,
    })
}

fn init_centroids(z: &DMatrix<f32>, k: usize, max_iter: usize) -> DMatrix<f32> {
    let labels = z.kmeans_rows(KmeansArgs {
        num_clusters: k,
        max_iter,
    });
    let d = z.ncols();
    let mut centroids = DMatrix::<f32>::zeros(k, d);
    let mut counts = vec![0usize; k];
    for (i, &c) in labels.iter().enumerate() {
        if c < k {
            for j in 0..d {
                centroids[(c, j)] += z[(i, j)];
            }
            counts[c] += 1;
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..d {
                centroids[(c, j)] /= counts[c] as f32;
            }
        } else {
            // Empty cluster: re-seed from a random row to keep K non-degenerate.
            let src = c % z.nrows();
            for j in 0..d {
                centroids[(c, j)] = z[(src, j)];
            }
        }
    }
    centroids
}

/// `(N×D, K×D) → N×K` matrix of squared Euclidean distances. Fills a
/// row-major flat buffer in parallel via `par_chunks_exact_mut` (one
/// alloc) before handing it to `DMatrix::from_row_slice`.
fn pairwise_sqdist_rows_to_rows(a: &DMatrix<f32>, b: &DMatrix<f32>) -> DMatrix<f32> {
    let n = a.nrows();
    let k = b.nrows();
    let d = a.ncols();
    debug_assert_eq!(b.ncols(), d);

    let mut buf = vec![0f32; n * k];
    buf.par_chunks_exact_mut(k)
        .enumerate()
        .for_each(|(i, row)| {
            for kk in 0..k {
                let mut s = 0f32;
                for j in 0..d {
                    let v = a[(i, j)] - b[(kk, j)];
                    s += v * v;
                }
                row[kk] = s;
            }
        });
    DMatrix::from_row_slice(n, k, &buf)
}

fn adaptive_sigma(dist_nk: &DMatrix<f32>) -> f32 {
    let n = dist_nk.nrows();
    if n == 0 {
        return 1.0;
    }
    let sum: f32 = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut m = f32::INFINITY;
            for k in 0..dist_nk.ncols() {
                if dist_nk[(i, k)] < m {
                    m = dist_nk[(i, k)];
                }
            }
            m
        })
        .sum();
    let mean_min = sum / n as f32;
    mean_min.max(1e-8)
}

/// Row-wise softmax of `−d/σ` with subtraction of per-row min for
/// numerical stability.
fn softmin_rows(d: &DMatrix<f32>, sigma: f32) -> DMatrix<f32> {
    let n = d.nrows();
    let k = d.ncols();
    let mut buf = vec![0f32; n * k];
    buf.par_chunks_exact_mut(k)
        .enumerate()
        .for_each(|(i, row)| {
            let mut min_d = f32::INFINITY;
            for kk in 0..k {
                if d[(i, kk)] < min_d {
                    min_d = d[(i, kk)];
                }
            }
            let mut zsum = 0f32;
            for kk in 0..k {
                let v = (-(d[(i, kk)] - min_d) / sigma).exp();
                row[kk] = v;
                zsum += v;
            }
            if zsum > 0.0 {
                for v in row.iter_mut() {
                    *v /= zsum;
                }
            } else {
                let u = 1.0 / k as f32;
                for v in row.iter_mut() {
                    *v = u;
                }
            }
        });
    DMatrix::from_row_slice(n, k, &buf)
}

/// MST over the K centroids using petgraph's `min_spanning_tree`. Edge
/// weights for ranking are the squared distances in `dist_kk`; the
/// returned weights are the Euclidean (sqrt) distances so downstream
/// geodesic distances are in latent-space units.
fn mst_from_sqdist(dist_kk: &DMatrix<f32>) -> (Vec<(usize, usize)>, Vec<f32>) {
    let k = dist_kk.nrows();
    if k <= 1 {
        return (Vec::new(), Vec::new());
    }
    // Build a complete undirected graph on K nodes; petgraph runs
    // Kruskal's MST internally on PartialOrd edge weights.
    let mut g: UnGraph<(), f32> = UnGraph::with_capacity(k, k * (k - 1) / 2);
    let nodes: Vec<NodeIndex> = (0..k).map(|_| g.add_node(())).collect();
    for a in 0..k {
        for b in (a + 1)..k {
            g.add_edge(nodes[a], nodes[b], dist_kk[(a, b)].max(0.0));
        }
    }
    let mst: UnGraph<(), f32> = UnGraph::from_elements(min_spanning_tree(&g));

    let mut edges = Vec::with_capacity(k - 1);
    let mut weights = Vec::with_capacity(k - 1);
    for e in mst.edge_references() {
        let a = e.source().index();
        let b = e.target().index();
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        edges.push((lo, hi));
        weights.push(e.weight().max(0.0).sqrt());
    }
    (edges, weights)
}

fn laplacian(k: usize, edges: &[(usize, usize)]) -> DMatrix<f32> {
    let mut l = DMatrix::<f32>::zeros(k, k);
    for &(a, b) in edges {
        l[(a, a)] += 1.0;
        l[(b, b)] += 1.0;
        l[(a, b)] -= 1.0;
        l[(b, a)] -= 1.0;
    }
    l
}

fn solve_spd(a: &DMatrix<f32>, b: &DMatrix<f32>) -> anyhow::Result<DMatrix<f32>> {
    let chol = a
        .clone()
        .cholesky()
        .ok_or_else(|| anyhow::anyhow!("Cholesky failed on principal-graph M-step system"))?;
    Ok(chol.solve(b))
}

fn objective(
    dist_nk: &DMatrix<f32>,
    r_nk: &DMatrix<f32>,
    y: &DMatrix<f32>,
    edges: &[(usize, usize)],
    sigma: f32,
    gamma: f32,
) -> f32 {
    let n = dist_nk.nrows();
    let k = dist_nk.ncols();
    let (data_term, entropy) = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut d_i = 0f32;
            let mut e_i = 0f32;
            for kk in 0..k {
                let r = r_nk[(i, kk)];
                d_i += r * dist_nk[(i, kk)];
                if r > 1e-12 {
                    e_i += r * r.ln();
                }
            }
            (d_i, e_i)
        })
        .reduce(|| (0f32, 0f32), |a, b| (a.0 + b.0, a.1 + b.1));
    let mut tree_term = 0f32;
    for &(a, b) in edges {
        let mut s = 0f32;
        for j in 0..y.ncols() {
            let v = y[(a, j)] - y[(b, j)];
            s += v * v;
        }
        tree_term += s;
    }
    data_term + sigma * entropy + 0.5 * gamma * tree_term
}

////////////////////////////////////////////////////////////////////////
// Cell projection + geodesic pseudotime
////////////////////////////////////////////////////////////////////////

/// Per-cell projection onto the principal graph.
#[derive(Debug, Clone, Copy)]
pub struct CellProjection {
    /// Index into `graph.edges` of the closest edge.
    pub nearest_edge: usize,
    /// Parameter t ∈ [0, 1] along that edge from `(j → k)`.
    pub t: f32,
    /// Squared distance from cell to its projected point.
    pub sqdist: f32,
}

/// Project each row of `z` to its nearest point on the principal graph.
pub fn project_cells_to_graph(z: &DMatrix<f32>, graph: &PrincipalGraph) -> Vec<CellProjection> {
    let n = z.nrows();
    let d = z.ncols();
    let nodes = &graph.nodes;
    let edges = &graph.edges;

    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut best = CellProjection {
                nearest_edge: 0,
                t: 0.0,
                sqdist: f32::INFINITY,
            };
            for (eidx, &(j, k)) in edges.iter().enumerate() {
                let mut dot = 0f32;
                let mut len2 = 0f32;
                for dd in 0..d {
                    let yj = nodes[(j, dd)];
                    let yk = nodes[(k, dd)];
                    let zi = z[(i, dd)];
                    dot += (zi - yj) * (yk - yj);
                    len2 += (yk - yj) * (yk - yj);
                }
                let t = if len2 > 1e-12 {
                    (dot / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let mut sd = 0f32;
                for dd in 0..d {
                    let yj = nodes[(j, dd)];
                    let yk = nodes[(k, dd)];
                    let proj = yj + t * (yk - yj);
                    let v = z[(i, dd)] - proj;
                    sd += v * v;
                }
                if sd < best.sqdist {
                    best = CellProjection {
                        nearest_edge: eidx,
                        t,
                        sqdist: sd,
                    };
                }
            }
            best
        })
        .collect()
}

/// Build a petgraph view of the principal tree (nodes = centroid index,
/// edges = MST with Euclidean weights). The graph is reconstructed lazily
/// because it's tiny (K ≈ 200) and avoids forcing PrincipalGraph itself
/// to carry a non-Clone graph type.
fn build_petgraph(graph: &PrincipalGraph) -> (UnGraph<(), f32>, Vec<NodeIndex>) {
    let k = graph.n_nodes();
    let mut g: UnGraph<(), f32> = UnGraph::with_capacity(k, graph.n_edges());
    let nodes: Vec<NodeIndex> = (0..k).map(|_| g.add_node(())).collect();
    for (&(a, b), &w) in graph.edges.iter().zip(&graph.edge_weights) {
        g.add_edge(nodes[a], nodes[b], w);
    }
    (g, nodes)
}

/// Geodesic distances from `root` to every node of the principal graph,
/// computed via `petgraph::algo::dijkstra`.
pub fn node_geodesic_from(graph: &PrincipalGraph, root: usize) -> Vec<f32> {
    let (g, nodes) = build_petgraph(graph);
    let dist_map = dijkstra(&g, nodes[root], None, |e| *e.weight());
    let mut out = vec![f32::INFINITY; graph.n_nodes()];
    for (nid, d) in dist_map {
        out[nid.index()] = d;
    }
    out
}

/// Per-cell pseudotime: geodesic distance from `root_node` to each cell's
/// projection on the principal graph.
pub fn pseudotime_from_root(
    graph: &PrincipalGraph,
    projections: &[CellProjection],
    root_node: usize,
) -> Vec<f32> {
    let node_dist = node_geodesic_from(graph, root_node);
    projections
        .iter()
        .map(|p| {
            let (j, k) = graph.edges[p.nearest_edge];
            let w = graph.edge_weights[p.nearest_edge];
            (node_dist[j] + p.t * w).min(node_dist[k] + (1.0 - p.t) * w)
        })
        .collect()
}

/// Find the centroid index closest to row `row` of `z`.
pub fn closest_node_to_row(z: &DMatrix<f32>, row: usize, graph: &PrincipalGraph) -> usize {
    let d = z.ncols();
    let mut best = 0usize;
    let mut best_sd = f32::INFINITY;
    for k in 0..graph.n_nodes() {
        let mut s = 0f32;
        for dd in 0..d {
            let v = z[(row, dd)] - graph.nodes[(k, dd)];
            s += v * v;
        }
        if s < best_sd {
            best_sd = s;
            best = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
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
}
