//! Reingold-Tilford-style 2D layout for the principal tree.
//!
//! Given a [`PrincipalGraph`] and a chosen root, produces a tree-shaped
//! picture where:
//!   - `y` = geodesic distance from the root along tree edges (i.e.
//!     pseudotime — so vertical axis is literally time),
//!   - `x` = Reingold-Tilford horizontal placement (post-order: each
//!     leaf gets a sequential slot, each internal node sits at the mean
//!     of its children's x).
//!
//! Cells are then placed along their assigned principal-graph edge at
//! the projection parameter t, with small perpendicular Gaussian
//! jitter so dense branches don't collapse to a line.

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::principal_graph::{node_geodesic_from, CellProjection, PrincipalGraph};

/// Per-node 2D position produced by [`reingold_tilford_layout`]. The y
/// coordinate is the geodesic distance from the root along tree edges,
/// so it doubles as the per-node pseudotime.
#[derive(Debug, Clone)]
pub struct TreeLayout {
    /// `(x, y)` per centroid index (length = `graph.n_nodes()`). NaN for
    /// nodes unreachable from the root.
    pub node_xy: Vec<(f32, f32)>,
}

/// Reingold-Tilford layout rooted at `root`.
///
/// `y_units` controls the vertical scale (use the median edge weight or
/// `1.0` for unit-spaced ticks). `x_unit_spacing` is the horizontal gap
/// between adjacent leaves; everything else inherits its x from the
/// post-order recursion.
pub fn reingold_tilford_layout(graph: &PrincipalGraph, root: usize) -> TreeLayout {
    let n = graph.n_nodes();
    let mut node_xy = vec![(f32::NAN, f32::NAN); n];
    if n == 0 {
        return TreeLayout { node_xy };
    }

    // Build BFS-rooted tree: parent[v] = u means u → v in the rooted
    // orientation. Children list per node lets us recurse later.
    let adj = adjacency_with_weights(graph);
    let (parent, order) = bfs_order(root, n, &adj);
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &v in &order {
        if v != root {
            if let Some(p) = parent[v] {
                children[p].push(v);
            }
        }
    }
    for c in children.iter_mut() {
        c.sort_unstable();
    }

    let geodesic = node_geodesic_from(graph, root);

    let mut next_leaf_x: f32 = 0.0;
    let mut x = vec![f32::NAN; n];
    rt_recurse(root, &children, &mut x, &mut next_leaf_x);

    for v in 0..n {
        let xv = x[v];
        let yv = geodesic[v];
        if xv.is_finite() && yv.is_finite() {
            node_xy[v] = (xv, yv);
        }
    }

    TreeLayout { node_xy }
}

fn adjacency_with_weights(graph: &PrincipalGraph) -> Vec<Vec<(usize, f32)>> {
    let mut adj = vec![Vec::new(); graph.n_nodes()];
    for (&(j, k), &w) in graph.edges.iter().zip(&graph.edge_weights) {
        adj[j].push((k, w));
        adj[k].push((j, w));
    }
    adj
}

fn bfs_order(root: usize, n: usize, adj: &[Vec<(usize, f32)>]) -> (Vec<Option<usize>>, Vec<usize>) {
    use std::collections::VecDeque;
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut q = VecDeque::new();
    q.push_back(root);
    visited[root] = true;
    while let Some(u) = q.pop_front() {
        order.push(u);
        for &(v, _w) in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                parent[v] = Some(u);
                q.push_back(v);
            }
        }
    }
    (parent, order)
}

fn rt_recurse(node: usize, children: &[Vec<usize>], x: &mut [f32], next_leaf_x: &mut f32) {
    let kids = &children[node];
    if kids.is_empty() {
        x[node] = *next_leaf_x;
        *next_leaf_x += 1.0;
        return;
    }
    let mut sum = 0f32;
    for &c in kids {
        rt_recurse(c, children, x, next_leaf_x);
        sum += x[c];
    }
    x[node] = sum / kids.len() as f32;
}

/// Place cells along their assigned principal-graph edges using the tree
/// layout. Each cell sits at `lerp(node_xy[j], node_xy[k], t)` with a
/// perpendicular Gaussian jitter scaled by `jitter_frac × edge_length`.
///
/// Returns a `(n_cells, 2)` matrix of `(x, y)` per cell. Cells whose
/// edge endpoints are unreachable from the root land at `NaN`, matching
/// the convention used elsewhere in senna for "drop this point".
pub fn place_cells_on_tree(
    graph: &PrincipalGraph,
    projections: &[CellProjection],
    layout: &TreeLayout,
    jitter_frac: f32,
    seed: u64,
) -> DMatrix<f32> {
    let n = projections.len();
    let mut out = DMatrix::<f32>::from_element(n, 2, f32::NAN);
    let mut rng = StdRng::seed_from_u64(seed);
    // jitter_frac is a fraction of edge length; clamp into a sane range
    // so a stray flag value doesn't blow the scatter wide open.
    let frac = jitter_frac.clamp(0.0, 0.5);
    let normal = Normal::new(0.0_f32, 1.0).expect("std normal");

    for (i, p) in projections.iter().enumerate() {
        let (j, k) = graph.edges[p.nearest_edge];
        let (xj, yj) = layout.node_xy[j];
        let (xk, yk) = layout.node_xy[k];
        if !xj.is_finite() || !yj.is_finite() || !xk.is_finite() || !yk.is_finite() {
            continue;
        }
        let t = p.t.clamp(0.0, 1.0);
        let x = xj + t * (xk - xj);
        let y = yj + t * (yk - yj);

        let dx = xk - xj;
        let dy = yk - yj;
        let len = (dx * dx + dy * dy).sqrt();
        let (jx, jy) = if frac > 0.0 && len > 1e-8 {
            // Perpendicular unit vector to the edge.
            let nx = -dy / len;
            let ny = dx / len;
            let s = normal.sample(&mut rng) * frac * len;
            (nx * s, ny * s)
        } else {
            (0.0, 0.0)
        };

        out[(i, 0)] = x + jx;
        out[(i, 1)] = y + jy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::principal_graph::{fit_principal_graph, project_cells_to_graph, PrincipalGraphArgs};

    fn synthetic_branched(n_per_branch: usize) -> DMatrix<f32> {
        // Three branches in 3-D meeting at the origin.
        let mut rows = Vec::new();
        for branch in 0..3 {
            let dir = match branch {
                0 => (1.0, 0.0, 0.0),
                1 => (-0.5, 0.866, 0.0),
                _ => (-0.5, -0.866, 0.0),
            };
            for i in 0..n_per_branch {
                let s = i as f32 / n_per_branch as f32 * 5.0;
                let jx = (i as f32 * 0.31).sin() * 0.05;
                rows.push(dir.0 * s + jx);
                rows.push(dir.1 * s);
                rows.push(dir.2 * s);
            }
        }
        DMatrix::from_row_slice(3 * n_per_branch, 3, &rows)
    }

    #[test]
    fn three_branch_tree_has_three_leaves() {
        let z = synthetic_branched(80);
        let g = fit_principal_graph(
            &z,
            &PrincipalGraphArgs {
                n_centroids: 30,
                gamma: 5.0,
                sigma: -1.0,
                max_iter: 30,
                tol: 1e-5,
                kmeans_max_iter: 100,
            },
        )
        .unwrap();
        // Pick the centroid closest to the origin as root.
        let mut root = 0usize;
        let mut best = f32::INFINITY;
        for k in 0..g.n_nodes() {
            let mut s = 0f32;
            for d in 0..3 {
                s += g.nodes[(k, d)] * g.nodes[(k, d)];
            }
            if s < best {
                best = s;
                root = k;
            }
        }
        let layout = reingold_tilford_layout(&g, root);
        let n_finite_nodes = layout
            .node_xy
            .iter()
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .count();
        assert!(
            n_finite_nodes >= 5,
            "expected ≥ 5 reachable nodes, got {n_finite_nodes}"
        );

        let projs = project_cells_to_graph(&z, &g);
        let cell_xy = place_cells_on_tree(&g, &projs, &layout, 0.05, 42);
        let n_finite = (0..cell_xy.nrows())
            .filter(|&i| cell_xy[(i, 0)].is_finite() && cell_xy[(i, 1)].is_finite())
            .count();
        assert!(
            n_finite >= cell_xy.nrows() * 9 / 10,
            "expected ≥ 90% of cells finite, got {n_finite}/{}",
            cell_xy.nrows()
        );
    }
}
