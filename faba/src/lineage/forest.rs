//! Per-tree principal-curve fitting over the velocity-informed rooted forest.
//!
//! Given the [`Branching`] (rooted forest over the centroids) and the per-edge
//! [`EdgeDirection`] calls, [`fit_forest_curves`] fits Slingshot curves independently per
//! tree (so each tree's pseudotime resets at its own root), merges them into a single
//! [`PrincipalCurves`] with global lineage/node ids, and computes the per-cell forest tree
//! id and root→node order confidence.

use anyhow::Result;
use matrix_util::branching::Branching;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::principal_curve::{
    fit_principal_curves, LineageCurve, PrincipalCurveArgs, PrincipalCurves,
};
use std::collections::{HashMap, HashSet};

use crate::lineage::orient::{undirected, EdgeCall, EdgeDirection};

/// Per-tree principal-curve fit over the rooted forest, plus the per-cell trust signals.
pub(crate) struct ForestFit {
    /// Curves + per-cell pseudotime/branch/weights merged across all trees.
    pub(crate) curves: PrincipalCurves,
    /// Per-cell forest tree id (`usize::MAX` if the cell's node is out of range).
    pub(crate) cell_tree: Vec<usize>,
    /// Per-cell order confidence = min edge confidence on the root→node path.
    pub(crate) order_conf: Vec<f32>,
}

/// The called direction `(from, to)` of an edge, or `None` if abstained.
fn called_direction(d: &EdgeDirection) -> Option<(usize, usize)> {
    match d.call {
        EdgeCall::Forward => Some((d.edge.0, d.edge.1)),
        EdgeCall::Reverse => Some((d.edge.1, d.edge.0)),
        EdgeCall::Abstain => None,
    }
}

/// Per-node order confidence: min edge confidence along the root→node path, where an edge
/// contributes its confidence only if its *called* direction agrees with the rooted
/// (parent→child) direction; abstained or contradicted edges contribute 0. Roots score 1.
fn node_order_confidence(
    br: &Branching,
    dirs_map: &HashMap<(usize, usize), &EdgeDirection>,
    k: usize,
) -> Vec<f32> {
    let mut conf = vec![f32::NAN; k];
    for v in 0..k {
        // Walk to the root, collecting the path, then fill min-confidence downward.
        let mut path = Vec::new();
        let mut x = v;
        while conf[x].is_nan() {
            match br.parent[x] {
                None => {
                    conf[x] = 1.0;
                    break;
                }
                Some(p) => {
                    path.push((x, p));
                    x = p;
                }
            }
        }
        // conf[x] is now known; propagate along the path (deepest last).
        for &(child, parent) in path.iter().rev() {
            let key = undirected(parent, child);
            let ec = match dirs_map.get(&key).and_then(|d| called_direction(d)) {
                Some((f, t)) if (f, t) == (parent, child) => dirs_map[&key].confidence,
                _ => 0.0, // abstained, contradicted, or missing → no asserted order
            };
            conf[child] = conf[parent].min(ec);
        }
    }
    conf
}

/// Nodes of one component ordered breadth-first from its root (parent→children).
fn bfs_order(br: &Branching, nodes: &[usize], root: usize) -> Vec<usize> {
    let inset: HashSet<usize> = nodes.iter().copied().collect();
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for &v in nodes {
        if let Some(p) = br.parent[v] {
            if inset.contains(&p) {
                children.entry(p).or_default().push(v);
            }
        }
    }
    let mut order = Vec::with_capacity(nodes.len());
    let mut queue = std::collections::VecDeque::from([root]);
    let mut seen: HashSet<usize> = HashSet::from([root]);
    while let Some(v) = queue.pop_front() {
        order.push(v);
        if let Some(cs) = children.get(&v) {
            for &c in cs {
                if seen.insert(c) {
                    queue.push_back(c);
                }
            }
        }
    }
    // include any stragglers (disconnected via abstained-only paths shouldn't occur)
    for &v in nodes {
        if seen.insert(v) {
            order.push(v);
        }
    }
    order
}

/// Fit principal curves independently per forest tree so each tree's pseudotime resets at
/// its own root, then merge the results into a single [`PrincipalCurves`] (global lineage
/// ids, global node ids) alongside the per-cell tree id and order confidence.
pub(crate) fn fit_forest_curves(
    theta: &DMatrix<f32>,
    centroids: &DMatrix<f32>,
    labels: &[usize],
    br: &Branching,
    dirs_map: &HashMap<(usize, usize), &EdgeDirection>,
    args: &PrincipalCurveArgs,
) -> Result<ForestFit> {
    let k = centroids.nrows();
    let n = theta.nrows();
    let d = centroids.ncols();
    let n_comp = br.roots.len();

    let node_conf = node_order_confidence(br, dirs_map, k);
    let cell_tree: Vec<usize> = labels
        .iter()
        .map(|&l| if l < k { br.tree[l] } else { usize::MAX })
        .collect();
    let order_conf: Vec<f32> = labels
        .iter()
        .map(|&l| if l < k { node_conf[l] } else { f32::NAN })
        .collect();

    let mut comp_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for v in 0..k {
        comp_nodes[br.tree[v]].push(v);
    }
    let mut comp_cells: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for (i, &l) in labels.iter().enumerate() {
        if l < k {
            comp_cells[br.tree[l]].push(i);
        }
    }

    let mut all_curves: Vec<LineageCurve> = Vec::new();
    let mut pseudotime = vec![f32::NAN; n];
    let mut branch = vec![0usize; n];
    let mut cluster = vec![0usize; n];
    // Per-component (component id, offset, l_c, weights, lineage_pt) stashed to fill the N×L
    // blocks once the total lineage count is known; cells are re-read from `comp_cells`.
    struct Stash {
        comp: usize,
        offset: usize,
        l_c: usize,
        w: DMatrix<f32>,
        lp: DMatrix<f32>,
    }
    let mut stashes: Vec<Stash> = Vec::new();
    let mut total_l = 0usize;

    for c in 0..n_comp {
        let nodes = &comp_nodes[c];
        let cells = &comp_cells[c];
        let root_g = br.roots[c];
        let offset = total_l;
        let local: HashMap<usize, usize> =
            nodes.iter().enumerate().map(|(li, &g)| (g, li)).collect();

        if nodes.len() >= 2 && cells.len() >= 2 {
            let mut csub = DMatrix::<f32>::zeros(nodes.len(), d);
            for (li, &g) in nodes.iter().enumerate() {
                for j in 0..d {
                    csub[(li, j)] = centroids[(g, j)];
                }
            }
            let mut tsub = DMatrix::<f32>::zeros(cells.len(), d);
            for (ci, &cell) in cells.iter().enumerate() {
                for j in 0..d {
                    tsub[(ci, j)] = theta[(cell, j)];
                }
            }
            let esub: Vec<(usize, usize)> = nodes
                .iter()
                .filter_map(|&v| br.parent[v].map(|p| (local[&p], local[&v])))
                .collect();
            let sub = fit_principal_curves(&tsub, &csub, &esub, local[&root_g], args)?;
            let l_c = sub.n_lineages();
            for cur in &sub.curves {
                let node_path: Vec<usize> = cur.node_path.iter().map(|&ln| nodes[ln]).collect();
                all_curves.push(LineageCurve {
                    node_path,
                    points: cur.points.clone(),
                    lambda_grid: cur.lambda_grid.clone(),
                });
            }
            for (ci, &cell) in cells.iter().enumerate() {
                pseudotime[cell] = sub.pseudotime[ci];
                branch[cell] = offset + sub.branch[ci];
                cluster[cell] = nodes[sub.cluster[ci]];
            }
            stashes.push(Stash {
                comp: c,
                offset,
                l_c,
                w: sub.weights,
                lp: sub.lineage_pseudotime,
            });
            total_l += l_c;
        } else {
            // Trivial tree (single node, or too few cells to fit a curve).
            let node_path = bfs_order(br, nodes, root_g);
            let rows = node_path.len().max(1);
            let mut pts = DMatrix::<f32>::zeros(rows, d);
            for (i, &g) in node_path.iter().enumerate() {
                for j in 0..d {
                    pts[(i, j)] = centroids[(g, j)];
                }
            }
            let lam: Vec<f32> = (0..rows).map(|i| i as f32).collect();
            all_curves.push(LineageCurve {
                node_path,
                points: pts,
                lambda_grid: lam,
            });
            for &cell in cells {
                pseudotime[cell] = f32::NAN;
                branch[cell] = offset;
                cluster[cell] = labels[cell];
            }
            stashes.push(Stash {
                comp: c,
                offset,
                l_c: 1,
                w: DMatrix::from_element(cells.len(), 1, 1.0),
                lp: DMatrix::from_element(cells.len(), 1, f32::NAN),
            });
            total_l += 1;
        }
    }

    let mut weights = DMatrix::<f32>::zeros(n, total_l.max(1));
    let mut lineage_pt = DMatrix::<f32>::from_element(n, total_l.max(1), f32::NAN);
    for st in &stashes {
        for (ci, &cell) in comp_cells[st.comp].iter().enumerate() {
            for l in 0..st.l_c {
                weights[(cell, st.offset + l)] = st.w[(ci, l)];
                lineage_pt[(cell, st.offset + l)] = st.lp[(ci, l)];
            }
        }
    }

    Ok(ForestFit {
        curves: PrincipalCurves {
            curves: all_curves,
            cluster,
            weights,
            lineage_pseudotime: lineage_pt,
            pseudotime,
            branch,
            n_iters: 0,
        },
        cell_tree,
        order_conf,
    })
}
