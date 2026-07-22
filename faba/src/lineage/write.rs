//! Every table `faba lineage` puts on disk.

use anyhow::{Context, Result};
use log::info;

use matrix_util::branching::Branching;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::principal_curve::PrincipalCurves;
use matrix_util::traits::IoOps;
use matrix_util::utils::median;

use super::layout::*;
use crate::lineage::orient::{undirected, EdgeCall, EdgeDirection};

/// `rows × [x, y]` 2D-coordinate table.
pub(super) fn write_xy(
    mat: &DMatrix<f32>,
    rows: &[Box<str>],
    header: &str,
    path: &str,
) -> Result<()> {
    let cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    mat.to_parquet_with_names(path, (Some(rows), Some(header)), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// Long format `[lineage, grid, x, y]`: projected principal-curve points.
pub(super) fn write_curves_2d(
    coords: &DMatrix<f32>,
    meta: &[(usize, usize)],
    path: &str,
) -> Result<()> {
    let total = coords.nrows();
    let mut mat = DMatrix::<f32>::zeros(total, 4);
    for i in 0..total {
        mat[(i, 0)] = meta[i].0 as f32;
        mat[(i, 1)] = meta[i].1 as f32;
        mat[(i, 2)] = coords[(i, 0)];
        mat[(i, 3)] = coords[(i, 1)];
    }
    let rows = numbered("row_", total);
    let cols: Vec<Box<str>> = vec!["lineage".into(), "grid".into(), "x".into(), "y".into()];
    mat.to_parquet_with_names(path, (Some(&rows), Some("row")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/////////////////////
// Parquet writers //
/////////////////////

/// Contiguous `{prefix}{0..n}` names for parquet row/column headers.
pub(super) fn numbered(prefix: &str, n: usize) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

/// `node_i × T{j}` matrix (centroids or node velocities).
pub(super) fn write_nodes(mat: &DMatrix<f32>, path: &str) -> Result<()> {
    let rows = numbered("node_", mat.nrows());
    let cols = numbered("T", mat.ncols());
    mat.to_parquet_with_names(path, (Some(&rows), Some("node")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// Geometry floor: an abstained edge can still connect via geometry.
pub(super) const BETA: f32 = 0.2;
/// Weight of a velocity-contradicted orientation — near zero so it is never selected.
pub(super) const BETA_LOW: f32 = 1e-3;

/// Build the directed arc set + per-node `root_affinity` for [`max_branching`]. Each
/// candidate edge yields two opposing arcs weighted by geometric affinity × direction
/// support; abstained edges contribute geometry only. A user `root_hint` is pinned as a
/// root via an infinite affinity. `root_affinity_arg` (τ_root) overrides the default
/// (median arc weight) and controls forest granularity.
pub(super) fn assemble_arcs(
    dirs: &[EdgeDirection],
    k: usize,
    root_affinity_arg: Option<f32>,
    root_hint: Option<usize>,
) -> (Vec<(usize, usize, f32)>, Vec<f32>) {
    // σ = median candidate geom_dist (scale of the affinity kernel).
    let d: Vec<f32> = dirs
        .iter()
        .map(|e| e.geom_dist)
        .filter(|x| *x > 0.0)
        .collect();
    let sigma = if d.is_empty() {
        1.0
    } else {
        median(&d).max(1e-6)
    };

    let mut arcs: Vec<(usize, usize, f32)> = Vec::with_capacity(dirs.len() * 2);
    for e in dirs {
        let (a, b) = e.edge;
        let s = (-(e.geom_dist / sigma).powi(2)).exp();
        let strong = s * (BETA + (1.0 - BETA) * e.confidence);
        let weak = s * BETA_LOW;
        let floor = s * BETA;
        match e.call {
            EdgeCall::Forward => {
                arcs.push((a, b, strong));
                arcs.push((b, a, weak));
            }
            EdgeCall::Reverse => {
                arcs.push((b, a, strong));
                arcs.push((a, b, weak));
            }
            EdgeCall::Abstain => {
                arcs.push((a, b, floor));
                arcs.push((b, a, floor));
            }
        }
    }

    let tau = root_affinity_arg
        .unwrap_or_else(|| median(&arcs.iter().map(|&(_, _, w)| w).collect::<Vec<_>>()));
    let mut root_affinity = vec![tau; k];
    if let Some(r) = root_hint {
        if r < k {
            root_affinity[r] = f32::INFINITY; // pin r as a root
        }
    }
    (arcs, root_affinity)
}

/// `edge_i × [from, to, geom_dist, velocity_flux, se, ci_lo, ci_hi, p, q, n_cells,
/// confidence, in_mst, selected, directed_from, directed_to, tree]` + `call` (Str).
/// Rows are all candidate edges. `directed_*`/`tree` are `NaN` for edges the branching
/// did not select; `call` is `forward`/`reverse`/`unassigned`.
pub(super) fn write_edge_directions(
    dirs: &[EdgeDirection],
    br: &Branching,
    path: &str,
) -> Result<()> {
    let m = dirs.len();
    let (mut from, mut to) = (vec![0f32; m], vec![0f32; m]);
    let (mut geom, mut flux) = (vec![0f32; m], vec![0f32; m]);
    let (mut se, mut ci_lo, mut ci_hi) = (vec![0f32; m], vec![0f32; m], vec![0f32; m]);
    let (mut p, mut q, mut ncell) = (vec![0f32; m], vec![0f32; m], vec![0f32; m]);
    let (mut conf, mut in_mst) = (vec![0f32; m], vec![0f32; m]);
    let (mut selected, mut dfrom, mut dto, mut tree) =
        (vec![0f32; m], vec![0f32; m], vec![0f32; m], vec![0f32; m]);
    let mut call: Vec<Box<str>> = Vec::with_capacity(m);

    for (i, e) in dirs.iter().enumerate() {
        let (a, b) = e.edge;
        from[i] = a as f32;
        to[i] = b as f32;
        geom[i] = e.geom_dist;
        flux[i] = e.flux;
        se[i] = e.se;
        ci_lo[i] = e.ci_lo;
        ci_hi[i] = e.ci_hi;
        p[i] = e.p;
        q[i] = e.q;
        ncell[i] = e.n_cells as f32;
        conf[i] = e.confidence;
        in_mst[i] = if e.in_mst { 1.0 } else { 0.0 };
        // Selected orientation from the branching (parent → child).
        let (sel, df, dt, tr) = if br.parent[b] == Some(a) {
            (1.0, a as f32, b as f32, br.tree[b] as f32)
        } else if br.parent[a] == Some(b) {
            (1.0, b as f32, a as f32, br.tree[a] as f32)
        } else {
            (0.0, f32::NAN, f32::NAN, f32::NAN)
        };
        selected[i] = sel;
        dfrom[i] = df;
        dto[i] = dt;
        tree[i] = tr;
        call.push(match e.call {
            EdgeCall::Forward => "forward".into(),
            EdgeCall::Reverse => "reverse".into(),
            EdgeCall::Abstain => "unassigned".into(),
        });
    }

    let rows = numbered("edge_", m);
    write_named_table(
        path,
        "edge",
        &rows,
        &[
            (Box::from("from"), Column::F32(&from)),
            (Box::from("to"), Column::F32(&to)),
            (Box::from("geom_dist"), Column::F32(&geom)),
            (Box::from("velocity_flux"), Column::F32(&flux)),
            (Box::from("se"), Column::F32(&se)),
            (Box::from("ci_lo"), Column::F32(&ci_lo)),
            (Box::from("ci_hi"), Column::F32(&ci_hi)),
            (Box::from("p"), Column::F32(&p)),
            (Box::from("q"), Column::F32(&q)),
            (Box::from("n_cells"), Column::F32(&ncell)),
            (Box::from("confidence"), Column::F32(&conf)),
            (Box::from("in_mst"), Column::F32(&in_mst)),
            (Box::from("selected"), Column::F32(&selected)),
            (Box::from("directed_from"), Column::F32(&dfrom)),
            (Box::from("directed_to"), Column::F32(&dto)),
            (Box::from("tree"), Column::F32(&tree)),
            (Box::from("call"), Column::Str(&call)),
        ],
    )
    .with_context(|| format!("writing {path}"))?;
    info!("Wrote {path}");
    Ok(())
}

/// `tree_c × [root, n_nodes, n_cells, mean_confidence]`: one row per forest tree.
pub(super) fn write_trees(
    br: &Branching,
    labels: &[usize],
    dirs_map: &DirsMap,
    path: &str,
) -> Result<()> {
    let k = br.parent.len();
    let n_comp = br.roots.len();

    let mut n_nodes = vec![0f32; n_comp];
    for v in 0..k {
        n_nodes[br.tree[v]] += 1.0;
    }
    let mut n_cells = vec![0f32; n_comp];
    for &l in labels {
        if l < k {
            n_cells[br.tree[l]] += 1.0;
        }
    }
    // Mean confidence of the selected (parent → child) edges within each tree.
    let mut conf_sum = vec![0f32; n_comp];
    let mut conf_cnt = vec![0f32; n_comp];
    for v in 0..k {
        if let Some(u) = br.parent[v] {
            if let Some(d) = dirs_map.get(&undirected(u, v)) {
                conf_sum[br.tree[v]] += d.confidence;
                conf_cnt[br.tree[v]] += 1.0;
            }
        }
    }
    let roots: Vec<f32> = br.roots.iter().map(|&r| r as f32).collect();
    let mean_conf: Vec<f32> = (0..n_comp)
        .map(|c| {
            if conf_cnt[c] > 0.0 {
                conf_sum[c] / conf_cnt[c]
            } else {
                f32::NAN
            }
        })
        .collect();
    let rows = numbered("tree_", n_comp);
    write_named_table(
        path,
        "tree",
        &rows,
        &[
            (Box::from("root"), Column::F32(&roots)),
            (Box::from("n_nodes"), Column::F32(&n_nodes)),
            (Box::from("n_cells"), Column::F32(&n_cells)),
            (Box::from("mean_confidence"), Column::F32(&mean_conf)),
        ],
    )
    .with_context(|| format!("writing {path}"))?;
    info!("Wrote {path} ({n_comp} tree(s))");
    Ok(())
}

/// Long format `[lineage, step, node]`: the ordered node path of each lineage.
pub(super) fn write_lineages(curves: &PrincipalCurves, path: &str) -> Result<()> {
    let total: usize = curves.curves.iter().map(|c| c.node_path.len()).sum();
    let mut mat = DMatrix::<f32>::zeros(total, 3);
    let mut r = 0usize;
    for (l, c) in curves.curves.iter().enumerate() {
        for (step, &node) in c.node_path.iter().enumerate() {
            mat[(r, 0)] = l as f32;
            mat[(r, 1)] = step as f32;
            mat[(r, 2)] = node as f32;
            r += 1;
        }
    }
    let rows = numbered("row_", total);
    let cols: Vec<Box<str>> = vec!["lineage".into(), "step".into(), "node".into()];
    mat.to_parquet_with_names(path, (Some(&rows), Some("row")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// `cell × [pseudotime, branch, tree, order_confidence]`: primary-lineage pseudotime +
/// global lineage id + forest tree id + the min edge confidence on the cell's root→node
/// path (0 where the ordering crosses an abstained/geometry-only edge). `pseudotime` and
/// `branch` stay the first two columns for back-compatibility with `faba dyn-assoc`.
pub(super) fn write_pseudotime(
    curves: &PrincipalCurves,
    cell_tree: &[usize],
    order_conf: &[f32],
    cell_names: &[Box<str>],
    path: &str,
) -> Result<()> {
    let n = curves.pseudotime.len();
    let mut mat = DMatrix::<f32>::zeros(n, 4);
    for i in 0..n {
        mat[(i, 0)] = curves.pseudotime[i];
        mat[(i, 1)] = curves.branch[i] as f32;
        mat[(i, 2)] = if cell_tree[i] == usize::MAX {
            f32::NAN
        } else {
            cell_tree[i] as f32
        };
        mat[(i, 3)] = order_conf[i];
    }
    let cols: Vec<Box<str>> = vec![
        "pseudotime".into(),
        "branch".into(),
        "tree".into(),
        "order_confidence".into(),
    ];
    mat.to_parquet_with_names(path, (Some(cell_names), Some("cell")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// `cell × {col_prefix}_{l}` (per-lineage weights or per-lineage pseudotime).
pub(super) fn write_cell_matrix(
    mat: &DMatrix<f32>,
    cell_names: &[Box<str>],
    col_prefix: &str,
    path: &str,
) -> Result<()> {
    let cols = numbered(&format!("{col_prefix}_"), mat.ncols());
    mat.to_parquet_with_names(path, (Some(cell_names), Some("cell")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// Long format `[lineage, grid, lambda, T0…]`: the smooth curve points.
pub(super) fn write_curves(curves: &PrincipalCurves, path: &str) -> Result<()> {
    let d = curves.curves.first().map_or(0, |c| c.points.ncols());
    let total: usize = curves.curves.iter().map(|c| c.points.nrows()).sum();
    let mut mat = DMatrix::<f32>::zeros(total, 3 + d);
    let mut r = 0usize;
    for (l, c) in curves.curves.iter().enumerate() {
        for g in 0..c.points.nrows() {
            mat[(r, 0)] = l as f32;
            mat[(r, 1)] = g as f32;
            mat[(r, 2)] = c.lambda_grid[g];
            for j in 0..d {
                mat[(r, 3 + j)] = c.points[(g, j)];
            }
            r += 1;
        }
    }
    let rows = numbered("row_", total);
    let mut cols: Vec<Box<str>> = vec!["lineage".into(), "grid".into(), "lambda".into()];
    cols.extend(numbered("T", d));
    mat.to_parquet_with_names(path, (Some(&rows), Some("row")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

#[cfg(test)]
#[path = "write_tests.rs"]
mod write_tests;
