//! Entry point for `faba lineage` — velocity-oriented lineage inference over a
//! `faba gem` embedding.
//!
//! Reads gem's raw parquet outputs by prefix (`{from}.latent.parquet` = θ,
//! `{from}.velocity.parquet` = δ), fits **K k-means centroids** on θ, an **MST**
//! over them ([`matrix_util::principal_graph::mst_from_sqdist`]), **orients** that
//! tree by the per-node mean velocity flux ([`crate::lineage::orient`]), and fits
//! **Slingshot-style principal curves** ([`matrix_util::principal_curve`]) rooted
//! at the velocity source. Outputs per-cell pseudotime + branch, the node graph,
//! and the smooth curves as parquet — the ordering the parked modality-enrichment
//! test will run against.
//!
//! NOTE: the underlying k-means (`matrix_util::…::kmeans_centroids`) is not
//! seeded, so centroid placement — and hence the exact root/lineage count — can
//! vary slightly run-to-run on the same input. There is no `--seed` for this
//! reason; reproducibility would need a seeded k-means in matrix-util.

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use log::{info, warn};
use std::path::Path;

use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::layout::{phate_layout_2d, project_cells_nystrom, PhateArgs};
use matrix_util::principal_curve::{fit_principal_curves, PrincipalCurveArgs, PrincipalCurves};
use matrix_util::principal_graph::{
    kmeans_centroids, mst_from_sqdist, pairwise_sqdist_rows_to_rows,
};
use matrix_util::traits::IoOps;

use crate::lineage::orient::{
    aggregate_node_velocity, directed_edges, edge_velocity_flux, pick_velocity_root,
};

/// 2D layout for plotting the trajectory.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum LayoutKind {
    /// No 2D layout (default).
    #[default]
    None,
    /// PHATE diffusion embedding — the trajectory-appropriate layout that
    /// preserves branch/continuum structure (unlike UMAP/t-SNE).
    Phate,
}

#[derive(Args, Debug)]
pub struct LineageArgs {
    #[arg(
        long,
        short = 'f',
        help = "gem output prefix (reads {from}.latent.parquet and {from}.velocity.parquet)"
    )]
    pub from: Box<str>,

    #[arg(long, short = 'o', help = "Output prefix (default: the gem prefix)")]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        help = "Number of MST node centroids K (default: min(cells / 10, 200))"
    )]
    pub n_centroids: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Gaussian kernel bandwidth in pseudotime units (0 = adaptive per curve)"
    )]
    pub curve_bandwidth: f32,

    #[arg(
        long,
        default_value_t = 100,
        help = "Points sampled along each fitted principal curve"
    )]
    pub curve_resolution: usize,

    #[arg(
        long,
        default_value_t = 15,
        help = "Max project-then-smooth iterations for the curves"
    )]
    pub max_iter: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        help = "Convergence tolerance on mean |Δpseudotime| / range"
    )]
    pub tol: f32,

    #[arg(
        long = "no-orient-velocity",
        help = "Do not orient the MST or pick the root by velocity flux"
    )]
    pub no_orient_velocity: bool,

    #[arg(
        long,
        help = "Force the root MST node by index (overrides velocity orientation)"
    )]
    pub root_node: Option<usize>,

    #[arg(
        long,
        help = "Force the root at the node nearest a named cell (overrides velocity)"
    )]
    pub root_cell: Option<Box<str>>,

    #[arg(
        long = "root-from-gem",
        help = "Anchor the root at gem's inferred root (the min-pseudotime cell in \
                {from}.pseudotime.parquet). Uses gem's velocity-DAG root inference — more \
                robust than the per-edge flux pick — while lineage still fits the curves. \
                Overridden by --root-node / --root-cell; falls back to flux if the file is \
                absent."
    )]
    pub root_from_gem: bool,

    #[arg(
        long,
        default_value_t = 100,
        help = "k-means iterations for centroid initialization"
    )]
    pub kmeans_iter: usize,

    #[arg(
        long = "no-normalize-latent",
        help = "Fit on raw θ instead of L2-normalized (cosine) θ [default: normalize]"
    )]
    pub no_normalize_latent: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = LayoutKind::None,
        help = "2D layout: 'phate' emits {out}.{cells,nodes,curves}_2d.parquet for plotting"
    )]
    pub layout: LayoutKind,

    #[arg(
        long,
        default_value_t = 15,
        help = "PHATE kNN adaptive bandwidth (only with --layout phate)"
    )]
    pub phate_knn: usize,

    #[arg(
        long,
        default_value_t = 20,
        help = "PHATE diffusion time t (only with --layout phate)"
    )]
    pub phate_t: usize,

    #[arg(
        long,
        default_value_t = 2000,
        help = "PHATE landmark budget: above this many cells, PHATE runs on a \
                landmark subsample + Nyström lift (scales linearly). Raise it \
                if the layout looks thin/stringy on very large data."
    )]
    pub phate_landmarks: usize,
}

/// Number of MST node centroids K: explicit `--n-centroids`, else `min(N/10, 200)`,
/// clamped to `[2, N]`.
fn choose_k(n: usize, requested: Option<usize>) -> usize {
    requested.unwrap_or_else(|| (n / 10).clamp(2, 200)).min(n)
}

/// Resolve the root MST node, in priority order: `--root-node` (validated), `--root-cell`
/// (the node of the named cell's cluster), `gem_root` (the centroid of gem's inferred
/// root cell, from `--root-from-gem`), the velocity-flux-picked root, else node 0.
fn resolve_root(
    root_node: Option<usize>,
    root_cell: Option<&str>,
    cell_names: &[Box<str>],
    labels: &[usize],
    k: usize,
    gem_root: Option<usize>,
    velocity_root: Option<usize>,
) -> Result<usize> {
    if let Some(r) = root_node {
        anyhow::ensure!(r < k, "--root-node {r} out of range (K = {k})");
        Ok(r)
    } else if let Some(name) = root_cell {
        let idx = cell_names
            .iter()
            .position(|c| c.as_ref() == name)
            .with_context(|| format!("--root-cell '{name}' not found in latent"))?;
        Ok(labels[idx])
    } else if let Some(r) = gem_root {
        Ok(r)
    } else {
        Ok(velocity_root.unwrap_or(0))
    }
}

/// Map gem's inferred root to an MST centroid (for `--root-from-gem`): read
/// `{prefix}.pseudotime.parquet`, take the barcode with minimum pseudotime (τ ≈ 0, the
/// velocity-DAG source), find it in the latent, and return its cluster label. `None`
/// (with a warning) when the file is absent or the barcode can't be matched — the caller
/// then falls back to the velocity-flux root.
fn gem_root_node(prefix: &str, cell_names: &[Box<str>], labels: &[usize]) -> Option<usize> {
    let path = format!("{prefix}.pseudotime.parquet");
    if !Path::new(&path).exists() {
        warn!("--root-from-gem: {path} absent; falling back to the velocity-flux root");
        return None;
    }
    let pt = match DMatrix::<f32>::from_parquet(&path) {
        Ok(pt) => pt,
        Err(e) => {
            warn!("--root-from-gem: cannot read {path} ({e}); falling back to velocity root");
            return None;
        }
    };
    // Column 0 is `pseudotime` (column 1 is `ambiguity`); take the row-minimum barcode.
    let rmin = (0..pt.mat.nrows()).min_by(|&a, &b| {
        pt.mat[(a, 0)]
            .partial_cmp(&pt.mat[(b, 0)])
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let root_bc = pt.rows.get(rmin)?.as_ref();
    match cell_names.iter().position(|c| c.as_ref() == root_bc) {
        Some(idx) => {
            info!(
                "--root-from-gem: root cell '{root_bc}' (τ≈min) → MST node {}",
                labels[idx]
            );
            Some(labels[idx])
        }
        None => {
            warn!("--root-from-gem: root barcode '{root_bc}' not in latent; using flux root");
            None
        }
    }
}

pub fn run_lineage(args: &LineageArgs) -> Result<()> {
    let prefix = args.from.as_ref();
    let out = args.out.as_deref().unwrap_or(prefix).to_string();
    mkdir_parent(&out)?;

    // ---- load frozen embedding θ ----
    // gem θ is cosine-oriented, so by default the whole fit (k-means → MST →
    // curves) runs on L2-normalized θ; this keeps a few extreme-magnitude cells
    // from dominating and matches the PHATE layout's geometry. `--no-normalize-latent`
    // reverts to the raw-Euclidean fit.
    let latent_path = format!("{prefix}.latent.parquet");
    let cell = DMatrix::<f32>::from_parquet(&latent_path)
        .with_context(|| format!("reading latent embedding {latent_path}"))?;
    let cell_names = cell.rows;
    let theta = if args.no_normalize_latent {
        cell.mat
    } else {
        l2_normalize_rows(&cell.mat)
    };
    let n = theta.nrows();
    anyhow::ensure!(n >= 2, "need ≥ 2 cells, got {n}");

    let k = choose_k(n, args.n_centroids);
    anyhow::ensure!(k >= 2, "need ≥ 2 centroids, got {k}");
    info!(
        "lineage: {n} cells × {} dims → {k} centroids",
        theta.ncols()
    );

    // ---- k-means centroids + MST ----
    let (centroids, labels) = kmeans_centroids(&theta, k, args.kmeans_iter);
    let (edges, weights) = mst_from_sqdist(&pairwise_sqdist_rows_to_rows(&centroids, &centroids));
    anyhow::ensure!(
        edges.len() == k - 1,
        "MST on {k} nodes should have {} edges, got {}",
        k - 1,
        edges.len()
    );

    // ---- velocity orientation (optional) ----
    let velocity_path = format!("{prefix}.velocity.parquet");
    let have_velocity = !args.no_orient_velocity && Path::new(&velocity_path).exists();
    let (node_velocity, flux) = if have_velocity {
        let vel = DMatrix::<f32>::from_parquet(&velocity_path)
            .with_context(|| format!("reading velocity {velocity_path}"))?;
        anyhow::ensure!(
            vel.mat.nrows() == n,
            "velocity rows ({}) != latent rows ({n})",
            vel.mat.nrows()
        );
        let nv = aggregate_node_velocity(&vel.mat, &labels, k);
        let fx = edge_velocity_flux(&centroids, &nv, &edges);
        (nv, fx)
    } else {
        if !args.no_orient_velocity {
            warn!("velocity file {velocity_path} absent; MST left unoriented, root defaults");
        }
        (
            DMatrix::<f32>::zeros(k, theta.ncols()),
            vec![0f32; edges.len()],
        )
    };
    let directed = directed_edges(&edges, &flux);

    // ---- root selection ----
    let velocity_root = have_velocity.then(|| pick_velocity_root(&edges, &flux, k));
    // Optional gem hand-off: anchor the root at gem's velocity-DAG-inferred root
    // (min-pseudotime cell), more robust than the per-edge flux pick.
    let gem_root = args
        .root_from_gem
        .then(|| gem_root_node(prefix, &cell_names, &labels))
        .flatten();
    let root = resolve_root(
        args.root_node,
        args.root_cell.as_deref(),
        &cell_names,
        &labels,
        k,
        gem_root,
        velocity_root,
    )?;
    info!("lineage root node = {root}");

    // ---- Slingshot principal curves ----
    let curves = fit_principal_curves(
        &theta,
        &centroids,
        &edges,
        root,
        &PrincipalCurveArgs {
            max_iter: args.max_iter,
            tol: args.tol,
            resolution: args.curve_resolution,
            bandwidth: args.curve_bandwidth,
        },
    )?;
    info!(
        "fit {} lineage(s) in {} iteration(s)",
        curves.n_lineages(),
        curves.n_iters
    );

    // ---- outputs ----
    write_nodes(&centroids, &format!("{out}.nodes.parquet"))?;
    write_nodes(&node_velocity, &format!("{out}.node_velocity.parquet"))?;
    write_edges(
        &edges,
        &weights,
        &flux,
        &directed,
        &format!("{out}.edges.parquet"),
    )?;
    write_lineages(&curves, &format!("{out}.lineages.parquet"))?;
    write_pseudotime(&curves, &cell_names, &format!("{out}.pseudotime.parquet"))?;
    write_cell_matrix(
        &curves.weights,
        &cell_names,
        "lineage",
        &format!("{out}.cell_lineage_weights.parquet"),
    )?;
    write_cell_matrix(
        &curves.lineage_pseudotime,
        &cell_names,
        "lineage",
        &format!("{out}.lineage_pseudotime.parquet"),
    )?;
    write_curves(&curves, &format!("{out}.curves.parquet"))?;

    // ---- optional PHATE 2D layout (cells + nodes + curves projected) ----
    if args.layout == LayoutKind::Phate {
        let phate = PhateArgs {
            t: args.phate_t,
            knn: args.phate_knn,
            ..PhateArgs::default()
        };
        emit_phate_layout(
            &theta,
            &centroids,
            &curves,
            &cell_names,
            &phate,
            args.phate_landmarks,
            &out,
        )?;
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////
// PHATE 2D layout
////////////////////////////////////////////////////////////////////////

/// Row-wise L2 normalization (unit vectors): Euclidean distance on the result
/// equals cosine distance on the input. Used for both the cosine θ fit and the
/// PHATE layout.
fn l2_normalize_rows(m: &DMatrix<f32>) -> DMatrix<f32> {
    let mut out = m.clone();
    for i in 0..out.nrows() {
        let norm = (0..out.ncols())
            .map(|j| out[(i, j)] * out[(i, j)])
            .sum::<f32>()
            .sqrt();
        // Leave a ~zero row unchanged: normalizing it would blow it up to an
        // arbitrary unit direction (e.g. a centroid of near-antipodal points).
        if norm > 1e-9 {
            for j in 0..out.ncols() {
                out[(i, j)] /= norm;
            }
        }
    }
    out
}

/// Choose the PHATE landmark set and its 2D layout. When `N ≤ n_landmarks` every
/// cell is a landmark (exact PHATE). Above that, PHATE runs on a deterministic
/// stride subsample of `n_landmarks` cells and the rest are lifted with the
/// Nyström projector — capping the O(n³) PHATE work at the landmark budget and
/// making the remainder linear in N. Returns `(landmark_features L×D, coords L×2)`.
fn phate_landmark_layout(
    theta_n: &DMatrix<f32>,
    phate: &PhateArgs,
    n_landmarks: usize,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let n = theta_n.nrows();
    if n <= n_landmarks || n_landmarks < 3 {
        return (theta_n.clone(), phate_layout_2d(theta_n, phate));
    }
    let (l, d) = (n_landmarks, theta_n.ncols());
    // Deterministic, evenly-spread stride subsample (cell order is arbitrary).
    let mut land = DMatrix::<f32>::zeros(l, d);
    for r in 0..l {
        let s = (r * n / l).min(n - 1);
        for j in 0..d {
            land[(r, j)] = theta_n[(s, j)];
        }
    }
    let coords = phate_layout_2d(&land, phate);
    (land, coords)
}

/// Lay the cells out with PHATE (trajectory-preserving), then place the node
/// centroids and principal-curve points into the *same* space via the alpha-decay
/// Nyström projection — so the trajectory overlays faithfully. `project_cells_nystrom`
/// takes points as columns (D × n), hence the transposes.
///
/// The layout runs on **L2-normalized θ (cosine geometry)** applied here
/// regardless of the fit mode — so the layout is always cosine, even under
/// `--no-normalize-latent`. (gem θ is cosine-oriented; a few extreme-magnitude
/// cells otherwise dominate the Euclidean diffusion distances.) For large N,
/// PHATE is run on a landmark subsample and every cell/node/curve point is lifted
/// onto that layout via the same Nyström projector.
fn emit_phate_layout(
    theta: &DMatrix<f32>,
    centroids: &DMatrix<f32>,
    curves: &PrincipalCurves,
    cell_names: &[Box<str>],
    phate: &PhateArgs,
    n_landmarks: usize,
    out: &str,
) -> Result<()> {
    let n = theta.nrows();
    let theta_n = l2_normalize_rows(theta);
    let (land_feat, land_2d) = phate_landmark_layout(&theta_n, phate, n_landmarks);
    let exact = land_feat.nrows() == n;
    info!(
        "PHATE layout: {n} cells ({})",
        if exact {
            "exact".to_string()
        } else {
            format!("{} landmarks + Nyström", land_feat.nrows())
        }
    );
    let land_t = land_feat.transpose(); // D × L
    let (knn, alpha) = (phate.knn, phate.alpha);

    // Cells: exact PHATE already placed them; else lift onto the landmark layout.
    let cells_2d = if exact {
        land_2d.clone()
    } else {
        project_cells_nystrom(&theta_n.transpose(), &land_t, &land_2d, knn, alpha)
    };

    // Nodes + curve points always lift onto the landmark layout via Nyström.
    let nodes_2d = project_cells_nystrom(
        &l2_normalize_rows(centroids).transpose(),
        &land_t,
        &land_2d,
        knn,
        alpha,
    );

    // Stack all lineage curve points (in θ space) + remember (lineage, grid).
    let d = theta.ncols();
    let total: usize = curves.curves.iter().map(|c| c.points.nrows()).sum();
    let mut cpts = DMatrix::<f32>::zeros(total, d);
    let mut meta: Vec<(usize, usize)> = Vec::with_capacity(total);
    let mut r = 0usize;
    for (l, c) in curves.curves.iter().enumerate() {
        for g in 0..c.points.nrows() {
            for j in 0..d {
                cpts[(r, j)] = c.points[(g, j)];
            }
            meta.push((l, g));
            r += 1;
        }
    }
    let curves_2d = project_cells_nystrom(
        &l2_normalize_rows(&cpts).transpose(),
        &land_t,
        &land_2d,
        knn,
        alpha,
    );

    write_xy(
        &cells_2d,
        cell_names,
        "cell",
        &format!("{out}.cells_2d.parquet"),
    )?;
    let node_names = numbered("node_", nodes_2d.nrows());
    write_xy(
        &nodes_2d,
        &node_names,
        "node",
        &format!("{out}.nodes_2d.parquet"),
    )?;
    write_curves_2d(&curves_2d, &meta, &format!("{out}.curves_2d.parquet"))?;
    Ok(())
}

/// `rows × [x, y]` 2D-coordinate table.
fn write_xy(mat: &DMatrix<f32>, rows: &[Box<str>], header: &str, path: &str) -> Result<()> {
    let cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    mat.to_parquet_with_names(path, (Some(rows), Some(header)), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// Long format `[lineage, grid, x, y]`: projected principal-curve points.
fn write_curves_2d(coords: &DMatrix<f32>, meta: &[(usize, usize)], path: &str) -> Result<()> {
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

////////////////////////////////////////////////////////////////////////
// Parquet writers
////////////////////////////////////////////////////////////////////////

/// Contiguous `{prefix}{0..n}` names for parquet row/column headers.
fn numbered(prefix: &str, n: usize) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

/// `node_i × T{j}` matrix (centroids or node velocities).
fn write_nodes(mat: &DMatrix<f32>, path: &str) -> Result<()> {
    let rows = numbered("node_", mat.nrows());
    let cols = numbered("T", mat.ncols());
    mat.to_parquet_with_names(path, (Some(&rows), Some("node")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// `edge_i × [from, to, weight, velocity_flux, directed_from, directed_to]`.
fn write_edges(
    edges: &[(usize, usize)],
    weights: &[f32],
    flux: &[f32],
    directed: &[(usize, usize)],
    path: &str,
) -> Result<()> {
    let mut mat = DMatrix::<f32>::zeros(edges.len(), 6);
    for i in 0..edges.len() {
        let (a, b) = edges[i];
        let (df, dt) = directed[i];
        mat[(i, 0)] = a as f32;
        mat[(i, 1)] = b as f32;
        mat[(i, 2)] = weights[i];
        mat[(i, 3)] = flux[i];
        mat[(i, 4)] = df as f32;
        mat[(i, 5)] = dt as f32;
    }
    let rows = numbered("edge_", edges.len());
    let cols: Vec<Box<str>> = vec![
        "from".into(),
        "to".into(),
        "weight".into(),
        "velocity_flux".into(),
        "directed_from".into(),
        "directed_to".into(),
    ];
    mat.to_parquet_with_names(path, (Some(&rows), Some("edge")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// Long format `[lineage, step, node]`: the ordered node path of each lineage.
fn write_lineages(curves: &PrincipalCurves, path: &str) -> Result<()> {
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

/// `cell × [pseudotime, branch]` (primary-lineage pseudotime + lineage id).
fn write_pseudotime(curves: &PrincipalCurves, cell_names: &[Box<str>], path: &str) -> Result<()> {
    let n = curves.pseudotime.len();
    let mut mat = DMatrix::<f32>::zeros(n, 2);
    for i in 0..n {
        mat[(i, 0)] = curves.pseudotime[i];
        mat[(i, 1)] = curves.branch[i] as f32;
    }
    let cols: Vec<Box<str>> = vec!["pseudotime".into(), "branch".into()];
    mat.to_parquet_with_names(path, (Some(cell_names), Some("cell")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// `cell × {col_prefix}_{l}` (per-lineage weights or per-lineage pseudotime).
fn write_cell_matrix(
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
fn write_curves(curves: &PrincipalCurves, path: &str) -> Result<()> {
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
mod tests;
