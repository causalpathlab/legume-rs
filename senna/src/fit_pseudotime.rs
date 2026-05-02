//! Pseudotime estimation via Monocle-style principal graph.
//!
//! Pipeline:
//!   1. Read a cell × K latent matrix (typically `senna topic`'s
//!      `.latent.parquet`).
//!   2. Fit a SimplePPT principal tree over the cells in latent space
//!      ([`principal_graph::fit_principal_graph`]).
//!   3. Project each cell to its nearest point on the tree and compute
//!      geodesic distance from a user-chosen root.

use crate::embed_common::*;
use crate::principal_graph::{
    closest_node_to_row, fit_principal_graph, project_cells_to_graph, pseudotime_from_root,
    CellProjection, PrincipalGraph, PrincipalGraphArgs,
};
use crate::run_manifest::{rel_to_manifest, resolve, RunManifest};
use crate::tree_layout::{place_cells_on_tree, reingold_tilford_layout};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct PseudotimeArgs {
    #[arg(
        long,
        short = 'l',
        help = "Latent representation file (cells × K)",
        long_help = "Cell × K latent matrix in parquet or TSV.\n\
                     Typical sources:\n  \
                     - senna topic / itopic / joint-topic → .latent.parquet\n  \
                     - senna svd                          → .latent.parquet\n\
                     The first column is expected to be cell names."
    )]
    latent: Option<Box<str>>,

    #[arg(
        long = "from",
        help = "Run manifest from `senna topic|itopic|joint-topic|svd|joint-svd`",
        long_help = "When given, latent is read from the manifest's outputs.latent\n\
                     path (resolved relative to the manifest directory). One of\n\
                     --latent or --from is required."
    )]
    from: Option<Box<str>>,

    #[arg(
        long,
        help = "Root cell name — pseudotime origin (looked up by row name)"
    )]
    root_cell: Option<Box<str>>,

    #[arg(
        long,
        help = "Root principal-graph node id (0 ≤ id < n-centroids); alternative to --root-cell"
    )]
    root_node: Option<usize>,

    #[arg(
        long,
        short = 'n',
        help = "Number of principal-graph nodes (default = min(cells/10, 200), bounded by ≥ 5)"
    )]
    n_centroids: Option<usize>,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Tree smoothness γ (higher → fewer wiggles; Monocle 3 default ≈ 10)"
    )]
    gamma: f32,

    #[arg(
        long,
        default_value_t = -1.0,
        help = "Soft-assignment bandwidth σ; ≤ 0 = adaptive (mean nearest-centroid dist²)"
    )]
    sigma: f32,

    #[arg(
        long,
        default_value_t = 25,
        help = "Maximum SimplePPT outer iterations"
    )]
    max_iter: usize,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "Relative objective change for early stop"
    )]
    tol: f32,

    #[arg(
        long,
        default_value_t = 100,
        help = "k-means iterations for centroid initialization"
    )]
    kmeans_iter: usize,

    #[arg(
        long,
        default_value_t = 0.08,
        help = "Tree-layout jitter (fraction of edge length, 0 = no jitter)",
        long_help = "Per-cell perpendicular Gaussian jitter applied when placing\n\
                     cells along principal-graph edges in the tree layout. Scaled\n\
                     by edge length so dense branches don't collapse to a thin\n\
                     line. 0 disables jitter (cells stack on the line)."
    )]
    tree_jitter: f32,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed for tree-layout cell jitter (reproducibility)"
    )]
    tree_jitter_seed: u64,

    #[arg(
        long,
        short = 'o',
        required = true,
        help = "Output file prefix",
        long_help = "Output prefix. Generates:\n  \
                     {out}.pseudotime.parquet           — cells × 1 pseudotime\n  \
                     {out}.principal_graph.nodes.parquet — K × D centroid coordinates\n  \
                     {out}.principal_graph.edges.parquet — E × 3 (from, to, weight)"
    )]
    out: Box<str>,
}

pub fn run_pseudotime(args: &PseudotimeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let manifest_ctx = load_manifest_ctx(args)?;
    let latent_path = resolve_latent_path(args, manifest_ctx.as_ref())?;
    info!("Reading latent matrix from {latent_path}");

    let MatWithNames {
        rows: cell_names,
        cols: feat_names,
        mat: latent,
    } = read_mat(&latent_path)?;

    info!(
        "Loaded latent: {} cells × {} dims",
        latent.nrows(),
        latent.ncols()
    );
    anyhow::ensure!(
        latent.nrows() >= 5,
        "need at least 5 cells to fit a principal graph"
    );

    let n_centroids = args
        .n_centroids
        .unwrap_or_else(|| default_k(latent.nrows()));
    info!(
        "SimplePPT: K={} centroids, γ={:.2}, σ={}, max_iter={}",
        n_centroids,
        args.gamma,
        if args.sigma > 0.0 {
            format!("{:.4}", args.sigma)
        } else {
            "adaptive".to_string()
        },
        args.max_iter
    );

    let pg_args = PrincipalGraphArgs {
        n_centroids,
        gamma: args.gamma,
        sigma: args.sigma,
        max_iter: args.max_iter,
        tol: args.tol,
        kmeans_max_iter: args.kmeans_iter,
    };

    let graph = fit_principal_graph(&latent, &pg_args)?;
    info!(
        "Fitted principal graph: {} nodes, {} edges, {} iter(s), final obj = {:.4}",
        graph.n_nodes(),
        graph.n_edges(),
        graph.n_iters,
        graph.final_objective
    );

    let projections = project_cells_to_graph(&latent, &graph);
    let mean_sd: f32 =
        projections.iter().map(|p| p.sqdist).sum::<f32>() / projections.len().max(1) as f32;
    info!(
        "Projected {} cells onto graph; mean projection-distance² = {:.4}",
        projections.len(),
        mean_sd
    );

    let root = resolve_root_node(args, &cell_names, &latent, &graph)?;
    info!("Pseudotime root: node {root}");

    let pseudotime = pseudotime_from_root(&graph, &projections, root);

    let pseudotime_path = format!("{}.pseudotime.parquet", args.out);
    let nodes_latent_path = format!("{}.principal_graph.nodes.parquet", args.out);
    let edges_path = format!("{}.principal_graph.edges.parquet", args.out);

    write_pseudotime(&pseudotime, &cell_names, &pseudotime_path)?;
    write_graph_nodes(&graph.nodes, &feat_names, &nodes_latent_path)?;
    write_graph_edges(&graph.edges, &graph.edge_weights, &edges_path)?;

    let nodes_2d_path = match manifest_ctx.as_ref() {
        Some(ctx) => project_centroids_to_2d(ctx, &latent, &graph, &cell_names, &args.out)?,
        None => None,
    };

    let tree_paths = write_tree_layout(
        &graph,
        &projections,
        root,
        &cell_names,
        TreeJitter {
            frac: args.tree_jitter,
            seed: args.tree_jitter_seed,
        },
        &args.out,
    )?;

    if let Some(ctx) = manifest_ctx {
        update_manifest(
            ctx,
            PseudotimeManifestUpdate {
                pseudotime: &pseudotime_path,
                nodes_latent: &nodes_latent_path,
                nodes_2d: nodes_2d_path.as_deref(),
                edges: &edges_path,
                root,
                tree: tree_paths.as_ref(),
            },
        )?;
    }

    Ok(())
}

/// Per-cell jitter knob for [`write_tree_layout`].
#[derive(Debug, Clone, Copy)]
struct TreeJitter {
    frac: f32,
    seed: u64,
}

/// Bundle of artifact paths handed to [`update_manifest`]. All paths are
/// the same strings already passed to the writer fns above; the struct
/// just keeps the call site readable.
struct PseudotimeManifestUpdate<'a> {
    pseudotime: &'a str,
    nodes_latent: &'a str,
    nodes_2d: Option<&'a str>,
    edges: &'a str,
    root: usize,
    tree: Option<&'a TreeLayoutPaths>,
}

struct TreeLayoutPaths {
    cell_coords: String,
    nodes_2d: String,
}

/// Compute the Reingold-Tilford tree layout, place cells on edges with
/// jitter, and write `{out}.tree_layout.{cell_coords,nodes_2d}.parquet`.
/// Returns `None` if no node ended up with a finite (x, y) — e.g. the
/// principal graph is empty.
fn write_tree_layout(
    graph: &PrincipalGraph,
    projections: &[CellProjection],
    root: usize,
    cell_names: &[Box<str>],
    jitter: TreeJitter,
    out: &str,
) -> anyhow::Result<Option<TreeLayoutPaths>> {
    let layout = reingold_tilford_layout(graph, root);
    let n_finite_nodes = layout
        .node_xy
        .iter()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .count();
    if n_finite_nodes == 0 {
        info!("tree layout: no finite nodes after RT — skipping tree-layout outputs");
        return Ok(None);
    }
    info!("tree layout: {n_finite_nodes} reachable nodes, root = node {root}");

    let cell_xy = place_cells_on_tree(graph, projections, &layout, jitter.frac, jitter.seed);

    // Match the layout.cell_coords schema (`x`, `y`) so `senna plot`'s
    // existing reader picks this up unchanged.
    let cell_coords_path = format!("{out}.tree_layout.cell_coords.parquet");
    let cell_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    cell_xy.to_parquet_with_names(
        &cell_coords_path,
        (Some(cell_names), Some("cell")),
        Some(&cell_cols),
    )?;
    info!("Wrote {cell_coords_path}");

    let nodes_2d_path = format!("{out}.tree_layout.nodes_2d.parquet");
    let mut node_mat = Mat::zeros(graph.n_nodes(), 2);
    for (i, &(x, y)) in layout.node_xy.iter().enumerate() {
        node_mat[(i, 0)] = x;
        node_mat[(i, 1)] = y;
    }
    let node_row_names: Vec<Box<str>> = (0..graph.n_nodes())
        .map(|i| format!("node_{i}").into_boxed_str())
        .collect();
    let node_col_names: Vec<Box<str>> = vec!["x".into(), "y".into()];
    node_mat.to_parquet_with_names(
        &nodes_2d_path,
        (Some(&node_row_names), Some("node")),
        Some(&node_col_names),
    )?;
    info!("Wrote {nodes_2d_path}");

    Ok(Some(TreeLayoutPaths {
        cell_coords: cell_coords_path,
        nodes_2d: nodes_2d_path,
    }))
}

fn default_k(n_cells: usize) -> usize {
    (n_cells / 10).clamp(5, 200)
}

/// Loaded manifest plus its on-disk path and resolved directory. Held
/// across the whole pseudotime run so we can both read existing entries
/// (e.g. `outputs.latent`, `layout.cell_coords`) and write new ones back.
struct ManifestCtx {
    manifest: RunManifest,
    path: PathBuf,
    dir: PathBuf,
}

fn load_manifest_ctx(args: &PseudotimeArgs) -> anyhow::Result<Option<ManifestCtx>> {
    let Some(from) = args.from.as_deref() else {
        return Ok(None);
    };
    let path = PathBuf::from(from);
    let (manifest, dir) = RunManifest::load(&path)?;
    Ok(Some(ManifestCtx {
        manifest,
        path,
        dir,
    }))
}

fn resolve_latent_path(args: &PseudotimeArgs, ctx: Option<&ManifestCtx>) -> anyhow::Result<String> {
    if let Some(p) = args.latent.as_deref() {
        return Ok(p.to_string());
    }
    let ctx = ctx.ok_or_else(|| anyhow::anyhow!("either --latent or --from is required"))?;
    let rel =
        ctx.manifest.outputs.latent.as_deref().ok_or_else(|| {
            anyhow::anyhow!("manifest {} has no outputs.latent", ctx.path.display())
        })?;
    Ok(resolve(&ctx.dir, rel).to_string_lossy().into_owned())
}

/// Project the K centroids into 2D using the run's cell layout when
/// available. The 2D position of each centroid is the mean of the 2D
/// coordinates of cells whose nearest centroid (in latent space) is that
/// node — the simplest faithful map from latent → layout space without
/// needing R or running a second embedding.
fn project_centroids_to_2d(
    ctx: &ManifestCtx,
    latent: &Mat,
    graph: &crate::principal_graph::PrincipalGraph,
    cell_names: &[Box<str>],
    out: &str,
) -> anyhow::Result<Option<String>> {
    let Some(rel) = ctx.manifest.layout.cell_coords.as_deref() else {
        info!(
            "manifest has no layout.cell_coords; skipping 2D centroid projection \
             (run `senna layout phate --from ...` first to enable plot overlay)"
        );
        return Ok(None);
    };
    let cell_coords_path = resolve(&ctx.dir, rel).to_string_lossy().into_owned();
    let MatWithNames {
        rows: layout_cells,
        cols: layout_cols,
        mat: layout,
    } = read_mat(&cell_coords_path)?;

    anyhow::ensure!(
        layout.nrows() == latent.nrows(),
        "layout cell_coords has {} rows but latent has {}; \
         re-run `senna layout` against the same training output",
        layout.nrows(),
        latent.nrows()
    );
    anyhow::ensure!(
        layout.ncols() >= 2,
        "layout cell_coords has {} columns; need ≥ 2 (x, y)",
        layout.ncols()
    );
    if layout_cells != cell_names {
        log::warn!(
            "layout {} and latent cell-name orderings differ — using positional alignment",
            cell_coords_path
        );
    }
    let xy_cols = pick_xy_columns(&layout_cols);

    let k = graph.n_nodes();
    let mut acc = Mat::zeros(k, 2);
    let mut counts = vec![0usize; k];
    for i in 0..latent.nrows() {
        let node = crate::principal_graph::closest_node_to_row(latent, i, graph);
        acc[(node, 0)] += layout[(i, xy_cols.0)];
        acc[(node, 1)] += layout[(i, xy_cols.1)];
        counts[node] += 1;
    }
    for k_idx in 0..k {
        if counts[k_idx] > 0 {
            acc[(k_idx, 0)] /= counts[k_idx] as f32;
            acc[(k_idx, 1)] /= counts[k_idx] as f32;
        } else {
            acc[(k_idx, 0)] = f32::NAN;
            acc[(k_idx, 1)] = f32::NAN;
        }
    }

    let path = format!("{out}.principal_graph.nodes_2d.parquet");
    let row_names: Vec<Box<str>> = (0..k)
        .map(|i| format!("node_{i}").into_boxed_str())
        .collect();
    let col_names: Vec<Box<str>> = vec!["x".into(), "y".into()];
    acc.to_parquet_with_names(&path, (Some(&row_names), Some("node")), Some(&col_names))?;
    info!("Wrote {path}");
    Ok(Some(path))
}

fn pick_xy_columns(cols: &[Box<str>]) -> (usize, usize) {
    let x = cols.iter().position(|c| c.as_ref() == "x").unwrap_or(0);
    let y = cols.iter().position(|c| c.as_ref() == "y").unwrap_or(1);
    (x, y)
}

fn update_manifest(
    mut ctx: ManifestCtx,
    update: PseudotimeManifestUpdate<'_>,
) -> anyhow::Result<()> {
    let rel = |p: &str| rel_to_manifest(&ctx.dir, p);
    ctx.manifest.pseudotime.pseudotime = Some(rel(update.pseudotime));
    ctx.manifest.pseudotime.nodes_latent = Some(rel(update.nodes_latent));
    ctx.manifest.pseudotime.nodes_2d = update.nodes_2d.map(rel);
    ctx.manifest.pseudotime.edges = Some(rel(update.edges));
    ctx.manifest.pseudotime.root_node = Some(update.root);
    ctx.manifest.pseudotime.tree_cell_coords = update.tree.map(|t| rel(&t.cell_coords));
    ctx.manifest.pseudotime.tree_nodes_2d = update.tree.map(|t| rel(&t.nodes_2d));
    ctx.manifest.save(&ctx.path)?;
    info!("Updated manifest {}", ctx.path.display());
    Ok(())
}

fn resolve_root_node(
    args: &PseudotimeArgs,
    cell_names: &[Box<str>],
    latent: &Mat,
    graph: &crate::principal_graph::PrincipalGraph,
) -> anyhow::Result<usize> {
    if let Some(id) = args.root_node {
        anyhow::ensure!(
            id < graph.n_nodes(),
            "--root-node {id} out of range (graph has {} nodes)",
            graph.n_nodes()
        );
        return Ok(id);
    }
    if let Some(name) = args.root_cell.as_deref() {
        let idx = cell_names
            .iter()
            .position(|c| c.as_ref() == name)
            .ok_or_else(|| anyhow::anyhow!("--root-cell '{name}' not found in latent rows"))?;
        return Ok(closest_node_to_row(latent, idx, graph));
    }
    // Default: pick the centroid closest to the first cell. Documented as
    // arbitrary so users always have a deterministic fallback.
    log::warn!(
        "no --root-cell or --root-node given; defaulting to the centroid \
         closest to the first cell in --latent (use --root-cell or --root-node \
         to set an explicit origin)"
    );
    Ok(closest_node_to_row(latent, 0, graph))
}

fn write_pseudotime(pseudotime: &[f32], cell_names: &[Box<str>], path: &str) -> anyhow::Result<()> {
    let mut mat = Mat::zeros(pseudotime.len(), 1);
    for (i, &t) in pseudotime.iter().enumerate() {
        mat[(i, 0)] = t;
    }
    let cols: Vec<Box<str>> = vec!["pseudotime".into()];
    mat.to_parquet_with_names(path, (Some(cell_names), Some("cell")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

fn write_graph_nodes(nodes: &Mat, feat_names: &[Box<str>], path: &str) -> anyhow::Result<()> {
    let row_names: Vec<Box<str>> = (0..nodes.nrows())
        .map(|i| format!("node_{i}").into_boxed_str())
        .collect();
    let col_names: Vec<Box<str>> = if feat_names.len() == nodes.ncols() {
        feat_names.to_vec()
    } else {
        axis_id_names("T", nodes.ncols())
    };
    nodes.to_parquet_with_names(path, (Some(&row_names), Some("node")), Some(&col_names))?;
    info!("Wrote {path}");
    Ok(())
}

fn write_graph_edges(edges: &[(usize, usize)], weights: &[f32], path: &str) -> anyhow::Result<()> {
    let mut mat = Mat::zeros(edges.len(), 3);
    for (i, (&(a, b), &w)) in edges.iter().zip(weights).enumerate() {
        mat[(i, 0)] = a as f32;
        mat[(i, 1)] = b as f32;
        mat[(i, 2)] = w;
    }
    let row_names: Vec<Box<str>> = (0..edges.len())
        .map(|i| format!("edge_{i}").into_boxed_str())
        .collect();
    let col_names: Vec<Box<str>> = vec!["from".into(), "to".into(), "weight".into()];
    mat.to_parquet_with_names(path, (Some(&row_names), Some("edge")), Some(&col_names))?;
    info!("Wrote {path}");
    Ok(())
}
