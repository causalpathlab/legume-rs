//! Pseudotime estimation via Monocle-style principal graph.
//!
//! Pipeline:
//!   1. Read a cell × K latent matrix (typically `senna topic`'s
//!      `.latent.parquet`).
//!   2. Fit a `SimplePPT` principal tree over the cells in latent space
//!      ([`legume_numeric::matrix::principal_graph::fit_principal_graph`]).
//!   3. Project each cell to its nearest point on the tree and compute
//!      geodesic distance from a user-chosen root.

use crate::mat_io::{axis_id_names, read_mat, Mat, MatWithNames};
use clap::Args;
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::principal_graph::{
    closest_node_to_row, fit_principal_graph, project_cells_to_graph, pseudotime_from_root,
    CellProjection, PrincipalGraph, PrincipalGraphArgs,
};
use legume_numeric::matrix::traits::IoOps;
use log::info;

//////////////////////////
// Pure pseudotime core //
//////////////////////////

/// How the pseudotime origin is specified. Cells/nodes are resolved against
/// the principal graph after it is fit, so any variant is valid here.
#[derive(Debug, Clone, Copy)]
pub enum RootSpec<'a> {
    /// Look up `cell_name` in the latent's row names, then snap to the
    /// closest principal-graph node.
    Cell(&'a str),
    /// Use this principal-graph node id directly.
    Node(usize),
    /// No root supplied — snap to the centroid closest to the first cell.
    /// `run_pseudotime` warns when it falls back to this variant.
    DefaultFirstCell,
}

/// Output of [`compute_pseudotime`]: everything needed both to write
/// parquet artifacts and to drive downstream consumers (orient-by-root,
/// tree layout, …) in-memory.
pub struct PseudotimeArtifacts {
    pub graph: PrincipalGraph,
    pub projections: Vec<CellProjection>,
    pub root: usize,
    pub pseudotime: Vec<f32>,
}

/// Pure core: fit the principal graph on `latent`, project cells, resolve
/// the root, and compute per-cell pseudotime. No I/O. Shared by
/// `run_pseudotime` and by `senna layout phate --orient-by-root`.
pub fn compute_pseudotime(
    latent: &Mat,
    cell_names: &[Box<str>],
    pg_args: &PrincipalGraphArgs,
    root: RootSpec<'_>,
) -> anyhow::Result<PseudotimeArtifacts> {
    anyhow::ensure!(
        latent.nrows() >= 5,
        "need at least 5 cells to fit a principal graph"
    );
    let graph = fit_principal_graph(latent, pg_args)?;
    let projections = project_cells_to_graph(latent, &graph);
    let root = resolve_root_node_spec(root, cell_names, latent, &graph)?;
    let pseudotime = pseudotime_from_root(&graph, &projections, root);
    Ok(PseudotimeArtifacts {
        graph,
        projections,
        root,
        pseudotime,
    })
}

fn resolve_root_node_spec(
    spec: RootSpec<'_>,
    cell_names: &[Box<str>],
    latent: &Mat,
    graph: &PrincipalGraph,
) -> anyhow::Result<usize> {
    match spec {
        RootSpec::Node(id) => {
            anyhow::ensure!(
                id < graph.n_nodes(),
                "--root-node {id} out of range (graph has {} nodes)",
                graph.n_nodes()
            );
            Ok(id)
        }
        RootSpec::Cell(name) => {
            let idx = cell_names
                .iter()
                .position(|c| c.as_ref() == name)
                .ok_or_else(|| anyhow::anyhow!("--root-cell '{name}' not found in latent rows"))?;
            Ok(closest_node_to_row(latent, idx, graph))
        }
        RootSpec::DefaultFirstCell => {
            log::warn!(
                "no --root-cell or --root-node given; defaulting to the centroid \
                 closest to the first cell in --latent (use --root-cell or --root-node \
                 to set an explicit origin)"
            );
            Ok(closest_node_to_row(latent, 0, graph))
        }
    }
}

#[derive(Args, Debug)]
pub struct PseudotimeArgs {
    #[arg(
        long,
        short = 'l',
        help = "Latent representation file (cells × K)",
        long_help = "Cell × K latent matrix in parquet or TSV. Typical sources:\n  \
                     - senna topic / masked-topic / joint-topic → .latent.parquet\n  \
                     - senna svd                          → .latent.parquet\n\
                     The first column is expected to be cell names."
    )]
    latent: Option<Box<str>>,

    #[arg(
        long = "from",
        help = "Run manifest from `senna topic|masked-topic|joint-topic|svd|joint-svd`",
        long_help = "When given, latent is read from the manifest's outputs.latent path.\n\
                     That path resolves relative to the manifest directory.\n\
                     One of --latent or --from is required."
    )]
    from: Option<Box<str>>,

    #[arg(
        long,
        help = "Root cell name — pseudotime origin (looked up by row name)"
    )]
    root_cell: Option<Box<str>>,

    #[arg(
        long,
        help = "Root principal-graph node id (0 ≤ id < n-centroids);\n\
                alternative to --root-cell"
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
        short = 'o',
        required = true,
        help = "Output file prefix",
        long_help = "Output prefix. Generates:\n  \
                     {out}.pseudotime.parquet           — cells × 1 pseudotime\n  \
                     {out}.principal_graph.nodes.parquet — K × D centroid coordinates\n  \
                     {out}.principal_graph.edges.parquet — E × 3 (from, to, weight)\n\
                     \n\
                     The Reingold-Tilford tree layout is no longer written here.\n\
                     Run `senna layout tree --from <manifest>` after this command,\n\
                     to produce {out}.tree_layout.{cell_coords,nodes_2d}.parquet."
    )]
    out: Box<str>,
}

impl PseudotimeArgs {
    /// `--latent`, when the caller named the matrix directly.
    #[must_use]
    pub fn latent(&self) -> Option<&str> {
        self.latent.as_deref()
    }

    /// `--from`, the run manifest the caller should resolve inputs against.
    /// Resolving it is `senna`'s job — see `senna::lineage_manifest`.
    #[must_use]
    pub fn manifest(&self) -> Option<&str> {
        self.from.as_deref()
    }
}

/// Paths a caller has already resolved for [`run_pseudotime`].
///
/// This crate never reads a run manifest, so resolving `--from` is the
/// caller's job: `senna::lineage_manifest` does it for the CLI, and a
/// caller working off explicit paths fills this directly.
pub struct PseudotimeInputs {
    /// Cell × K latent matrix (parquet or delimited text).
    pub latent: String,
    /// The run's 2D cell layout, when it has one. Present ⇒ the K centroids
    /// are also projected into layout space.
    pub cell_coords: Option<String>,
}

/// Artifact paths [`run_pseudotime`] wrote, plus the root it resolved — what
/// a manifest-updating caller needs to record the run.
pub struct PseudotimeOutputs {
    pub pseudotime: String,
    pub nodes_latent: String,
    pub nodes_2d: Option<String>,
    pub edges: String,
    pub root: usize,
}

pub fn run_pseudotime(
    args: &PseudotimeArgs,
    inputs: &PseudotimeInputs,
) -> anyhow::Result<PseudotimeOutputs> {
    mkdir_parent(&args.out)?;

    let latent_path = inputs.latent.as_str();
    info!("Reading latent matrix from {latent_path}");

    let MatWithNames {
        rows: cell_names,
        cols: feat_names,
        mat: latent,
    } = read_mat(latent_path)?;

    info!(
        "Loaded latent: {} cells × {} dims",
        latent.nrows(),
        latent.ncols()
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

    let root_spec = if let Some(id) = args.root_node {
        RootSpec::Node(id)
    } else if let Some(name) = args.root_cell.as_deref() {
        RootSpec::Cell(name)
    } else {
        RootSpec::DefaultFirstCell
    };

    let PseudotimeArtifacts {
        graph,
        projections,
        root,
        pseudotime,
    } = compute_pseudotime(&latent, &cell_names, &pg_args, root_spec)?;

    info!(
        "Fitted principal graph: {} nodes, {} edges, {} iter(s), final obj = {:.4}",
        graph.n_nodes(),
        graph.n_edges(),
        graph.n_iters,
        graph.final_objective
    );
    let mean_sd: f32 =
        projections.iter().map(|p| p.sqdist).sum::<f32>() / projections.len().max(1) as f32;
    info!(
        "Projected {} cells onto graph; mean projection-distance² = {:.4}",
        projections.len(),
        mean_sd
    );
    info!("Pseudotime root: node {root}");

    let pseudotime_path = format!("{}.pseudotime.parquet", args.out);
    let nodes_latent_path = format!("{}.principal_graph.nodes.parquet", args.out);
    let edges_path = format!("{}.principal_graph.edges.parquet", args.out);

    write_pseudotime(&pseudotime, &cell_names, &pseudotime_path)?;
    write_graph_nodes(&graph.nodes, &feat_names, &nodes_latent_path)?;
    write_graph_edges(&graph.edges, &graph.edge_weights, &edges_path)?;

    let nodes_2d_path = inputs
        .cell_coords
        .as_deref()
        .map(|cell_coords| {
            project_centroids_to_2d(cell_coords, &latent, &graph, &cell_names, &args.out)
        })
        .transpose()?;

    info!(
        "Run `senna layout tree --from <manifest>` to produce the Reingold-Tilford \
         tree layout from this pseudotime run."
    );

    Ok(PseudotimeOutputs {
        pseudotime: pseudotime_path,
        nodes_latent: nodes_latent_path,
        nodes_2d: nodes_2d_path,
        edges: edges_path,
        root,
    })
}

fn default_k(n_cells: usize) -> usize {
    (n_cells / 10).clamp(5, 200)
}

/// Project the K centroids into 2D using the run's cell layout. The 2D
/// position of each centroid is the mean of the 2D coordinates of cells whose
/// nearest centroid (in latent space) is that node — the simplest faithful map
/// from latent → layout space without needing R or running a second embedding.
fn project_centroids_to_2d(
    cell_coords_path: &str,
    latent: &Mat,
    graph: &PrincipalGraph,
    cell_names: &[Box<str>],
    out: &str,
) -> anyhow::Result<String> {
    let MatWithNames {
        rows: layout_cells,
        cols: layout_cols,
        mat: layout,
    } = read_mat(cell_coords_path)?;

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
            "layout {cell_coords_path} and latent cell-name orderings differ — using positional alignment"
        );
    }
    let xy_cols = pick_xy_columns(&layout_cols);

    let k = graph.n_nodes();
    let mut acc = Mat::zeros(k, 2);
    let mut counts = vec![0usize; k];
    for i in 0..latent.nrows() {
        let node = closest_node_to_row(latent, i, graph);
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
    Ok(path)
}

fn pick_xy_columns(cols: &[Box<str>]) -> (usize, usize) {
    let x = cols.iter().position(|c| c.as_ref() == "x").unwrap_or(0);
    let y = cols.iter().position(|c| c.as_ref() == "y").unwrap_or(1);
    (x, y)
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
