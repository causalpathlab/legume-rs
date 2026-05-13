//! `senna layout tree` — Reingold-Tilford tree layout from a manifest's
//! pseudotime entry.
//!
//! Reads `manifest.pseudotime.{nodes_latent, edges, root_node}` written by
//! `senna fit-pseudotime`, reconstructs the principal graph, re-projects
//! cells onto it, then runs the RT layout + per-cell jitter. Writes
//! `{out}.tree_layout.{cell_coords,nodes_2d}.parquet` and updates
//! `manifest.pseudotime.tree_{cell_coords,nodes_2d}`.

use crate::embed_common::*;
use crate::principal_graph::{project_cells_to_graph, PrincipalGraph};
use crate::run_manifest::{rel_to_manifest, resolve, RunManifest};
use crate::tree_layout::{place_cells_on_tree, reingold_tilford_layout};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct LayoutTreeArgs {
    #[arg(
        long = "from",
        required = true,
        help = "Run manifest JSON written by `senna fit-pseudotime`"
    )]
    from: Box<str>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (defaults to manifest.prefix)",
        long_help = "Output prefix. Produces:\n  \
                     {out}.tree_layout.cell_coords.parquet — N × 2 cell positions\n  \
                     {out}.tree_layout.nodes_2d.parquet    — K × 2 node positions"
    )]
    out: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0.08,
        help = "Tree-layout jitter (fraction of edge length, 0 = no jitter)",
        long_help = "Per-cell perpendicular Gaussian jitter applied when placing\n\
                     cells along principal-graph edges. Scaled by edge length so\n\
                     dense branches don't collapse to a thin line. 0 disables\n\
                     jitter (cells stack on the line)."
    )]
    tree_jitter: f32,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed for tree-layout cell jitter (reproducibility)"
    )]
    tree_jitter_seed: u64,
}

pub fn fit_layout_tree(args: &LayoutTreeArgs) -> anyhow::Result<()> {
    let manifest_path = PathBuf::from(args.from.as_ref());
    let (mut manifest, manifest_dir) = RunManifest::load(&manifest_path)?;

    let pt = &manifest.pseudotime;
    let root_node = pt.root_node.ok_or_else(|| {
        anyhow::anyhow!(
            "manifest {} has no pseudotime entries; run \
             `senna fit-pseudotime --from {} --root-cell <name>` first",
            manifest_path.display(),
            manifest_path.display()
        )
    })?;
    let nodes_latent_rel = pt.nodes_latent.as_deref().ok_or_else(|| {
        anyhow::anyhow!("manifest has no pseudotime.nodes_latent — re-run `senna fit-pseudotime`")
    })?;
    let edges_rel = pt.edges.as_deref().ok_or_else(|| {
        anyhow::anyhow!("manifest has no pseudotime.edges — re-run `senna fit-pseudotime`")
    })?;
    let latent_rel = manifest
        .outputs
        .latent
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("manifest has no outputs.latent"))?;

    let nodes_path = resolve(&manifest_dir, nodes_latent_rel)
        .to_string_lossy()
        .into_owned();
    let edges_path = resolve(&manifest_dir, edges_rel)
        .to_string_lossy()
        .into_owned();
    let latent_path = resolve(&manifest_dir, latent_rel)
        .to_string_lossy()
        .into_owned();

    info!("Reading principal graph nodes from {nodes_path}");
    let MatWithNames { mat: nodes, .. } = read_mat(&nodes_path)?;
    info!("Reading principal graph edges from {edges_path}");
    let MatWithNames { mat: edges_mat, .. } = read_mat(&edges_path)?;
    info!("Reading latent matrix from {latent_path}");
    let MatWithNames {
        rows: cell_names,
        mat: latent,
        ..
    } = read_mat(&latent_path)?;

    let graph = reconstruct_principal_graph(nodes, &edges_mat)?;

    anyhow::ensure!(
        root_node < graph.n_nodes(),
        "manifest pseudotime.root_node ({root_node}) out of range (graph has {} nodes)",
        graph.n_nodes()
    );

    let projections = project_cells_to_graph(&latent, &graph);

    let layout = reingold_tilford_layout(&graph, root_node);
    let n_finite_nodes = layout
        .node_xy
        .iter()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .count();
    anyhow::ensure!(
        n_finite_nodes > 0,
        "Reingold-Tilford layout produced no finite nodes — \
         check that the principal graph is connected and the root \
         lies in the largest component"
    );
    info!("tree layout: {n_finite_nodes} reachable nodes, root = node {root_node}");

    let cell_xy = place_cells_on_tree(
        &graph,
        &projections,
        &layout,
        args.tree_jitter,
        args.tree_jitter_seed,
    );

    let out_prefix = args
        .out
        .as_deref()
        .map(|s| s.to_string())
        .unwrap_or_else(|| manifest.prefix.clone());
    mkdir_parent(&out_prefix)?;

    let cell_coords_path = format!("{out_prefix}.tree_layout.cell_coords.parquet");
    let coord_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    cell_xy.to_parquet_with_names(
        &cell_coords_path,
        (Some(&cell_names), Some("cell")),
        Some(&coord_cols),
    )?;
    info!("Wrote {cell_coords_path}");

    let nodes_2d_path = format!("{out_prefix}.tree_layout.nodes_2d.parquet");
    let mut node_mat = Mat::zeros(graph.n_nodes(), 2);
    for (i, &(x, y)) in layout.node_xy.iter().enumerate() {
        node_mat[(i, 0)] = x;
        node_mat[(i, 1)] = y;
    }
    let node_row_names: Vec<Box<str>> = (0..graph.n_nodes())
        .map(|i| format!("node_{i}").into_boxed_str())
        .collect();
    node_mat.to_parquet_with_names(
        &nodes_2d_path,
        (Some(&node_row_names), Some("node")),
        Some(&coord_cols),
    )?;
    info!("Wrote {nodes_2d_path}");

    let rel = |p: &str| rel_to_manifest(&manifest_dir, p);
    manifest.pseudotime.tree_cell_coords = Some(rel(&cell_coords_path));
    manifest.pseudotime.tree_nodes_2d = Some(rel(&nodes_2d_path));
    manifest.save(&manifest_path)?;
    info!("Updated manifest {}", manifest_path.display());

    Ok(())
}

/// Rebuild a [`PrincipalGraph`] from the parquet artifacts written by
/// `senna fit-pseudotime`. `n_iters` and `final_objective` are not
/// persisted and are left at placeholder zeros — neither the RT layout
/// nor cell projection consult them.
fn reconstruct_principal_graph(nodes: Mat, edges_mat: &Mat) -> anyhow::Result<PrincipalGraph> {
    anyhow::ensure!(
        edges_mat.ncols() >= 3,
        "principal_graph.edges parquet must have ≥ 3 columns (from, to, weight); got {}",
        edges_mat.ncols()
    );
    let n_nodes = nodes.nrows();
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(edges_mat.nrows());
    let mut edge_weights: Vec<f32> = Vec::with_capacity(edges_mat.nrows());
    for i in 0..edges_mat.nrows() {
        let a = edges_mat[(i, 0)] as usize;
        let b = edges_mat[(i, 1)] as usize;
        anyhow::ensure!(
            a < n_nodes && b < n_nodes,
            "edge {i} references node ({a}, {b}) out of range for K={n_nodes}"
        );
        edges.push((a, b));
        edge_weights.push(edges_mat[(i, 2)]);
    }
    Ok(PrincipalGraph {
        nodes,
        edges,
        edge_weights,
        n_iters: 0,
        final_objective: 0.0,
    })
}
