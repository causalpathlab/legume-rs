//! Entry point for `faba lineage` — velocity-informed lineage inference over a
//! `faba gem` embedding.
//!
//! Reads a θ/δ pair by prefix — `cell_embedding` + `velocity` on an embedding run,
//! `latent` + `velocity_factor` on a topic one (see [`super::input`]) — fits
//! **K k-means centroids** on θ and an **MST**
//! over them ([`matrix_util::principal_graph::mst_from_sqdist`]), tests the velocity
//! **direction** of every candidate edge ([`crate::lineage::orient`]), and turns that
//! into a **rooted forest** by maximum-weight branching ([`matrix_util::branching`]):
//! contradictions are cut, weak parents rewired, and each tree rooted at its velocity
//! source. **Slingshot-style principal curves** ([`matrix_util::principal_curve`]) are
//! then fit per tree ([`crate::lineage::forest`]). Outputs per-cell pseudotime + branch
//! + tree + order confidence, the candidate-edge graph, the trees, and the curves.
//!
//! NOTE: centroid placement uses a **seeded** k-means
//! (`matrix_util::…::kmeans_centroids_seeded`, kmeans++ from `--seed`), so the whole
//! fit — centroids, MST, edge directions, forest, curves — is reproducible for a seed.

use anyhow::Result;
use log::{info, warn};

use graph_embedding_util::type_annotation::{Abstain, MarkerBootstrapConfig};
use matrix_util::branching::max_branching;
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::layout::PhateArgs;
use matrix_util::principal_curve::PrincipalCurveArgs;
use matrix_util::principal_graph::{
    kmeans_centroids_seeded, mst_from_sqdist, pairwise_sqdist_rows_to_rows,
};
use std::collections::HashMap;

use super::args::*;
use super::cluster::*;
use super::input::*;
use super::layout::*;
use super::root::*;
use super::traj_annotation::*;
use super::write::*;
use crate::lineage::forest::fit_forest_curves;
use crate::lineage::orient::{
    aggregate_node_velocity, candidate_edges, edge_directionality, mst_only_directions, EdgeCall,
    EdgeDirection, EdgeDirectionConfig,
};

pub fn run_lineage(args: &LineageArgs) -> Result<()> {
    let prefix = args.from.as_ref();
    let out = args.out.as_deref().unwrap_or(prefix).to_string();
    mkdir_parent(&out)?;
    anyhow::ensure!(
        args.root_type.is_none() || args.markers.is_some(),
        "--root-type needs --markers (the node cell-type calls come from the marker annotation)"
    );

    /////////////////////////////
    // load frozen embedding θ //
    /////////////////////////////
    // WHICH manifold and in WHAT metric are both decisions with preconditions, so they
    // live in `super::input`. On a topic run `auto` reads the SIMPLEX (latent.parquet)
    // rather than the θ·α co-embedding: θ·α confines every cell to the convex hull of
    // α's K rows, so a diffuse softmax θ compresses the population toward that hull's
    // centroid — blobby for reasons no layout algorithm can undo.
    if args.normalize_latent {
        warn!(
            "--normalize-latent is deprecated and now a no-op: cosine is the default \
             geometry for a cell embedding. Use --latent-geometry to choose explicitly."
        );
    }
    let theta_from = resolve_theta_from(args.theta_from, prefix)?;
    let geometry = resolve_geometry(args.latent_geometry, theta_from);
    let loaded = load_theta(prefix, theta_from, args.no_orient_velocity)?;
    let LoadedTheta {
        cell_names,
        theta: theta_native,
        velocity,
    } = loaded;
    let n = theta_native.nrows();
    anyhow::ensure!(n >= 2, "need ≥ 2 cells, got {n}");

    // The fit (k-means → MST → curves) and the layout both run on the transformed θ.
    // `theta_native` survives alongside it for the velocity field: arrows are projected
    // from the space δ is actually expressed in, onto whichever 2D coordinates result.
    let theta = apply_geometry(&theta_native, geometry);
    info!("fit + layout geometry: {geometry:?}");

    // `--markers` scores against the co-embedded gene vectors, which are H-space — so it
    // reads cell_embedding even when the trajectory was fitted on the K-space simplex.
    let raw_theta: Option<DMatrix<f32>> = args
        .markers
        .is_some()
        .then(|| load_marker_theta(prefix, &cell_names))
        .transpose()?;

    let k = choose_k(n, args.n_centroids);
    anyhow::ensure!(k >= 2, "need ≥ 2 centroids, got {k}");
    info!(
        "lineage: {n} cells × {} dims → {k} centroids",
        theta.ncols()
    );

    /////////////////////////////
    // k-means centroids + MST //
    /////////////////////////////
    // The GROUPING runs in θ (identity, default), θ+δ (nascent), or [θ|δ] (concat) per
    // --cluster-space: velocity separates transcriptionally-central cells θ alone cannot. The
    // trajectory centroids are recomputed in RAW θ from the resulting labels, so the manifold
    // geometry (MST, layout, marker scoring) is unchanged — only the partition differs. For the
    // Identity default this is exactly the old θ k-means (centroid = mean θ of its cells).
    let cluster_feats = cluster_features(&theta, velocity.as_ref(), args.cluster_space);
    if args.cluster_space != LayoutSpace::Identity {
        info!(
            "annotation grouping on {:?} space ({} dims)",
            args.cluster_space,
            cluster_feats.ncols()
        );
    }
    let (_cf_centroids, labels) =
        kmeans_centroids_seeded(&cluster_feats, k, args.kmeans_iter, args.seed);
    let centroids = theta_centroids_from_labels(&theta, &labels, k);
    let (edges, _mst_weights) =
        mst_from_sqdist(&pairwise_sqdist_rows_to_rows(&centroids, &centroids));
    anyhow::ensure!(
        edges.len() == k - 1,
        "MST on {k} nodes should have {} edges, got {}",
        k - 1,
        edges.len()
    );

    let node_velocity = match &velocity {
        Some(v) => aggregate_node_velocity(v, &labels, k),
        None => DMatrix::<f32>::zeros(k, theta.ncols()),
    };

    // Candidate edges (MST ∪ kNN centroids) with a statistically-tested velocity direction.
    let cand_edges = candidate_edges(&centroids, &edges, args.edge_cand_knn);
    let dirs: Vec<EdgeDirection> = match (&velocity, args.no_edge_direction) {
        (Some(v), false) => edge_directionality(
            &centroids,
            v,
            &labels,
            &cand_edges,
            &edges,
            &EdgeDirectionConfig {
                n_boot: args.edge_direction_n_boot,
                n_perm: args.edge_direction_n_perm,
                alpha: args.edge_alpha,
                min_cells: args.edge_min_cells,
                seed: args.seed,
            },
        ),
        // No velocity / disabled: keep the MST only, as geometry-only (all-abstain) edges.
        _ => mst_only_directions(&centroids, &edges),
    };
    let n_called = dirs.iter().filter(|d| d.call != EdgeCall::Abstain).count();
    info!(
        "edge directions: {n_called}/{} candidate edge(s) confidently oriented",
        dirs.len()
    );

    // Marker node calls (before rooting, so `--root-type` can ground a root hint). Also
    // writes `{out}.lineage_annot.*`.
    //
    // The fit above is kind-agnostic — `cell_embedding.parquet` and `velocity.parquet`
    // are an H-space pair from either producer, so k-means → MST → curves means the same
    // thing on both. The MARKER call is not: it is the co-embedded nearest-centroid
    // statistic, which is what `faba annotate --mode` exists to arbitrate, and on a topic
    // model `annotate` now defaults AWAY from it. Say so rather than let the two commands
    // answer the same question differently without comment.
    if args.markers.is_some()
        && faba::manifest::detect_reporting(prefix) == Some(faba::manifest::RunKind::Topic)
    {
        warn!(
            "--markers on a TOPIC model ({}): the node calls below are the co-embedded \
             nearest-centroid statistic, which `faba annotate` no longer defaults to for \
             this kind of run. The trajectory itself is unaffected — only the names on its \
             nodes. For the topic-native cell call, run `faba annotate --mode enrichment` \
             on the same prefix and read the two together.",
            faba::manifest::path(prefix)
        );
    }
    let node_calls = match (args.markers.as_deref(), raw_theta.as_ref()) {
        (Some(markers), Some(raw)) => Some(compute_node_calls(&AnnotateTrajArgs {
            prefix,
            out: &out,
            markers,
            raw_theta: raw,
            cell_names: &cell_names,
            labels: &labels,
            k,
            num_perm: args.marker_num_perm,
            obo: args.marker_obo.as_deref(),
            label_cl: args.marker_label_cl.as_deref(),
            bootstrap: (!args.no_bootstrap_markers).then_some(MarkerBootstrapConfig {
                n_boot: args.marker_n_boot,
                abstain: Abstain::Support(args.marker_min_support),
                set_coverage: 0.8,
                max_set_size: 3,
                recluster: true,
            }),
            theta: &theta,
            kmeans_iter: args.kmeans_iter,
            seed: args.seed,
        })?),
        _ => None,
    };

    ////////////////////////////////////////////////////////////////
    // max-weight branching: cut + rewire + root into a forest     //
    ////////////////////////////////////////////////////////////////
    // Optional root hint (user / marker type / gem source) pins one node as a root.
    let type_root = args
        .root_type
        .as_deref()
        .and_then(|t| node_calls.as_ref().and_then(|c| root_type_node(c, t)));
    let gem_root = args
        .root_from_gem
        .then(|| gem_root_node(prefix, &cell_names, &labels, k))
        .flatten();
    let root_hint = resolve_root_hint(
        args.root_node,
        args.root_cell.as_deref(),
        &cell_names,
        &labels,
        k,
        type_root,
        gem_root,
    )?;

    let (arcs, root_affinity) = assemble_arcs(&dirs, k, args.root_affinity, root_hint);
    let branching = max_branching(k, &arcs, &root_affinity);
    info!(
        "forest: {} tree(s), {} directed edge(s) selected over {k} nodes",
        branching.roots.len(),
        branching.parent.iter().filter(|p| p.is_some()).count()
    );

    /////////////////////////////////////////////
    // per-tree Slingshot curves + pseudotime   //
    /////////////////////////////////////////////
    let dirs_map: HashMap<(usize, usize), &EdgeDirection> =
        dirs.iter().map(|d| (d.edge, d)).collect();
    let forest = fit_forest_curves(
        &theta,
        &centroids,
        &labels,
        &branching,
        &dirs_map,
        &PrincipalCurveArgs {
            max_iter: args.max_iter,
            tol: args.tol,
            resolution: args.curve_resolution,
            bandwidth: args.curve_bandwidth,
        },
    )?;
    let curves = &forest.curves;
    info!(
        "fit {} lineage(s) across {} tree(s)",
        curves.n_lineages(),
        branching.roots.len()
    );
    // Cells on a trivial tree (a lone centroid, or a tree with too few cells to fit a curve)
    // get no position on any trajectory: their pseudotime is NaN. That is a real, reportable
    // state, not a defect — but it is easy to miss in a parquet, and every downstream model
    // (`faba assoc`) has to drop those cells, so say so here, at the source.
    let n_unplaced = curves.pseudotime.iter().filter(|t| !t.is_finite()).count();
    if n_unplaced > 0 {
        log::warn!(
            "{n_unplaced}/{} cell(s) have no pseudotime (their tree is too small to carry a \
             curve); they are written as NaN and are skipped by `faba assoc`",
            curves.pseudotime.len()
        );
    }

    /////////////
    // outputs //
    /////////////
    write_nodes(&centroids, &format!("{out}.nodes.parquet"))?;
    write_nodes(&node_velocity, &format!("{out}.node_velocity.parquet"))?;
    write_edge_directions(&dirs, &branching, &format!("{out}.edges.parquet"))?;
    write_trees(
        &branching,
        &labels,
        &dirs_map,
        &format!("{out}.trees.parquet"),
    )?;
    write_lineages(curves, &format!("{out}.lineages.parquet"))?;
    write_pseudotime(
        curves,
        &forest.cell_tree,
        &forest.order_conf,
        &cell_names,
        &format!("{out}.pseudotime.parquet"),
    )?;
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
    write_curves(curves, &format!("{out}.curves.parquet"))?;

    ///////////////////////////////////////////////////////////////////////
    // labeled trajectory annotation (node roles from the rooted forest) //
    ///////////////////////////////////////////////////////////////////////
    if let Some(calls) = &node_calls {
        write_trajectory_annotation(
            calls,
            &branching,
            &format!("{out}.trajectory_annotation.parquet"),
        )?;
    }

    ///////////////////////////////////////////////////////////////////
    // optional 2D layout (cells + nodes + curves + velocity field)  //
    ///////////////////////////////////////////////////////////////////
    // The field travels on the NATIVE pair, not the metric-transformed θ: an arrow
    // is a claim about the data, and the metric only decides where the plot puts
    // the cells it is drawn between.
    let native_field = NativeField {
        theta: &theta_native,
        velocity: velocity.as_ref(),
    };
    if args.layout == LayoutKind::Phate {
        let phate = PhateArgs {
            t: args.phate_t,
            knn: args.phate_knn,
            ..PhateArgs::default()
        };
        // Warp the layout along flow only when the directions are trustworthy.
        let frac_called = if dirs.is_empty() {
            0.0
        } else {
            n_called as f32 / dirs.len() as f32
        };
        let velocity_aware = match args.velocity_aware_layout {
            VelocityLayout::On => true,
            VelocityLayout::Off => false,
            VelocityLayout::Auto => frac_called >= 0.5,
        };
        if velocity_aware {
            info!(
                "PHATE: velocity-aware warp ({:.0}% of edges oriented)",
                100.0 * frac_called
            );
        }
        emit_phate_layout(
            &theta,
            native_field,
            &centroids,
            curves,
            &cell_names,
            &phate,
            args.phate_landmarks,
            args.seed,
            &out,
            velocity_aware.then_some((&dirs_map, &branching, labels.as_slice())),
        )?;
    }
    if args.layout == LayoutKind::Umap {
        emit_umap_layout(
            &theta,
            native_field,
            args.layout_space,
            geometry,
            &centroids,
            curves,
            &cell_names,
            args.phate_knn,
            args.layout_pcs,
            args.seed,
            &out,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
