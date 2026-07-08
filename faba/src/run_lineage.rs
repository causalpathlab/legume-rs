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

use graph_embedding_util::type_annotation::{
    annotate_with_communities, CommunityCalls, InputEmbeddings, TermOraConfig,
};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::layout::{phate_layout_2d, project_cells_nystrom, PhateArgs};
use matrix_util::parquet::{write_named_table, Column};
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
    /// No 2D layout.
    None,
    /// PHATE diffusion embedding (default) — the trajectory-appropriate layout
    /// that preserves branch/continuum structure (unlike UMAP/t-SNE), so it is
    /// on by default; pass `--layout none` to skip it.
    #[default]
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
        long = "root-type",
        help = "Root the trajectory at the highest-confidence node of this cell type \
                (needs --markers), e.g. `--root-type HSC_MPP`. Marker-grounded and robust \
                to unreliable velocity; overrides --root-from-gem / velocity but is itself \
                overridden by --root-node / --root-cell."
    )]
    pub root_type: Option<Box<str>>,

    #[arg(
        long = "root-from-gem",
        help = "Anchor the root at gem's velocity-DAG source: the modal MST node of the \
                low-τ region in {from}.dag_pseudotime.parquet. More robust than the per-edge \
                flux pick, while lineage still fits the curves. Overridden by \
                --root-node / --root-cell / --root-type; falls back to the flux root if the \
                file is absent or gem's DAG has no terminal structure (lineage_qc.json)."
    )]
    pub root_from_gem: bool,

    #[arg(
        long,
        help = "Marker TSV (gene<TAB>celltype) to name trajectory nodes by cell type",
        long_help = "Annotate each trajectory node with a cell type by term over-representation \
                     (the `faba annotate` core) run over the MST-node grouping, so the call \
                     carries the same permutation-calibrated confidence.\n\
                     Input: a `gene<TAB>celltype` TSV (tab/comma/space delimited).\n\
                     Reads gene β from `{from}.beta_dictionary.parquet` and raw θ from \
                     `{from}.latent.parquet`. Writes `{out}.lineage_annot.*` (per-cell calls \
                     keyed by MST node) and `{out}.trajectory_annotation.parquet` \
                     (node → role[root|terminal|internal] → cell_type → confidence)."
    )]
    pub markers: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 500,
        help = "With --markers: permutation draws calibrating each node's over-representation"
    )]
    pub marker_num_perm: usize,

    #[arg(
        long,
        help = "Cell Ontology OBO file for the --markers ontology layer (needs --marker-label-cl)",
        long_help = "Optional. Adds a TreeBH Cell-Ontology layer over the per-node marker calls, \
                     as in `faba annotate`. Give the OBO graph here and the marker-type → CL id \
                     map via --marker-label-cl (both required together)."
    )]
    pub marker_obo: Option<Box<str>>,

    #[arg(
        long,
        help = "Curated `label<TAB>CL:id` TSV pairing marker types to CL ids (with --marker-obo)"
    )]
    pub marker_label_cl: Option<Box<str>>,

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
        default_value_t = LayoutKind::Phate,
        help = "2D layout for plotting (default: phate). Emits {out}.{cells,nodes,curves}_2d.parquet; 'none' to skip"
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
/// (the node of the named cell's cluster), `type_root` (`--root-type`, a marker-named
/// node), `gem_root` (gem's velocity-DAG source, from `--root-from-gem`), the
/// velocity-flux-picked root, else node 0.
#[allow(clippy::too_many_arguments)]
fn resolve_root(
    root_node: Option<usize>,
    root_cell: Option<&str>,
    cell_names: &[Box<str>],
    labels: &[usize],
    k: usize,
    type_root: Option<usize>,
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
    } else if let Some(r) = type_root.or(gem_root).or(velocity_root) {
        Ok(r)
    } else {
        Ok(0)
    }
}

/// Map gem's inferred root to an MST centroid (for `--root-from-gem`): read
/// `{prefix}.dag_pseudotime.parquet` and return the MST node that dominates gem's low-τ
/// (velocity-DAG source) region — the **modal** cluster among the lowest-τ cells, which
/// is robust to a single τ≈0 outlier (a rare cell the old min-τ-cell pick could land on).
/// Returns `None` (with a warning) — so the caller falls back to the velocity-flux root —
/// when the file is absent/unreadable, no low-τ barcode matches the latent, or gem's
/// `lineage_qc.json` reports zero terminal fates (a structureless DAG with meaningless τ).
fn gem_root_node(
    prefix: &str,
    cell_names: &[Box<str>],
    labels: &[usize],
    k: usize,
) -> Option<usize> {
    if gem_dag_n_terminals(prefix) == Some(0) {
        warn!(
            "--root-from-gem: gem's velocity-DAG has no terminal structure (lineage_qc.json); \
             using the velocity-flux root instead"
        );
        return None;
    }
    let path = format!("{prefix}.dag_pseudotime.parquet");
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
    // Barcode → MST node lookup, then vote the modal node over the lowest-τ cells.
    let bc_label: std::collections::HashMap<&str, usize> = cell_names
        .iter()
        .zip(labels)
        .map(|(c, &l)| (c.as_ref(), l))
        .collect();
    let nrow = pt.mat.nrows();
    let mut order: Vec<usize> = (0..nrow).collect();
    order.sort_by(|&a, &b| {
        pt.mat[(a, 0)]
            .partial_cmp(&pt.mat[(b, 0)])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_low = (nrow / 20).clamp(5.min(nrow), nrow); // lowest ~5% of τ, ≥ 5 cells (or all)
    let mut votes = vec![0usize; k];
    for &r in order.iter().take(n_low) {
        if let Some(&lab) = pt.rows.get(r).and_then(|bc| bc_label.get(bc.as_ref())) {
            if lab < k {
                votes[lab] += 1;
            }
        }
    }
    let root = (0..k).max_by_key(|&c| votes[c])?;
    if votes[root] == 0 {
        warn!("--root-from-gem: no low-τ barcode matched the latent; using flux root");
        return None;
    }
    info!(
        "--root-from-gem: low-τ region ({n_low} cells) → MST node {root} ({} votes)",
        votes[root]
    );
    Some(root)
}

/// Terminal-fate count from gem's `{prefix}.lineage_qc.json`. `None` when the file is
/// absent/unreadable or the field is missing — the caller then does NOT veto the gem
/// root (no signal). `Some(0)` means a structureless DAG whose τ is meaningless.
fn gem_dag_n_terminals(prefix: &str) -> Option<usize> {
    let s = std::fs::read_to_string(format!("{prefix}.lineage_qc.json")).ok()?;
    let qc: serde_json::Value = serde_json::from_str(&s).ok()?;
    qc.get("n_terminals")?.as_u64().map(|n| n as usize)
}

pub fn run_lineage(args: &LineageArgs) -> Result<()> {
    let prefix = args.from.as_ref();
    let out = args.out.as_deref().unwrap_or(prefix).to_string();
    mkdir_parent(&out)?;
    anyhow::ensure!(
        args.root_type.is_none() || args.markers.is_some(),
        "--root-type needs --markers (the node cell-type calls come from the marker annotation)"
    );

    // ---- load frozen embedding θ ----
    // gem θ is cosine-oriented, so by default the whole fit (k-means → MST →
    // curves) runs on L2-normalized θ; this keeps a few extreme-magnitude cells
    // from dominating and matches the PHATE layout's geometry. `--no-normalize-latent`
    // reverts to the raw-Euclidean fit.
    let latent_path = format!("{prefix}.latent.parquet");
    let cell = DMatrix::<f32>::from_parquet(&latent_path)
        .with_context(|| format!("reading latent embedding {latent_path}"))?;
    let cell_names = cell.rows;
    // Raw θ kept only for `--markers` node annotation, which scores in the same raw
    // latent space `faba annotate` uses (the L2-normalized `theta` below drives the fit).
    let raw_theta: Option<DMatrix<f32>> = args.markers.is_some().then(|| cell.mat.clone());
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

    // ---- marker node calls (computed here, before root selection, so `--root-type`
    // can pick the root from them; written as `{out}.trajectory_annotation.parquet`
    // after the root/orientation are known). Also writes `{out}.lineage_annot.*`. ----
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
        })?),
        _ => None,
    };

    // ---- root selection ----
    let velocity_root = have_velocity.then(|| pick_velocity_root(&edges, &flux, k));
    // Marker-grounded root (`--root-type`): the highest-confidence node of the named type.
    let type_root = args
        .root_type
        .as_deref()
        .and_then(|t| node_calls.as_ref().and_then(|c| root_type_node(c, t)));
    // gem hand-off: anchor at gem's velocity-DAG source unless that DAG is structureless.
    let gem_root = args
        .root_from_gem
        .then(|| gem_root_node(prefix, &cell_names, &labels, k))
        .flatten();
    let root = resolve_root(
        args.root_node,
        args.root_cell.as_deref(),
        &cell_names,
        &labels,
        k,
        type_root,
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

    // ---- labeled trajectory annotation (node roles need the oriented root) ----
    if let Some(calls) = &node_calls {
        write_trajectory_annotation(&out, calls, root, &directed, k)?;
    }

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
// Marker annotation of the trajectory
////////////////////////////////////////////////////////////////////////

/// Inputs for [`annotate_trajectory`] — bundled to keep the fan-in a struct.
struct AnnotateTrajArgs<'a> {
    prefix: &'a str,
    out: &'a str,
    markers: &'a str,
    /// Raw θ `[N × H]` — the same latent space `faba annotate` scores in.
    raw_theta: &'a DMatrix<f32>,
    cell_names: &'a [Box<str>],
    /// Per-cell MST-node id (the k-means `labels`) — the annotation clustering.
    labels: &'a [usize],
    /// Number of MST nodes.
    k: usize,
    num_perm: usize,
    obo: Option<&'a str>,
    label_cl: Option<&'a str>,
}

/// Name each trajectory node by cell type: run the `faba annotate` term-ORA core over the
/// MST-node grouping (raw θ vs the gem β dictionary), giving every node a permutation-
/// calibrated call. Writes `{out}.lineage_annot.*` and returns the per-node
/// [`CommunityCalls`]. Run BEFORE root selection (it doesn't depend on the root) so
/// `--root-type` can pick the root from these calls; the caller writes
/// `{out}.trajectory_annotation.parquet` afterwards via [`write_trajectory_annotation`].
fn compute_node_calls(a: &AnnotateTrajArgs) -> Result<CommunityCalls> {
    let beta_path = format!("{}.beta_dictionary.parquet", a.prefix);
    let beta = DMatrix::<f32>::from_parquet(&beta_path)
        .with_context(|| format!("--markers needs the gem β dictionary {beta_path}"))?;
    let cfg = TermOraConfig {
        n_perm: a.num_perm,
        obo: a.obo.map(str::to_owned),
        label_cl: a.label_cl.map(str::to_owned),
        ..TermOraConfig::default()
    };
    let input = InputEmbeddings {
        feature_emb: &beta.mat,
        gene_names: &beta.rows,
        cell_emb: a.raw_theta,
        cell_names: a.cell_names,
    };
    annotate_with_communities(
        &input,
        a.markers,
        &format!("{}.lineage_annot", a.out),
        true, // IDF-weight markers, as `faba annotate` does by default
        a.labels,
        a.k,
        &cfg,
    )
}

/// `--root-type`: the MST node whose per-node call matches `root_type` (case-insensitive)
/// with the highest confidence, or `None` (with a warning) when no node carries that type.
fn root_type_node(calls: &CommunityCalls, root_type: &str) -> Option<usize> {
    let node = (0..calls.labels.len())
        .filter(|&i| calls.labels[i].eq_ignore_ascii_case(root_type))
        .max_by(|&a, &b| {
            calls.confidence[a]
                .partial_cmp(&calls.confidence[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    match node {
        Some(i) => {
            info!(
                "--root-type '{root_type}' → MST node {i} (confidence {:.3})",
                calls.confidence[i]
            );
            Some(i)
        }
        None => {
            warn!("--root-type '{root_type}' matched no trajectory node; using the next root rule");
            None
        }
    }
}

/// Cross the per-node calls with the oriented MST → the labeled trajectory: one row per
/// MST node — `role` (root | terminal | internal), `cell_type`, `confidence`.
fn write_trajectory_annotation(
    out: &str,
    calls: &CommunityCalls,
    root: usize,
    directed: &[(usize, usize)],
    k: usize,
) -> Result<()> {
    // A terminal is a directed leaf: it never appears as a `from`.
    let mut has_out = vec![false; k];
    for &(f, _) in directed {
        if f < k {
            has_out[f] = true;
        }
    }
    let node_names = numbered("node_", k);
    let roles: Vec<Box<str>> = (0..k)
        .map(|node| {
            if node == root {
                Box::from("root")
            } else if !has_out[node] {
                Box::from("terminal")
            } else {
                Box::from("internal")
            }
        })
        .collect();
    let path = format!("{out}.trajectory_annotation.parquet");
    write_named_table(
        &path,
        "node",
        &node_names,
        &[
            (Box::from("role"), Column::Str(&roles)),
            (Box::from("cell_type"), Column::Str(&calls.labels)),
            (Box::from("confidence"), Column::F32(&calls.confidence)),
        ],
    )
    .with_context(|| format!("writing {path}"))?;
    info!("wrote {path} ({k} nodes; root={root})");
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
