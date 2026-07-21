//! Entry point for `faba lineage` — velocity-informed lineage inference over a
//! `faba gem` embedding.
//!
//! Reads gem's raw parquet outputs by prefix (`{from}.cell_embedding.parquet` = θ,
//! `{from}.velocity.parquet` = δ), fits **K k-means centroids** on θ and an **MST**
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

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use log::{info, warn};
use std::path::Path;

use graph_embedding_util::type_annotation::{
    annotate_with_communities, Abstain, CommunityCalls, InputEmbeddings, MarkerBootstrapConfig,
    Regroup, TermOraConfig,
};
use matrix_util::branching::{max_branching, Branching};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::layout::{phate_layout_2d, project_cells_nystrom, PhateArgs};
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::principal_curve::{PrincipalCurveArgs, PrincipalCurves};
use matrix_util::principal_graph::{
    kmeans_centroids_seeded, mst_from_sqdist, pairwise_sqdist_rows_to_rows,
};
use matrix_util::traits::{IoOps, MatOps};
use matrix_util::utils::median;
use std::collections::HashMap;

use crate::lineage::forest::fit_forest_curves;
use crate::lineage::orient::{
    aggregate_node_velocity, candidate_edges, edge_directionality, mst_only_directions, undirected,
    EdgeCall, EdgeDirection, EdgeDirectionConfig,
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
    /// t-UMAP on a **cosine** kNN graph — sharper cluster separation than PHATE when
    /// the embedding is magnitude-heavy (PHATE's diffusion over-elongates those into
    /// convoluted arms). Cells, MST nodes, and curve points are embedded **jointly**
    /// in one fuzzy-kNN fit so they share the 2D space (no PHATE-style Nyström here).
    Umap,
}

/// Feature space the t-UMAP layout embeds on (`--layout umap`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum LayoutSpace {
    /// θ only — the identity manifold (current state).
    Identity,
    /// θ + δ — the NASCENT state (where each cell is heading). Splays the manifold
    /// toward the fates, so branches separate best on cosine t-UMAP. The default.
    #[default]
    Nascent,
    /// [θ | δ] concatenated — identity and velocity as separate cosine channels.
    Concat,
}

/// Whether the PHATE layout is warped along the confident velocity directions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum VelocityLayout {
    /// Warp only when enough edges are confidently oriented (default).
    #[default]
    Auto,
    /// Always warp along the selected directions.
    On,
    /// Never warp — the pure-θ PHATE manifold.
    Off,
}

// The advanced tuning knobs carry `hide_short_help` so `faba lineage -h` shows only the
// common flags; `--help` lists everything. `help_heading` buckets each flag into a category.
#[derive(Args, Debug)]
pub struct LineageArgs {
    #[arg(
        long,
        short = 'f',
        help_heading = "Input/output",
        help = "gem output prefix (reads {from}.cell_embedding.parquet and {from}.velocity.parquet)"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        short = 'o',
        help_heading = "Input/output",
        help = "Output prefix (default: the gem prefix)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        help_heading = "Centroids & MST",
        help = "Number of MST node centroids K (default: min(cells / 10, 200))"
    )]
    pub n_centroids: Option<usize>,

    #[arg(
        long,
        default_value_t = 42,
        help_heading = "Centroids & MST",
        help = "RNG seed (reproducible centroids, edge directions, forest)"
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 100,
        hide_short_help = true,
        help_heading = "Centroids & MST",
        help = "k-means iterations for centroid initialization"
    )]
    pub kmeans_iter: usize,

    #[arg(
        long = "normalize-latent",
        hide_short_help = true,
        help_heading = "Centroids & MST",
        help = "L2-normalize θ (cosine geometry) for the fit AND layout \
                [default: raw θ / Euclidean]"
    )]
    pub normalize_latent: bool,

    #[arg(
        long = "no-edge-direction",
        help_heading = "Velocity direction & forest",
        help = "Skip the per-edge velocity direction test; forest = the geometric MST",
        long_help = "Skip the per-edge velocity direction test.\n\
            Every candidate edge is then geometry-only (abstained),\n\
            so the max-weight branching reduces to the geometric MST rooted by the hint chain —\n\
            the legacy behaviour with no velocity-informed cut/rewire."
    )]
    pub no_edge_direction: bool,

    #[arg(
        long = "no-orient-velocity",
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Ignore velocity entirely (skip loading {from}.velocity.parquet)"
    )]
    pub no_orient_velocity: bool,

    #[arg(
        long,
        default_value_t = 4,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Nearest centroids added to the MST to form the directionality candidate set"
    )]
    pub edge_cand_knn: usize,

    #[arg(
        long,
        default_value_t = 200,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Cell bootstrap resamples for each edge's direction CI/SE"
    )]
    pub edge_direction_n_boot: usize,

    #[arg(
        long,
        default_value_t = 500,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Sign-flip permutation draws for each edge's direction p-value"
    )]
    pub edge_direction_n_perm: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "q cutoff and CI level (the abstain bar) for calling an edge's direction"
    )]
    pub edge_alpha: f64,

    #[arg(
        long,
        default_value_t = 2,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Minimum cells on an edge before its direction can be called (else abstain)"
    )]
    pub edge_min_cells: usize,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Forest granularity τ_root: virtual no-parent weight; higher ⇒ more trees. \
                Default = median selected arc weight."
    )]
    pub root_affinity: Option<f32>,

    #[arg(
        long = "root-type",
        help_heading = "Root selection",
        help = "Root the trajectory at the highest-confidence node of this cell type \
                (needs --markers), e.g. `--root-type HSC_MPP`. Marker-grounded and robust \
                to unreliable velocity; overrides --root-from-gem / velocity but is itself \
                overridden by --root-node / --root-cell."
    )]
    pub root_type: Option<Box<str>>,

    #[arg(
        long = "root-from-gem",
        help_heading = "Root selection",
        help = "Anchor the root at gem's velocity-DAG source: the modal MST node of the \
                low-τ region in {from}.dag_pseudotime.parquet. More robust than the per-edge \
                flux pick, while lineage still fits the curves. Overridden by \
                --root-node / --root-cell / --root-type; falls back to the flux root if the \
                file is absent or gem's DAG has no terminal structure (lineage_qc.json)."
    )]
    pub root_from_gem: bool,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Root selection",
        help = "Force the root MST node by index (overrides velocity orientation)"
    )]
    pub root_node: Option<usize>,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Root selection",
        help = "Force the root at the node nearest a named cell (overrides velocity)"
    )]
    pub root_cell: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0.0,
        hide_short_help = true,
        help_heading = "Principal curves",
        help = "Gaussian kernel bandwidth in pseudotime units (0 = adaptive per curve)"
    )]
    pub curve_bandwidth: f32,

    #[arg(
        long,
        default_value_t = 100,
        hide_short_help = true,
        help_heading = "Principal curves",
        help = "Points sampled along each fitted principal curve"
    )]
    pub curve_resolution: usize,

    #[arg(
        long,
        default_value_t = 15,
        hide_short_help = true,
        help_heading = "Principal curves",
        help = "Max project-then-smooth iterations for the curves"
    )]
    pub max_iter: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        hide_short_help = true,
        help_heading = "Principal curves",
        help = "Convergence tolerance on mean |Δpseudotime| / range"
    )]
    pub tol: f32,

    #[arg(
        long,
        help_heading = "Marker annotation",
        help = "Marker TSV (gene<TAB>celltype) to name trajectory nodes by cell type",
        long_help = "Annotate each trajectory node with a cell type by term over-representation (the `faba annotate` core)\n\
                     run over the MST-node grouping, so the call carries the same permutation-calibrated confidence.\n\
                     Input: a `gene<TAB>celltype` TSV (tab/comma/space delimited).\n\
                     Reads the co-embedded gene vectors from `{from}.feature_embedding.parquet` (spliced rows)\n\
                     and raw θ from `{from}.cell_embedding.parquet`.\n\
                     Writes `{out}.lineage_annot.*` (per-cell calls keyed by MST node)\n\
                     and `{out}.trajectory_annotation.parquet` (node → role[root|terminal|internal] → cell_type → confidence)."
    )]
    pub markers: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 500,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "With --markers: permutation draws calibrating each node's over-representation"
    )]
    pub marker_num_perm: usize,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "Cell Ontology OBO file for the --markers ontology layer (needs --marker-label-cl)",
        long_help = "Optional. Adds a TreeBH Cell-Ontology layer over the per-node marker calls, as in `faba annotate`.\n\
                     Give the OBO graph here and the marker-type → CL id map via --marker-label-cl (both required together)."
    )]
    pub marker_obo: Option<Box<str>>,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "Curated `label<TAB>CL:id` TSV pairing marker types to CL ids (with --marker-obo)"
    )]
    pub marker_label_cl: Option<Box<str>>,

    #[arg(
        long = "no-bootstrap-markers",
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "[--markers] Turn OFF the stability bootstrap on the node calls",
        long_help = "Turn OFF the stability bootstrap on the node calls, naming each node by a bare point estimate.\n\n\
            The bootstrap is ON by default.\n\
            Each draw resamples every type's marker panel with replacement AND re-derives the k-means grouping;\n\
            the consensus is what ships, so a node's name carries the fraction of resamples that agreed on it.\n\n\
            This matters most for --root-type,\n\
            which picks the trajectory root as the highest-confidence node of a given type.\n\
            Without the bootstrap that `confidence` is a softmaxed test statistic rather than a reproducibility —\n\
            and the whole trajectory hangs off it.\n\n\
            Costs ~6 min at --marker-n-boot 200:\n\
            the replicate k-means has nothing to cache, unlike `faba annotate`'s kNN graph"
    )]
    pub no_bootstrap_markers: bool,

    #[arg(
        long,
        default_value_t = 200,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "Bootstrap resamples on the node calls (--no-bootstrap-markers to disable)"
    )]
    pub marker_n_boot: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "[--markers] Minimum fraction of resamples the top label must win for \
                a node to be called at all (ignored under --no-bootstrap-markers)"
    )]
    pub marker_min_support: f32,

    #[arg(
        long,
        value_enum,
        default_value_t = LayoutKind::Phate,
        help_heading = "PHATE layout",
        help = "2D layout for plotting (default: phate). Emits {out}.{cells,nodes,curves}_2d.parquet; 'none' to skip"
    )]
    pub layout: LayoutKind,

    #[arg(
        long = "layout-space",
        value_enum,
        default_value_t = LayoutSpace::Nascent,
        help = "Feature space for --layout umap: identity (θ), nascent (θ+δ, default), or concat ([θ|δ])"
    )]
    pub layout_space: LayoutSpace,

    #[arg(
        long = "cluster-space",
        value_enum,
        default_value_t = LayoutSpace::Identity,
        help = "Feature space for the k-means grouping that drives annotation: identity (θ, default), \
                nascent (θ+δ), or concat ([θ|δ] — velocity separates committing progenitors θ alone cannot). \
                Marker scoring stays in θ regardless.",
        long_help = "Which cell features the annotation k-means groups on. `identity` (θ, the spliced \
                     state) is the default — cell TYPE is an identity question. `concat` ([θ|δ], each \
                     channel L2-normalised) additionally splits cells by their VELOCITY direction, so two \
                     transcriptionally-central cells heading to different fates land in different clusters — \
                     useful on a progenitor-enriched (e.g. CD34+) sample where θ alone can't resolve the \
                     committing structure. `nascent` (θ+δ) blends the two. The trajectory centroids and \
                     marker scoring are always recomputed in raw θ, so only the GROUPING changes."
    )]
    pub cluster_space: LayoutSpace,

    #[arg(
        long,
        value_enum,
        default_value_t = VelocityLayout::Auto,
        help_heading = "PHATE layout",
        help = "Warp the PHATE layout along confident velocity directions: auto (when \
                enough edges are oriented), on, or off"
    )]
    pub velocity_aware_layout: VelocityLayout,

    #[arg(
        long,
        default_value_t = 15,
        hide_short_help = true,
        help_heading = "PHATE layout",
        help = "PHATE kNN adaptive bandwidth (only with --layout phate)"
    )]
    pub phate_knn: usize,

    #[arg(
        long,
        default_value_t = 0,
        hide_short_help = true,
        help_heading = "PHATE layout",
        help = "PHATE diffusion time t (0 = auto-select at the von-Neumann-entropy knee)"
    )]
    pub phate_t: usize,

    #[arg(
        long,
        default_value_t = 2000,
        hide_short_help = true,
        help_heading = "PHATE layout",
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

/// Feature matrix the annotation k-means groups on, per `--cluster-space`. `identity` = raw θ;
/// `nascent` = θ+δ; `concat` = `[θ̂ | δ̂]` with each channel L2-normalised per row so identity and
/// velocity contribute equally to the Euclidean k-means. Falls back to θ when velocity is absent.
fn cluster_features(
    theta: &DMatrix<f32>,
    velocity: Option<&DMatrix<f32>>,
    space: LayoutSpace,
) -> DMatrix<f32> {
    match (space, velocity) {
        (LayoutSpace::Identity, _) | (_, None) => theta.clone(),
        (LayoutSpace::Nascent, Some(v)) => {
            let mut f = theta.clone();
            f += v;
            f
        }
        (LayoutSpace::Concat, Some(v)) => {
            let (tn, vn) = (l2_normalize_rows(theta), l2_normalize_rows(v));
            let (n, h) = (theta.nrows(), theta.ncols());
            let mut f = DMatrix::<f32>::zeros(n, 2 * h);
            f.view_mut((0, 0), (n, h)).copy_from(&tn);
            f.view_mut((0, h), (n, h)).copy_from(&vn);
            f
        }
    }
}

/// Recompute the `k` trajectory centroids in RAW θ from the grouping labels (mean θ of each
/// cluster's cells), so the manifold geometry (MST, layout, marker scoring) stays θ-based even
/// when the GROUPING used θ+δ or `[θ|δ]`. For the `identity` default this equals the θ k-means
/// centroids exactly.
fn theta_centroids_from_labels(theta: &DMatrix<f32>, labels: &[usize], k: usize) -> DMatrix<f32> {
    let h = theta.ncols();
    let mut c = DMatrix::<f32>::zeros(k, h);
    let mut cnt = vec![0f32; k];
    for (i, &l) in labels.iter().enumerate() {
        cnt[l] += 1.0;
        for j in 0..h {
            c[(l, j)] += theta[(i, j)];
        }
    }
    for l in 0..k {
        if cnt[l] > 0.0 {
            for j in 0..h {
                c[(l, j)] /= cnt[l];
            }
        }
    }
    c
}

/// Resolve the root MST node, in priority order: `--root-node` (validated), `--root-cell`
/// (the node of the named cell's cluster), `type_root` (`--root-type`, a marker-named
/// node), `gem_root` (gem's velocity-DAG source, from `--root-from-gem`), the
/// velocity-flux-picked root, else node 0.
fn resolve_root_hint(
    root_node: Option<usize>,
    root_cell: Option<&str>,
    cell_names: &[Box<str>],
    labels: &[usize],
    k: usize,
    type_root: Option<usize>,
    gem_root: Option<usize>,
) -> Result<Option<usize>> {
    if let Some(r) = root_node {
        anyhow::ensure!(r < k, "--root-node {r} out of range (K = {k})");
        Ok(Some(r))
    } else if let Some(name) = root_cell {
        let idx = cell_names
            .iter()
            .position(|c| c.as_ref() == name)
            .with_context(|| format!("--root-cell '{name}' not found in latent"))?;
        Ok(Some(labels[idx]))
    } else {
        Ok(type_root.or(gem_root))
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

    /////////////////////////////
    // load frozen embedding θ //
    /////////////////////////////
    // The fit (k-means → MST → curves) and the PHATE layout run on raw θ (Euclidean) by
    // default. `--normalize-latent` L2-normalizes θ instead (cosine geometry): gem θ is
    // cosine-oriented, so that can help when a few extreme-magnitude cells otherwise
    // dominate the raw distances.
    let latent_path = format!("{prefix}.cell_embedding.parquet");
    let cell = DMatrix::<f32>::from_parquet(&latent_path)
        .with_context(|| format!("reading cell embedding {latent_path}"))?;
    let cell_names = cell.rows;
    // Raw θ kept only for `--markers` node annotation, which scores in the same raw
    // latent space `faba annotate` uses (the `theta` below drives the fit).
    let raw_theta: Option<DMatrix<f32>> = args.markers.is_some().then(|| cell.mat.clone());
    let theta = if args.normalize_latent {
        l2_normalize_rows(&cell.mat)
    } else {
        cell.mat
    };
    let n = theta.nrows();
    anyhow::ensure!(n >= 2, "need ≥ 2 cells, got {n}");

    let k = choose_k(n, args.n_centroids);
    anyhow::ensure!(k >= 2, "need ≥ 2 centroids, got {k}");
    info!(
        "lineage: {n} cells × {} dims → {k} centroids",
        theta.ncols()
    );

    // Velocity δ, loaded EARLY — the grouping below may cluster on it (--cluster-space), and the
    // directed forest orients on it further down.
    let velocity_path = format!("{prefix}.velocity.parquet");
    let have_velocity = !args.no_orient_velocity && Path::new(&velocity_path).exists();
    let velocity: Option<DMatrix<f32>> = if have_velocity {
        let vel = DMatrix::<f32>::from_parquet(&velocity_path)
            .with_context(|| format!("reading velocity {velocity_path}"))?;
        anyhow::ensure!(
            vel.mat.nrows() == n,
            "velocity rows ({}) != latent rows ({n})",
            vel.mat.nrows()
        );
        Some(vel.mat)
    } else {
        if !args.no_orient_velocity {
            warn!("velocity file {velocity_path} absent; forest falls back to the geometric MST");
        }
        None
    };

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

    /////////////////////////////////////////////////////////////////
    // optional PHATE 2D layout (cells + nodes + curves projected) //
    /////////////////////////////////////////////////////////////////
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
            &centroids,
            curves,
            &cell_names,
            &phate,
            args.phate_landmarks,
            args.seed,
            &out,
            velocity_aware.then_some((&dirs_map, &branching, &labels)),
            args.normalize_latent,
        )?;
    }
    if args.layout == LayoutKind::Umap {
        emit_umap_layout(
            &theta,
            velocity.as_ref(),
            args.layout_space,
            &centroids,
            curves,
            &cell_names,
            args.phate_knn,
            args.seed,
            &out,
        )?;
    }
    Ok(())
}

/// t-UMAP 2D layout on a **cosine** kNN graph (an alternative to PHATE). Cells, MST
/// nodes, and principal-curve points are stacked into ONE matrix, L2-normalized
/// (cosine geometry), fed through a single fuzzy-kNN graph, and embedded jointly by
/// `matrix_util::umap` — so all three share the 2D space (t-UMAP has no PHATE-style
/// Nyström out-of-sample). The cell rows use the `space` representation (θ, θ+δ, or
/// [θ|δ]); the backbone (nodes/curves) stays on θ (δ is a small increment, so the
/// joint embedding stays coherent). Emits `{out}.{cells,nodes,curves}_2d.parquet`,
/// plus `{out}.velocity_grid_2d.parquet` (scVelo-style gridded arrows) when δ exists.
#[allow(clippy::too_many_arguments)]
fn emit_umap_layout(
    theta: &DMatrix<f32>,
    velocity: Option<&DMatrix<f32>>,
    space: LayoutSpace,
    centroids: &DMatrix<f32>,
    curves: &PrincipalCurves,
    cell_names: &[Box<str>],
    knn: usize,
    seed: u64,
    out: &str,
) -> Result<()> {
    use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
    use matrix_util::umap::Umap;

    let (n, h) = (theta.nrows(), theta.ncols());
    let vel = velocity.filter(|_| space != LayoutSpace::Identity);
    // Effective feature width: Concat doubles it ([θ|δ]); Identity/Nascent keep H.
    let d = if space == LayoutSpace::Concat && vel.is_some() {
        2 * h
    } else {
        h
    };

    // CELLS ONLY — embedding just the cells (per the `space` representation) keeps the
    // trajectory backbone from distorting the manifold (the R exercises are cells-only);
    // the backbone is projected onto the fitted layout afterwards.
    let mut feats = DMatrix::<f32>::zeros(n, d);
    for i in 0..n {
        for j in 0..h {
            let d_ij = vel.map_or(0.0, |v| v[(i, j)]);
            match space {
                LayoutSpace::Identity => feats[(i, j)] = theta[(i, j)],
                LayoutSpace::Nascent => feats[(i, j)] = theta[(i, j)] + d_ij,
                LayoutSpace::Concat => {
                    feats[(i, j)] = theta[(i, j)];
                    feats[(i, h + j)] = d_ij;
                }
            }
        }
    }

    // scale=T (MatOps::scale_columns) → cosine (L2-normalize rows) → t-UMAP.
    let feats_n = l2_normalize_rows(&feats.scale_columns());
    info!("t-UMAP layout (cosine + scale, space={space:?}): {n} cells, knn={knn}");
    let graph = KnnGraph::from_rows(
        &feats_n,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;
    let w = graph.fuzzy_kernel_weights();
    let edges: Vec<(usize, usize, f32)> = graph
        .edges
        .iter()
        .zip(w.iter())
        .map(|(&(i, j), &wt)| (i, j, wt))
        .collect();

    // Deterministic 2D init in [−10, 10] (seeded LCG — no rng dependency).
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut init = vec![0f32; n * 2];
    for v in init.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = (((s >> 33) as f32) / ((1u32 << 31) as f32) - 1.0) * 10.0;
    }
    // t-UMAP (a=b=1) — the uwot::tumap kernel; more spread than standard UMAP.
    let coords = Umap {
        seed,
        ..Umap::tumap()
    }
    .fit(&edges, n, &init);
    let mut cells_2d = DMatrix::<f32>::zeros(n, 2);
    for i in 0..n {
        cells_2d[(i, 0)] = coords[i * 2];
        cells_2d[(i, 1)] = coords[i * 2 + 1];
    }
    write_xy(
        &cells_2d,
        cell_names,
        "cell",
        &format!("{out}.cells_2d.parquet"),
    )?;

    // Project the backbone (nodes + curve points, θ space) onto the cells-only layout:
    // each lands at the mean 2D of its θ-nearest cells (t-UMAP has no Nyström).
    let n_nodes = centroids.nrows();
    let node_names = numbered("node_", n_nodes);
    write_xy(
        &project_onto_cells(centroids, theta, &cells_2d, knn),
        &node_names,
        "node",
        &format!("{out}.nodes_2d.parquet"),
    )?;
    let total_curve: usize = curves.curves.iter().map(|c| c.points.nrows()).sum();
    let mut curve_pts = DMatrix::<f32>::zeros(total_curve, h);
    let mut meta: Vec<(usize, usize)> = Vec::with_capacity(total_curve);
    let mut r = 0usize;
    for (l, c) in curves.curves.iter().enumerate() {
        for g in 0..c.points.nrows() {
            for j in 0..h {
                curve_pts[(r, j)] = c.points[(g, j)];
            }
            meta.push((l, g));
            r += 1;
        }
    }
    write_curves_2d(
        &project_onto_cells(&curve_pts, theta, &cells_2d, knn),
        &meta,
        &format!("{out}.curves_2d.parquet"),
    )?;

    // scVelo-style gridded velocity arrows (≤ a few hundred): project δ into 2D via
    // the θ-neighbour transition, then average onto a coarse grid.
    if let Some(v) = velocity {
        let grid = velocity_grid_arrows(&cells_2d, theta, v, knn);
        write_velocity_grid(&grid, &format!("{out}.velocity_grid_2d.parquet"))?;
    }
    Ok(())
}

/// Project points (θ space, `[m × H]`) onto a cells-only 2D layout: each lands at the
/// mean 2D of its `knn` θ-nearest cells (a simple k-NN out-of-sample, since t-UMAP has
/// no PHATE-style Nyström). Parallel over the `m` points. `cell_theta` `[n × H]`,
/// `cells_2d` `[n × 2]`.
fn project_onto_cells(
    pts: &DMatrix<f32>,
    cell_theta: &DMatrix<f32>,
    cells_2d: &DMatrix<f32>,
    knn: usize,
) -> DMatrix<f32> {
    use rayon::prelude::*;
    let (m, h) = (pts.nrows(), pts.ncols());
    let n = cell_theta.nrows();
    let k = knn.clamp(1, n.max(1));
    let rows: Vec<(f32, f32)> = (0..m)
        .into_par_iter()
        .map(|p| {
            let mut dist: Vec<(f32, usize)> = (0..n)
                .map(|c| {
                    let dd = (0..h)
                        .map(|j| (pts[(p, j)] - cell_theta[(c, j)]).powi(2))
                        .sum::<f32>();
                    (dd, c)
                })
                .collect();
            if k < n {
                dist.select_nth_unstable_by(k - 1, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let (mut sx, mut sy) = (0f32, 0f32);
            for &(_, c) in dist.iter().take(k) {
                sx += cells_2d[(c, 0)];
                sy += cells_2d[(c, 1)];
            }
            (sx / k as f32, sy / k as f32)
        })
        .collect();
    let mut out = DMatrix::<f32>::zeros(m, 2);
    for (p, &(x, y)) in rows.iter().enumerate() {
        out[(p, 0)] = x;
        out[(p, 1)] = y;
    }
    out
}

/// Running sums for one lattice bin of the velocity grid: the member cells' layout
/// positions and their projected 2D velocities, plus the member count that turns
/// those sums into means (and gates the bin on `MIN_PER_CELL`).
#[derive(Default, Clone, Copy)]
struct GridBin {
    sum_x: f32,
    sum_y: f32,
    sum_dx: f32,
    sum_dy: f32,
    n: u32,
}

impl GridBin {
    /// Mean position and mean velocity of the bin's members. Callers check
    /// `n >= MIN_PER_CELL` first, so `n` is nonzero here.
    fn means(&self) -> (f32, f32, f32, f32) {
        let c = self.n as f32;
        (
            self.sum_x / c,
            self.sum_y / c,
            self.sum_dx / c,
            self.sum_dy / c,
        )
    }
}

/// scVelo-style velocity projection + gridding. For each cell, the 2D arrow is the
/// θ-neighbour transition-weighted mean displacement (weight = `max(0, cos(δ_i,
/// θ_j−θ_i))`); those are then averaged onto a `GRID×GRID` grid, keeping only cells
/// with ≥ `MIN_PER_CELL` members. Returns `(x, y, dx, dy)` per occupied grid cell
/// (a few hundred), unit-ish arrows scaled to the grid pitch.
fn velocity_grid_arrows(
    cells_2d: &DMatrix<f32>,
    theta: &DMatrix<f32>,
    delta: &DMatrix<f32>,
    knn: usize,
) -> Vec<(f32, f32, f32, f32)> {
    use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
    const GRID: usize = 30;
    const MIN_PER_CELL: usize = 5;
    let (n, h) = (theta.nrows(), theta.ncols());
    if n == 0 {
        return Vec::new();
    }
    // θ-neighbour graph (identity space) for the transition projection.
    let Ok(graph) = KnnGraph::from_rows(
        theta,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    ) else {
        return Vec::new();
    };
    let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j) in &graph.edges {
        nbrs[i].push(j);
        nbrs[j].push(i);
    }
    // Per-cell 2D velocity via θ-transition weights on δ.
    let mut cell_vel = vec![(0f32, 0f32); n];
    for i in 0..n {
        let (mut vx, mut vy, mut wsum) = (0f32, 0f32, 0f32);
        let di = (0..h)
            .map(|c| delta[(i, c)] * delta[(i, c)])
            .sum::<f32>()
            .sqrt();
        if di < 1e-8 {
            continue;
        }
        for &j in &nbrs[i] {
            // cos(δ_i, θ_j − θ_i): the arrow points along the developmental-FORWARD flow.
            // gem's δ is the nascent (unspliced) increment on top of θ, so `θ+δ` is the
            // emerging/future state — the cell is heading toward `+δ`, i.e. from the
            // progenitor hub outward to the committed compartments.
            let (mut dot, mut dj2) = (0f32, 0f32);
            for c in 0..h {
                let dth = theta[(j, c)] - theta[(i, c)];
                dot += delta[(i, c)] * dth;
                dj2 += dth * dth;
            }
            let cos = if dj2 > 1e-12 {
                dot / (di * dj2.sqrt())
            } else {
                0.0
            };
            let wt = cos.max(0.0);
            if wt > 0.0 {
                let (dx, dy) = (
                    cells_2d[(j, 0)] - cells_2d[(i, 0)],
                    cells_2d[(j, 1)] - cells_2d[(i, 1)],
                );
                let dn = (dx * dx + dy * dy).sqrt().max(1e-8);
                vx += wt * dx / dn;
                vy += wt * dy / dn;
                wsum += wt;
            }
        }
        if wsum > 0.0 {
            cell_vel[i] = (vx / wsum, vy / wsum);
        }
    }
    // Grid-average onto a GRID×GRID lattice over the layout bounds.
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for i in 0..n {
        xmin = xmin.min(cells_2d[(i, 0)]);
        xmax = xmax.max(cells_2d[(i, 0)]);
        ymin = ymin.min(cells_2d[(i, 1)]);
        ymax = ymax.max(cells_2d[(i, 1)]);
    }
    let (wx, wy) = ((xmax - xmin).max(1e-6), (ymax - ymin).max(1e-6));
    let pitch = (wx / GRID as f32).min(wy / GRID as f32);
    let mut acc: HashMap<(usize, usize), GridBin> = HashMap::new();
    for i in 0..n {
        let gx = (((cells_2d[(i, 0)] - xmin) / wx * GRID as f32) as usize).min(GRID - 1);
        let gy = (((cells_2d[(i, 1)] - ymin) / wy * GRID as f32) as usize).min(GRID - 1);
        let e = acc.entry((gx, gy)).or_default();
        e.sum_x += cells_2d[(i, 0)];
        e.sum_y += cells_2d[(i, 1)];
        e.sum_dx += cell_vel[i].0;
        e.sum_dy += cell_vel[i].1;
        e.n += 1;
    }
    let mut out = Vec::new();
    for bin in acc.into_values() {
        if (bin.n as usize) < MIN_PER_CELL {
            continue;
        }
        let (mx, my, mdx, mdy) = bin.means();
        let mag = (mdx * mdx + mdy * mdy).sqrt();
        if mag < 1e-6 {
            continue;
        }
        // Scale each arrow to ~one grid pitch (unit direction × pitch).
        out.push((mx, my, mdx / mag * pitch, mdy / mag * pitch));
    }
    out
}

/// Write `{out}.velocity_grid_2d.parquet` — gridded arrows `[x, y, dx, dy]`.
fn write_velocity_grid(arrows: &[(f32, f32, f32, f32)], path: &str) -> Result<()> {
    let mut m = DMatrix::<f32>::zeros(arrows.len(), 4);
    for (i, &(x, y, dx, dy)) in arrows.iter().enumerate() {
        m[(i, 0)] = x;
        m[(i, 1)] = y;
        m[(i, 2)] = dx;
        m[(i, 3)] = dy;
    }
    let cols: Vec<Box<str>> = ["x", "y", "dx", "dy"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    m.to_parquet_with_names(path, (None, None), Some(&cols))?;
    info!("Wrote {path} ({} gridded velocity arrows)", arrows.len());
    Ok(())
}

/////////////////////////////////////////
// Marker annotation of the trajectory //
/////////////////////////////////////////

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
    /// Stability bootstrap over the marker panel + the k-means grouping (`None` ⇒ point estimate).
    bootstrap: Option<MarkerBootstrapConfig>,
    /// θ `[N × H]`, kept so a replicate can re-run k-means on it under a fresh seed.
    theta: &'a DMatrix<f32>,
    kmeans_iter: usize,
    seed: u64,
}

/// Name each trajectory node by cell type: run the `faba annotate` term-ORA core over the
/// MST-node grouping (raw θ vs the gem β dictionary), giving every node a permutation-
/// calibrated call. Writes `{out}.lineage_annot.*` and returns the per-node
/// [`CommunityCalls`]. Run BEFORE root selection (it doesn't depend on the root) so
/// `--root-type` can pick the root from these calls; the caller writes
/// `{out}.trajectory_annotation.parquet` afterwards via [`write_trajectory_annotation`].
fn compute_node_calls(a: &AnnotateTrajArgs) -> Result<CommunityCalls> {
    // The co-embedded feature vectors, not β — see `faba::gem::marker_embedding` for why a
    // Euclidean nearest-centroid call against β is not a well-posed question.
    let beta = faba::gem::marker_embedding::load_gene_embedding(
        a.prefix,
        faba::gem::marker_embedding::Modality::Spliced,
    )?;
    let cfg = TermOraConfig {
        n_perm: a.num_perm,
        // `--seed` drives the whole fit; it should drive the annotation's randomness too. It was
        // silently falling through to `TermOraConfig`'s own default of 42, so varying `--seed`
        // moved the centroids but left the permutation null (and now the bootstrap) untouched.
        min_markers: 3,
        seed: a.seed,
        obo: a.obo.map(str::to_owned),
        label_cl: a.label_cl.map(str::to_owned),
        panel_perm: 0,
        support_perm: 0,
        bootstrap: a.bootstrap.clone(),
        ..TermOraConfig::default()
    };
    let input = InputEmbeddings {
        feature_emb: &beta.mat,
        gene_names: &beta.rows,
        cell_emb: a.raw_theta,
        cell_names: a.cell_names,
    };

    // One replicate's grouping: the same k-means, reseeded.
    //
    // The trajectory's own nodes stay put — they are the structure, and `--seed` is meant to
    // reproduce them. This redraws the grouping *inside* the annotation bootstrap only, which is
    // what gives the resampling something to disagree about. Holding the partition fixed and
    // resampling the panel alone is close to a no-op: a node's argmax does not flip because a
    // few markers were redrawn, so every call comes back with support ≈ 1 and nothing abstains.
    // k-means++ from a fresh seed lands on genuinely different nodes, and a label that survives
    // *that* is a label worth printing on a trajectory.
    let regroup = |seed: u64| -> Result<Vec<usize>> {
        let (_, labels) = kmeans_centroids_seeded(a.theta, a.k, a.kmeans_iter, seed);
        Ok(labels)
    };
    let regroup: Option<&Regroup<'_>> = a.bootstrap.as_ref().map(|_| &regroup as &Regroup<'_>);

    annotate_with_communities(
        &input,
        a.markers,
        &format!("{}.lineage_annot", a.out),
        true, // IDF-weight markers, as `faba annotate` does by default
        a.labels,
        a.k,
        regroup,
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

/// Cross the per-node calls with the rooted forest → the labeled trajectory: one row per
/// node — `role` (root | terminal | internal), `cell_type`, `confidence`. Terminals are
/// derived from the rooted children (a node with no children), not from the orientation,
/// so abstained edges cannot misclassify a leaf.
fn write_trajectory_annotation(calls: &CommunityCalls, br: &Branching, path: &str) -> Result<()> {
    let k = br.parent.len();
    let mut has_child = vec![false; k];
    for p in br.parent.iter().flatten() {
        if *p < k {
            has_child[*p] = true;
        }
    }
    let node_names = numbered("node_", k);
    let roles: Vec<Box<str>> = (0..k)
        .map(|node| {
            if br.parent[node].is_none() {
                Box::from("root")
            } else if !has_child[node] {
                Box::from("terminal")
            } else {
                Box::from("internal")
            }
        })
        .collect();
    write_named_table(
        path,
        "node",
        &node_names,
        &[
            (Box::from("role"), Column::Str(&roles)),
            (Box::from("cell_type"), Column::Str(&calls.labels)),
            (Box::from("confidence"), Column::F32(&calls.confidence)),
        ],
    )
    .with_context(|| format!("writing {path}"))?;
    info!("wrote {path} ({k} nodes; {} root(s))", br.roots.len());
    Ok(())
}

/////////////////////
// PHATE 2D layout //
/////////////////////

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
/// cell is a landmark (exact PHATE). Above that, PHATE runs on `n_landmarks`
/// **k-means centroids** and the rest are lifted with the Nyström projector — capping
/// the O(n³) PHATE work at the landmark budget and making the remainder linear in N.
/// Returns `(landmark_features L×D, coords L×2)`.
fn phate_landmark_layout(
    theta_n: &DMatrix<f32>,
    phate: &PhateArgs,
    n_landmarks: usize,
    seed: u64,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let n = theta_n.nrows();
    if n <= n_landmarks || n_landmarks < 3 {
        return (theta_n.clone(), phate_layout_2d(theta_n, phate));
    }
    // Landmarks = k-means centroids (density-representative), like reference PHATE's
    // spectral landmarks. A plain stride subsample under-represents structure and the
    // Nyström lift then smears it — collapsing the layout onto its principal curve.
    // A loose 15-iteration cap is plenty: landmarks only seed the Nyström base, so tight
    // k-means convergence isn't needed (and k ≈ 2000 rarely stabilizes exactly anyway).
    let (land, _labels) = kmeans_centroids_seeded(theta_n, n_landmarks, 15, seed);
    let coords = phate_layout_2d(&land, phate);
    (land, coords)
}

/// Lay the cells out with PHATE (trajectory-preserving), then place the node
/// centroids and principal-curve points into the *same* space via the alpha-decay
/// Nyström projection — so the trajectory overlays faithfully. `project_cells_nystrom`
/// takes points as columns (D × n), hence the transposes.
///
/// The layout runs on raw θ (Euclidean) by default; `--normalize-latent` switches it (and the
/// fit) to L2-normalized θ (cosine geometry). For large N, PHATE runs on **k-means landmarks**
/// and every cell/node/curve point is lifted onto that layout via the same Nyström projector.
/// Edge → its tested direction, keyed by the canonical `(min, max)` node pair.
type DirsMap<'a> = HashMap<(usize, usize), &'a EdgeDirection>;

/// Warp step as a fraction of the mean selected-edge length.
const WARP_STEP_FRAC: f32 = 0.15;

/// Nudge each node along the net 2D flow of its confident selected edges (child downstream,
/// parent upstream), magnitude ∝ confidence and `WARP_STEP_FRAC` of the mean edge length;
/// cells follow their node. Abstained/geometry-only regions stay put.
fn warp_layout_along_flow(
    nodes_2d: &mut DMatrix<f32>,
    cells_2d: &mut DMatrix<f32>,
    dirs_map: &DirsMap,
    br: &Branching,
    labels: &[usize],
) {
    let k = nodes_2d.nrows();
    let mut disp = DMatrix::<f32>::zeros(k, 2);
    let (mut len_sum, mut len_cnt) = (0f32, 0f32);
    for v in 0..k {
        let Some(p) = br.parent[v] else { continue };
        let Some(d) = dirs_map.get(&undirected(p, v)) else {
            continue;
        };
        if d.call == EdgeCall::Abstain {
            continue;
        }
        let dx = nodes_2d[(v, 0)] - nodes_2d[(p, 0)];
        let dy = nodes_2d[(v, 1)] - nodes_2d[(p, 1)];
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        len_sum += len;
        len_cnt += 1.0;
        let (ux, uy) = (d.confidence * dx / len, d.confidence * dy / len);
        disp[(v, 0)] += ux;
        disp[(v, 1)] += uy;
        disp[(p, 0)] -= ux;
        disp[(p, 1)] -= uy;
    }
    let step = if len_cnt > 0.0 {
        WARP_STEP_FRAC * len_sum / len_cnt
    } else {
        0.0
    };
    for v in 0..k {
        nodes_2d[(v, 0)] += step * disp[(v, 0)];
        nodes_2d[(v, 1)] += step * disp[(v, 1)];
    }
    for i in 0..cells_2d.nrows() {
        let l = labels[i];
        if l < k {
            cells_2d[(i, 0)] += step * disp[(l, 0)];
            cells_2d[(i, 1)] += step * disp[(l, 1)];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_phate_layout(
    theta: &DMatrix<f32>,
    centroids: &DMatrix<f32>,
    curves: &PrincipalCurves,
    cell_names: &[Box<str>],
    phate: &PhateArgs,
    n_landmarks: usize,
    seed: u64,
    out: &str,
    warp: Option<(&DirsMap, &Branching, &[usize])>,
    normalize: bool,
) -> Result<()> {
    let n = theta.nrows();
    // Cosine metric = rows projected to the unit sphere (Euclidean distance there is
    // 2(1−cos)); with `normalize = false` PHATE runs on raw θ, i.e. true Euclidean.
    let norm = |m: &DMatrix<f32>| {
        if normalize {
            l2_normalize_rows(m)
        } else {
            m.clone()
        }
    };
    let theta_n = norm(theta);
    let (land_feat, land_2d) = phate_landmark_layout(&theta_n, phate, n_landmarks, seed);
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
    let mut cells_2d = if exact {
        land_2d.clone()
    } else {
        project_cells_nystrom(&theta_n.transpose(), &land_t, &land_2d, knn, alpha)
    };

    // Nodes + curve points always lift onto the landmark layout via Nyström.
    let mut nodes_2d =
        project_cells_nystrom(&norm(centroids).transpose(), &land_t, &land_2d, knn, alpha);

    if let Some((dirs_map, br, labels)) = warp {
        warp_layout_along_flow(&mut nodes_2d, &mut cells_2d, dirs_map, br, labels);
    }

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
    let curves_2d = project_cells_nystrom(&norm(&cpts).transpose(), &land_t, &land_2d, knn, alpha);

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

/////////////////////
// Parquet writers //
/////////////////////

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

/// Geometry floor: an abstained edge can still connect via geometry.
const BETA: f32 = 0.2;
/// Weight of a velocity-contradicted orientation — near zero so it is never selected.
const BETA_LOW: f32 = 1e-3;

/// Build the directed arc set + per-node `root_affinity` for [`max_branching`]. Each
/// candidate edge yields two opposing arcs weighted by geometric affinity × direction
/// support; abstained edges contribute geometry only. A user `root_hint` is pinned as a
/// root via an infinite affinity. `root_affinity_arg` (τ_root) overrides the default
/// (median arc weight) and controls forest granularity.
fn assemble_arcs(
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
fn write_edge_directions(dirs: &[EdgeDirection], br: &Branching, path: &str) -> Result<()> {
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
fn write_trees(br: &Branching, labels: &[usize], dirs_map: &DirsMap, path: &str) -> Result<()> {
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

/// `cell × [pseudotime, branch, tree, order_confidence]`: primary-lineage pseudotime +
/// global lineage id + forest tree id + the min edge confidence on the cell's root→node
/// path (0 where the ordering crosses an abstained/geometry-only edge). `pseudotime` and
/// `branch` stay the first two columns for back-compatibility with `faba dyn-assoc`.
fn write_pseudotime(
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
