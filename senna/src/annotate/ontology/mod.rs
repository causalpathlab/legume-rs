//! `senna annotate-ontology` — hierarchical, multi-resolution cell-type calling.
//!
//! A post-processor on the cluster × celltype matrix written by
//! `annotate-by-enrichment`. It places each cluster on the Cell Ontology `is_a`
//! tree at the *deepest resolution the data supports*, abstaining on sibling
//! ties, via the TreeBH procedure (Bogomolov–Peterson–Benjamini–Sabatti,
//! "Hypotheses on a tree", Biometrika 2021; DOI 10.1093/biomet/asaa086).
//!
//! Pipeline:
//!   1. read the restandardized-ES z-scores (`cluster_celltype_es_std`) and map
//!      each celltype column to a CL id via a curated `label→CL` TSV;
//!   2. build the sub-DAG induced by those leaf CL ids (reduced to a tree by
//!      keeping each node's most-specific induced parent), and augment every
//!      labelled node with a `:self` leaf carrying its own z so internal-level
//!      signal (e.g. generic "T") is not lost;
//!   3. per cluster: leaf p = Φ(−z), Simes-combine bottom-up, run TreeBH
//!      top-down; the deepest rejected label = the assignment, "no rejected
//!      child" = abstention, "nothing rejected" = `cannot_explain`.
//!
//! Outputs `{out}.ontology_assignment.tsv` and (for soft viz colouring)
//! `{out}.ontology_node_mass.parquet` (Σ of descendant-leaf Q per node).

mod treebh;

use super::args::AnnotateOntologyArgs;
use crate::embed_common::Mat;
use crate::run_manifest::{self, RunManifest};
use anyhow::{anyhow, Context, Result};
use auxiliary_data::ontology::Ontology;
use log::{info, warn};
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::IoOps;
use rustc_hash::{FxHashMap, FxHashSet};
use special::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Upper-tail standard-normal probability `P(Z > z) = ½·erfc(z/√2)` via the
/// exact complementary error function (`special::Error`, already a senna dep) —
/// converts a z-score to a one-sided p-value. NaN → 1 (no evidence); clamped to
/// a small positive floor.
fn norm_sf(z: f64) -> f64 {
    if z.is_nan() {
        return 1.0; // clamp would propagate NaN
    }
    (0.5 * (z / std::f64::consts::SQRT_2).compl_error()).clamp(1e-12, 1.0)
}

/// Parse a `label<TAB>CL:id` TSV (flexible delimiter; `#` comments; header row
/// with a non-`CL:` second column is skipped).
fn parse_label_cl(path: &str) -> Result<FxHashMap<Box<str>, Box<str>>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read label→CL map: {path}"))?;
    let mut map: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Split on TAB/comma only — NOT space: cell-type labels may contain
        // spaces (e.g. "CD4 T cell"), which must match the marker-file column.
        let mut it = line
            .split(['\t', ','])
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let (Some(label), Some(cl)) = (it.next(), it.next()) else {
            continue;
        };
        if !cl.starts_with("CL:") {
            continue; // header or malformed
        }
        if let Some(prev) = map.get(label) {
            if prev.as_ref() != cl {
                return Err(anyhow!(
                    "label {label:?} mapped to two different CL ids in {path}: {prev} vs {cl}"
                ));
            }
        }
        map.insert(label.into(), cl.into());
    }
    if map.is_empty() {
        return Err(anyhow!("no `label<TAB>CL:id` rows parsed from {path}"));
    }
    Ok(map)
}

/// Replace the `cluster_celltype_q.parquet` suffix of the manifest's Q path with
/// another enrichment artifact suffix.
fn sibling_artifact(q_path: &str, suffix: &str) -> Result<String> {
    q_path
        .strip_suffix("cluster_celltype_q.parquet")
        .map(|stem| format!("{stem}{suffix}"))
        .ok_or_else(|| {
            anyhow!(
                "manifest cluster_celltype_q path {q_path:?} does not end in \
                 'cluster_celltype_q.parquet'; cannot locate sibling {suffix}"
            )
        })
}

/// The augmented induced tree: CL nodes (tree-reduced) + a `:self` leaf per
/// labelled node. Cluster-independent; built once.
struct TreeModel {
    children: Vec<Vec<usize>>,
    depth: Vec<usize>,    // tree depth from virtual root (root = 0)
    cl_id: Vec<Box<str>>, // CL id (self-leaf shares its CL node's id)
    disp: Vec<Box<str>>,  // display name
    is_self_leaf: Vec<bool>,
    label_col: Vec<Option<usize>>, // Some(col) only for self-leaves
    postorder: Vec<usize>,         // children before parents
    sub_cols: Vec<Vec<usize>>,     // label columns at-or-below each node
    root: usize,
}

impl TreeModel {
    /// Build the **collapsed label tree**: nodes are the marker labels
    /// themselves (each label's parent = its nearest *labelled* CL ancestor),
    /// plus a `:self` leaf per label carrying its own evidence. CL's generic
    /// Steiner ancestors (lymphocyte, leukocyte, …) and dead cross-cutting axes
    /// are dropped — they only add depth and dilute the bottom-up Simes, and the
    /// resolvable abstention levels are the labels we actually have markers for.
    fn build(
        onto: &Ontology,
        col_cl: &[Box<str>], // CL id per celltype column (all present)
        col_label: &[Box<str>],
    ) -> Result<Self> {
        let ncol = col_cl.len();
        // CL ancestor set per column (used only to relate labels to each other).
        let anc: Vec<FxHashSet<Box<str>>> =
            col_cl.iter().map(|cl| onto.ancestors_or_self(cl)).collect();
        let anc_size: Vec<usize> = anc.iter().map(FxHashSet::len).collect();

        // Layout: 0 = virtual root; 1..=ncol = label nodes (col c → c+1);
        // ncol+1.. = self-leaves (col c → ncol+1+c).
        let root = 0usize;
        let n = 1 + 2 * ncol;
        let struct_of = |c: usize| 1 + c;
        let selfleaf_of = |c: usize| 1 + ncol + c;

        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut cl_id: Vec<Box<str>> = vec!["ROOT".into(); n];
        let mut disp: Vec<Box<str>> = vec!["root".into(); n];
        let mut is_self_leaf = vec![false; n];
        let mut label_col: Vec<Option<usize>> = vec![None; n];

        for c in 0..ncol {
            let s = struct_of(c);
            cl_id[s] = col_cl[c].clone();
            disp[s] = onto
                .name(&col_cl[c])
                .map_or_else(|| col_cl[c].clone(), Into::into);
            let sl = selfleaf_of(c);
            cl_id[sl] = col_cl[c].clone();
            disp[sl] = format!("{}:self", col_label[c]).into();
            is_self_leaf[sl] = true;
            label_col[sl] = Some(c);
            children[s].push(sl);
        }
        // Nearest labelled ancestor → tree edges among labels.
        for c in 0..ncol {
            let mut best: Option<usize> = None;
            for c2 in 0..ncol {
                if c2 == c || col_cl[c2] == col_cl[c] {
                    continue; // self / same CL node ⇒ not a strict ancestor
                }
                if anc[c].contains(&col_cl[c2]) {
                    best = match best {
                        Some(b) if anc_size[b] >= anc_size[c2] => Some(b),
                        _ => Some(c2), // deepest (most ancestors) wins ⇒ nearest
                    };
                }
            }
            let s = struct_of(c);
            let p = best.map_or(root, struct_of);
            children[p].push(s);
        }

        Ok(Self::finalize(
            children,
            cl_id,
            disp,
            is_self_leaf,
            label_col,
            root,
        ))
    }

    /// Compute depth / postorder / `sub_cols` from the assembled node arrays and
    /// pack a `TreeModel`.
    fn finalize(
        mut children: Vec<Vec<usize>>,
        cl_id: Vec<Box<str>>,
        disp: Vec<Box<str>>,
        is_self_leaf: Vec<bool>,
        label_col: Vec<Option<usize>>,
        root: usize,
    ) -> Self {
        for ch in &mut children {
            ch.sort_unstable();
        }
        let n = children.len();
        let mut depth = vec![0usize; n];
        let mut postorder = Vec::with_capacity(n);
        let mut stack = vec![(root, false)];
        while let Some((v, processed)) = stack.pop() {
            if processed {
                postorder.push(v);
                continue;
            }
            stack.push((v, true));
            for &c in &children[v] {
                depth[c] = depth[v] + 1;
                stack.push((c, false));
            }
        }
        let mut sub_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &node in &postorder {
            if let Some(col) = label_col[node] {
                sub_cols[node].push(col);
            }
            let kids = children[node].clone();
            for c in kids {
                let cc = sub_cols[c].clone();
                sub_cols[node].extend(cc);
            }
            sub_cols[node].sort_unstable();
            sub_cols[node].dedup();
        }
        Self {
            children,
            depth,
            cl_id,
            disp,
            is_self_leaf,
            label_col,
            postorder,
            sub_cols,
            root,
        }
    }
}

/// Per-cluster assignment record.
struct Assignment {
    cluster: Box<str>,
    assigned_cl: Box<str>,
    assigned_name: Box<str>,
    depth: usize,
    abstained: bool,
    unresolved_below: Vec<Box<str>>,
    supported: Vec<Box<str>>,
    multi_lineage: bool,
    cannot_explain: bool,
}

pub fn run(args: &AnnotateOntologyArgs) -> Result<()> {
    let out: String = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => run_manifest::derive_out_prefix(&args.from),
    };
    mkdir_parent(&out)?;

    // ----- manifest → enrichment matrices -----
    let (mut manifest, manifest_dir) = RunManifest::load(Path::new(args.from.as_ref()))?;
    let q_rel = manifest
        .annotate
        .cluster_celltype_q
        .clone()
        .ok_or_else(|| {
            anyhow!(
                "manifest has no `annotate.cluster_celltype_q` — run \
             `senna annotate-by-enrichment --from {} -m <markers>` first",
                args.from
            )
        })?;
    let q_abs = run_manifest::resolve(&manifest_dir, &q_rel)
        .to_string_lossy()
        .into_owned();

    // Score preference: explicit --use-perm-p → pooled count p; otherwise the
    // correlation-preserving permutation z when present, else the row-
    // randomization restandardized ES. Both z forms use leaf p = Φ(−z).
    let (score_abs, from_z) = if args.use_perm_p {
        (
            sibling_artifact(&q_abs, "cluster_celltype_p.parquet")?,
            false,
        )
    } else {
        let perm_z = sibling_artifact(&q_abs, "cluster_celltype_perm_z.parquet")?;
        if Path::new(&perm_z).exists() {
            (perm_z, true)
        } else {
            (
                sibling_artifact(&q_abs, "cluster_celltype_es_std.parquet")?,
                true,
            )
        }
    };

    let score = Mat::from_parquet_with_row_names(&score_abs, Some(0))
        .with_context(|| format!("reading score matrix {score_abs}"))?;
    let q = Mat::from_parquet_with_row_names(&q_abs, Some(0))
        .with_context(|| format!("reading Q matrix {q_abs}"))?;
    anyhow::ensure!(
        score.cols == q.cols,
        "score and Q matrices have mismatched celltype columns"
    );
    anyhow::ensure!(
        score.rows == q.rows,
        "score and Q matrices have mismatched cluster rows (different order/clustering)"
    );
    let cluster_names = score.rows.clone();
    let celltype_names = score.cols.clone();
    let (n_clusters, n_types) = (score.mat.nrows(), score.mat.ncols());
    info!(
        "loaded {n_clusters} clusters × {n_types} celltypes from {} ({})",
        score_abs,
        if from_z { "z→p" } else { "permutation p" }
    );

    let score_in = if from_z {
        OntologyScore::Z(&score.mat)
    } else {
        OntologyScore::Pvalue(&score.mat)
    };
    let (assign_path, mass_path) = annotate_ontology_core(
        &OntologyParams {
            out: &out,
            label_cl: &args.label_cl,
            obo: &args.obo,
            fdr_q: args.fdr_q,
            by: args.by,
        },
        score_in,
        Some(&q.mat),
        &cluster_names,
        &celltype_names,
    )?;

    manifest.annotate.ontology_assignment =
        Some(run_manifest::rel_to_manifest(&manifest_dir, &assign_path));
    manifest.annotate.ontology_node_mass =
        Some(run_manifest::rel_to_manifest(&manifest_dir, &mass_path));
    manifest.save(Path::new(args.from.as_ref()))?;

    info!("senna annotate-ontology complete");
    Ok(())
}

/// Parameters for the shared ontology-annotation core (front-end agnostic).
pub(crate) struct OntologyParams<'a> {
    pub out: &'a str,
    pub label_cl: &'a str,
    pub obo: &'a str,
    pub fdr_q: f64,
    pub by: bool,
}

/// How to interpret a `units × celltype` score matrix — keeps the matrix and its
/// meaning inseparable (no separate `from_z` flag for callers to keep in sync).
pub(crate) enum OntologyScore<'a> {
    /// z-scores → leaf p = Φ(−z).
    Z(&'a Mat),
    /// already one-sided p-values in (0, 1].
    Pvalue(&'a Mat),
}

impl OntologyScore<'_> {
    fn mat(&self) -> &Mat {
        match self {
            Self::Z(m) | Self::Pvalue(m) => m,
        }
    }
    fn leaf_p(&self, k: usize, col: usize) -> f64 {
        match self {
            Self::Z(m) => norm_sf(m[(k, col)] as f64),
            Self::Pvalue(m) => (m[(k, col)] as f64).clamp(1e-12, 1.0),
        }
    }
}

/// Per-row softmax — derives a node-mass posterior when a front-end supplies no
/// Q, so the fabrication lives here once rather than in each caller.
fn row_softmax(m: &Mat) -> Mat {
    let mut out = Mat::zeros(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        let mx = (0..m.ncols()).fold(f32::NEG_INFINITY, |a, j| a.max(m[(i, j)]));
        let mut s = 0.0f32;
        for j in 0..m.ncols() {
            let e = (m[(i, j)] - mx).exp();
            out[(i, j)] = e;
            s += e;
        }
        let s = s.max(1e-12);
        for j in 0..m.ncols() {
            out[(i, j)] /= s;
        }
    }
    out
}

/// Shared TreeBH ontology annotation over a `units × celltype` score matrix.
///
/// `score` carries its own interpretation ([`OntologyScore`]); `q` is the soft
/// posterior used only for node-mass viz (`None` ⇒ derived from `score`). The
/// engine is indifferent to where the matrix came from — **any front-end that
/// emits a units × celltype z-matrix** (cluster enrichment, per-cell/community
/// projection, …) can call this. Writes `{out}.ontology_assignment.tsv` +
/// `{out}.ontology_node_mass.parquet` and returns their paths; manifest wiring is
/// the caller's job.
pub(crate) fn annotate_ontology_core(
    params: &OntologyParams,
    score: OntologyScore,
    q: Option<&Mat>,
    cluster_names: &[Box<str>],
    celltype_names: &[Box<str>],
) -> Result<(String, String)> {
    let out = params.out;
    let sm = score.mat();
    let (n_clusters, n_types) = (sm.nrows(), sm.ncols());
    if let Some(q) = q {
        anyhow::ensure!(
            q.nrows() == n_clusters && q.ncols() == n_types,
            "score and Q matrices have mismatched shape"
        );
    }
    // Node-mass source: the supplied posterior Q, else a row-softmax of `score`.
    let q_derived: Option<Mat> = if q.is_none() {
        Some(row_softmax(sm))
    } else {
        None
    };
    let q_mass: &Mat = q.unwrap_or_else(|| q_derived.as_ref().unwrap());

    // ----- label → CL -----
    let label_cl = parse_label_cl(params.label_cl)?;
    let mut col_cl: Vec<Box<str>> = Vec::with_capacity(n_types);
    let mut missing: Vec<Box<str>> = Vec::new();
    for ct in celltype_names {
        match label_cl.get(ct) {
            Some(cl) => col_cl.push(cl.clone()),
            None => missing.push(ct.clone()),
        }
    }
    if !missing.is_empty() {
        return Err(anyhow!(
            "celltype labels absent from {} : {:?}",
            params.label_cl,
            missing
        ));
    }

    let onto = Ontology::load_obo(params.obo)?;
    info!(
        "loaded Cell Ontology: {} terms from {}",
        onto.len(),
        params.obo
    );
    for cl in &col_cl {
        anyhow::ensure!(
            onto.contains(cl),
            "CL id {cl} not in ontology {}",
            params.obo
        );
    }
    let tree = TreeModel::build(&onto, &col_cl, celltype_names)?;

    run_treebh(
        &tree,
        &score,
        q_mass,
        cluster_names,
        celltype_names,
        out,
        params.fdr_q,
        params.by,
    )
}

/// Per-cluster TreeBH loop + output writing over a *pre-built* tree (the CL
/// collapsed-label tree). `col_names` index the score-matrix columns (celltype
/// labels). The call per cluster is the deepest resolved label — the most
/// specific cell type the data supports.
#[allow(clippy::too_many_arguments)]
fn run_treebh(
    tree: &TreeModel,
    score: &OntologyScore,
    q_mass: &Mat,
    cluster_names: &[Box<str>],
    col_names: &[Box<str>],
    out: &str,
    fdr_q: f64,
    by: bool,
) -> Result<(String, String)> {
    let n_clusters = cluster_names.len();

    // ----- per-cluster TreeBH -----
    let mut assignments: Vec<Assignment> = Vec::with_capacity(n_clusters);
    // K × (struct CL nodes) soft mass for viz.
    let struct_nodes: Vec<usize> = (0..tree.children.len())
        .filter(|&i| !tree.is_self_leaf[i] && i != tree.root)
        .collect();
    let mut mass = Mat::zeros(n_clusters, struct_nodes.len());

    // Own celltype column of a label (struct) node — carried by its `:self` child.
    let own_col = |v: usize| -> Option<usize> {
        tree.children[v]
            .iter()
            .find(|&&c| tree.is_self_leaf[c])
            .and_then(|&c| tree.label_col[c])
    };

    for k in 0..n_clusters {
        // leaf p-values for this cluster (internal nodes stay None for Simes).
        let mut leaf_p: Vec<Option<f64>> = vec![None; tree.children.len()];
        for (node, slot) in leaf_p.iter_mut().enumerate() {
            if let Some(col) = tree.label_col[node] {
                *slot = Some(score.leaf_p(k, col));
            }
        }
        let cp = treebh::combine_bottom_up(&tree.children, &tree.postorder, &leaf_p);
        let rejected = treebh::descend(&tree.children, tree.root, &cp, fdr_q, by);

        // A rejected STRUCT (label) node = TreeBH rejected "cluster is type-v or
        // a descendant" (reading self-leaves alone would drop a label rejected at
        // its parent's family whose own `:self` missed the shrunk threshold).
        // Leaf-most = a rejected label with no rejected struct CHILD; since
        // descend reaches a child only when its parent is rejected, "no rejected
        // child" ⟺ "no rejected descendant".
        let rejected_label = |v: usize| v != tree.root && !tree.is_self_leaf[v] && rejected[v];
        let leaf_most: Vec<usize> = (0..tree.children.len())
            .filter(|&v| {
                rejected_label(v)
                    && !tree.children[v]
                        .iter()
                        .any(|&c| !tree.is_self_leaf[c] && rejected[c])
            })
            .collect();

        // Assign from the leaf-most (deepest) rejected labels — the most specific
        // cell type the data resolves.
        let candidates = &leaf_most;
        let (assigned, abstained, unresolved, multi, cannot): (
            Option<usize>,
            bool,
            Vec<Box<str>>,
            bool,
            bool,
        ) = if candidates.is_empty() {
            (None, false, Vec::new(), false, true)
        } else {
            let primary = *candidates
                .iter()
                .max_by(|&&a, &&b| {
                    tree.depth[a]
                        .cmp(&tree.depth[b])
                        .then_with(|| tree.cl_id[b].cmp(&tree.cl_id[a]))
                })
                .unwrap();
            // finer labels strictly below `primary` (its own label excluded) that
            // were NOT resolved — the abstention candidate set.
            let own: FxHashSet<usize> = own_col(primary).into_iter().collect();
            let unresolved: Vec<Box<str>> = tree.sub_cols[primary]
                .iter()
                .filter(|c| !own.contains(c))
                .map(|&c| col_names[c].clone())
                .collect();
            (
                Some(primary),
                !unresolved.is_empty(),
                unresolved,
                candidates.len() > 1,
                false,
            )
        };

        let mut supported: Vec<Box<str>> = (0..tree.children.len())
            .filter(|&v| rejected_label(v))
            .filter_map(|v| own_col(v).map(|c| col_names[c].clone()))
            .collect();
        supported.sort_unstable();
        supported.dedup();

        // `cannot_explain` clusters get an explicit non-CL sentinel, never the
        // internal ROOT node id.
        let (assigned_cl, assigned_name, depth) = match assigned {
            Some(v) => (tree.cl_id[v].clone(), tree.disp[v].clone(), tree.depth[v]),
            None => ("NA".into(), "unassigned".into(), 0usize),
        };
        assignments.push(Assignment {
            cluster: cluster_names[k].clone(),
            assigned_cl,
            assigned_name,
            depth,
            abstained,
            unresolved_below: unresolved,
            supported,
            multi_lineage: multi,
            cannot_explain: cannot,
        });

        // Soft mass = Σ descendant-leaf Q.
        for (j, &node) in struct_nodes.iter().enumerate() {
            let m: f32 = tree.sub_cols[node].iter().map(|&c| q_mass[(k, c)]).sum();
            mass[(k, j)] = m;
        }
    }

    // ----- write outputs -----
    let assign_path = format!("{out}.ontology_assignment.tsv");
    {
        let mut f = File::create(&assign_path)?;
        writeln!(
            f,
            "cluster\tassigned_cl\tassigned_name\tdepth\tabstained\t\
             unresolved_below\tsupported_labels\tmulti_lineage\tcannot_explain"
        )?;
        for a in &assignments {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                a.cluster,
                a.assigned_cl,
                a.assigned_name,
                a.depth,
                a.abstained,
                join_or_dash(&a.unresolved_below),
                join_or_dash(&a.supported),
                a.multi_lineage,
                a.cannot_explain,
            )?;
        }
    }
    info!("wrote {assign_path}");

    let mass_path = format!("{out}.ontology_node_mass.parquet");
    // Column name per struct node = its label (unique; the score-matrix columns),
    // so two labels mapping to the same CL id don't collide into duplicate headers.
    let mass_cols: Vec<Box<str>> = struct_nodes
        .iter()
        .map(|&n| own_col(n).map_or_else(|| tree.cl_id[n].clone(), |c| col_names[c].clone()))
        .collect();
    mass.to_parquet_with_names(
        &mass_path,
        (Some(cluster_names), Some("cluster")),
        Some(&mass_cols),
    )?;
    info!("wrote {mass_path}");

    summarize(&assignments);

    Ok((assign_path, mass_path))
}

fn join_or_dash(v: &[Box<str>]) -> String {
    if v.is_empty() {
        "-".to_string()
    } else {
        v.iter().map(|s| s.as_ref()).collect::<Vec<_>>().join(",")
    }
}

fn summarize(assignments: &[Assignment]) {
    let n = assignments.len();
    let abst = assignments.iter().filter(|a| a.abstained).count();
    let cant = assignments.iter().filter(|a| a.cannot_explain).count();
    let multi = assignments.iter().filter(|a| a.multi_lineage).count();
    eprintln!("\nOntology annotation ({n} clusters)");
    eprintln!("  abstained (resolution-limited): {abst}");
    eprintln!("  cannot_explain (no rejection):  {cant}");
    eprintln!("  multi-lineage:                  {multi}\n");
    let mut counts: FxHashMap<Box<str>, usize> = FxHashMap::default();
    for a in assignments {
        *counts.entry(a.assigned_name.clone()).or_default() += 1;
    }
    let mut rows: Vec<(Box<str>, usize)> = counts.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, c) in rows {
        eprintln!("  {name:32} {c}");
    }
    eprintln!();
    if cant == n && n > 0 {
        warn!("no cluster could be explained — check the score, --fdr-q, and label→CL map");
    }
}
