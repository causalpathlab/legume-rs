//! Hierarchical, multi-resolution cell-type calling (TreeBH) over a
//! `units × celltype` score matrix and a cell-type ontology.
//!
//! A post-processor on any `units × celltype` enrichment matrix (cluster ×
//! celltype z from `senna annotate-by-enrichment`, cluster × term ORA from
//! `faba gem-annotate`, …). It places each unit (cluster) on the ontology
//! `is_a` tree at the *deepest resolution the data supports*, abstaining on
//! sibling ties, via the TreeBH procedure (Bogomolov–Peterson–Benjamini–
//! Sabatti, "Hypotheses on a tree", Biometrika 2021; DOI 10.1093/biomet/asaa086).
//!
//! Pipeline:
//!   1. map each celltype column to an ontology id via a curated `label→id` map;
//!   2. build the sub-DAG induced by those leaf ids (reduced to a tree by
//!      keeping each node's most-specific induced parent), and augment every
//!      labelled node with a `:self` leaf carrying its own z so internal-level
//!      signal (e.g. generic "T") is not lost;
//!   3. per unit: leaf p = Φ(−z), Simes-combine bottom-up, run TreeBH top-down;
//!      the deepest rejected label = the assignment, "no rejected child" =
//!      abstention, "nothing rejected" = `cannot_explain`.
//!
//! **Generic over the ontology.** The core never names a concrete ontology
//! type; it reaches the ontology only through three closures
//! (`ancestors_or_self` / `name_of` / `contains`), so this crate needs no OBO
//! parser dependency — the caller loads the ontology (e.g.
//! `auxiliary_data::ontology::Ontology`) and injects the closures.
//!
//! Writes `{out}.ontology_assignment.tsv` and (for soft viz colouring)
//! `{out}.ontology_node_mass.parquet` (Σ of descendant-leaf Q per node).

use crate::treebh;
use crate::Mat;
use anyhow::{anyhow, Result};
use log::{info, warn};
use matrix_util::traits::IoOps;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fs::File;
use std::io::Write;

/// Read-only ontology access the TreeBH core needs, injected as closures so this
/// crate stays free of any OBO-parser dependency. Bundles the three operations
/// the core calls; the caller builds one over its loaded ontology.
pub struct OntologyAccess<'a> {
    /// All `is_a` ancestors of `id` **including `id` itself**.
    pub ancestors_or_self: &'a dyn Fn(&str) -> FxHashSet<Box<str>>,
    /// Human-readable display name for `id`, if any.
    pub name_of: &'a dyn Fn(&str) -> Option<Box<str>>,
    /// Whether `id` is present in the ontology.
    pub contains: &'a dyn Fn(&str) -> bool,
}

/// Upper-tail standard-normal probability `P(Z > z) = ½·erfc(z/√2)` via the
/// exact complementary error function — converts a z-score to a one-sided
/// p-value. NaN → 1 (no evidence); clamped to a small positive floor.
fn norm_sf(z: f64) -> f64 {
    use special::Error;
    if z.is_nan() {
        return 1.0; // clamp would propagate NaN
    }
    (0.5 * (z / std::f64::consts::SQRT_2).compl_error()).clamp(1e-12, 1.0)
}

/// Parse a `label<TAB>id` TSV (flexible delimiter; `#` comments; a header row
/// whose second column fails `id_ok` is skipped). `id_ok` validates the id
/// column (e.g. starts with `CL:`), so a header line is dropped silently.
pub fn parse_label_map(
    path: &str,
    id_ok: impl Fn(&str) -> bool,
) -> Result<FxHashMap<Box<str>, Box<str>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("failed to read label→id map {path}: {e}"))?;
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
        let (Some(label), Some(id)) = (it.next(), it.next()) else {
            continue;
        };
        if !id_ok(id) {
            continue; // header or malformed
        }
        if let Some(prev) = map.get(label) {
            if prev.as_ref() != id {
                return Err(anyhow!(
                    "label {label:?} mapped to two different ids in {path}: {prev} vs {id}"
                ));
            }
        }
        map.insert(label.into(), id.into());
    }
    if map.is_empty() {
        return Err(anyhow!("no `label<TAB>id` rows parsed from {path}"));
    }
    Ok(map)
}

/// The augmented induced tree: ontology nodes (tree-reduced) + a `:self` leaf
/// per labelled node. Unit-independent; built once.
struct TreeModel {
    children: Vec<Vec<usize>>,
    depth: Vec<usize>,    // tree depth from virtual root (root = 0)
    cl_id: Vec<Box<str>>, // ontology id (self-leaf shares its node's id)
    disp: Vec<Box<str>>,  // display name
    is_self_leaf: Vec<bool>,
    label_col: Vec<Option<usize>>, // Some(col) only for self-leaves
    postorder: Vec<usize>,         // children before parents
    sub_cols: Vec<Vec<usize>>,     // label columns at-or-below each node
    root: usize,
}

impl TreeModel {
    /// Build the **collapsed label tree**: nodes are the marker labels
    /// themselves (each label's parent = its nearest *labelled* ancestor),
    /// plus a `:self` leaf per label carrying its own evidence. Generic Steiner
    /// ancestors (lymphocyte, leukocyte, …) and dead cross-cutting axes are
    /// dropped — they only add depth and dilute the bottom-up Simes, and the
    /// resolvable abstention levels are the labels we actually have markers for.
    fn build(
        onto: &OntologyAccess<'_>,
        col_cl: &[Box<str>], // ontology id per celltype column (all present)
        col_label: &[Box<str>],
    ) -> Result<Self> {
        let ncol = col_cl.len();
        // Ancestor set per column (used only to relate labels to each other).
        let anc: Vec<FxHashSet<Box<str>>> = col_cl
            .iter()
            .map(|cl| (onto.ancestors_or_self)(cl))
            .collect();
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
            disp[s] = (onto.name_of)(&col_cl[c]).unwrap_or_else(|| col_cl[c].clone());
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
                    continue; // self / same node ⇒ not a strict ancestor
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

/// Per-unit assignment record.
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

/// Parameters for the ontology-annotation core (front-end agnostic).
pub struct OntologyParams<'a> {
    pub out: &'a str,
    pub fdr_q: f64,
    pub by: bool,
}

/// How to interpret a `units × celltype` score matrix — keeps the matrix and its
/// meaning inseparable (no separate `from_z` flag for callers to keep in sync).
pub enum OntologyScore<'a> {
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
/// posterior used only for node-mass viz (`None` ⇒ derived from `score`).
/// `label_to_id` maps each celltype label → an ontology id; `onto` injects the
/// (generic) ontology access. **Any front-end that emits a units × celltype
/// matrix** (cluster enrichment, per-cell/community projection, term ORA, …)
/// can call this. Writes `{out}.ontology_assignment.tsv` +
/// `{out}.ontology_node_mass.parquet` and returns their paths; manifest wiring
/// is the caller's job.
pub fn annotate_ontology_core(
    params: &OntologyParams,
    score: OntologyScore,
    q: Option<&Mat>,
    label_to_id: &FxHashMap<Box<str>, Box<str>>,
    onto: &OntologyAccess<'_>,
    cluster_names: &[Box<str>],
    celltype_names: &[Box<str>],
) -> Result<(String, String)> {
    let out = params.out;
    let sm = score.mat();
    let (n_clusters, n_types) = (sm.nrows(), sm.ncols());
    anyhow::ensure!(
        cluster_names.len() == n_clusters,
        "cluster_names ({}) != score rows ({n_clusters})",
        cluster_names.len()
    );
    anyhow::ensure!(
        celltype_names.len() == n_types,
        "celltype_names ({}) != score cols ({n_types})",
        celltype_names.len()
    );
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

    /////////////////////////
    // label → ontology id //
    /////////////////////////
    let mut col_cl: Vec<Box<str>> = Vec::with_capacity(n_types);
    let mut missing: Vec<Box<str>> = Vec::new();
    for ct in celltype_names {
        match label_to_id.get(ct) {
            Some(cl) => col_cl.push(cl.clone()),
            None => missing.push(ct.clone()),
        }
    }
    if !missing.is_empty() {
        return Err(anyhow!(
            "celltype labels absent from label→id map: {missing:?}"
        ));
    }
    for cl in &col_cl {
        anyhow::ensure!((onto.contains)(cl), "ontology id {cl} not in ontology");
    }
    let tree = TreeModel::build(onto, &col_cl, celltype_names)?;

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

/// Per-unit TreeBH loop + output writing over a *pre-built* tree (the collapsed
/// label tree). `col_names` index the score-matrix columns (celltype labels).
/// The call per unit is the deepest resolved label — the most specific cell
/// type the data supports.
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

    /////////////////////
    // per-unit TreeBH //
    /////////////////////
    let mut assignments: Vec<Assignment> = Vec::with_capacity(n_clusters);
    // K × (struct ontology nodes) soft mass for viz.
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
        // leaf p-values for this unit (internal nodes stay None for Simes).
        let mut leaf_p: Vec<Option<f64>> = vec![None; tree.children.len()];
        for (node, slot) in leaf_p.iter_mut().enumerate() {
            if let Some(col) = tree.label_col[node] {
                *slot = Some(score.leaf_p(k, col));
            }
        }
        let cp = treebh::combine_bottom_up(&tree.children, &tree.postorder, &leaf_p);
        let rejected = treebh::descend(&tree.children, tree.root, &cp, fdr_q, by);

        // A rejected STRUCT (label) node = TreeBH rejected "unit is type-v or a
        // descendant" (reading self-leaves alone would drop a label rejected at
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

        // `cannot_explain` units get an explicit non-id sentinel, never the
        // internal ROOT node id.
        let (assigned_cl, assigned_name, depth) = match assigned {
            Some(v) => (tree.cl_id[v].clone(), tree.disp[v].clone(), tree.depth[v]),
            None => ("NA".into(), crate::UNASSIGNED_LABEL.into(), 0usize),
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

    ///////////////////
    // write outputs //
    ///////////////////
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
    // so two labels mapping to the same id don't collide into duplicate headers.
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
        warn!("no cluster could be explained — check the score, --fdr-q, and label→id map");
    }
}
