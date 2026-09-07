//! Data-preparation helpers for `fit_layout` (tsne/phate):
//! - Per-PB mean-feature accumulation and coverage-based tail pruning.
//! - Raw-gene-space log1p-CPM construction for PB landmarks.
//! - SVD preprocessing for dimensionality reduction.

use crate::embed_common::*;
use matrix_util::traits::RandomizedAlgs;
use rayon::prelude::*;

/// Accumulate the mean feature vector for each PB group.
///
/// `feat_kn` is `(k × n_cells)` (e.g. the random projection). Cells
/// whose group is `usize::MAX` or ≥ `n_pb` are skipped. Returns
/// `(k × n_pb)` column-major so it pairs directly with `proj_kn`.
pub(super) fn aggregate_features_by_group(
    feat_kn: &Mat,
    pb_membership: &[usize],
    n_pb: usize,
) -> Mat {
    let k = feat_kn.nrows();

    // Bucket cell indices by PB in one O(n_cells) pass, then average
    // each PB's cells in parallel (disjoint PB columns = no contention).
    let mut cells_by_pb: Vec<Vec<usize>> = vec![Vec::new(); n_pb];
    for (cell_idx, &g) in pb_membership.iter().enumerate() {
        if g < n_pb {
            cells_by_pb[g].push(cell_idx);
        }
    }

    let cols: Vec<Vec<f32>> = cells_by_pb
        .par_iter()
        .map(|cells| {
            if cells.is_empty() {
                return vec![0.0f32; k];
            }
            let mut acc = vec![0.0f32; k];
            for &c in cells {
                let src = feat_kn.column(c);
                for (a, v) in acc.iter_mut().zip(src.iter()) {
                    *a += *v;
                }
            }
            let inv = 1.0 / cells.len() as f32;
            for v in &mut acc {
                *v *= inv;
            }
            acc
        })
        .collect();

    let mut out = Mat::zeros(k, n_pb);
    for (p, col) in cols.iter().enumerate() {
        out.column_mut(p).copy_from_slice(col);
    }
    out
}

/// Return the sorted list of PB indices whose cell counts, taken in
/// descending size order, cumulatively cover at least `coverage` of the
/// total cells. Result is in ascending-index order so that downstream
/// slicing preserves relative positions. If `coverage >= 1.0` all PBs are
/// returned.
pub(super) fn select_pb_coverage(pb_size: &[usize], coverage: f32) -> Vec<usize> {
    let n = pb_size.len();
    if coverage >= 1.0 || n == 0 {
        return (0..n).collect();
    }
    let total: usize = pb_size.iter().sum();
    let target = (coverage.clamp(0.0, 1.0) * total as f32).ceil() as usize;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pb_size[b].cmp(&pb_size[a]));

    let mut cum = 0usize;
    let mut cut = 0usize;
    for (i, &idx) in order.iter().enumerate() {
        cum += pb_size[idx];
        cut = i + 1;
        if cum >= target {
            break;
        }
    }
    let mut kept: Vec<usize> = order[..cut].to_vec();
    kept.sort_unstable();
    kept
}

/// Persist the per-cell random projection so `senna layout` can reuse it
/// without re-running `load_and_collapse` on raw data. Writes
/// `{prefix}.cell_proj.parquet` with cells as rows and projection
/// dimensions as columns.
///
/// `proj_kn` is `(proj_dim × n_cells)` column-major (matches
/// `PreparedData.proj_kn`); this helper transposes to row-major cells
/// before serializing.
pub(crate) fn write_cell_proj(
    prefix: &str,
    proj_kn: &Mat,
    cell_names: &[Box<str>],
    keep_idx: Option<&[usize]>,
) -> anyhow::Result<String> {
    let path = format!("{prefix}.cell_proj.parquet");
    let n_cells = proj_kn.ncols();
    anyhow::ensure!(
        cell_names.len() == n_cells,
        "cell_names len {} != proj_kn cols {}",
        cell_names.len(),
        n_cells
    );
    let proj_nk = proj_kn.transpose();
    let col_names: Vec<Box<str>> = (0..proj_nk.ncols())
        .map(|i| format!("p{i}").into_boxed_str())
        .collect();
    let n_emitted = if let Some((mat, names)) =
        crate::output_helpers::cell_subset(&proj_nk, cell_names, keep_idx)
    {
        let n = mat.nrows();
        mat.to_parquet_with_names(&path, (Some(&names), Some("cell")), Some(&col_names))?;
        n
    } else {
        proj_nk.to_parquet_with_names(&path, (Some(cell_names), Some("cell")), Some(&col_names))?;
        n_cells
    };
    info!(
        "Wrote cell projection: {} cells × {} dims → {path}",
        n_emitted,
        proj_nk.ncols()
    );
    Ok(path)
}

/// Serialize the post-refinement cell→pseudobulk membership per
/// coarsening level so a downstream `--from` chain can skip the
/// HNSW + binary-sort + DC-SBM refinement step.
///
/// `cell_to_pb_per_level[i]` is the membership for level `i`, finest-
/// last (matching `PreparedData.collapsed_levels`'s order after the
/// internal `reverse()`). Output is a `[N, num_levels]` `f32` parquet
/// (`f32` for compatibility with the existing parquet writer; the
/// downstream loader casts back to `usize`) with cells as rows and
/// `level_0..level_{L-1}` as columns. PB indices on disk are 0-based.
pub(crate) fn write_cell_to_pb(
    prefix: &str,
    cell_to_pb_per_level: &[Vec<usize>],
    cell_names: &[Box<str>],
    keep_idx: Option<&[usize]>,
) -> anyhow::Result<String> {
    anyhow::ensure!(
        !cell_to_pb_per_level.is_empty(),
        "cell_to_pb_per_level is empty — refusing to write empty membership"
    );
    let n_cells = cell_names.len();
    for (l, lvl) in cell_to_pb_per_level.iter().enumerate() {
        anyhow::ensure!(
            lvl.len() == n_cells,
            "level {l} has {} cells but cell_names has {n_cells}",
            lvl.len()
        );
    }
    let num_levels = cell_to_pb_per_level.len();
    let mut mat = Mat::zeros(n_cells, num_levels);
    for (l, lvl) in cell_to_pb_per_level.iter().enumerate() {
        for (i, &pb) in lvl.iter().enumerate() {
            mat[(i, l)] = pb as f32;
        }
    }
    let col_names: Vec<Box<str>> = (0..num_levels)
        .map(|i| format!("level_{i}").into_boxed_str())
        .collect();
    let path = format!("{prefix}.cell_to_pb.parquet");
    let n_emitted =
        if let Some((m, names)) = crate::output_helpers::cell_subset(&mat, cell_names, keep_idx) {
            let n = m.nrows();
            m.to_parquet_with_names(&path, (Some(&names), Some("cell")), Some(&col_names))?;
            n
        } else {
            mat.to_parquet_with_names(&path, (Some(cell_names), Some("cell")), Some(&col_names))?;
            n_cells
        };
    info!("Wrote cell→pb membership: {n_emitted} cells × {num_levels} levels → {path}");
    Ok(path)
}

////////////////////////////
// Residual-bit tree JSON //
////////////////////////////

#[derive(serde::Serialize)]
struct PbTreeGeneView<'a> {
    gene: &'a str,
    loading: f32,
    lfc: f32,
    weight: f32,
}

#[derive(serde::Serialize)]
struct PbTreeSplitView<'a> {
    id: String,
    depth: usize,
    path: usize,
    left: String,
    right: String,
    n_cells: usize,
    n_left: usize,
    n_right: usize,
    n_left_per_batch: &'a [usize],
    n_right_per_batch: &'a [usize],
    n_genes: usize,
    s1: f32,
    s2: f32,
    sigma: f32,
    mp_edge: f32,
    s1_over_edge: f32,
    passes_edge: bool,
    applied: bool,
    llr_split: f64,
    ve_ratio: f32,
    pos: Vec<PbTreeGeneView<'a>>,
    neg: Vec<PbTreeGeneView<'a>>,
}

#[derive(serde::Serialize)]
struct PbTreeRootView<'a> {
    root: usize,
    n_cells: usize,
    n_genes: usize,
    ve_ratio_residual: f32,
    ve_ratio_marginal: f32,
    splits: Vec<PbTreeSplitView<'a>>,
}

#[derive(serde::Serialize)]
struct PbTreeLeafView<'a> {
    code: usize,
    coarse_group: usize,
    path: usize,
    pb_ids: &'a [usize],
}

#[derive(serde::Serialize)]
struct PbTreeView<'a> {
    coarse_bits: usize,
    depth: usize,
    num_cells: usize,
    num_batches: usize,
    edge_margin: f32,
    reassigned_cells: usize,
    roots: Vec<PbTreeRootView<'a>>,
    leaves: Vec<PbTreeLeafView<'a>>,
}

fn split_id(root: usize, depth: usize, path: usize) -> String {
    format!("{root}:{depth}:{path}")
}

/// Write `{prefix}.pb_tree.json`: the tree behind the finest pseudobulk
/// partition, with each split's contrast genes named. The full
/// loading vectors stay out of the file; the top genes per side carry the
/// interpretation.
pub(crate) fn write_pb_tree(
    prefix: &str,
    tree: &data_beans_alg::collapse_data::PbTree,
    gene_names: &[Box<str>],
) -> anyhow::Result<String> {
    let name = |g: usize| gene_names.get(g).map_or("", |n| n.as_ref());
    let gene_view = |cg: &data_beans_alg::collapse_data::ContrastGene| PbTreeGeneView {
        gene: name(cg.gene),
        loading: cg.loading,
        lfc: cg.lfc,
        weight: cg.weight,
    };
    let roots: Vec<PbTreeRootView<'_>> = tree
        .roots
        .iter()
        .map(|nd| PbTreeRootView {
            root: nd.root,
            n_cells: nd.n_cells,
            n_genes: nd.n_genes,
            ve_ratio_residual: nd.ve_ratio_residual,
            ve_ratio_marginal: nd.ve_ratio_marginal,
            splits: nd
                .splits
                .iter()
                .map(|sp| PbTreeSplitView {
                    id: split_id(sp.root, sp.depth, sp.path),
                    depth: sp.depth,
                    path: sp.path,
                    left: split_id(sp.root, sp.depth + 1, sp.path << 1),
                    right: split_id(sp.root, sp.depth + 1, (sp.path << 1) | 1),
                    n_cells: sp.n_cells,
                    n_left: sp.n_left,
                    n_right: sp.n_right,
                    n_left_per_batch: &sp.n_left_per_batch,
                    n_right_per_batch: &sp.n_right_per_batch,
                    n_genes: sp.n_genes,
                    s1: sp.s1,
                    s2: sp.s2,
                    sigma: sp.sigma,
                    mp_edge: sp.mp_edge,
                    s1_over_edge: if sp.mp_edge > 0.0 {
                        sp.s1 / sp.mp_edge
                    } else {
                        f32::NAN
                    },
                    passes_edge: sp.passes_edge,
                    applied: sp.applied,
                    llr_split: sp.llr_split,
                    ve_ratio: sp.ve_ratio,
                    pos: sp.pos.iter().map(gene_view).collect(),
                    neg: sp.neg.iter().map(gene_view).collect(),
                })
                .collect(),
        })
        .collect();
    let low_mask = (1usize << tree.coarse_bits) - 1;
    let leaves: Vec<PbTreeLeafView<'_>> = tree
        .leaf_to_finest_pb
        .iter()
        .map(|(code, pbs)| PbTreeLeafView {
            code: *code,
            coarse_group: code & low_mask,
            path: code >> tree.coarse_bits,
            pb_ids: pbs,
        })
        .collect();
    let view = PbTreeView {
        coarse_bits: tree.coarse_bits,
        depth: tree.depth,
        num_cells: tree.num_cells,
        num_batches: tree.num_batches,
        edge_margin: tree.edge_margin,
        reassigned_cells: tree.reassigned_cells,
        roots,
        leaves,
    };
    let path = format!("{prefix}.pb_tree.json");
    std::fs::write(&path, serde_json::to_string_pretty(&view)?)?;
    let n_splits: usize = tree.roots.iter().map(|n| n.splits.len()).sum();
    info!(
        "Wrote pseudobulk tree: {} roots, {} splits → {path}",
        tree.roots.len(),
        n_splits
    );
    Ok(path)
}

/// Apply SVD preprocessing: reduce matrix to top N components.
/// Returns U * diag(S) where (U, S, V) = rsvd(mat, `n_components`).
pub(super) fn apply_svd_preprocessing(mat: &Mat, n_components: usize) -> anyhow::Result<Mat> {
    use anyhow::Context;

    let (n_rows, n_cols) = (mat.nrows(), mat.ncols());
    let n_components = n_components.min(n_rows).min(n_cols);

    info!("Running randomized SVD: {n_rows} × {n_cols} → {n_components} components");
    let (u, s, _v) = mat.rsvd(n_components).context("Randomized SVD failed")?;

    let mut reduced = Mat::zeros(n_rows, n_components);
    for i in 0..n_components {
        let col = u.column(i) * s[i];
        reduced.set_column(i, &col);
    }

    info!("SVD done, reduced to {n_components} dims");
    Ok(reduced)
}
