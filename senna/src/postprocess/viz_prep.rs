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
    let n_emitted = match crate::output_helpers::cell_subset(&proj_nk, cell_names, keep_idx) {
        Some((mat, names)) => {
            let n = mat.nrows();
            mat.to_parquet_with_names(&path, (Some(&names), Some("cell")), Some(&col_names))?;
            n
        }
        None => {
            proj_nk.to_parquet_with_names(&path, (Some(cell_names), Some("cell")), Some(&col_names))?;
            n_cells
        }
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
    let n_emitted = match crate::output_helpers::cell_subset(&mat, cell_names, keep_idx) {
        Some((m, names)) => {
            let n = m.nrows();
            m.to_parquet_with_names(&path, (Some(&names), Some("cell")), Some(&col_names))?;
            n
        }
        None => {
            mat.to_parquet_with_names(&path, (Some(cell_names), Some("cell")), Some(&col_names))?;
            n_cells
        }
    };
    info!(
        "Wrote cell→pb membership: {} cells × {} levels → {path}",
        n_emitted, num_levels
    );
    Ok(path)
}

/// Apply SVD preprocessing: reduce matrix to top N components.
/// Returns U * diag(S) where (U, S, V) = rsvd(mat, n_components).
pub(super) fn apply_svd_preprocessing(mat: &Mat, n_components: usize) -> anyhow::Result<Mat> {
    use anyhow::Context;

    let (n_rows, n_cols) = (mat.nrows(), mat.ncols());
    let n_components = n_components.min(n_rows).min(n_cols);

    info!(
        "Running randomized SVD: {} × {} → {} components",
        n_rows, n_cols, n_components
    );
    let (u, s, _v) = mat.rsvd(n_components).context("Randomized SVD failed")?;

    let mut reduced = Mat::zeros(n_rows, n_components);
    for i in 0..n_components {
        let col = u.column(i) * s[i];
        reduced.set_column(i, &col);
    }

    info!("SVD done, reduced to {} dims", n_components);
    Ok(reduced)
}
