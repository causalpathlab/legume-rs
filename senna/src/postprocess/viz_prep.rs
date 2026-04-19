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

/// Load the dictionary β (features × K) and auto-exponentiate if it was
/// written in log-probability space. Validates that `K` matches the latent.
pub(super) fn load_dictionary(path: &str, k: usize) -> anyhow::Result<Mat> {
    use matrix_util::common_io::file_ext;

    let ext = file_ext(path)?;
    let MatWithNames { mat: beta, .. } = match ext.as_ref() {
        "parquet" => Mat::from_parquet_with_row_names(path, Some(0))?,
        _ => Mat::read_data_with_names(path, &['\t', ',', ' '], Some(0), Some(0))?,
    };

    if beta.ncols() != k {
        return Err(anyhow::anyhow!(
            "Dictionary has {} columns but latent has {} — K must match",
            beta.ncols(),
            k
        ));
    }

    info!(
        "Loaded dictionary: {} features × {} atoms",
        beta.nrows(),
        beta.ncols()
    );
    // If the dictionary was written in log space (all entries ≤ 0),
    // exp + column-normalize to probabilities.
    let max_v = beta.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_v > 0.0 {
        return Ok(beta);
    }
    info!(
        "dictionary: detected log-space input (max={max_v:.3} ≤ 0); exponentiating to probabilities"
    );
    let mut beta = beta;
    for v in beta.iter_mut() {
        *v = v.exp();
    }
    for j in 0..beta.ncols() {
        let s: f32 = beta.column(j).iter().sum::<f32>().max(1e-12);
        for i in 0..beta.nrows() {
            beta[(i, j)] /= s;
        }
    }
    Ok(beta)
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
#[allow(dead_code)] // wired to training subcommands in a follow-up commit
pub(crate) fn write_cell_proj(
    prefix: &str,
    proj_kn: &Mat,
    cell_names: &[Box<str>],
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
    proj_nk.to_parquet_with_names(
        &path,
        (Some(cell_names), Some("cell")),
        Some(&col_names),
    )?;
    info!(
        "Wrote cell projection: {} cells × {} dims → {path}",
        n_cells,
        proj_nk.ncols()
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
