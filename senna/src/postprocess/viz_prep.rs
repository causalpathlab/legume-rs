//! Data-preparation helpers for `fit_visualize`:
//! - Per-PB mean-feature accumulation and coverage-based tail pruning.
//! - Raw-gene-space log1p-CPM construction for PB landmarks.

use crate::embed_common::*;
use rayon::prelude::*;

/// Column-wise log1p-CPM of a `(D × n_pb)` PB count matrix. Returns
/// `(D × n_pb)` so `compute_cosine_similarity` (which operates on
/// *columns*) can be called directly without a transpose.
///
/// Columns (PBs) are scaled to sum = `scale` (default 1e4 for CPM),
/// then `log1p` is applied elementwise. No HVG selection, no z-scoring
/// — cosine similarity handles the remaining scale.
pub(super) fn log1p_cpm_pb(mu_dp: &Mat, scale: f32) -> Mat {
    let d = mu_dp.nrows();
    let n_pb = mu_dp.ncols();
    let cols: Vec<Vec<f32>> = (0..n_pb)
        .into_par_iter()
        .map(|p| {
            let src = mu_dp.column(p);
            let denom = src.iter().sum::<f32>().max(1e-12);
            let s = scale / denom;
            (0..d).map(|g| (src[g] * s).ln_1p()).collect()
        })
        .collect();
    let mut out = Mat::zeros(d, n_pb);
    for (p, col) in cols.iter().enumerate() {
        out.column_mut(p).copy_from_slice(col);
    }
    out
}

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
