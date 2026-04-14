//! Data-preparation helpers for `fit_visualize`:
//! - Parquet/TSV loaders for the latent and the dictionary, with automatic
//!   log-space detection (topic models write `log p(·)`, SVD writes real
//!   scores — both flow through the same loader).
//! - Whitening: `L` such that `L L^T = β^T β`, with a Cholesky path and a
//!   symmetric-eigen pseudo-Cholesky fallback for rank-deficient dictionaries.
//! - Per-PB mean-latent accumulation and coverage-based tail pruning.

use crate::embed_common::*;

/// Load the latent file, align its rows to the data-column order, and
/// auto-exponentiate if it was written in log-probability space.
pub(super) fn load_latent_file(path: &str, data_vec: &SparseIoVec) -> anyhow::Result<Mat> {
    use matrix_util::common_io::*;
    use rustc_hash::FxHashMap as HashMap;

    let ext = file_ext(path)?;
    let MatWithNames {
        rows: latent_cells,
        cols: _,
        mat: latent_nk,
    } = match ext.as_ref() {
        "parquet" => Mat::from_parquet_with_row_names(path, Some(0))?,
        _ => Mat::read_data_with_names(path, &['\t', ',', ' '], Some(0), Some(0))?,
    };

    let data_cells = data_vec.column_names()?;
    let latent_idx: HashMap<&str, usize> = latent_cells
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_ref(), i))
        .collect();

    let missing: Vec<_> = data_cells
        .iter()
        .filter(|c| !latent_idx.contains_key(c.as_ref()))
        .take(5)
        .collect();
    if !missing.is_empty() {
        return Err(anyhow::anyhow!(
            "Latent file missing {} cells from data (e.g., {:?})",
            data_cells
                .iter()
                .filter(|c| !latent_idx.contains_key(c.as_ref()))
                .count(),
            missing
        ));
    }

    let k = latent_nk.ncols();
    let mut reordered = Mat::zeros(data_cells.len(), k);
    for (i, cell) in data_cells.iter().enumerate() {
        let src_idx = latent_idx[cell.as_ref()];
        reordered.row_mut(i).copy_from(&latent_nk.row(src_idx));
    }

    if latent_cells.len() != data_cells.len() {
        info!(
            "Latent has {} cells, data has {} cells; using {} common cells",
            latent_cells.len(),
            data_cells.len(),
            data_cells.len()
        );
    }

    Ok(exp_if_log_space(reordered, "latent", NormalizeAxis::Rows))
}

/// Load the dictionary β (features × K) and auto-exponentiate if it was
/// written in log-probability space. Validates that `K` matches the latent.
pub(super) fn load_dictionary(path: &str, k: usize) -> anyhow::Result<Mat> {
    use matrix_util::common_io::*;

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
    Ok(exp_if_log_space(beta, "dictionary", NormalizeAxis::Cols))
}

#[derive(Clone, Copy)]
enum NormalizeAxis {
    /// Each row sums to 1 after exp (e.g., topic proportions per cell).
    Rows,
    /// Each column sums to 1 after exp (e.g., gene distribution per topic).
    Cols,
}

/// If `m` has all non-positive values (`max ≤ 0`), treat it as log-
/// probabilities, exponentiate in-place, and normalize along the requested
/// axis. Leaves Gaussian / signed matrices (e.g. SVD scores) untouched.
fn exp_if_log_space(mut m: Mat, tag: &str, axis: NormalizeAxis) -> Mat {
    let max_v = m.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_v > 0.0 {
        return m;
    }
    info!(
        "{}: detected log-space input (max={:.3} ≤ 0); exponentiating to probabilities",
        tag, max_v
    );
    for v in m.iter_mut() {
        *v = v.exp();
    }
    match axis {
        NormalizeAxis::Rows => {
            for i in 0..m.nrows() {
                let s: f32 = m.row(i).iter().sum::<f32>().max(1e-12);
                for j in 0..m.ncols() {
                    m[(i, j)] /= s;
                }
            }
        }
        NormalizeAxis::Cols => {
            for j in 0..m.ncols() {
                let s: f32 = m.column(j).iter().sum::<f32>().max(1e-12);
                for i in 0..m.nrows() {
                    m[(i, j)] /= s;
                }
            }
        }
    }
    m
}

/// Compute `L` such that `L Lᵀ = βᵀ β`.
///
/// Euclidean distance on `z = μ · Lᵀ` equals reconstruction distance
/// `‖β(μ_a − μ_b)‖₂` in feature space, so every downstream geometric method
/// inherits redundancy-invariance from the dictionary: dead atoms contribute
/// zero rows to `L`, and redundant atoms collapse through the factor.
///
/// Tries Cholesky first; falls back to a symmetric-eigen pseudo-Cholesky
/// `L = U · diag(√max(λ, 0))` when β is rank-deficient.
pub(super) fn compute_whitening(beta: &Mat) -> Mat {
    let b_kk = beta.transpose() * beta;
    let k = b_kk.nrows();

    // Cholesky consumes its input on success, so a clone is needed to keep
    // `b_kk` around for the eigen fallback if Cholesky reports rank deficiency.
    if let Some(chol) = b_kk.clone().cholesky() {
        return chol.l();
    }

    info!("β^T β is rank-deficient; using symmetric-eigen pseudo-Cholesky");
    let eig = b_kk.symmetric_eigen();
    let max_eig = eig
        .eigenvalues
        .iter()
        .cloned()
        .fold(0.0f32, f32::max)
        .max(1e-12);
    let tol = 1e-6 * max_eig;

    let mut dead = 0usize;
    let mut sqrt_lam = Mat::zeros(k, k);
    for i in 0..k {
        let lam = eig.eigenvalues[i];
        if lam > tol {
            sqrt_lam[(i, i)] = lam.sqrt();
        } else {
            dead += 1;
        }
    }
    if dead > 0 {
        info!(
            "{}/{} dictionary atoms have negligible norm in β^T β (effectively dead or redundant)",
            dead, k
        );
    }
    eig.eigenvectors * sqrt_lam
}

/// Whiten per-PB mean latent: `z_pb = μ̄_pb · Lᵀ`. Output is `n_pb × K`,
/// with Euclidean / cosine distances on rows equaling reconstruction-space
/// distances.
pub(super) fn whiten_pb_features(pb_latent_mean: &Mat, l_kk: &Mat) -> Mat {
    pb_latent_mean * l_kk.transpose()
}

/// Accumulate the mean latent vector for each PB (indexed by the group
/// membership assigned to `data_vec`).
pub(super) fn compute_pb_latent(latent_nk: &Mat, data_vec: &SparseIoVec) -> anyhow::Result<Mat> {
    let n_cells = latent_nk.nrows();
    let k = latent_nk.ncols();

    let pb_membership = data_vec.get_group_membership(0..n_cells)?;
    let n_pb = *pb_membership.iter().max().unwrap_or(&0) + 1;

    let mut pb_sum = Mat::zeros(n_pb, k);
    let mut pb_count = vec![0usize; n_pb];

    for (cell_idx, &pb_idx) in pb_membership.iter().enumerate() {
        for col in 0..k {
            pb_sum[(pb_idx, col)] += latent_nk[(cell_idx, col)];
        }
        pb_count[pb_idx] += 1;
    }

    for (pb_idx, &count) in pb_count.iter().enumerate() {
        if count > 0 {
            pb_sum.row_mut(pb_idx).scale_mut(1.0 / count as f32);
        }
    }

    Ok(pb_sum)
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
    kept.sort();
    kept
}
