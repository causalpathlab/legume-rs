//! Shared anchor-selection helpers used by both the training-time β prior
//! (`topic::anchor_prior`) and post-training cell annotation
//! (`postprocess::fit_annotate`).
//!
//! The core idea: given a matrix of pseudobulk expression profiles, pick a
//! small number of "anchor" rows that are maximally orthogonal to one another
//! via greedy Gram-Schmidt. Annotation-side labeling of those anchors against
//! marker genes lives in `postprocess::fit_annotate`.
//!
//! These helpers are intentionally dependency-free beyond `nalgebra` so the
//! module can sit at the crate root and be imported by both the training and
//! postprocessing sides without dragging candle / data-beans into each other's
//! graph.

use crate::embed_common::Mat;
use crate::logging::new_progress_bar;
use nalgebra::DVector;
use rayon::prelude::*;

/// Write `softmax(col)` from a source column view into a destination column
/// view of the same length. Used per anchor when building anchor simplex
/// profiles from log1p expression.
pub(crate) fn softmax_col_into(
    src: nalgebra::DVectorView<f32>,
    mut dst: nalgebra::DVectorViewMut<f32>,
) {
    let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (i, &v) in src.iter().enumerate() {
        let e = (v - max).exp();
        dst[i] = e;
        sum += e;
    }
    if sum > 1e-12 {
        dst /= sum;
    }
}

/// Return a column-z-scored copy of `x_pg` (PB rows × gene columns).
/// Each gene column is shifted and scaled so its mean is 0 and its std is 1.
/// Constant columns get zero (their contribution to residuals is nil anyway).
pub(crate) fn zscore_columns(x_pg: &Mat) -> Mat {
    let mut out = x_pg.clone();
    let n = out.nrows() as f32;
    if n < 2.0 {
        return out;
    }
    for mut col in out.column_iter_mut() {
        let mean = col.iter().sum::<f32>() / n;
        let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let sd = var.sqrt();
        if sd > 1e-8 {
            for v in col.iter_mut() {
                *v = (*v - mean) / sd;
            }
        } else {
            col.fill(0.0);
        }
    }
    out
}

/// Greedy Gram-Schmidt vertex selection. Returns `k` row indices of
/// `x_pg` (PB rows × gene columns) chosen to maximize residual norm at
/// each step. Rows are projected out of all remaining rows after each pick.
///
/// Parallelized over rows: residual is stored as a contiguous row-vector
/// list so the per-iteration argmax-norm and projection both run with
/// `rayon` over the row axis. The K-anchor outer loop stays sequential —
/// each pick depends on the prior projection.
pub(crate) fn gram_schmidt_anchors(x_pg: &Mat, k: usize) -> Vec<usize> {
    let n = x_pg.nrows();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    // `x_pg` is column-major (nalgebra `DMatrix`), so reading rows strides
    // by `n` and trashes cache at D≈36k. One transpose puts each original
    // row into a contiguous column of `xt`, after which extracting rows is
    // a sequential read per worker.
    let xt = x_pg.transpose();
    let mut residual: Vec<DVector<f32>> = (0..n)
        .into_par_iter()
        .map(|i| xt.column(i).into_owned())
        .collect();
    let mut picked: Vec<usize> = Vec::with_capacity(k);
    let mut available: Vec<usize> = (0..n).collect();

    let prog_bar = new_progress_bar(k as u64);
    prog_bar.set_message("Anchor selection");

    for _ in 0..k {
        let (best_pos, best_row) = available
            .par_iter()
            .enumerate()
            .map(|(pos, &row)| (pos, row, residual[row].norm_squared()))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(pos, row, _)| (pos, row))
            .expect("non-empty available");
        picked.push(best_row);
        available.swap_remove(best_pos);

        // Project the anchor out of every row: r_j ← r_j − (r_j · v) v / ‖v‖²,
        // where v = residual[best_row]. We fold the normalization into the
        // axpy scalar (`-dot/norm2`) to skip the per-iteration `DVector`
        // allocation for the unit vector.
        let norm2 = residual[best_row].norm_squared();
        if norm2 < 1e-12 {
            prog_bar.inc(1);
            continue;
        }
        let anchor = residual[best_row].clone();
        residual.par_iter_mut().for_each(|r| {
            let dot = r.dot(&anchor);
            r.axpy(-dot / norm2, &anchor, 1.0);
        });
        // `axpy` leaves the anchor row at ~0 (modulo round-off); pin it to
        // exact zero so future projections stay clean.
        residual[best_row].fill(0.0);
        prog_bar.inc(1);
    }
    prog_bar.finish_and_clear();

    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_recovery() {
        let d = 12;
        let mut x = Mat::zeros(6, d);
        for g in 0..4 {
            x[(0, g)] = 5.0;
        }
        for g in 4..8 {
            x[(1, g)] = 5.0;
        }
        for g in 8..12 {
            x[(2, g)] = 5.0;
        }
        for g in 0..d {
            x[(3, g)] = 0.5 * (g as f32).sin();
            x[(4, g)] = 1.0;
            x[(5, g)] = 0.3 * (g % 3) as f32;
        }
        let picked = gram_schmidt_anchors(&x, 3);
        let mut sorted = picked.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![0, 1, 2],
            "expected pure rows, got {:?}",
            picked
        );
    }

    #[test]
    fn zscore_is_unit_std() {
        let x = Mat::from_row_slice(
            4,
            3,
            &[1.0, 2.0, 5.0, 2.0, 3.0, 5.0, 3.0, 4.0, 5.0, 4.0, 5.0, 5.0],
        );
        let z = zscore_columns(&x);
        for j in 0..x.ncols() {
            let col: Vec<f32> = z.column(j).iter().copied().collect();
            let mean = col.iter().sum::<f32>() / col.len() as f32;
            assert!(mean.abs() < 1e-5);
            if j < 2 {
                let var = col.iter().map(|v| v * v).sum::<f32>() / col.len() as f32;
                assert!((var - 1.0).abs() < 1e-5);
            } else {
                assert!(col.iter().all(|&v| v == 0.0));
            }
        }
    }
}
