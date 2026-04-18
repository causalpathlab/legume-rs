//! Shared anchor-selection helpers used by both the training-time β prior
//! (`topic::anchor_prior`) and post-training cell annotation
//! (`postprocess::fit_annotate`).
//!
//! The core idea: given a matrix of pseudobulk expression profiles, pick a
//! small number of "anchor" rows that are maximally orthogonal to one another
//! via greedy Gram-Schmidt, then optionally label each anchor with a cell-type
//! name by scoring marker-gene z-scores against a margin threshold.
//!
//! These helpers are intentionally dependency-free beyond `nalgebra` so the
//! module can sit at the crate root and be imported by both the training and
//! postprocessing sides without dragging candle / data-beans into each other's
//! graph.

use crate::embed_common::Mat;
use crate::marker_support::MarkerInfo;

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
/// Uses nalgebra's vectorized row operations so the inner loop stays dense
/// and cache-friendly even at D≈36k.
pub(crate) fn gram_schmidt_anchors(x_pg: &Mat, k: usize) -> Vec<usize> {
    let n = x_pg.nrows();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    // Residual copy. Rows get orthogonalized against the chosen anchors.
    let mut residual = x_pg.clone();
    let mut picked: Vec<usize> = Vec::with_capacity(k);
    let mut available: Vec<usize> = (0..n).collect();

    for _ in 0..k {
        // Argmax residual L2 norm over still-available rows.
        let (best_pos, &best_row) = available
            .iter()
            .enumerate()
            .max_by(|(_, &a), (_, &b)| {
                let na: f32 = residual.row(a).iter().map(|&v| v * v).sum();
                let nb: f32 = residual.row(b).iter().map(|&v| v * v).sum();
                na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty available");
        picked.push(best_row);
        available.swap_remove(best_pos);

        // Normalize the picked anchor row, then project it out of every
        // other row: r_j ← r_j - (r_j · u) · u, where u is the unit-length
        // anchor direction.
        let anchor_row = residual.row(best_row).clone_owned();
        let norm2: f32 = anchor_row.iter().map(|&v| v * v).sum();
        if norm2 < 1e-12 {
            continue;
        }
        let norm = norm2.sqrt();
        let unit = anchor_row / norm;
        for i in 0..n {
            if i == best_row {
                continue;
            }
            let dot: f32 = residual
                .row(i)
                .iter()
                .zip(unit.iter())
                .map(|(a, b)| a * b)
                .sum();
            let mut row_i = residual.row_mut(i);
            for (v, &u) in row_i.iter_mut().zip(unit.iter()) {
                *v -= dot * u;
            }
        }
        residual.row_mut(best_row).fill(0.0);
    }

    picked
}

/// Score each anchor PB against every celltype using a weighted one-sample
/// z-statistic over the celltype's TF-IDF-weighted markers. Promiscuous
/// markers contribute less; celltype-unique markers contribute most.
///
/// Score for celltype c at PB p:
///   `score_c = (Σ_g w_g · z_g) / sqrt(Σ_g w_g²)`
/// where the sum runs over genes with `membership_gc[(g, c)] > 0` and
/// `w_g = membership_gc[(g, c)]` is the TF-IDF weight set at load time.
///
/// Under H0 "marker z-scores are iid N(0,1)" the score is again N(0,1)
/// regardless of celltype size or weight distribution — so the
/// `margin_threshold` between top1 and top2 stays comparable across
/// celltypes.
///
/// Assign each anchor the best-scoring celltype iff
/// `top1 - top2 >= margin_threshold`. Anchors that don't clear the
/// margin — or whose best celltype has no markers — become `novel_{i}`.
///
/// When multiple anchors pick the same celltype, later anchors get a
/// numeric suffix so the labels stay unique (e.g. `T_cells`, `T_cells_2`).
pub(crate) fn label_anchors(
    x_zscored: &Mat,
    anchor_pb_idx: &[usize],
    markers: &MarkerInfo,
    margin_threshold: f32,
) -> (Vec<Box<str>>, Vec<(f32, f32)>) {
    let n_ct = markers.celltypes.len();
    let mut marker_rows_per_ct: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_ct];
    for (c, col) in markers.membership_gc.column_iter().enumerate() {
        for (g, &w) in col.iter().enumerate() {
            if w > 0.0 {
                marker_rows_per_ct[c].push((g, w));
            }
        }
    }
    let denom_per_ct: Vec<f32> = (0..n_ct)
        .map(|c| markers.membership_gc.column(c).norm())
        .collect();

    let mut used_counts = rustc_hash::FxHashMap::<Box<str>, usize>::default();
    let mut labels: Vec<Box<str>> = Vec::with_capacity(anchor_pb_idx.len());
    let mut scores: Vec<(f32, f32)> = Vec::with_capacity(anchor_pb_idx.len());

    for (i, &pb) in anchor_pb_idx.iter().enumerate() {
        let pb_row = x_zscored.row(pb);

        let mut per_ct: Vec<(usize, f32)> = (0..n_ct)
            .filter_map(|c| {
                let rows = &marker_rows_per_ct[c];
                if rows.is_empty() {
                    return None;
                }
                let denom = denom_per_ct[c];
                if denom <= 0.0 {
                    return None;
                }
                let sum_wz: f32 = rows.iter().map(|&(g, w)| w * pb_row[g]).sum();
                let score = sum_wz / denom;
                Some((c, score))
            })
            .collect();
        per_ct.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (top1_ct, top1_z) = per_ct.first().copied().unwrap_or((usize::MAX, 0.0));
        let top2_z = per_ct.get(1).map_or(f32::NEG_INFINITY, |&(_, s)| s);
        scores.push((top1_z, top2_z));

        let clears_margin = top1_ct != usize::MAX && (top1_z - top2_z) >= margin_threshold;
        if !clears_margin {
            labels.push(format!("novel_{i}").into_boxed_str());
            continue;
        }
        let base = markers.celltypes[top1_ct].clone();
        let n = used_counts.entry(base.clone()).or_insert(0);
        *n += 1;
        let label = if *n == 1 {
            base
        } else {
            format!("{base}_{n}").into_boxed_str()
        };
        labels.push(label);
    }

    (labels, scores)
}

/// Strip `_<n>` suffix from disambiguated multi-anchor labels, so we can
/// look them up in the marker file's celltype list.
pub(crate) fn base_celltype_label(label: &str) -> &str {
    if let Some(pos) = label.rfind('_') {
        let tail = &label[pos + 1..];
        if tail.parse::<usize>().is_ok() {
            return &label[..pos];
        }
    }
    label
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

    #[test]
    fn base_label_strips_numeric_suffix() {
        assert_eq!(base_celltype_label("T_cells"), "T_cells");
        assert_eq!(base_celltype_label("T_cells_2"), "T_cells");
        assert_eq!(base_celltype_label("novel_5"), "novel");
    }
}
