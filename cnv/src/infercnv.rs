//! Scalable inferCNV-style CNV signal construction.
//!
//! Given a `[G × N]` expression (or log-expression) matrix and a
//! [`GenomeOrder`], build a genome-ordered log-ratio matrix by:
//!
//! 1. Reordering rows onto canonical chromosomes.
//! 2. Subtracting a per-gene reference mean (explicit reference columns, or
//!    the all-column mean if none are given).
//! 3. Chromosome-bounded sliding-window averaging. A prefix-sum + count
//!    scan makes this **O(G)** per column, not O(G·W).
//! 4. Optional symmetric clipping and per-column centering.
//!
//! Columns are independent and filled in parallel via
//! [`legume_numeric::matrix::dmatrix_util::build_columns_par`].
//!
//! This is the expression-side counterpart of [`crate::per_sample`]: feed the
//! returned matrix into [`crate::per_sample::call_per_sample_cnv`] (with
//! `n_topics = 1`) to run the HMM on the smoothed signal.
//!
//! # References
//!
//! - Patel et al. (2014); Tirosh et al. (2016) — expression-based CNV.
//! - inferCNV, Broad Institute. <https://github.com/broadinstitute/infercnv>
//!   (windowed smoothing, reference subtraction, dynamic range clipping).

use crate::genome_order::GenomeOrder;
use legume_numeric::matrix::dmatrix_util::build_columns_par;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Configuration for [`infercnv_signal`].
#[derive(Debug, Clone)]
pub struct InferCnvConfig {
    /// Sliding window length in genes (odd preferred). inferCNV default is 101.
    /// Values `< 1` disable smoothing.
    pub window: usize,
    /// Clip log-ratios to `±clip` after smoothing. `None` leaves values unbounded.
    pub clip: Option<f32>,
    /// Subtract each column's mean after smoothing (and clipping).
    pub center_cells: bool,
}

impl Default for InferCnvConfig {
    fn default() -> Self {
        Self {
            window: 101,
            clip: Some(3.0),
            center_cells: true,
        }
    }
}

/// Build a genome-ordered inferCNV log-ratio matrix `[G_ord × N]`.
///
/// `expr[g, j]` is the (log) expression of gene `g` in column `j`.
/// `ref_cols` are column indices treated as the reference; an empty slice
/// uses every column as the reference (inferCNV's no-annotation fallback).
pub fn infercnv_signal(
    expr: &DMatrix<f32>,
    order: &GenomeOrder,
    ref_cols: &[usize],
    config: &InferCnvConfig,
) -> anyhow::Result<DMatrix<f32>> {
    let ordered = order.reorder_rows(expr)?;
    let log_ratio = subtract_reference(&ordered, ref_cols)?;
    Ok(smooth_ordered(&log_ratio, order, config))
}

/// Subtract the per-gene reference mean from every column.
///
/// `ref_cols` empty ⇒ mean over all columns. Non-finite entries are ignored
/// in the mean; a gene with no finite reference values yields 0.
pub fn subtract_reference(expr: &DMatrix<f32>, ref_cols: &[usize]) -> anyhow::Result<DMatrix<f32>> {
    let g = expr.nrows();
    let n = expr.ncols();
    if ref_cols.iter().any(|&j| j >= n) {
        anyhow::bail!(
            "inferCNV: reference column index out of range (ncols={n}, refs={ref_cols:?})"
        );
    }
    let refs: Vec<usize> = if ref_cols.is_empty() {
        (0..n).collect()
    } else {
        ref_cols.to_vec()
    };

    let gene_means: Vec<f32> = (0..g)
        .into_par_iter()
        .map(|gi| {
            let mut sum = 0f32;
            let mut count = 0f32;
            for &j in &refs {
                let v = expr[(gi, j)];
                if v.is_finite() {
                    sum += v;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                sum / count
            } else {
                0.0
            }
        })
        .collect();

    Ok(build_columns_par(g, n, |j, col| {
        for gi in 0..g {
            let v = expr[(gi, j)];
            col[gi] = if v.is_finite() {
                v - gene_means[gi]
            } else {
                f32::NAN
            };
        }
    }))
}

/// Chromosome-bounded window smooth of an already genome-ordered `[G_ord × N]`
/// matrix, with optional clip and column centering.
///
/// `order.chr_boundaries` must describe `signal`'s row axis (i.e. `signal`
/// came from [`GenomeOrder::reorder_rows`]).
pub fn smooth_ordered(
    signal: &DMatrix<f32>,
    order: &GenomeOrder,
    config: &InferCnvConfig,
) -> DMatrix<f32> {
    let g = signal.nrows();
    let n = signal.ncols();
    debug_assert_eq!(g, order.len());

    let bounds = &order.chr_boundaries;
    let window = config.window;
    let clip = config.clip;
    let center = config.center_cells;

    build_columns_par(g, n, |j, col| {
        // Column-major: one contiguous memcpy per cell.
        col.copy_from_slice(signal.column(j).as_slice());
        if window >= 1 {
            smooth_column_inplace(col, bounds, window);
        }
        if let Some(c) = clip {
            let c = c.abs();
            for v in col.iter_mut() {
                if v.is_finite() {
                    *v = v.clamp(-c, c);
                }
            }
        }
        if center {
            center_finite(col);
        }
    })
}

/// In-place chromosome-bounded moving average on one column.
///
/// Window of length `w` is centered when `w` is odd (`half = w/2` on each
/// side). At chromosome ends the window shrinks (no padding, no cross-chr
/// bleed). Non-finite values are excluded from both numerator and denominator.
fn smooth_column_inplace(col: &mut [f32], chr_boundaries: &[(Box<str>, usize, usize)], w: usize) {
    if w <= 1 || col.is_empty() {
        return;
    }
    let half = w / 2;
    // Work on a copy of the raw values; write averages back into `col`.
    let raw = col.to_vec();

    for &(_, start, end) in chr_boundaries {
        if start >= end || end > raw.len() {
            continue;
        }
        let len = end - start;
        // prefix_sum[i] = sum of finite raw[start .. start+i]
        // prefix_n[i]   = count of finite raw[start .. start+i]
        let mut prefix_sum = vec![0f32; len + 1];
        let mut prefix_n = vec![0f32; len + 1];
        for i in 0..len {
            let v = raw[start + i];
            let (add_s, add_n) = if v.is_finite() { (v, 1.0) } else { (0.0, 0.0) };
            prefix_sum[i + 1] = prefix_sum[i] + add_s;
            prefix_n[i + 1] = prefix_n[i] + add_n;
        }
        for i in 0..len {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(len);
            let n = prefix_n[hi] - prefix_n[lo];
            col[start + i] = if n > 0.0 {
                (prefix_sum[hi] - prefix_sum[lo]) / n
            } else {
                f32::NAN
            };
        }
    }
}

fn center_finite(col: &mut [f32]) {
    let mut sum = 0f32;
    let mut n = 0f32;
    for &v in col.iter() {
        if v.is_finite() {
            sum += v;
            n += 1.0;
        }
    }
    if n <= 0.0 {
        return;
    }
    let mean = sum / n;
    for v in col.iter_mut() {
        if v.is_finite() {
            *v -= mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome_order::GenePosition;
    use approx::assert_relative_eq;

    fn order_two_chr() -> GenomeOrder {
        // chr1: genes 0,1,2  chr2: genes 3,4
        let genes = vec![
            GenePosition {
                gene_idx: 0,
                chromosome: "chr1".into(),
                position: 100,
            },
            GenePosition {
                gene_idx: 1,
                chromosome: "chr1".into(),
                position: 200,
            },
            GenePosition {
                gene_idx: 2,
                chromosome: "chr1".into(),
                position: 300,
            },
            GenePosition {
                gene_idx: 3,
                chromosome: "chr2".into(),
                position: 100,
            },
            GenePosition {
                gene_idx: 4,
                chromosome: "chr2".into(),
                position: 200,
            },
        ];
        GenomeOrder::from_positions(&genes)
    }

    fn naive_smooth(col: &[f32], bounds: &[(Box<str>, usize, usize)], w: usize) -> Vec<f32> {
        let half = w / 2;
        let mut out = col.to_vec();
        for &(_, start, end) in bounds {
            let len = end - start;
            for i in 0..len {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(len);
                let mut s = 0f32;
                let mut n = 0f32;
                for k in lo..hi {
                    let v = col[start + k];
                    if v.is_finite() {
                        s += v;
                        n += 1.0;
                    }
                }
                out[start + i] = if n > 0.0 { s / n } else { f32::NAN };
            }
        }
        out
    }

    #[test]
    fn prefix_sum_matches_naive() {
        let order = order_two_chr();
        let raw = vec![1.0, 2.0, 3.0, 10.0, 20.0];
        let mut prefix = raw.clone();
        smooth_column_inplace(&mut prefix, &order.chr_boundaries, 3);
        let naive = naive_smooth(&raw, &order.chr_boundaries, 3);
        for (a, b) in prefix.iter().zip(naive.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6);
        }
        // chr1 gene 0: mean(1,2)=1.5; gene 1: mean(1,2,3)=2; gene 2: mean(2,3)=2.5
        assert_relative_eq!(prefix[0], 1.5);
        assert_relative_eq!(prefix[1], 2.0);
        assert_relative_eq!(prefix[2], 2.5);
        // chr2 must not see chr1
        assert_relative_eq!(prefix[3], 15.0);
        assert_relative_eq!(prefix[4], 15.0);
    }

    #[test]
    fn no_cross_chromosome_bleed() {
        let order = order_two_chr();
        let raw = vec![100.0, 100.0, 100.0, 0.0, 0.0];
        let mut col = raw.clone();
        smooth_column_inplace(&mut col, &order.chr_boundaries, 101);
        assert!(col[3].abs() < 1e-6, "chr2 should stay 0, got {}", col[3]);
        assert!(col[4].abs() < 1e-6, "chr2 should stay 0, got {}", col[4]);
        assert_relative_eq!(col[0], 100.0);
    }

    #[test]
    fn nan_excluded_from_window() {
        let order = order_two_chr();
        let raw = vec![1.0, f32::NAN, 3.0, 0.0, 0.0];
        let mut col = raw.clone();
        smooth_column_inplace(&mut col, &order.chr_boundaries, 3);
        // gene 1: only 1 and 3 are finite → 2.0
        assert_relative_eq!(col[1], 2.0);
        // gene 0: 1 and NaN → 1.0
        assert_relative_eq!(col[0], 1.0);
    }

    #[test]
    fn subtract_reference_uses_named_columns() {
        // 3 genes × 4 cells; refs = {0,1} with mean 1, query cells 10
        let expr = DMatrix::from_row_slice(
            3,
            4,
            &[
                1.0, 1.0, 10.0, 10.0, //
                1.0, 1.0, 10.0, 10.0, //
                1.0, 1.0, 10.0, 10.0,
            ],
        );
        let lr = subtract_reference(&expr, &[0, 1]).unwrap();
        assert_relative_eq!(lr[(0, 0)], 0.0);
        assert_relative_eq!(lr[(0, 2)], 9.0);
    }

    #[test]
    fn infercnv_signal_reorders_and_smooths() {
        // genes stored as chr2, chr1 — reorder should put chr1 first
        let genes = vec![
            GenePosition {
                gene_idx: 0,
                chromosome: "chr2".into(),
                position: 100,
            },
            GenePosition {
                gene_idx: 1,
                chromosome: "chr1".into(),
                position: 50,
            },
        ];
        let order = GenomeOrder::from_positions(&genes);
        let expr = DMatrix::from_row_slice(2, 2, &[5.0, 5.0, 1.0, 3.0]);
        let cfg = InferCnvConfig {
            window: 1,
            clip: None,
            center_cells: false,
        };
        let out = infercnv_signal(&expr, &order, &[0], &cfg).unwrap();
        assert_eq!(out.nrows(), 2);
        // row 0 is original gene 1 (chr1): 1-1=0, 3-1=2
        assert_relative_eq!(out[(0, 0)], 0.0);
        assert_relative_eq!(out[(0, 1)], 2.0);
        // row 1 is original gene 0 (chr2): 5-5=0, 5-5=0
        assert_relative_eq!(out[(1, 0)], 0.0);
        assert_relative_eq!(out[(1, 1)], 0.0);
    }

    #[test]
    fn clip_and_center() {
        let order = order_two_chr();
        let signal = DMatrix::from_column_slice(5, 1, &[10.0, 10.0, 10.0, -10.0, -10.0]);
        let cfg = InferCnvConfig {
            window: 1,
            clip: Some(3.0),
            center_cells: true,
        };
        let out = smooth_ordered(&signal, &order, &cfg);
        // after clip: [3,3,3,-3,-3], mean = 0.6, centered:
        for i in 0..3 {
            assert_relative_eq!(out[(i, 0)], 3.0 - 0.6, epsilon = 1e-5);
        }
        for i in 3..5 {
            assert_relative_eq!(out[(i, 0)], -3.0 - 0.6, epsilon = 1e-5);
        }
    }

    #[test]
    fn bad_ref_index_errors() {
        let expr = DMatrix::from_element(2, 2, 1.0);
        let err = subtract_reference(&expr, &[5]).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }
}
