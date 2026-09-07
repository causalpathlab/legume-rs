//! Exact all-pairs k-nearest neighbours of the rows of a dense matrix.
//!
//! [`super::ColumnDict`] indexes a point set for arbitrary queries. When every
//! point is itself a query, an index build is paid for once and amortised over
//! nothing, and above the dictionary's exact threshold that build dominates.
//! Here the Gram matrix does the work as one GEMM per row block —
//! `‖x_i − x_j‖² = ‖x_i‖² + ‖x_j‖² − 2 x_i·x_j` — blocks in parallel, each row
//! keeping a short list of its best candidates as it scans its column of the
//! block. Two guards make the answer exact rather than merely fast: the
//! columns are centred first, which leaves every pairwise distance unchanged
//! but removes the shared offset that makes the Gram form cancel
//! catastrophically, and the short list is re-scored by direct differences
//! before the final `k` are chosen. Ties break by index, non-finite distances
//! come last, and the result does not depend on the thread count.

use crate::knn::metric::l2_sq;
use crate::utils::generate_minibatch_intervals;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Bytes of Gram block one task holds; sets the default row block.
const BLOCK_BYTES: usize = 64 << 20;
/// Candidates re-scored by direct differences per row, as a multiple of `k`.
const RESCORE_FACTOR: usize = 2;

/// Exact `k` nearest neighbours of every row of `x` among the other rows:
/// `(indices, Euclidean distances)` per row, nearest first, ties by index.
/// `k` is clamped to the number of other rows. A row with a non-finite entry
/// is at a non-finite distance from everything, and such distances come last.
pub fn knn_rows_l2(x: &DMatrix<f32>, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n = x.nrows();
    let block = (BLOCK_BYTES / (4 * n.max(1))).clamp(16, 1024);
    knn_rows_l2_blocked(x, k, block)
}

/// [`knn_rows_l2`] with an explicit row block (columns of the Gram block held
/// per task).
pub(crate) fn knn_rows_l2_blocked(
    x: &DMatrix<f32>,
    k: usize,
    block: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let (n, s) = (x.nrows(), x.ncols());
    let k = k.min(n.saturating_sub(1));
    if n == 0 || k == 0 {
        return (vec![Vec::new(); n], vec![Vec::new(); n]);
    }

    // Centre each column over its finite entries (one non-finite value must
    // not poison the column); a column of `x` is contiguous.
    let mut xc = x.clone();
    xc.as_mut_slice().par_chunks_mut(n).for_each(|col| {
        let (sum, count) = col
            .iter()
            .filter(|v| v.is_finite())
            .fold((0.0f64, 0usize), |(s, c), &v| (s + v as f64, c + 1));
        if count > 0 {
            let mean = (sum / count as f64) as f32;
            col.iter_mut().for_each(|v| *v -= mean);
        }
    });
    // A row of `x` is a contiguous column of `xt`: the GEMM operand and the
    // re-scoring both read it that way.
    let xt = xc.transpose();
    let norms: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| xt.column(i).norm_squared())
        .collect();
    let shortlist = (RESCORE_FACTOR * k).min(n - 1);

    let blocks = generate_minibatch_intervals(n, 0, Some(block.max(1)));
    let bar = crate::progress::new_progress_bar(blocks.len() as u64)
        .with_message(format!("all-pairs kNN {n} x {s}, k={k}"));
    let rows: Vec<(Vec<usize>, Vec<f32>)> = blocks
        .into_par_iter()
        .flat_map_iter(|(lo, hi)| {
            // `[n, hi − lo]`: column `i − lo` holds every candidate's dot
            // product with row `i`, contiguously.
            let gram = &xc * xt.columns(lo, hi - lo);
            let mut cand: Vec<(f32, usize)> = Vec::with_capacity(shortlist + 1);
            let out: Vec<(Vec<usize>, Vec<f32>)> = (lo..hi)
                .map(|i| {
                    cand.clear();
                    let g = gram.column(i - lo);
                    for (j, (&gj, &nj)) in g.as_slice().iter().zip(&norms).enumerate() {
                        if j != i {
                            keep_smallest(
                                &mut cand,
                                (sort_key(norms[i] + nj - 2.0 * gj), j),
                                shortlist,
                            );
                        }
                    }
                    // Direct differences settle the order and give the true distances.
                    let xi = xt.column(i);
                    let mut scored: Vec<(f32, usize)> = cand
                        .iter()
                        .map(|&(_, j)| (sort_key(l2_sq(xi.as_slice(), xt.column(j).as_slice())), j))
                        .collect();
                    scored.sort_unstable_by(by_distance_then_index);
                    scored.truncate(k);
                    scored.into_iter().map(|(d2, j)| (j, d2.sqrt())).unzip()
                })
                .collect();
            bar.inc(1);
            out
        })
        .collect();
    bar.finish_and_clear();
    rows.into_iter().unzip()
}

/// A squared distance as a sort key: clamped at zero against rounding, and a
/// non-finite value made a *positive* NaN. `f32::max` would turn NaN into
/// zero, and an arithmetic NaN may carry the sign bit, which `total_cmp`
/// orders first; a positive NaN sorts after every finite value.
fn sort_key(d2: f32) -> f32 {
    if d2.is_finite() {
        d2.max(0.0)
    } else {
        f32::NAN
    }
}

fn by_distance_then_index(a: &(f32, usize), b: &(f32, usize)) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// Insert `item` into the sorted `buf` if it ranks among the `cap` smallest,
/// keeping `buf` sorted and at most `cap` long. Almost every candidate is
/// rejected on the one comparison against the current worst.
fn keep_smallest(buf: &mut Vec<(f32, usize)>, item: (f32, usize), cap: usize) {
    if buf.len() == cap && by_distance_then_index(&item, &buf[cap - 1]) != Ordering::Less {
        return;
    }
    let pos = buf.partition_point(|x| by_distance_then_index(x, &item) == Ordering::Less);
    buf.insert(pos, item);
    buf.truncate(cap);
}

#[cfg(test)]
#[path = "all_pairs_tests.rs"]
mod tests;
