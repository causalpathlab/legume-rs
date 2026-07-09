//! Top-K feature selection: per-row, per-CSC-column, and `IndexedSample`
//! construction routines.
//!
//! Scoring is `log1p(value) · weight` and selection keeps the top-K
//! candidates with strictly positive score. The returned indices are
//! sorted ascending; the corresponding `values` are the **raw** row
//! values (the `log1p` only re-ranks).

use super::types::IndexedSample;
use indicatif::ParallelProgressIterator;
use matrix_util::traits::CandleDataLoaderOps;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;

///////////////////////////
// Public weighted top-K //
///////////////////////////

/// Top-K selection ranked by `log1p(row[i]) · weights[i]`, returning the
/// **raw** `row[i]` for the selected indices.
///
/// The `log1p` compression on the scoring side is essential when
/// `weights` are NB-Fisher info: without it, a housekeeping gene with
/// raw count `v=200` and `w=0.04` scores `8`, beating a marker with
/// `v=10`, `w=0.24` (score `2.4`). With `log1p`, the same pair scores
/// `0.21` vs `0.58` — markers win.
///
/// Returns at most `k` indices: features whose score is non-positive
/// (`v == 0` or `w == 0`) are dropped before selection, so a cell with
/// fewer non-zero positively-weighted features than `k` returns a short
/// vector. Every downstream consumer uses `s.indices.len().min(k)` when
/// packing into `[N, K]` buffers and leaves the rest at the zero-init.
pub fn top_k_indices_weighted(row: &[f32], weights: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    debug_assert_eq!(row.len(), weights.len());
    top_k_from_entries(
        row.iter()
            .zip(weights.iter())
            .enumerate()
            .map(|(i, (&v, &w))| (i as u32, v, w)),
        k,
    )
}

/// Shared top-K core for [`top_k_indices_weighted`] and
/// [`csc_columns_to_indexed_samples`].
///
/// `entries` yields `(feature_id, raw_value, weight)` triples — for the
/// sparse path only the column's stored nonzeros need be supplied.
/// Selection ranks by `log1p(value) · weight`, drops non-positive
/// scores, keeps the top `k`, and returns indices sorted ascending with
/// their **raw** values.
pub(crate) fn top_k_from_entries<I>(entries: I, k: usize) -> (Vec<u32>, Vec<f32>)
where
    I: Iterator<Item = (u32, f32, f32)>,
{
    let mut scored: Vec<(f32, u32, f32)> = entries
        .filter_map(|(i, v, w)| {
            let score = v.max(0.0).ln_1p() * w;
            (score > 0.0).then_some((score, i, v))
        })
        .collect();

    let k = k.min(scored.len());
    if k == 0 {
        return (Vec::new(), Vec::new());
    }

    scored.select_nth_unstable_by(k - 1, |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut top: Vec<(u32, f32)> = scored[..k].iter().map(|&(_, i, v)| (i, v)).collect();
    top.sort_unstable_by_key(|&(i, _)| i);

    let idx: Vec<u32> = top.iter().map(|&(i, _)| i).collect();
    let values: Vec<f32> = top.iter().map(|&(_, v)| v).collect();
    (idx, values)
}

////////////////////////////////////
// CSC and dense-row entry points //
////////////////////////////////////

/// Build [`IndexedSample`]s straight from a sparse `[D, N]` CSC matrix —
/// columns are samples (cells), rows are features. Each column's stored
/// nonzeros are scored with `log1p(value) · weight` and the top
/// `context_size` are kept. Avoids ever materializing the dense `[N, D]`
/// matrix the dense path needs.
///
/// `gene_remap = Some(new_to_train)` remaps each stored row index from a
/// held-out gene axis to the training axis before scoring (rows that
/// don't map are dropped) — for `predict` on a differing gene set, where
/// `shortlist_weights` is indexed in training-gene space. `None` when
/// the CSC is already on the training axis.
///
/// Sequential over columns by design: callers already run inside a
/// block-level rayon map, so a nested par_iter here would only add
/// task-splitting overhead.
pub fn csc_columns_to_indexed_samples(
    x_dn: &CscMatrix<f32>,
    shortlist_weights: &[f32],
    context_size: usize,
    gene_remap: Option<&[Option<usize>]>,
) -> Vec<IndexedSample> {
    debug_assert_eq!(
        x_dn.nrows(),
        gene_remap.map_or(shortlist_weights.len(), <[_]>::len)
    );
    (0..x_dn.ncols())
        .map(|j| {
            let col = x_dn.col(j);
            let pairs = col.row_indices().iter().zip(col.values().iter());
            let (indices, values) = match gene_remap {
                Some(rm) => top_k_from_entries(
                    pairs.filter_map(|(&r, &v)| {
                        rm[r].map(|rt| (rt as u32, v, shortlist_weights[rt]))
                    }),
                    context_size,
                ),
                None => top_k_from_entries(
                    pairs.map(|(&r, &v)| (r as u32, v, shortlist_weights[r])),
                    context_size,
                ),
            };
            IndexedSample { indices, values }
        })
        .collect()
}

/// Build [`IndexedSample`]s from a `CandleDataLoaderOps` source by
/// applying weighted top-K to each row. Parallel over rows via rayon.
pub fn build_indexed_samples<D: CandleDataLoaderOps + Sync>(
    data: &D,
    n_samples: usize,
    context_size: usize,
    shortlist_weights: &[f32],
    label: &str,
) -> Vec<IndexedSample> {
    let prog_bar = super::labeled_bar(label, n_samples as u64);
    let out = (0..n_samples)
        .into_par_iter()
        .progress_with(prog_bar.clone())
        .map(|i| {
            let row = data.row_to_f32_vec(i);
            let (indices, values) = top_k_indices_weighted(&row, shortlist_weights, context_size);
            IndexedSample { indices, values }
        })
        .collect();
    prog_bar.finish_and_clear();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_indices_weighted() {
        // raw row top-3 by value would be {3,5,1} (0.9, 0.7, 0.5).
        // Down-weight idx 3 and 5 ("housekeeping"); idx 0,2,4 should win.
        let row = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.7];
        let weights = vec![1.0, 0.1, 1.0, 0.01, 1.0, 0.05];
        let (indices, values) = top_k_indices_weighted(&row, &weights, 3);
        // log1p scores:
        //   0: ln1p(0.1)*1.0  ≈ 0.0953
        //   1: ln1p(0.5)*0.1  ≈ 0.0405
        //   2: ln1p(0.3)*1.0  ≈ 0.2624
        //   3: ln1p(0.9)*0.01 ≈ 0.0064
        //   4: ln1p(0.2)*1.0  ≈ 0.1823
        //   5: ln1p(0.7)*0.05 ≈ 0.0265
        // Top-3 = {2, 4, 0}, sorted by index → {0, 2, 4}.
        assert_eq!(indices, vec![0, 2, 4]);
        assert_eq!(values, vec![0.1, 0.3, 0.2]);
    }

    #[test]
    fn test_top_k_drops_zero_valued_features() {
        // Only 2 of 6 features are non-zero; with k=5, the result must be
        // length 2 (no zero-padding into top-K).
        let row = vec![0.0, 0.4, 0.0, 0.0, 0.7, 0.0];
        let weights = vec![1.0; 6];
        let (indices, values) = top_k_indices_weighted(&row, &weights, 5);
        assert_eq!(indices, vec![1, 4]);
        assert_eq!(values, vec![0.4, 0.7]);

        // All-zero row → empty result.
        let zero_row = vec![0.0; 6];
        let (idx0, val0) = top_k_indices_weighted(&zero_row, &weights, 3);
        assert!(idx0.is_empty());
        assert!(val0.is_empty());

        // Zero-weight features are also dropped (score = log1p(v) * 0 = 0).
        let row_wz = vec![0.4, 0.7, 0.3];
        let w_wz = vec![0.0, 1.0, 0.0];
        let (idx_wz, val_wz) = top_k_indices_weighted(&row_wz, &w_wz, 3);
        assert_eq!(idx_wz, vec![1]);
        assert_eq!(val_wz, vec![0.7]);
    }

    #[test]
    fn test_csc_columns_match_dense_top_k() {
        let n = 4;
        let d = 6;
        let dense = [
            [0.1f32, 0.5, 0.3, 0.9, 0.2, 0.7],
            [0.8, 0.1, 0.6, 0.2, 0.9, 0.3],
            [0.0, 0.7, 0.0, 0.4, 0.6, 0.0],
            [0.2, 0.3, 0.8, 0.1, 0.5, 0.9],
        ];
        let weights = vec![1.0f32, 0.1, 1.0, 0.01, 1.0, 0.05];

        let mut coo = nalgebra_sparse::CooMatrix::<f32>::new(d, n);
        for (j, row) in dense.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    coo.push(i, j, v);
                }
            }
        }
        let csc = CscMatrix::from(&coo);

        let from_csc = csc_columns_to_indexed_samples(&csc, &weights, 3, None);
        assert_eq!(from_csc.len(), n);
        for (j, row) in dense.iter().enumerate() {
            let (exp_idx, exp_val) = top_k_indices_weighted(row, &weights, 3);
            assert_eq!(from_csc[j].indices, exp_idx, "sample {j} indices");
            assert_eq!(from_csc[j].values, exp_val, "sample {j} values");
        }
    }
}
