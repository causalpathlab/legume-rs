//! Streaming per-cluster aggregation over `SparseIoVec`.
//!
//! Walks raw count columns in parallel column-blocks, accumulating
//! `T[k, g] = Σ_{n ∈ k} y[g, n]`, the per-group gene-sum tensor laid out
//! row-major as `Vec<f64>` of length `k · m`. Used by:
//!   - `cluster_bhc::run_cluster_bhc` for Dirichlet-Multinomial BHC sufficient
//!     stats;
//!   - `annotate` to derive NB-Fisher-adjusted per-cluster mean expression.
//!
//! Cells with `labels[n] >= k` (or `usize::MAX` from `remove_small_clusters`)
//! are silently skipped — callers handle "unassigned" out of band.

use crate::embed_common::*;
use rayon::prelude::*;

/// Streaming per-group gene-sum aggregator. Returns a `Vec<f64>` of length
/// `k * m` laid out row-major (`gene_sum[k * m + g]` = total count of gene
/// `g` over cells with `label == k`).
pub fn accumulate_gene_sum(
    data_vec: &SparseIoVec,
    labels: &[usize],
    k: usize,
    m: usize,
    block_size: usize,
) -> anyhow::Result<Vec<f64>> {
    let (gs, _) = accumulate_gene_sum_pair(data_vec, labels, k, &[], 0, m, block_size)?;
    Ok(gs)
}

/// Like `accumulate_gene_sum` but accumulates two independent groupings
/// (`labels_a` / `labels_b`) in a single column-block sweep. When
/// `labels_b` is empty the second buffer is skipped — used to share code
/// with the single-grouping case. Halves zarr I/O for callers that need
/// both per-cluster and per-batch sums (e.g. `senna annotate`).
pub fn accumulate_gene_sum_pair(
    data_vec: &SparseIoVec,
    labels_a: &[usize],
    k_a: usize,
    labels_b: &[usize],
    k_b: usize,
    m: usize,
    block_size: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let n = labels_a.len();
    if !labels_b.is_empty() {
        anyhow::ensure!(
            labels_b.len() == n,
            "labels_a/labels_b length mismatch: {} vs {}",
            n,
            labels_b.len()
        );
    }
    let blocks: Vec<(usize, usize)> = (0..n)
        .step_by(block_size.max(1))
        .map(|lb| (lb, (lb + block_size).min(n)))
        .collect();

    let per_block: Vec<(Vec<f64>, Vec<f64>)> = blocks
        .par_iter()
        .map(|&(lb, ub)| block_gene_sum(data_vec, labels_a, k_a, labels_b, k_b, lb, ub, m))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut sum_a = vec![0.0f64; k_a * m];
    let mut sum_b = vec![0.0f64; k_b * m];
    for (a, b) in per_block {
        for (acc, x) in sum_a.iter_mut().zip(a.iter()) {
            *acc += x;
        }
        for (acc, x) in sum_b.iter_mut().zip(b.iter()) {
            *acc += x;
        }
    }
    Ok((sum_a, sum_b))
}

#[allow(clippy::too_many_arguments)]
fn block_gene_sum(
    data_vec: &SparseIoVec,
    labels_a: &[usize],
    k_a: usize,
    labels_b: &[usize],
    k_b: usize,
    lb: usize,
    ub: usize,
    m: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let csc = data_vec.read_columns_csc(lb..ub)?;
    let mut sum_a = vec![0.0f64; k_a * m];
    let mut sum_b = vec![0.0f64; k_b * m];
    let do_b = !labels_b.is_empty();

    for j in 0..csc.ncols() {
        let global = lb + j;
        let col = csc.col(j);
        let rows = col.row_indices();
        let vals = col.values();

        let kk_a = labels_a[global];
        if kk_a < k_a {
            let row = &mut sum_a[kk_a * m..(kk_a + 1) * m];
            for (&g, &v) in rows.iter().zip(vals.iter()) {
                row[g] += v as f64;
            }
        }
        if do_b {
            let kk_b = labels_b[global];
            if kk_b < k_b {
                let row = &mut sum_b[kk_b * m..(kk_b + 1) * m];
                for (&g, &v) in rows.iter().zip(vals.iter()) {
                    row[g] += v as f64;
                }
            }
        }
    }
    Ok((sum_a, sum_b))
}

/// Convert a row-major `gene_sum` tensor (`k · m`) into a `m × k` mean-profile
/// matrix with optional per-gene weights: `μ[g, c] = w[g] · sum[c, g] /
/// Σ_g sum[c, g]`. Columns are populated in parallel over rayon.
///
/// `weights.len()` must equal `m` (or be empty for unweighted means).
pub fn weighted_mean_profile(
    gene_sum: &[f64],
    n_groups: usize,
    n_genes: usize,
    weights: &[f32],
) -> Mat {
    debug_assert_eq!(gene_sum.len(), n_groups * n_genes);
    debug_assert!(weights.is_empty() || weights.len() == n_genes);

    let mut out = Mat::zeros(n_genes, n_groups);
    // nalgebra DMatrix is column-major: columns live as contiguous slices
    // of length `n_genes` in `as_mut_slice`.
    let cols_buf = out.as_mut_slice();
    let weighted = !weights.is_empty();

    cols_buf
        .par_chunks_mut(n_genes)
        .enumerate()
        .for_each(|(c, col)| {
            let row = &gene_sum[c * n_genes..(c + 1) * n_genes];
            let s = row.iter().sum::<f64>().max(1.0);
            let inv = (1.0 / s) as f32;
            if weighted {
                for (gi, slot) in col.iter_mut().enumerate() {
                    *slot = row[gi] as f32 * inv * weights[gi];
                }
            } else {
                for (gi, slot) in col.iter_mut().enumerate() {
                    *slot = row[gi] as f32 * inv;
                }
            }
        });
    out
}
