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
    let mut out = accumulate_gene_sum_multi(data_vec, &[(labels, k)], m, block_size)?;
    Ok(out.pop().expect("one grouping in, one out"))
}

/// Like `accumulate_gene_sum` but accumulates two independent groupings
/// (`labels_a` / `labels_b`) in a single column-block sweep. When
/// `labels_b` is empty the second buffer is skipped — used to share code
/// with the single-grouping case. Halves zarr I/O for callers that need
/// both per-cluster and per-batch sums (e.g. `senna annotate-by-enrichment`).
pub fn accumulate_gene_sum_pair(
    data_vec: &SparseIoVec,
    labels_a: &[usize],
    k_a: usize,
    labels_b: &[usize],
    k_b: usize,
    m: usize,
    block_size: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    if labels_b.is_empty() {
        let sum_a = accumulate_gene_sum(data_vec, labels_a, k_a, m, block_size)?;
        return Ok((sum_a, vec![0.0; k_b * m]));
    }
    let mut out =
        accumulate_gene_sum_multi(data_vec, &[(labels_a, k_a), (labels_b, k_b)], m, block_size)?;
    let sum_b = out.pop().expect("two groupings in, two out");
    let sum_a = out.pop().expect("two groupings in, two out");
    Ok((sum_a, sum_b))
}

/// Accumulate any number of independent groupings in one pass over the columns.
///
/// Reading the columns is the expensive part — it is the raw matrix off disk —
/// so a caller that needs several partitions of the same cells (deconvolve pools
/// over archetype granularities) must not re-stream the matrix per partition.
/// Each grouping is `(labels, k)`; cells whose label is outside `0..k` are
/// skipped, which is how callers park cells they mean to exclude.
pub fn accumulate_gene_sum_multi(
    data_vec: &SparseIoVec,
    groupings: &[(&[usize], usize)],
    m: usize,
    block_size: usize,
) -> anyhow::Result<Vec<Vec<f64>>> {
    anyhow::ensure!(!groupings.is_empty(), "no groupings to accumulate");
    let n = groupings[0].0.len();
    for (labels, _) in groupings {
        anyhow::ensure!(
            labels.len() == n,
            "grouping length mismatch: {} vs {n}",
            labels.len()
        );
    }
    let blocks: Vec<(usize, usize)> = (0..n)
        .step_by(block_size.max(1))
        .map(|lb| (lb, (lb + block_size).min(n)))
        .collect();

    let per_block: Vec<Vec<Vec<f64>>> = blocks
        .par_iter()
        .map(|&(lb, ub)| block_gene_sum(data_vec, groupings, lb, ub, m))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut sums: Vec<Vec<f64>> = groupings
        .iter()
        .map(|&(_, k)| vec![0.0f64; k * m])
        .collect();
    for block in per_block {
        for (acc, part) in sums.iter_mut().zip(block) {
            for (a, x) in acc.iter_mut().zip(part.iter()) {
                *a += x;
            }
        }
    }
    Ok(sums)
}

fn block_gene_sum(
    data_vec: &SparseIoVec,
    groupings: &[(&[usize], usize)],
    lb: usize,
    ub: usize,
    m: usize,
) -> anyhow::Result<Vec<Vec<f64>>> {
    let csc = data_vec.read_columns_csc(lb..ub)?;
    let mut sums: Vec<Vec<f64>> = groupings
        .iter()
        .map(|&(_, k)| vec![0.0f64; k * m])
        .collect();

    for j in 0..csc.ncols() {
        let global = lb + j;
        let col = csc.col(j);
        let rows = col.row_indices();
        let vals = col.values();

        for (sum, &(labels, k)) in sums.iter_mut().zip(groupings) {
            let kk = labels[global];
            if kk >= k {
                continue;
            }
            let row = &mut sum[kk * m..(kk + 1) * m];
            for (&g, &v) in rows.iter().zip(vals.iter()) {
                row[g] += f64::from(v);
            }
        }
    }
    Ok(sums)
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
