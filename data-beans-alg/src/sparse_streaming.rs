//! Lock-free parallel streaming pass over a `SparseIoVec` that yields a
//! merged `SparseRunningStatistics<f32>`.
//!
//! Both HVG selection (`hvg::select_hvg_streaming`) and NB-Fisher
//! weighting (`gene_weighting::compute_nb_fisher_weights`) need the same
//! per-feature `(npos, sum, sum_sq, n_cells)` aggregates from a sparse
//! backend. Sharing the skeleton keeps the parallel / lock-free
//! invariants in one place and gives every caller a labelled progress
//! bar with consistent styling.

use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use matrix_util::sparse_stat::SparseRunningStatistics;
use matrix_util::utils::generate_minibatch_intervals;
use rayon::prelude::*;

/// Stream cells through `data_vec` in `block_size` chunks, accumulating
/// per-feature NB sufficient statistics into one `SparseRunningStatistics`
/// per rayon worker, then merge to one result.
///
/// `progress_label` is shown in the indicatif progress bar (e.g.
/// `"HVG"`, `"NB-Fisher"`).
pub fn streaming_sparse_running_stats(
    data_vec: &SparseIoVec,
    block_size: Option<usize>,
    progress_label: &str,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let n_features = data_vec.num_rows();
    let n_total = data_vec.num_columns();
    let jobs = generate_minibatch_intervals(n_total, n_features, block_size);

    let prog_bar = matrix_util::progress::new_progress_bar(jobs.len() as u64)
        .with_message(format!("{progress_label} blocks"));

    let stats: SparseRunningStatistics<f32> = jobs
        .par_iter()
        .progress_with(prog_bar.clone())
        .try_fold(
            || SparseRunningStatistics::<f32>::new(n_features),
            |mut acc, &(lb, ub)| -> anyhow::Result<SparseRunningStatistics<f32>> {
                let chunk = data_vec.read_columns_csc(lb..ub)?;
                acc.add_csc(&chunk);
                Ok(acc)
            },
        )
        .try_reduce(
            || SparseRunningStatistics::<f32>::new(n_features),
            |mut a, b| {
                a.merge(&b);
                Ok(a)
            },
        )?;
    prog_bar.finish_and_clear();

    Ok(stats)
}

/// The same pass, accumulating a second set of statistics on a folded axis.
///
/// Returns `(row_stats, folded_stats)`. `row_to_gene[r]` names the folded
/// bucket row `r` belongs to; buckets are `0..n_folded`.
///
/// The fold happens inside the block read, so it costs `O(nnz)` arithmetic and
/// no extra I/O. It exists because two of the four statistics do NOT survive a
/// fold applied afterwards:
///
/// - `s2` of a sum is not the sum of `s2`. The cross term `2ab` is gone, so a
///   variance rebuilt from per-row squares understates a bucket whose rows
///   covary, which two splice tracks of one gene certainly do.
/// - `npos` of a sum is not the sum of `npos`. A cell detected on both rows of
///   a bucket is one detection, not two, and the difference is exactly the
///   bimodality that breaks a detection-based cutoff on a channelized matrix.
///
/// `s1` and the mean do survive, which is why a caller that needs only those
/// can fold a row-axis result afterwards instead of asking for this.
pub fn streaming_sparse_running_stats_folded(
    data_vec: &SparseIoVec,
    block_size: Option<usize>,
    progress_label: &str,
    row_to_gene: &[u32],
    n_folded: usize,
) -> anyhow::Result<(SparseRunningStatistics<f32>, SparseRunningStatistics<f32>)> {
    let n_features = data_vec.num_rows();
    anyhow::ensure!(
        row_to_gene.len() == n_features,
        "row_to_gene has {} entries for a {}-row matrix",
        row_to_gene.len(),
        n_features
    );
    if let Some(&bad) = row_to_gene.iter().max() {
        anyhow::ensure!(
            (bad as usize) < n_folded,
            "row_to_gene names bucket {} but only {} were declared",
            bad,
            n_folded
        );
    }

    let n_total = data_vec.num_columns();
    let jobs = generate_minibatch_intervals(n_total, n_features, block_size);

    let prog_bar = matrix_util::progress::new_progress_bar(jobs.len() as u64)
        .with_message(format!("{progress_label} blocks"));

    // The dense scratch lives in the fold accumulator, so it is allocated once
    // per rayon worker rather than once per block, and it is cleared in
    // O(touched) rather than O(n_folded).
    type Acc = (
        SparseRunningStatistics<f32>,
        SparseRunningStatistics<f32>,
        Vec<f32>,
        Vec<usize>,
        Vec<f32>,
    );

    let (row_stats, folded_stats) = jobs
        .par_iter()
        .progress_with(prog_bar.clone())
        .try_fold(
            || {
                (
                    SparseRunningStatistics::<f32>::new(n_features),
                    SparseRunningStatistics::<f32>::new(n_folded),
                    vec![0.0f32; n_folded],
                    Vec::<usize>::new(),
                    Vec::<f32>::new(),
                )
            },
            |(mut rows, mut folded, mut buf, mut touched, mut vals),
             &(lb, ub)|
             -> anyhow::Result<Acc> {
                let chunk = data_vec.read_columns_csc(lb..ub)?;
                rows.add_csc(&chunk);
                for col in chunk.col_iter() {
                    touched.clear();
                    for (&r, &v) in col.row_indices().iter().zip(col.values().iter()) {
                        let g = row_to_gene[r] as usize;
                        if buf[g] == 0.0 {
                            touched.push(g);
                        }
                        buf[g] += v;
                    }
                    // `add_sparse_column` accumulates per row and never reads
                    // the order, so the touched list is left unsorted.
                    vals.clear();
                    vals.extend(touched.iter().map(|&g| buf[g]));
                    folded.add_sparse_column(&touched, &vals);
                    for &g in touched.iter() {
                        buf[g] = 0.0;
                    }
                }
                Ok((rows, folded, buf, touched, vals))
            },
        )
        .map(|acc| acc.map(|(rows, folded, ..)| (rows, folded)))
        .try_reduce(
            || {
                (
                    SparseRunningStatistics::<f32>::new(n_features),
                    SparseRunningStatistics::<f32>::new(n_folded),
                )
            },
            |(mut ra, mut fa), (rb, fb)| {
                ra.merge(&rb);
                fa.merge(&fb);
                Ok((ra, fa))
            },
        )?;
    prog_bar.finish_and_clear();

    Ok((row_stats, folded_stats))
}
