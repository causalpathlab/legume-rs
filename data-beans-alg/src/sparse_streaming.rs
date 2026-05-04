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
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
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

    let tmpl = format!("{progress_label} {{bar:40}} {{pos}}/{{len}} blocks ({{eta}})");
    let pb = ProgressBar::new(jobs.len() as u64).with_style(
        ProgressStyle::with_template(&tmpl)
            .unwrap()
            .progress_chars("##-"),
    );

    let stats: SparseRunningStatistics<f32> = jobs
        .par_iter()
        .progress_with(pb.clone())
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
    pb.finish_and_clear();

    Ok(stats)
}
