use crate::sparse_data_visitors::*;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;

use indicatif::ParallelProgressIterator;
use log::warn;
use matrix_util::sparse_stat::SparseRunningStatistics;
use matrix_util::traits::RunningStatOps;
use matrix_util::utils::partition_by_membership;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use fnv::FnvHashMap as HashMap;

#[derive(Clone)]
pub struct SqueezeCutoffs {
    pub row: usize,
    pub column: usize,
}

/// squeeze out rows and columns with excessive zero values
pub fn squeeze_by_nnz(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    cutoffs: SqueezeCutoffs,
    block_size: usize,
    preload: bool,
) -> anyhow::Result<()> {
    let col_stat = collect_column_stat(data, block_size)?;
    let row_stat = collect_row_stat(data, block_size)?;

    let file = data.get_backend_file_name();
    let backend = data.backend_type();

    let mut data = open_sparse_matrix(file, &backend)?;
    if preload {
        data.preload_columns()?;
    }

    fn nnz_index(nnz: &[f32], cutoff: usize) -> Option<Vec<usize>> {
        let ret: Vec<usize> = nnz
            .iter()
            .enumerate()
            .filter(|&(_, &x)| (x as usize) >= cutoff)
            .map(|(i, _)| i)
            .collect();

        (!ret.is_empty()).then_some(ret)
    }

    let row_nnz_vec = row_stat.count_positives();
    let col_nnz_vec = col_stat.count_positives();
    let row_idx = nnz_index(&row_nnz_vec, cutoffs.row);
    let col_idx = nnz_index(&col_nnz_vec, cutoffs.column);

    if row_idx.is_none() {
        warn!(
            "No rows can be kept with this cutoff {}!\n\
	     \n\
	     We will stop squeezing on the rows.\n\
	     \n",
            cutoffs.row
        );
    }

    if col_idx.is_none() {
        warn!(
            "No columns can be kept with this cutoff {}!\n\
	     \n\
	     We will stop squeezing on the columns.\n\
	     \n",
            cutoffs.column
        );
    }

    data.subset_columns_rows(col_idx.as_ref(), row_idx.as_ref())
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIoVec` across many data matrices
/// * `block_size` - a block size for each parallelized job
pub fn collect_row_stat_across_vec(
    data: &SparseIoVec,
    block_size: usize,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let mut row_stat = SparseRunningStatistics::new(data.num_rows());
    data.visit_columns_by_block(
        &row_stat_vec_visitor,
        &EmptyArgs {},
        &mut row_stat,
        Some(block_size),
    )?;
    Ok(row_stat)
}

/// collect row statistics for each group of columns
/// * `data` - `SparseIo`
/// * `column_membership` - a hashmap assign columns to groups
/// * `block_size` - a block size for each parallelized job
#[allow(clippy::type_complexity)]
pub fn collect_stratified_row_stat_across_vec(
    data: &SparseIoVec,
    column_membership: &HashMap<Box<str>, Box<str>>,
    block_size: usize,
) -> anyhow::Result<(Vec<Box<str>>, Vec<SparseRunningStatistics<f32>>)> {
    let column_names = data.column_names()?;
    let default = "".to_string().into_boxed_str();
    let membership = column_names
        .into_iter()
        .map(|k| column_membership.get(&k).unwrap_or(&default).clone())
        .collect::<Vec<_>>();

    let partitions = partition_by_membership(&membership, None);
    let mut group_names = Vec::with_capacity(partitions.len());
    let mut group_stats = Vec::with_capacity(partitions.len());

    for (k, cols) in partitions {
        let jobs = create_jobs(cols.len(), Some(block_size));
        let mut row_stat = SparseRunningStatistics::new(data.num_rows());
        let arc_stat = Arc::new(Mutex::new(&mut row_stat));

        jobs.par_iter()
            .progress_count(jobs.len() as u64)
            .for_each(|&(lb, ub)| {
                let cols_sub = cols[lb..ub].iter().cloned();
                let csc = data
                    .read_columns_csc(cols_sub)
                    .expect("failed to read data");
                let mut stat = arc_stat.lock().expect("failed to lock row_stat");
                stat.add_csc(&csc);
            });

        group_names.push(k);
        group_stats.push(row_stat);
    }

    Ok((group_names, group_stats))
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIo`
/// * `block_size` - a block size for each parallelized job
pub fn collect_row_stat(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: usize,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let nrows = data.num_rows().unwrap_or(0);
    let mut row_stat = SparseRunningStatistics::new(nrows);
    let arc_stat = Arc::new(Mutex::new(&mut row_stat));

    let jobs = create_jobs(data.num_columns().unwrap_or(0), Some(block_size));

    jobs.par_iter()
        .progress_count(jobs.len() as u64)
        .for_each(|&(lb, ub)| {
            let csc = data
                .read_columns_csc((lb..ub).collect())
                .expect("failed to read data");
            let mut stat = arc_stat.lock().expect("failed to lock row_stat");
            stat.add_csc(&csc);
        });

    Ok(row_stat)
}

/// collect column-wise sufficient statistics for Q/C
/// * `data` - `SparseIoVec` across many data matrices
/// * `select_rows` - selected row indices
/// * `block_size` - a block size for each parallelized job
pub fn collect_column_stat_across_vec(
    data: &SparseIoVec,
    select_rows: Option<&[usize]>,
    block_size: usize,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let mut col_stat = SparseRunningStatistics::new(data.num_columns());

    data.visit_columns_by_block(
        &col_stat_selected_rows_visitor,
        select_rows.unwrap_or(&[]),
        &mut col_stat,
        Some(block_size),
    )?;
    Ok(col_stat)
}

/// collect column-wise sufficient statistics for Q/C
/// * `data` - `SparseIo`
/// * `block_size` - a block size for each parallelized job
pub fn collect_column_stat(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: usize,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let ncols = data.num_columns().unwrap_or(0);
    let mut col_stat = SparseRunningStatistics::new(ncols);
    let arc_stat = Arc::new(Mutex::new(&mut col_stat));

    let jobs = create_jobs(ncols, Some(block_size));

    jobs.par_iter()
        .progress_count(jobs.len() as u64)
        .for_each(|&(lb, ub)| {
            let batch_ncols = ub - lb;
            let csc = data
                .read_columns_csc((lb..ub).collect())
                .expect("failed to read data");

            // Build column-indexed triplets
            let mut col_triplets: Vec<(usize, usize, f32)> = Vec::new();
            for (local_col, col) in csc.col_iter().enumerate() {
                let global_col = lb + local_col;
                for &val in col.values() {
                    col_triplets.push((global_col, 0, val));
                }
            }

            let mut stat = arc_stat.lock().expect("failed to lock col_stat");
            stat.add_triplets(batch_ncols, &col_triplets);
        });

    Ok(col_stat)
}

struct EmptyArgs {}

fn row_stat_vec_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    _: &EmptyArgs,
    arc_stat: Arc<Mutex<&mut SparseRunningStatistics<f32>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let csc = data.read_columns_csc(lb..ub)?;

    let mut stat = arc_stat.lock().expect("failed to lock row_stat");
    stat.add_csc(&csc);
    Ok(())
}

fn col_stat_selected_rows_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    select_row_indices: &[usize],
    arc_stat: Arc<Mutex<&mut SparseRunningStatistics<f32>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let batch_ncols = ub - lb;

    let csc = data.read_columns_csc(lb..ub)?;

    // Build column-indexed triplets: for each column in csc, accumulate stats at that column index
    let mut col_triplets: Vec<(usize, usize, f32)> = Vec::new();

    for (local_col, col) in csc.col_iter().enumerate() {
        let global_col = lb + local_col;
        let rows = col.row_indices();
        let vals = col.values();

        if select_row_indices.is_empty() {
            for (&_row, &val) in rows.iter().zip(vals.iter()) {
                // For column stats: index by column, dummy row index
                col_triplets.push((global_col, 0, val));
            }
        } else {
            for (&row, &val) in rows.iter().zip(vals.iter()) {
                if select_row_indices.contains(&row) {
                    col_triplets.push((global_col, 0, val));
                }
            }
        }
    }

    let mut stat = arc_stat.lock().expect("failed to lock col_stat");
    stat.add_triplets(batch_ncols, &col_triplets);

    Ok(())
}
