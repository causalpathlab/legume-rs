use crate::sparse_data_visitors::*;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;

use indicatif::ParallelProgressIterator;
use matrix_util::ndarray_stat::RunningStatistics;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

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
) -> anyhow::Result<()> {
    let col_stat = collect_column_stat(data, block_size)?;
    let row_stat = collect_row_stat(data, block_size)?;

    let file = data.get_backend_file_name();
    let backend = data.backend_type();

    let mut data = open_sparse_matrix(file, &backend)?;
    data.preload_columns()?;

    fn nnz_index(nnz: &[f32], cutoff: usize) -> Option<Vec<usize>> {
        let ret: Vec<usize> = nnz
            .iter()
            .enumerate()
            .filter(|&(_, &x)| (x as usize) >= cutoff)
            .map(|(i, _)| i)
            .collect();

        (!ret.is_empty()).then_some(ret)
    }

    let row_nnz_vec = row_stat.count_positives().to_vec();
    let col_nnz_vec = col_stat.count_positives().to_vec();
    let row_idx = nnz_index(&row_nnz_vec, cutoffs.row);
    let col_idx = nnz_index(&col_nnz_vec.to_vec(), cutoffs.column);

    data.subset_columns_rows(col_idx.as_ref(), row_idx.as_ref())
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIoVec` across many data matrices
/// * `block_size` - parallel block size
pub fn collect_row_stat_across_vec(
    data: &SparseIoVec,
    block_size: usize,
) -> anyhow::Result<RunningStatistics<Ix1>> {
    let mut row_stat = RunningStatistics::new(Ix1(data.num_rows()?));
    data.visit_columns_by_block(
        &row_stat_vec_visitor,
        &EmptyArgs {},
        &mut row_stat,
        Some(block_size),
    )?;
    Ok(row_stat)
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIo`
/// * `block_size` - parallel block size
pub fn collect_row_stat(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: usize,
) -> anyhow::Result<RunningStatistics<Ix1>> {
    let nrows = data.num_rows().unwrap_or(0);
    let mut row_stat = RunningStatistics::new(Ix1(nrows));
    let arc_stat = Arc::new(Mutex::new(&mut row_stat));

    let jobs = create_jobs(data.num_columns().unwrap_or(0), Some(block_size));

    jobs.par_iter()
        .progress_count(jobs.len() as u64)
        .for_each(|&(lb, ub)| {
            let xx = data
                .read_columns_ndarray((lb..ub).collect())
                .expect("failed to read data");
            let mut stat = arc_stat.lock().expect("failed to lock row_stat");
            for x in xx.axis_iter(Axis(1)) {
                stat.add(&x);
            }
        });

    Ok(row_stat)
}

/// collect column-wise sufficient statistics for Q/C
/// * `data` - `SparseIoVec` across many data matrices
/// * `select_rows` - selected row indices
/// * `block_size` - parallel block size
pub fn collect_column_stat_across_vec(
    data: &SparseIoVec,
    select_rows: Option<&[usize]>,
    block_size: usize,
) -> anyhow::Result<RunningStatistics<Ix1>> {
    let mut col_stat = RunningStatistics::new(Ix1(data.num_columns()?));

    data.visit_columns_by_block(
        &col_stat_selected_rows_visitor,
        select_rows.unwrap_or(&[]),
        &mut col_stat,
        Some(block_size),
    )?;
    Ok(col_stat)
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIo`
/// * `block_size` - parallel block size
pub fn collect_column_stat(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: usize,
) -> anyhow::Result<RunningStatistics<Ix1>> {
    let ncols = data.num_columns().unwrap_or(0);
    let mut col_stat = RunningStatistics::new(Ix1(ncols));
    let arc_stat = Arc::new(Mutex::new(&mut col_stat));

    let jobs = create_jobs(data.num_columns().unwrap_or(0), Some(block_size));

    jobs.par_iter()
        .progress_count(jobs.len() as u64)
        .for_each(|&(lb, ub)| {
            let xx = data
                .read_columns_ndarray((lb..ub).collect())
                .expect("failed to read data");
            let mut stat = arc_stat.lock().expect("failed to lock row_stat");
            for x in xx.axis_iter(Axis(0)) {
                for j in lb..ub {
                    let i = j - lb;
                    stat.add_element(&[j], x[i]);
                }
            }
        });

    Ok(col_stat)
}

struct EmptyArgs {}

fn row_stat_vec_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    _: &EmptyArgs,
    arc_stat: Arc<Mutex<&mut RunningStatistics<Ix1>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let xx = data.read_columns_ndarray(lb..ub)?;

    let mut stat = arc_stat.lock().expect("failed to lock row_stat");
    for x in xx.axis_iter(Axis(1)) {
        stat.add(&x);
    }
    Ok(())
}

fn col_stat_selected_rows_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    select_row_indices: &[usize],
    arc_stat: Arc<Mutex<&mut RunningStatistics<Ix1>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;

    let xx = if select_row_indices.is_empty() {
        data.read_columns_ndarray(lb..ub)?
    } else {
        data.read_columns_ndarray(lb..ub)?
            .select(Axis(0), select_row_indices)
    };

    let mut stat = arc_stat.lock().expect("failed to lock col_stat");

    for x in xx.axis_iter(Axis(0)) {
        for target_column_index in lb..ub {
            let source_column_index = target_column_index - lb;
            stat.add_element(&[target_column_index], x[source_column_index]);
        }
    }

    Ok(())
}
