//! Read-side statistics and write-side streaming for one sparse backend.
//!
//! Everything here goes through data-beans: stats via the streaming
//! collectors, and the filtered copy via
//! [`data_beans::column_subset::stream_column_selection`], which writes the
//! survivors slab by slab without materialising them. `qc` never edits a file
//! in place.

use crate::common::*;
use data_beans::column_subset::stream_column_selection;
use data_beans::hdf5_io::resolve_backend_file;
use legume_numeric::matrix::traits::RunningStatOps;

pub type Backend = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

pub fn open_matrix(path: &str) -> anyhow::Result<Backend> {
    let (backend, file) = resolve_backend_file(path, None)?;
    open_sparse_matrix(&file, &backend)
}

/// `(rows, cols, nnz)` of a backend, from its attributes alone.
pub fn shape(data: &dyn SparseIo<IndexIter = Vec<usize>>) -> (usize, usize, usize) {
    (
        data.num_rows().unwrap_or(0),
        data.num_columns().unwrap_or(0),
        data.num_non_zeros().unwrap_or(0),
    )
}

/// Per-row `(nnz, sum)` over every column.
pub fn row_nnz_sum(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: Option<usize>,
) -> anyhow::Result<(Vec<usize>, Vec<f64>)> {
    let st = collect_row_stat(data, block_size)?;
    let nnz = st.count_positives().iter().map(|&x| x as usize).collect();
    let sum = st.sum().iter().map(|&x| x as f64).collect();
    Ok((nnz, sum))
}

/// Per-row count of positive entries restricted to `cols` (ascending global
/// column ids): the "cells per feature" a cut is measured on once the cell
/// axis has been decided. Counts only, never entries.
pub fn row_nnz_over_columns(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    cols: &[usize],
) -> anyhow::Result<Vec<usize>> {
    let nrow = data.num_rows().unwrap_or(0);
    let mut nnz = vec![0usize; nrow];
    for (lb, ub) in
        legume_numeric::matrix::utils::generate_minibatch_intervals(cols.len(), 0, Some(8192))
    {
        let (_, _, triplets) = data.read_triplets_by_columns(cols[lb..ub].to_vec())?;
        for (r, _, x) in triplets {
            if x > 0.0 {
                nnz[r as usize] += 1;
            }
        }
    }
    Ok(nnz)
}

/// Where a filtered matrix goes: `{out_dir}/{stem}` in the input's backend,
/// zipped when the input was.
pub struct OutSpec<'a> {
    pub out_dir: &'a str,
    pub stem: &'a str,
    pub backend: &'a SparseIoBackend,
    pub zip: bool,
}

impl OutSpec<'_> {
    pub fn path(&self) -> crate::quant::BackendOutputPath {
        crate::quant::BackendOutputPath::new(self.out_dir, self.stem, self.backend, self.zip)
    }
}

/// Shape of one written matrix, for the summary table.
#[derive(Debug, Clone)]
pub struct Written {
    pub target: Box<str>,
    pub nrow: usize,
    pub ncol: usize,
    pub nnz: usize,
}

/// Stream `cols` × `rows` (both ascending global ids) into a fresh backend.
/// `None` when the selection is empty on either axis: an empty matrix is not
/// a useful file, and the caller logs it.
pub fn write_subset(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    cols: &[usize],
    rows: &[usize],
    row_names: &[Box<str>],
    col_names: &[Box<str>],
    spec: &OutSpec<'_>,
) -> anyhow::Result<Option<Written>> {
    if cols.is_empty() || rows.is_empty() {
        return Ok(None);
    }
    let out = spec.path();
    remove_file(&out.write_path)?;
    remove_file(&out.target_path)?;
    let out_rows: Vec<Box<str>> = rows.iter().map(|&i| row_names[i].clone()).collect();
    let out_cols: Vec<Box<str>> = cols.iter().map(|&j| col_names[j].clone()).collect();
    let (nrow, ncol, nnz) = stream_column_selection(
        data,
        cols,
        Some(rows),
        &out_rows,
        &out_cols,
        &out.write_path,
        spec.backend,
    )?;
    out.finalize()?;
    Ok(Some(Written {
        target: out.target_path.clone(),
        nrow,
        ncol,
        nnz,
    }))
}

/// Ascending global ids of the columns whose name is in `keep` (every column
/// when `None`) and which carry at least one stored entry, read off the
/// resident indptr rather than a stat pass. faba stores positive entries
/// only, so stored nnz is positive nnz.
pub fn select_columns(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    col_names: &[Box<str>],
    keep: Option<&rustc_hash::FxHashSet<Box<str>>>,
) -> Vec<usize> {
    (0..col_names.len())
        .filter(|&j| {
            data.column_nnz(j).unwrap_or(0) > 0 && keep.is_none_or(|k| k.contains(&col_names[j]))
        })
        .collect()
}
