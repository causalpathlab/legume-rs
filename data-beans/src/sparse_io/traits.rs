#![allow(dead_code, unused_imports)]

pub use candle_util::candle_core::Tensor;
pub use nalgebra::DMatrix;
pub use nalgebra_sparse::{csc::CscMatrix, csr::CsrMatrix};
pub use ndarray::prelude::*;

pub const MAX_ROW_NAME_IDX: usize = 3;
pub const MAX_COLUMN_NAME_IDX: usize = 10;
pub const COLUMN_SEP: &str = "@";
pub const ROW_SEP: &str = "_";

use super::helpers::*;

use crate::sparse_data_visitors::styled_progress_bar;
use clap::ValueEnum;
use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::mtx_io::*;
use matrix_util::traits::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum SparseIoBackend {
    Zarr,
    HDF5,
}

/// Identifies one of the six 1-D datasets inside a sparse backend.
/// Used by the streaming write API so we don't have to add six separate
/// abstract methods per dtype × (csc|csr) × (data|indices|indptr).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsKey {
    CscData,
    CscIndices,
    CscIndptr,
    CsrData,
    CsrIndices,
    CsrIndptr,
}

pub trait SparseIo: Sync + Send {
    type IndexIter: IntoIterator<Item = usize> + FromIterator<usize>;

    ////////////////////////////
    // default implementation //
    ////////////////////////////

    /// Read columns within the range and return dense `ndarray::Array2`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_ndarray(&self, columns: Self::IndexIter) -> anyhow::Result<Array2<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        Array2::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Read columns within the range and return dense `candle_core::Tensor`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_tensor(&self, columns: Self::IndexIter) -> anyhow::Result<Tensor> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        Tensor::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Read columns within the range and return dense `nalgebrea::DMatrix`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_dmatrix(&self, columns: Self::IndexIter) -> anyhow::Result<DMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Read columns within the range and return sparse `CsrMatrix`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_csr(&self, columns: Self::IndexIter) -> anyhow::Result<CsrMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Read columns within the range and return sparse `CsrMatrix`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_csc(&self, columns: Self::IndexIter) -> anyhow::Result<CscMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Zero-copy view of preloaded column-major CSC arrays as
    /// `(indptr, indices, data)`. Returns `None` when the backend has
    /// not preloaded columns or doesn't support direct array access.
    /// Callers (e.g. `SparseIoVec::read_columns_csc`) use this to skip
    /// the triplet roundtrip when columns are already in memory.
    fn csc_column_arrays(&self) -> Option<(&[u64], &[u64], &[f32])> {
        None
    }

    /// Read rows within the range and return dense `ndarray::Array2`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_ndarray(&self, rows: Self::IndexIter) -> anyhow::Result<Array2<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        Array2::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Read rows within the range and return dense `candle_core::Tensor`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_tensor(&self, rows: Self::IndexIter) -> anyhow::Result<Tensor> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        Tensor::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Read rows within the range and return dense `nalgebra::DMatrix`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_dmatrix(&self, rows: Self::IndexIter) -> anyhow::Result<DMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Read rows within the range and return sparse `CsrMatrix`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_csr(&self, rows: Self::IndexIter) -> anyhow::Result<CsrMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Read rows within the range and return sparse `CscMatrix`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_csc(&self, rows: Self::IndexIter) -> anyhow::Result<CscMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /////////////////////////////
    // `mtx` related functions //
    /////////////////////////////

    /// Read mtx file and populate the data into HDF5 for faster row-by-row access
    /// * `mtx_file`: mtx file to be read into HDF5 backend
    fn import_mtx_file_by_row(&mut self, mtx_file: &str) -> anyhow::Result<()> {
        let (mut mtx_triplets, mtx_shape) = read_mtx_triplets(mtx_file)?;
        info!("read mtx file: {}", mtx_file);
        if mtx_triplets.is_empty() {
            return Err(anyhow::anyhow!("No data in mtx file"));
        }
        self.record_mtx_shape(mtx_shape)?;
        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Read mtx file and populate the data into HDF5 for faster column-by-column access
    /// * `mtx_file`: mtx file to be read into HDF5 backend
    fn import_mtx_file_by_col(&mut self, mtx_file: &str) -> anyhow::Result<()> {
        let (mut mtx_triplets, mtx_shape) = read_mtx_triplets(mtx_file)?;
        info!("read mtx file: {}", mtx_file);
        if mtx_triplets.is_empty() {
            return Err(anyhow::anyhow!("No data in mtx file"));
        }
        self.record_mtx_shape(mtx_shape)?;
        self.record_triplets_by_col(&mut mtx_triplets)
    }

    /////////////////////////////////
    // `dmatrix` related functions //
    /////////////////////////////////

    /// Add dmatrix to zarr backend by row (CSR format)
    /// * `array` - 2D array to be added to the backend
    fn import_dmatrix_by_row(&mut self, matrix: &DMatrix<f32>) -> anyhow::Result<()> {
        let (nrow, ncol) = matrix.shape();
        let mut mtx_triplets = dmatrix_to_triplets(matrix);
        let mtx_shape = (nrow, ncol, mtx_triplets.len());
        self.record_mtx_shape(Some(mtx_shape))?;
        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Add dmatrix to zarr backend by column (CSC format)
    /// * `array` - 2D array to be added to the backend
    fn import_dmatrix_by_col(&mut self, matrix: &DMatrix<f32>) -> anyhow::Result<()> {
        let (nrow, ncol) = matrix.shape();
        let mut mtx_triplets = dmatrix_to_triplets(matrix);
        let mtx_shape = (nrow, ncol, mtx_triplets.len());
        self.record_mtx_shape(Some(mtx_shape))?;
        self.record_triplets_by_col(&mut mtx_triplets)
    }

    /////////////////////////////////
    // `ndarray` related functions //
    /////////////////////////////////

    /// Add ndarray to zarr backend by row (CSR format)
    /// * `array` - 2D array to be added to the backend
    fn import_ndarray_by_row(&mut self, array: &Array2<f32>) -> anyhow::Result<()> {
        let nrow = array.shape()[0];
        let ncol = array.shape()[1];

        // dbg!("importing ndarray by row...");
        let mut mtx_triplets = ndarray_to_triplets(array);

        let nnz = mtx_triplets.len();
        let mtx_shape = (nrow, ncol, nnz);
        self.record_mtx_shape(Some(mtx_shape))?;

        // dbg!(format!("populated: {} elements", mtx_triplets.len()));

        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Add ndarray to zarr backend by column (CSC format)
    /// * `array` - 2D array to be added to the backend
    fn import_ndarray_by_col(&mut self, array: &Array2<f32>) -> anyhow::Result<()> {
        let nrow = array.shape()[0];
        let ncol = array.shape()[1];

        // dbg!("importing ndarray by column...");
        let mut mtx_triplets = ndarray_to_triplets(array);

        let nnz = mtx_triplets.len();
        let mtx_shape = (nrow, ncol, nnz);
        self.record_mtx_shape(Some(mtx_shape))?;

        // dbg!(format!("populated: {} elements", mtx_triplets.len()));

        self.record_triplets_by_col(&mut mtx_triplets)
    }

    //////////////////////
    // backend-specific //
    //////////////////////

    /// Read rows within the range and return a vector of triplets (row, column, value)
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    #[allow(clippy::type_complexity)]
    fn read_triplets_by_rows(
        &self,
        rows: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)>;

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    #[allow(clippy::type_complexity)]
    fn read_triplets_by_columns(
        &self,
        columns: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)>;

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `col` : usize
    ///
    #[allow(clippy::type_complexity)]
    fn read_triplets_by_single_column(
        &self,
        col: usize,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)>;

    /// Export the data to a mtx file. This will take time.
    /// * `mtx_file`: mtx file to be written
    fn to_mtx_file(&self, mtx_file: &str) -> anyhow::Result<()>;

    /// Number of rows in the underlying data matrix
    fn num_rows(&self) -> Option<usize>;

    /// Number of columns in the underlying data matrix
    fn num_columns(&self) -> Option<usize>;

    /// Number of non-zero elements
    fn num_non_zeros(&self) -> Option<usize>;

    /// Re-open handles on the CURRENT backend path after its contents were
    /// replaced from outside (a finished temp file renamed into place). The
    /// zarr store is path-addressed so this is a cache refresh; hdf5 holds an
    /// open file handle that would otherwise point at the deleted inode.
    fn reopen_backend(&mut self) -> anyhow::Result<()>;

    /// Maintained by [`append_csc_slab`](Self::append_csc_slab) — never call
    /// this yourself: padding the cursor masks exactly the under-append the
    /// finalize audit exists to catch.
    #[doc(hidden)]
    /// Advance the streaming-write cursor by `n` entries. Called by
    /// [`append_csc_slab`](Self::append_csc_slab); backends keep the count so
    /// [`finalize_streaming_csc`](Self::finalize_streaming_csc) can audit the
    /// declared nnz against what was actually appended — the one violation the
    /// written indptr cannot reveal, because an over-declared tail leaves it
    /// perfectly monotone with the phantom hiding between the last written
    /// pointer and the sentinel.
    fn note_streamed_nnz(&mut self, n: u64);

    /// Entries appended so far in this streaming build.
    #[doc(hidden)]
    fn streamed_nnz(&self) -> u64;

    /// Zero the cursor. Called by [`begin_streaming_csc`](Self::begin_streaming_csc).
    #[doc(hidden)]
    fn reset_streamed_nnz(&mut self);

    /// The resident by-column indptr, loaded at `open()`. Empty when the
    /// backend carries no `/by_column/indptr` array — `read_column_indptr`
    /// silently does nothing on that failure, and the accessors below must
    /// report that absence rather than read zeros out of it.
    fn column_indptr(&self) -> &[u64];

    /// Exact nnz of one column, from the resident indptr — no I/O.
    ///
    /// `None` for an out-of-range column or when the indptr is absent. This is
    /// what lets a streaming writer declare a column subset's total nnz up
    /// front without a counting pass over the data.
    fn column_nnz(&self, col: usize) -> Option<u64> {
        let indptr = self.column_indptr();
        let hi = *indptr.get(col + 1)?;
        let lo = *indptr.get(col)?;
        hi.checked_sub(lo)
    }

    /// Set row names for the matrix
    /// * `row_name_file`: a file each line contains row name words
    fn register_row_names_file(&mut self, row_name_file: &str);

    /// Set column names for the matrix
    /// * `column_name_file`: a file each line contains column name words
    fn register_column_names_file(&mut self, column_name_file: &str);

    /// Set row names for the matrix
    /// * `rows`: a vector of row names
    fn register_row_names_vec(&mut self, rows: &[Box<str>]);

    /// Set column names for the matrix
    /// * `columns`: a vector of column names
    fn register_column_names_vec(&mut self, columns: &[Box<str>]);

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `name_file`: a file each line contains name words
    /// * `name_columns`: range of columns to be used for name
    /// * `name_sep`: separator for name columns
    fn register_names_file(
        &mut self,
        key: &str,
        name_file: &str,
        name_columns: Range<usize>,
        name_sep: &str,
    ) -> anyhow::Result<()>;

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `names`: a file each line contains name words
    fn register_names_vec(&mut self, key: &str, names: &[Box<str>]) -> anyhow::Result<()>;

    fn row_names(&self) -> anyhow::Result<Vec<Box<str>>>;

    fn column_names(&self) -> anyhow::Result<Vec<Box<str>>>;

    /// Get back the registered names
    /// * `key`: key for the registered names
    fn retrieve_registered_names(&self, key: &str) -> anyhow::Result<Vec<Box<str>>>;

    /////////////////////////////
    // major structural change //
    /////////////////////////////

    /// Select the columns of the data and create a new backend file
    /// * `columns`: columns to be subsetted
    /// * `rows`: if something, subset the rows
    fn subset_columns_rows(
        &mut self,
        columns: Option<&Vec<usize>>,
        rows: Option<&Vec<usize>>,
    ) -> anyhow::Result<()> {
        let ncol_data = self
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("missing shape information"))?;
        let nrow_data = self
            .num_rows()
            .ok_or_else(|| anyhow::anyhow!("missing shape information"))?;

        // An empty selection is refused, not honoured: this method DESTROYS the
        // original backend, and writing a zero-column husk over real data is
        // almost certainly a caller mistake rather than an intention.
        // Empty and duplicated selections are refused, not honoured: this
        // method DESTROYS the original, and a duplicate collapses in the
        // old→new map while still counting toward the new shape — slabs then
        // land at wrong offsets and the finalize audit rejects the build with a
        // message about nnz tiling that names nothing the caller did.
        let distinct = |sel: &[usize], what: &str| -> anyhow::Result<()> {
            anyhow::ensure!(!sel.is_empty(), "subset: empty {what} selection");
            let mut seen = sel.to_vec();
            seen.sort_unstable();
            seen.dedup();
            anyhow::ensure!(
                seen.len() == sel.len(),
                "subset: the {what} selection repeats an index ({} of {} are distinct)",
                seen.len(),
                sel.len()
            );
            Ok(())
        };
        if let Some(cols) = columns {
            distinct(cols, "column")?;
        }
        if let Some(rs) = rows {
            distinct(rs, "row")?;
        }

        //////////////////////////////////////////////////////
        // 0. Create a mapping from old to new columns/rows //
        //////////////////////////////////////////////////////

        let (old2new_cols, new_col_names) =
            take_subset_indices_names_if_needed(columns, Some(ncol_data), self.column_names()?);
        let (old2new_rows, new_row_names) =
            take_subset_indices_names_if_needed(rows, Some(nrow_data), self.row_names()?);
        let (new_ncol, new_nrow) = (new_col_names.len(), new_row_names.len());
        anyhow::ensure!(new_ncol > 0, "subset: no column survived the selection");
        anyhow::ensure!(new_nrow > 0, "subset: no row survived the selection");

        // Old columns in NEW order — the selection's order is the output order.
        let mut cols_new_order: Vec<(u64, u64)> =
            old2new_cols.iter().map(|(&o, &n)| (n, o)).collect();
        cols_new_order.sort_unstable();

        // Dense old-row → new-row map, and whether it preserves order. A
        // monotone map keeps within-column rows ascending after renumbering, so
        // no per-column sort is needed; a reordering map costs one small sort
        // per column.
        let mut row_map: Vec<Option<u64>> = vec![None; nrow_data];
        for (&old, &new) in &old2new_rows {
            row_map[old as usize] = Some(new);
        }
        let monotone_rows = row_map.iter().flatten().is_sorted_by(|a, b| a < b);

        ///////////////////////////////////////////////////////
        // 1. Exact per-new-column nnz, without materialising //
        ///////////////////////////////////////////////////////

        // No row filter: straight off the resident indptr, zero I/O. With one:
        // a counting pass — reads every selected column once and keeps counts,
        // never entries.
        let full_rows = rows.is_none();
        let per_col_nnz: Vec<u64> = if full_rows {
            cols_new_order
                .iter()
                .map(|&(_, old)| {
                    self.column_nnz(old as usize)
                        .ok_or_else(|| anyhow::anyhow!("subset: no indptr for column {old}"))
                })
                .collect::<anyhow::Result<_>>()?
        } else {
            // Block reads, never one column at a time: a single-column read
            // pays the cached-subset machinery per call, which measured two
            // orders of magnitude slower at imaging scale. Counts only, never
            // entries.
            let mut counts = vec![0u64; cols_new_order.len()];
            let coarse = matrix_util::utils::generate_minibatch_intervals(
                cols_new_order.len(),
                0,
                Some(8192),
            );
            for (lb, ub) in coarse {
                let old_cols: Vec<usize> = cols_new_order[lb..ub]
                    .iter()
                    .map(|&(_, o)| o as usize)
                    .collect();
                let (_, _, triplets) =
                    self.read_triplets_by_columns(old_cols.into_iter().collect())?;
                for (i, c_local, _) in triplets {
                    if row_map[i as usize].is_some() {
                        counts[lb + c_local as usize] += 1;
                    }
                }
            }
            counts
        };
        let new_nnz: u64 = per_col_nnz.iter().sum();

        ///////////////////////////////////////////////////////////
        // 2. Stream the survivors into a TEMPORARY sibling file //
        ///////////////////////////////////////////////////////////

        // Written beside the original (same filesystem, so the final rename is
        // atomic) and swapped in only when complete. The old implementation
        // deleted the original FIRST and rewrote it from a RAM buffer, so a
        // crash mid-write lost the data outright — and that buffer held every
        // surviving triplet, which is the memory wall this replaces.
        // CONTRACT of the swap: the original is untouched until the temporary
        // sibling is complete and finalized; a failure mid-stream leaves the
        // original intact plus a `{path}.subset_tmp` leftover (cleaned on the
        // next attempt); the unrecoverable window is only remove→rename below.
        // A zip-archived backend is refused up front — streaming to a sibling
        // DIRECTORY and renaming it over the `.zip` name would silently change
        // the on-disk format under the old extension, and the old code's
        // "store is read-only" failure was at least loud.
        let final_path = self.get_backend_file_name().to_string();
        anyhow::ensure!(
            !final_path.ends_with(".zip"),
            "subset: {final_path} is a zip archive; convert it to a directory \
             backend first (data-beans convert)"
        );
        let temp_path = format!("{final_path}.subset_tmp");
        if std::path::Path::new(&temp_path).exists() {
            crate::sparse_io::remove_backend_path(&temp_path)?;
        }

        {
            let backend_kind = self.backend_type();
            let mut out = crate::sparse_io::create_sparse_streaming_empty(
                Some(&temp_path),
                Some(&backend_kind),
            )?;
            out.begin_streaming_csc((new_nrow, new_ncol, new_nnz as usize))?;

            // Blocks bounded by bytes of surviving triplets, from the measured
            // per-column counts — not by a fixed column count.
            let blocks = matrix_util::utils::byte_budget_intervals(
                &per_col_nnz,
                crate::sparse_io::SLAB_BUDGET_BYTES,
                crate::sparse_io::TRIPLET_BYTES,
            );

            let mut nnz_offset = 0u64;
            for (lb, ub) in blocks {
                // ONE block read per slab (see the counting pass above for why).
                // The read returns LOCAL column ids in the requested order, rows
                // ascending within each column.
                let old_cols: Vec<usize> = cols_new_order[lb..ub]
                    .iter()
                    .map(|&(_, o)| o as usize)
                    .collect();
                let (_, _, triplets) =
                    self.read_triplets_by_columns(old_cols.into_iter().collect())?;

                let n_block = ub - lb;
                let mut per_col: Vec<Vec<(u64, f32)>> = vec![Vec::new(); n_block];
                for (i, c_local, x) in triplets {
                    if let Some(new_row) = row_map[i as usize] {
                        per_col[c_local as usize].push((new_row, x));
                    }
                }
                let mut local_colptr = Vec::with_capacity(n_block);
                let mut row_indices = Vec::new();
                let mut values = Vec::new();
                for entries in &mut per_col {
                    if !monotone_rows {
                        // Renumbering scrambled this column's order; restore the
                        // ascending-rows invariant the writer enforces.
                        entries.sort_unstable_by_key(|&(r, _)| r);
                    }
                    local_colptr.push(row_indices.len() as u64);
                    for &(r, x) in entries.iter() {
                        row_indices.push(r);
                        values.push(x);
                    }
                }
                out.append_csc_slab(lb as u64, nnz_offset, &local_colptr, &row_indices, &values)?;
                nnz_offset += values.len() as u64;
            }

            out.finalize_streaming_csc()?;
            out.build_csr_from_csc_streaming()?;
            out.register_row_names_vec(&new_row_names);
            out.register_column_names_vec(&new_col_names);
        }

        ////////////////////////////////////
        // 3. Swap the finished file in  //
        ////////////////////////////////////

        self.remove_backend_file()?;
        std::fs::rename(&temp_path, &final_path)?;
        self.reopen_backend()?;
        self.clean_preloaded_columns();
        self.clean_preloaded_rows();
        info!("registered new data to {}", self.get_backend_file_name());
        Ok(())
    }

    /// Reposition rows in a new order specified by `remap`
    /// * `row_names_order` - a vector of row names in the new order
    fn reorder_rows(&mut self, row_names_order: &[Box<str>]) -> anyhow::Result<()> {
        let new_col_names = self.column_names()?.clone();
        let name2new = build_name2index_map(row_names_order);

        let block_size = 100;

        let old2new: HashMap<u64, u64> = self
            .row_names()?
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx_old, name)| {
                name2new
                    .get(&name)
                    .map(|&idx_new| (idx_old as u64, idx_new as u64))
            })
            .collect();

        if let Some(ncol) = self.num_columns() {
            /////////////////////////////////////////////////////
            // 1. triplets after filtering and reordering rows //
            /////////////////////////////////////////////////////

            let arc_triplets = Arc::new(Mutex::new(vec![]));

            let nblock = ncol.div_ceil(block_size);

            info!("remapping triplets ...");

            (0..nblock)
                .into_par_iter()
                .progress_with(styled_progress_bar(nblock as u64, "blocks"))
                .map(|b| {
                    let lb = (b * block_size) as u64;
                    let ub = ((b + 1) * block_size).min(ncol) as u64;
                    (lb, ub)
                })
                .for_each(|(lb, ub)| {
                    let (_, _, _triplets_b) = self
                        .read_triplets_by_columns(((lb as usize)..(ub as usize)).collect())
                        .unwrap();

                    let _triplets_b = _triplets_b.into_iter().filter_map(|(i, j_loc, x)| {
                        let j_glob = j_loc + lb;
                        old2new.get(&i).map(|&i_new| (i_new, j_glob, x))
                    });

                    {
                        let mut triplets = arc_triplets.lock().unwrap();
                        triplets.extend(_triplets_b);
                    }
                });

            /////////////////////////////////////
            // 2. Remove previous backend file //
            /////////////////////////////////////
            self.remove_backend_file()?;

            ///////////////////////////////
            // 3. populate a new backend //
            ///////////////////////////////
            self.initialize_backend()?;

            // populate data from mtx triplets
            {
                let mut row_col_val_triplets =
                    arc_triplets.lock().expect("failed to lock triplets");

                let nnz = row_col_val_triplets.len();
                debug_assert!(row_col_val_triplets.len() <= nnz); // subset
                let new_nrow = row_names_order.len();
                let mtx_shape = (new_nrow, ncol, nnz);

                info!("sorting triplets ...");

                self.record_mtx_shape(Some(mtx_shape))?;
                self.record_triplets_by_col(&mut row_col_val_triplets)?;
                self.record_triplets_by_row(&mut row_col_val_triplets)?;
            }
            self.read_column_indptr()?;
            self.read_row_indptr()?;

            self.register_row_names_vec(row_names_order);
            self.register_column_names_vec(&new_col_names);
            info!("registered new data to {}", self.get_backend_file_name());
        }

        self.clean_preloaded_columns();
        self.clean_preloaded_rows();
        Ok(())
    }
    // fn reorder_rows(&mut self, row_names_order: &[Box<str>]) -> anyhow::Result<()>;

    /// Remove backend file
    fn remove_backend_file(&self) -> anyhow::Result<()>;

    /// Initialize backend
    fn initialize_backend(&mut self) -> anyhow::Result<()>;

    fn record_mtx_shape(&mut self, mtx_shape: Option<(usize, usize, usize)>) -> anyhow::Result<()>;

    /// Helper function to add triplets to zarr backend by row (CSR format)
    fn record_triplets_by_row(
        &mut self,
        row_col_val_triplets: &mut Vec<(u64, u64, f32)>,
    ) -> anyhow::Result<()> {
        let nrow = self.num_rows().expect("should have `nrow`");

        if row_col_val_triplets.is_empty() {
            let csr_rowptr = vec![0u64; nrow + 1];
            return self.record_csr_dataset_backend(&[], &[], &csr_rowptr);
        }

        row_col_val_triplets.par_sort_by_key(|&(_, col, _)| col);
        row_col_val_triplets.par_sort_by_key(|&(row, _, _)| row);

        let mut csr_rowptr = vec![];
        let mut csr_cols = vec![];
        let mut csr_vals = vec![];

        let nnz = row_col_val_triplets.len();

        let first = row_col_val_triplets[0].0 as usize;
        csr_rowptr.resize(first, 0);

        csr_rowptr.push(0);
        csr_cols.push(row_col_val_triplets[0].1);
        csr_vals.push(row_col_val_triplets[0].2);

        for i in 1..nnz {
            let lb = row_col_val_triplets[i - 1].0;
            let ub = row_col_val_triplets[i].0;
            for _ in lb..ub {
                csr_rowptr.push(i as u64);
            }
            csr_cols.push(row_col_val_triplets[i].1);
            csr_vals.push(row_col_val_triplets[i].2);
        }

        let last = row_col_val_triplets[nnz - 1].0 as usize;
        for _ in last..nrow {
            csr_rowptr.push(nnz as u64);
        }

        self.record_csr_dataset_backend(&csr_cols, &csr_vals, &csr_rowptr)
    }

    fn record_triplets_by_col(
        &mut self,
        row_col_val_triplets: &mut Vec<(u64, u64, f32)>,
    ) -> anyhow::Result<()> {
        let ncol = self.num_columns().expect("should have `ncol`");

        if row_col_val_triplets.is_empty() {
            let csc_colptr = vec![0u64; ncol + 1];
            return self.record_csc_dataset_backend(&[], &[], &csc_colptr);
        }

        row_col_val_triplets.par_sort_by_key(|&(row, _, _)| row);
        row_col_val_triplets.par_sort_by_key(|&(_, col, _)| col);

        let mut csc_colptr: Vec<u64> = vec![];
        let mut csc_rows: Vec<u64> = vec![];
        let mut csc_vals: Vec<f32> = vec![];

        let nnz = row_col_val_triplets.len();

        let first = row_col_val_triplets[0].1 as usize;
        csc_colptr.resize(first, 0);

        csc_colptr.push(0);
        csc_rows.push(row_col_val_triplets[0].0);
        csc_vals.push(row_col_val_triplets[0].2);

        for i in 1..nnz {
            let lb = row_col_val_triplets[i - 1].1;
            let ub = row_col_val_triplets[i].1;
            for _ in lb..ub {
                csc_colptr.push(i as u64);
            }
            csc_rows.push(row_col_val_triplets[i].0);
            csc_vals.push(row_col_val_triplets[i].2);
        }

        let last = row_col_val_triplets[nnz - 1].1 as usize;
        for _ in last..ncol {
            csc_colptr.push(nnz as u64);
        }

        self.record_csc_dataset_backend(&csc_rows, &csc_vals, &csc_colptr)
    }

    /// CSR data structure in Zarr backend
    ///
    /// ```text
    ///     └── by_row
    ///         ├── data
    ///         ├── indices (column indices)
    ///         └── isndptr (row pointers)
    /// ```
    fn record_csr_dataset_backend(
        &mut self,
        csr_cols: &[u64],
        csr_vals: &[f32],
        csr_rowptr: &[u64],
    ) -> anyhow::Result<()>;

    /// Helper function to add CSC dataset to HDF5 backend
    ///
    /// ```text
    /// Helper function to record the CSC dataset
    ///     ├── by_column
    ///     │   ├── data
    ///     │   ├── indices (row indices)
    ///     │   └── indptr (column pointers)
    /// ```
    fn record_csc_dataset_backend(
        &mut self,
        csc_rows: &[u64],
        csc_vals: &[f32],
        csc_colptr: &[u64],
    ) -> anyhow::Result<()>;

    /// Create a fixed-size 1-D backend dataset of `len` elements for the
    /// given CSC/CSR slot. No data is written yet.
    fn cs_create(&mut self, key: CsKey, len: usize) -> anyhow::Result<()>;

    /// Write a `u64` slab at `offset` in the specified dataset.
    /// Used for CSC/CSR `indices` and `indptr`.
    fn cs_write_u64(&mut self, key: CsKey, offset: u64, data: &[u64]) -> anyhow::Result<()>;

    /// Write an `f32` slab at `offset` in the specified dataset.
    /// Used for CSC/CSR `data`.
    fn cs_write_f32(&mut self, key: CsKey, offset: u64, data: &[f32]) -> anyhow::Result<()>;

    /// Begin a streaming CSC build for a sparse matrix of known shape.
    /// Pre-creates `/by_column/{data, indices, indptr}` at their final sizes
    /// so subsequent [`append_csc_slab`](Self::append_csc_slab) calls write
    /// into disjoint hyperslabs without further allocation.
    fn begin_streaming_csc(&mut self, shape: (usize, usize, usize)) -> anyhow::Result<()> {
        // A reused handle must not inherit a previous build's cursor: the
        // finalize audit compares appended-vs-declared, and a stale count turns
        // a correct build into a false accusation.
        self.reset_streamed_nnz();
        let (_, ncol, nnz) = shape;
        self.record_mtx_shape(Some(shape))?;
        self.cs_create(CsKey::CscData, nnz)?;
        self.cs_create(CsKey::CscIndices, nnz)?;
        self.cs_create(CsKey::CscIndptr, ncol + 1)?;
        Ok(())
    }

    /// Append one contiguous CSC column band.
    ///
    /// * `col_offset` — global column index where this band starts
    /// * `nnz_offset` — global nnz offset where this band's values land
    /// * `local_colptr` — length `batch_ncol`, values in `[0, batch_nnz]`,
    ///   will be shifted by `nnz_offset` before writing
    /// * `row_indices` — length `batch_nnz`
    /// * `values`      — length `batch_nnz`
    fn append_csc_slab(
        &mut self,
        col_offset: u64,
        nnz_offset: u64,
        local_colptr: &[u64],
        row_indices: &[u64],
        values: &[f32],
    ) -> anyhow::Result<()> {
        // These checks exist because a violation does NOT fail loudly on its
        // own: unwritten regions read back as the zarr fill value, so a bad
        // slab yields a backend that opens and reads cleanly while carrying
        // poisoned or duplicated entries. Cheap (one pass over the slab, no
        // I/O) next to the compressed writes below.
        anyhow::ensure!(
            row_indices.len() == values.len(),
            "append_csc_slab: {} row indices vs {} values",
            row_indices.len(),
            values.len()
        );
        anyhow::ensure!(
            local_colptr.first().copied() == Some(0) || local_colptr.is_empty(),
            "append_csc_slab: local_colptr must start at 0"
        );
        anyhow::ensure!(
            local_colptr.windows(2).all(|w| w[0] <= w[1]),
            "append_csc_slab: local_colptr must be monotone non-decreasing"
        );
        if let Some(&last) = local_colptr.last() {
            anyhow::ensure!(
                last <= values.len() as u64,
                "append_csc_slab: colptr claims {last} entries, slab holds {}",
                values.len()
            );
        }
        if let Some(nrow) = self.num_rows() {
            if let Some(&bad) = row_indices.iter().find(|&&r| r >= nrow as u64) {
                anyhow::bail!("append_csc_slab: row index {bad} outside the {nrow}-row matrix");
            }
        }
        // Ascending rows within each column: readers document it as an
        // invariant, and the h5ad export hands the arrays to scipy as-is.
        for (c, &start) in local_colptr.iter().enumerate() {
            let end = local_colptr
                .get(c + 1)
                .copied()
                .unwrap_or(values.len() as u64) as usize;
            anyhow::ensure!(
                row_indices[start as usize..end]
                    .windows(2)
                    .all(|w| w[0] < w[1]),
                "append_csc_slab: rows within column {} of this band must be \
                 strictly ascending — repeated rows usually mean duplicate \
                 (row, col) coordinates in the source (an MTX with repeated \
                 entries, or a union remap folding rows together)",
                col_offset as usize + c
            );
        }

        let shifted: Vec<u64> = local_colptr.iter().map(|&p| p + nnz_offset).collect();
        self.cs_write_u64(CsKey::CscIndptr, col_offset, &shifted)?;
        self.cs_write_u64(CsKey::CscIndices, nnz_offset, row_indices)?;
        self.cs_write_f32(CsKey::CscData, nnz_offset, values)?;
        self.note_streamed_nnz(values.len() as u64);
        Ok(())
    }

    /// Finalize CSC streaming by writing the final indptr sentinel at
    /// position `ncol`, equal to the total nnz.
    fn finalize_streaming_csc(&mut self) -> anyhow::Result<()> {
        let ncol = self
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("ncol not set before finalize_streaming_csc"))?;
        let nnz = self
            .num_non_zeros()
            .ok_or_else(|| anyhow::anyhow!("nnz not set before finalize_streaming_csc"))?;
        self.cs_write_u64(CsKey::CscIndptr, ncol as u64, &[nnz as u64])?;
        self.read_column_indptr()?;

        // The tiling check the per-slab guards cannot do. A gap or overlap in
        // the nnz offsets, or an over-declared total, leaves the WRITTEN
        // indptr non-monotone or short of the declared nnz — and unwritten
        // indptr slots read back as the fill value 0, so column j would claim
        // the whole array prefix. `indptr[ncol] - indptr[0]` equals the
        // declaration by construction, which is why the old debug_assert on it
        // could never fire; the shape of the vector between the endpoints is
        // what carries the truth.
        let indptr = self.column_indptr();
        anyhow::ensure!(
            indptr.len() == ncol + 1,
            "finalize_streaming_csc: indptr has {} entries, expected {}",
            indptr.len(),
            ncol + 1
        );
        anyhow::ensure!(
            indptr.first().copied() == Some(0),
            "finalize_streaming_csc: indptr[0] = {:?}, expected 0 — the first \
             slab was never appended",
            indptr.first()
        );
        if let Some(w) = indptr.windows(2).position(|w| w[0] > w[1]) {
            anyhow::bail!(
                "finalize_streaming_csc: indptr decreases at column {w} — slabs \
                 were appended with a gap or overlap in their nnz offsets"
            );
        }
        // The appended count is the ground truth the indptr cannot carry: an
        // over-declared nnz leaves the written indptr perfectly monotone with
        // the phantom tail hiding between the last written pointer and the
        // sentinel — and the sentinel itself was written by this function, so
        // comparing against it can only ever agree.
        let appended = self.streamed_nnz();
        anyhow::ensure!(
            appended == nnz as u64,
            "finalize_streaming_csc: {appended} entries appended but {nnz} \
             declared — the difference reads back as fill values wearing real \
             entries' positions"
        );
        Ok(())
    }

    /// Build `/by_row/{data, indices, indptr}` by transposing the already-
    /// written CSC data on disk. Uses two passes over CSC with bounded
    /// auxiliary memory (~`24 B × nrow` plus one row-band worth of CSR).
    fn build_csr_from_csc_streaming(&mut self) -> anyhow::Result<()> {
        let nrow = self
            .num_rows()
            .ok_or_else(|| anyhow::anyhow!("nrow not set before build_csr_from_csc_streaming"))?;
        let ncol = self
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("ncol not set before build_csr_from_csc_streaming"))?;
        let nnz = self
            .num_non_zeros()
            .ok_or_else(|| anyhow::anyhow!("nnz not set before build_csr_from_csc_streaming"))?;

        if nnz == 0 {
            self.cs_create(CsKey::CsrData, 0)?;
            self.cs_create(CsKey::CsrIndices, 0)?;
            self.cs_create(CsKey::CsrIndptr, nrow + 1)?;
            let zeros = vec![0u64; nrow + 1];
            self.cs_write_u64(CsKey::CsrIndptr, 0, &zeros)?;
            self.read_row_indptr()?;
            return Ok(());
        }

        info!("transpose pass 1: counting row nnz");
        let mut row_counts = vec![0u64; nrow];
        const COL_BLOCK: usize = 1024;
        let mut col_lo = 0usize;
        while col_lo < ncol {
            let col_hi = (col_lo + COL_BLOCK).min(ncol);
            let cols: Self::IndexIter = (col_lo..col_hi).collect();
            let (_, _, triplets) = self.read_triplets_by_columns(cols)?;
            for (row_i, _, _) in &triplets {
                row_counts[*row_i as usize] += 1;
            }
            col_lo = col_hi;
        }

        let mut rowptr = vec![0u64; nrow + 1];
        let mut acc = 0u64;
        for i in 0..nrow {
            rowptr[i] = acc;
            acc += row_counts[i];
        }
        rowptr[nrow] = acc;
        debug_assert_eq!(acc, nnz as u64);

        self.cs_create(CsKey::CsrData, nnz)?;
        self.cs_create(CsKey::CsrIndices, nnz)?;
        self.cs_create(CsKey::CsrIndptr, nrow + 1)?;
        self.cs_write_u64(CsKey::CsrIndptr, 0, &rowptr)?;

        // Per-band buffers carry 12 B/nnz (u64 col + f32 val); cap aggregate
        // at ~256 MB so the row-banded scatter stays within a fixed budget
        // regardless of nnz.
        const TRANSPOSE_BAND_BYTES: usize = 256 * 1024 * 1024;
        let avg_density = nnz.div_ceil(nrow.max(1));
        let band_rows = (TRANSPOSE_BAND_BYTES / (12 * avg_density.max(1)))
            .max(1)
            .min(nrow);

        info!(
            "transpose pass 2: scatter (band of {} rows, {} bands)",
            band_rows,
            nrow.div_ceil(band_rows)
        );

        let mut band_lo = 0usize;
        while band_lo < nrow {
            let band_hi = (band_lo + band_rows).min(nrow);
            let band_nnz_start = rowptr[band_lo];
            let band_nnz_end = rowptr[band_hi];
            let band_nnz = (band_nnz_end - band_nnz_start) as usize;

            if band_nnz == 0 {
                band_lo = band_hi;
                continue;
            }

            let mut out_indices = vec![0u64; band_nnz];
            let mut out_values = vec![0f32; band_nnz];
            let mut cursor = vec![0u64; band_hi - band_lo];

            let mut col_lo = 0usize;
            while col_lo < ncol {
                let col_hi = (col_lo + COL_BLOCK).min(ncol);
                let cols: Self::IndexIter = (col_lo..col_hi).collect();
                let (_, _, triplets) = self.read_triplets_by_columns(cols)?;
                for &(row_i, col_j_local, x) in &triplets {
                    let row_i_us = row_i as usize;
                    if row_i_us >= band_lo && row_i_us < band_hi {
                        let band_idx = row_i_us - band_lo;
                        // col_j_local is already the global column index because
                        // read_triplets_by_columns returns columns in the passed order
                        // (0..batch for standalone call). We passed col_lo..col_hi,
                        // which returns local indices 0..(col_hi - col_lo) — so add col_lo.
                        let col_j_global = col_j_local + col_lo as u64;
                        let offset_in_band =
                            (rowptr[band_lo + band_idx] - band_nnz_start) + cursor[band_idx];
                        out_indices[offset_in_band as usize] = col_j_global;
                        out_values[offset_in_band as usize] = x;
                        cursor[band_idx] += 1;
                    }
                }
                col_lo = col_hi;
            }

            self.cs_write_u64(CsKey::CsrIndices, band_nnz_start, &out_indices)?;
            self.cs_write_f32(CsKey::CsrData, band_nnz_start, &out_values)?;

            band_lo = band_hi;
        }

        self.read_row_indptr()?;
        Ok(())
    }

    /// preload row index pointers
    fn read_row_indptr(&mut self) -> anyhow::Result<()>;

    /// preload column index pointers
    fn read_column_indptr(&mut self) -> anyhow::Result<()>;

    /// preload all the columns for faster processing
    fn preload_columns(&mut self) -> anyhow::Result<()>;

    /// unload the memory
    fn clean_preloaded_columns(&mut self);

    /// preload all the rows for faster processing
    fn preload_rows(&mut self) -> anyhow::Result<()>;

    /// unload the row memory
    fn clean_preloaded_rows(&mut self);

    /// backend file name
    fn get_backend_file_name(&self) -> &str;

    /// backend file type
    fn backend_type(&self) -> SparseIoBackend;
}
