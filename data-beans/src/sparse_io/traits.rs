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
        if let (Some(ncol), Some(nrow), Some(nnz)) =
            (self.num_columns(), self.num_rows(), self.num_non_zeros())
        {
            let (nrow_data, ncol_data, nnz) = (nrow, ncol, nnz);

            //////////////////////////////////////////////////////
            // 0. Create a mapping from old to new columns/rows //
            //////////////////////////////////////////////////////

            let (old2new_cols, new_col_names) =
                take_subset_indices_names_if_needed(columns, Some(ncol_data), self.column_names()?);

            let (old2new_rows, new_row_names) =
                take_subset_indices_names_if_needed(rows, Some(nrow_data), self.row_names()?);

            let (new_ncol, new_nrow) = (new_col_names.len(), new_row_names.len());

            /////////////////////////////////////////////////////////
            // 1. Create remapped Mtx only taking a subset of rows //
            /////////////////////////////////////////////////////////

            let arc_triplets = Arc::new(Mutex::new(vec![]));
            let block_size = 100;
            let nblock = ncol_data.div_ceil(block_size);

            info!("remapping triplets ...");

            (0..nblock)
                .into_par_iter()
                .progress_count(nblock as u64)
                .map(|b| {
                    let lb: u64 = (b * block_size) as u64;
                    let ub: u64 = ((b + 1) * block_size).min(ncol) as u64;
                    (lb, ub)
                })
                .for_each(|(lb, ub)| {
                    let mut record = vec![];

                    (lb..ub)
                        .filter_map(|j| {
                            if let Some(&j_new) = old2new_cols.get(&j) {
                                Some((j, j_new))
                            } else {
                                None
                            }
                        })
                        .for_each(|(j, j_new)| {
                            let (_, _, _triplets_j) = self
                                .read_triplets_by_single_column(j as usize)
                                .expect("failed to read a single column");

                            record.extend(_triplets_j.into_iter().filter_map(|(i, _, x)| {
                                old2new_rows.get(&i).map(|&i_new| (i_new, j_new, x))
                            }));
                        });
                    {
                        arc_triplets
                            .lock()
                            .expect("failed to lock triplets")
                            .extend(record);
                    }
                });

            /////////////////////////////////////
            // 2. Remove previous backend file //
            /////////////////////////////////////
            self.remove_backend_file()?;

            info!(
                "removing and recreating the backend {}",
                self.get_backend_file_name()
            );

            ///////////////////////////////
            // 3. populate a new backend //
            ///////////////////////////////
            self.initialize_backend()?;

            // populate data from mtx triplets
            {
                let mut row_col_val_triplets =
                    arc_triplets.lock().expect("failed to lock triplets");

                debug_assert!(row_col_val_triplets.len() <= nnz); // subset
                let nnz = row_col_val_triplets.len();
                let mtx_shape = (new_nrow, new_ncol, nnz);

                info!("sorting triplets ...");

                self.record_mtx_shape(Some(mtx_shape))?;
                self.record_triplets_by_col(&mut row_col_val_triplets)?;
                self.record_triplets_by_row(&mut row_col_val_triplets)?;
            }
            self.read_column_indptr()?;
            self.read_row_indptr()?;

            self.register_row_names_vec(&new_row_names);
            self.register_column_names_vec(&new_col_names);
            info!("registered new data to {}", self.get_backend_file_name());
        } else {
            return Err(anyhow::anyhow!("missing shape information"));
        }

        self.clean_preloaded_columns();
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
                .progress_count(nblock as u64)
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
        debug_assert_eq!(row_indices.len(), values.len());
        let shifted: Vec<u64> = local_colptr.iter().map(|&p| p + nnz_offset).collect();
        self.cs_write_u64(CsKey::CscIndptr, col_offset, &shifted)?;
        self.cs_write_u64(CsKey::CscIndices, nnz_offset, row_indices)?;
        self.cs_write_f32(CsKey::CscData, nnz_offset, values)?;
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

    /// backend file name
    fn get_backend_file_name(&self) -> &str;

    /// backend file type
    fn backend_type(&self) -> SparseIoBackend;
}
