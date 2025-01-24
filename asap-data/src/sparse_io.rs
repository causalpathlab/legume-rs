use crate::sparse_matrix_hdf5;
use crate::sparse_matrix_zarr;

pub use candle_core::Tensor;
pub use nalgebra::DMatrix;
pub use nalgebra_sparse::{csc::CscMatrix, csr::CsrMatrix};
pub use ndarray::prelude::*;

use clap::ValueEnum;
use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::mtx_io::*;
use matrix_util::traits::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum SparseIoBackend {
    Zarr,
    HDF5,
}

#[allow(dead_code)]
/// Open a sparse matrix io (backend)
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn open_sparse_matrix(
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    match backend {
        SparseIoBackend::Zarr => Ok(Box::new(sparse_matrix_zarr::SparseMtxData::open(
            &backend_file,
        )?)),
        SparseIoBackend::HDF5 => Ok(Box::new(sparse_matrix_hdf5::SparseMtxData::open(
            &backend_file,
        )?)),
    }
}

#[allow(dead_code)]
/// Open a sparse matrix io (backend)
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn create_sparse_from_triplets(
    triplets: Vec<(u64, u64, f32)>,
    mtx_shape: (usize, usize, usize),
    backend_file: Option<&str>,
    backend: Option<&SparseIoBackend>,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    let mut triplets: Vec<(u64, u64, f32)> = triplets.clone();

    match backend {
        Some(SparseIoBackend::Zarr) => {
            let mut ret = Box::new(sparse_matrix_zarr::SparseMtxData::new(backend_file)?);
            ret.record_mtx_shape(Some(mtx_shape))?;
            ret.record_triplets_by_col(&mut triplets)?;
            ret.record_triplets_by_row(&mut triplets)?;
            Ok(ret)
        }
        Some(SparseIoBackend::HDF5) => {
            let mut ret = Box::new(sparse_matrix_hdf5::SparseMtxData::new(backend_file)?);

            ret.record_mtx_shape(Some(mtx_shape))?;
            ret.record_triplets_by_col(&mut triplets)?;
            ret.record_triplets_by_row(&mut triplets)?;
            Ok(ret)
        }
        _ => return Err(anyhow::anyhow!("backend not supported")),
    }
}

#[allow(dead_code)]
/// Create a sparse matrix io (backend) with 10x mtx
/// * `mtx_file`: file path to the 10x mtx
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn create_sparse_from_mtx_file(
    mtx_file: &str,
    backend_file: Option<&str>,
    backend: Option<&SparseIoBackend>,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    match backend {
        Some(SparseIoBackend::HDF5) => Ok(Box::new(
            sparse_matrix_hdf5::SparseMtxData::from_mtx_file(mtx_file, backend_file, Some(true))?,
        )),
        _ => Ok(Box::new(sparse_matrix_zarr::SparseMtxData::from_mtx_file(
            mtx_file,
            backend_file,
            Some(true),
        )?)),
    }
}

#[allow(dead_code)]
/// Create a sparse matrix io (backend) with dense `Array2`
/// * `data`: data matrix
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn create_sparse_from_ndarray(
    data: &Array2<f32>,
    backend_file: Option<&str>,
    backend: Option<&SparseIoBackend>,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    match backend {
        Some(SparseIoBackend::HDF5) => Ok(Box::new(
            sparse_matrix_hdf5::SparseMtxData::from_ndarray(data, backend_file, Some(true))?,
        )),
        _ => Ok(Box::new(sparse_matrix_zarr::SparseMtxData::from_ndarray(
            data,
            backend_file,
            Some(true),
        )?)),
    }
}

#[allow(dead_code)]
/// Create a sparse matrix io (backend) with dense `DMatrix`
/// * `data`: data matrix
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn create_sparse_from_dmatrix(
    data: &DMatrix<f32>,
    backend_file: Option<&str>,
    backend: Option<&SparseIoBackend>,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    match backend {
        Some(SparseIoBackend::HDF5) => Ok(Box::new(
            sparse_matrix_hdf5::SparseMtxData::from_dmatrix(data, backend_file, Some(true))?,
        )),
        _ => Ok(Box::new(sparse_matrix_zarr::SparseMtxData::from_dmatrix(
            data,
            backend_file,
            Some(true),
        )?)),
    }
}

#[allow(dead_code)]
pub trait SparseIo: Sync + Send {
    type IndexIter: IntoIterator<Item = usize> + FromIterator<usize>;

    ////////////////////////////
    // default implementation //
    ////////////////////////////

    /// Read columns within the range and return dense `ndarray::Array2`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_ndarray(self: &Self, columns: Self::IndexIter) -> anyhow::Result<Array2<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        Array2::<f32>::from_nonzero_triplets(nrow, ncol as usize, triplets)
    }

    /// Read columns within the range and return dense `candle_core::Tensor`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_tensor(self: &Self, columns: Self::IndexIter) -> anyhow::Result<Tensor> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        Tensor::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /// Read columns within the range and return dense `nalgebrea::DMatrix`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_dmatrix(self: &Self, columns: Self::IndexIter) -> anyhow::Result<DMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /// Read columns within the range and return sparse `CsrMatrix`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_csr(&self, columns: Self::IndexIter) -> anyhow::Result<CsrMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /// Read columns within the range and return sparse `CsrMatrix`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns_csc(&self, columns: Self::IndexIter) -> anyhow::Result<CscMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_columns(columns)?;
        CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /// Read rows within the range and return dense `ndarray::Array2`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_ndarray(&self, rows: Self::IndexIter) -> anyhow::Result<Array2<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        Array2::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /// Read rows within the range and return dense `candle_core::Tensor`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_tensor(self: &Self, rows: Self::IndexIter) -> anyhow::Result<Tensor> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        Tensor::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /// Read rows within the range and return dense `nalgebra::DMatrix`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_dmatrix(&self, rows: Self::IndexIter) -> anyhow::Result<DMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /// Read rows within the range and return sparse `CsrMatrix`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_csr(&self, rows: Self::IndexIter) -> anyhow::Result<CsrMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /// Read rows within the range and return sparse `CscMatrix`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows_csc(&self, rows: Self::IndexIter) -> anyhow::Result<CscMatrix<f32>> {
        let (nrow, ncol, triplets) = self.read_triplets_by_rows(rows)?;
        CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    /////////////////////////////
    // `mtx` related functions //
    /////////////////////////////

    /// Read mtx file and populate the data into HDF5 for faster row-by-row access
    /// * `mtx_file`: mtx file to be read into HDF5 backend
    fn import_mtx_file_by_row(self: &mut Self, mtx_file: &str) -> anyhow::Result<()> {
        let (mut mtx_triplets, mtx_shape) = read_mtx_triplets(mtx_file)?;
        info!("read mtx file: {}", mtx_file);
        if mtx_triplets.len() == 0 {
            return Err(anyhow::anyhow!("No data in mtx file"));
        }
        self.record_mtx_shape(mtx_shape)?;
        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Read mtx file and populate the data into HDF5 for faster column-by-column access
    /// * `mtx_file`: mtx file to be read into HDF5 backend
    fn import_mtx_file_by_col(self: &mut Self, mtx_file: &str) -> anyhow::Result<()> {
        let (mut mtx_triplets, mtx_shape) = read_mtx_triplets(mtx_file)?;
        info!("read mtx file: {}", mtx_file);
        if mtx_triplets.len() == 0 {
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
        let mut mtx_triplets = dmatrix_to_triplets(&matrix);
        let mtx_shape = (nrow, ncol, mtx_triplets.len());
        self.record_mtx_shape(Some(mtx_shape))?;
        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Add dmatrix to zarr backend by column (CSC format)
    /// * `array` - 2D array to be added to the backend
    fn import_dmatrix_by_col(&mut self, matrix: &DMatrix<f32>) -> anyhow::Result<()> {
        let (nrow, ncol) = matrix.shape();
        let mut mtx_triplets = dmatrix_to_triplets(&matrix);
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
        let mut mtx_triplets = ndarray_to_triplets(&array);

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
        let mut mtx_triplets = ndarray_to_triplets(&array);

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
    fn read_triplets_by_rows(
        &self,
        rows: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, f32)>)>;

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_columns(
        &self,
        columns: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, f32)>)>;

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `col` : usize
    ///
    fn read_triplets_by_single_column(
        &self,
        col: usize,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, f32)>)>;

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
    fn register_row_names_vec(&mut self, rows: &Vec<Box<str>>);

    /// Set column names for the matrix
    /// * `columns`: a vector of column names
    fn register_column_names_vec(&mut self, columns: &Vec<Box<str>>);

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
    fn register_names_vec(&mut self, key: &str, names: &Vec<Box<str>>) -> anyhow::Result<()>;

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
            let (nrow_data, ncol_data, nnz) = (nrow as usize, ncol as usize, nnz as usize);

            //////////////////////////////////////////////////////
            // 0. Create a mapping from old to new columns/rows //
            //////////////////////////////////////////////////////

            let (old2new_cols, new_col_names) =
                take_subset_indices_names_if_needed(columns, Some(ncol_data), self.column_names()?);

            let (old2new_rows, new_row_names) =
                take_subset_indices_names_if_needed(rows, Some(nrow_data), self.row_names()?);

            let (new_ncol, new_nrow) = (old2new_cols.len(), old2new_rows.len());

            /////////////////////////////////////////////////////////
            // 1. Create remapped Mtx only taking a subset of rows //
            /////////////////////////////////////////////////////////

            let arc_triplets = Arc::new(Mutex::new(vec![]));
            let block_size = 100;
            let nblock = (ncol_data + block_size - 1) / block_size;

            info!("remapping triplets ...");

            (0..nblock)
                .into_par_iter()
                .progress_count(nblock as u64)
                .map(|b| {
                    let lb: usize = b * block_size;
                    let ub: usize = ((b + 1) * block_size).min(ncol);
                    (lb, ub)
                })
                .for_each(|(lb, ub)| {
                    let mut record = vec![];

                    (lb..ub)
                        .into_iter()
                        .filter_map(|j| {
                            if let Some(&j_new) = old2new_cols.get(&j) {
                                Some((j, j_new))
                            } else {
                                None
                            }
                        })
                        .for_each(|(j, j_new)| {
                            let (_, _, _triplets_j) = self
                                .read_triplets_by_single_column(j)
                                .expect("failed to read a single column");

                            record.extend(_triplets_j.into_iter().filter_map(|(i, _, x)| {
                                old2new_rows
                                    .get(&i)
                                    .map(|&i_new| (i_new as u64, j_new as u64, x))
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

        Ok(())
    }

    /// Reposition rows in a new order specified by `remap`
    /// * `row_names_order` - a vector of row names in the new order
    fn reorder_rows(&mut self, row_names_order: &Vec<Box<str>>) -> anyhow::Result<()> {
        let new_row_names = row_names_order.clone();
        let new_col_names = self.column_names()?.clone();
        let name2new = build_name2index_map(&new_row_names);

        let block_size = 100;

        let old2new: HashMap<usize, usize> = self
            .row_names()?
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx_old, name)| name2new.get(&name).map(|&idx_new| (idx_old, idx_new)))
            .collect();

        if let Some(ncol) = self.num_columns() {
            /////////////////////////////////////////////////////
            // 1. triplets after filtering and reordering rows //
            /////////////////////////////////////////////////////

            let arc_triplets = Arc::new(Mutex::new(vec![]));

            let nblock = (ncol + block_size - 1) / block_size;

            info!("remapping triplets ...");

            (0..nblock)
                .into_par_iter()
                .progress_count(nblock as u64)
                .map(|b| {
                    let lb: usize = b * block_size;
                    let ub: usize = ((b + 1) * block_size).min(ncol);
                    (lb, ub)
                })
                .for_each(|(lb, ub)| {
                    let (_, _, _triplets_b) =
                        self.read_triplets_by_columns((lb..ub).collect()).unwrap();
                    let _triplets_b = _triplets_b.into_iter().filter_map(|(i, j, x)| {
                        old2new.get(&i).map(|&i_new| (i_new as u64, j as u64, x))
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
                let new_nrow = new_row_names.len();
                let mtx_shape = (new_nrow, ncol, nnz);

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
        }

        Ok(())
    }
    // fn reorder_rows(&mut self, row_names_order: &Vec<Box<str>>) -> anyhow::Result<()>;

    /// Remove backend file
    fn remove_backend_file(&self) -> anyhow::Result<()>;

    /// Initialize backend
    fn initialize_backend(&mut self) -> anyhow::Result<()>;

    fn record_mtx_shape(
        self: &mut Self,
        mtx_shape: Option<(usize, usize, usize)>,
    ) -> anyhow::Result<()>;

    /// Helper function to add triplets to zarr backend by row (CSR format)
    fn record_triplets_by_row(
        &mut self,
        row_col_val_triplets: &mut Vec<(u64, u64, f32)>,
    ) -> anyhow::Result<()> {
        debug_assert!(row_col_val_triplets.len() > 0);

        row_col_val_triplets.sort_by_key(|&(_, col, _)| col);
        row_col_val_triplets.sort_by_key(|&(row, _, _)| row);

        let mut csr_rowptr = vec![];
        let mut csr_cols = vec![];
        let mut csr_vals = vec![];

        let nrow = self.num_rows().expect("should have `nrow`");
        let nnz = row_col_val_triplets.len();

        // fill in rowptr 0 to the first row index
        let first = row_col_val_triplets[0].0;
        for _ in 0..first {
            csr_rowptr.push(0);
        }

        // for the first row/triplet
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

        // fill in the rest of the rowptr
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
        debug_assert!(row_col_val_triplets.len() > 0);

        row_col_val_triplets.sort_by_key(|&(row, _, _)| row);
        row_col_val_triplets.sort_by_key(|&(_, col, _)| col);
        // dbg!(&row_col_val_triplets);

        let mut csc_colptr: Vec<u64> = vec![];
        let mut csc_rows: Vec<u64> = vec![];
        let mut csc_vals: Vec<f32> = vec![];

        let ncol = self.num_columns().expect("should have `ncol`");
        let nnz = row_col_val_triplets.len();

        // fill in colptr 0 to the first column index
        let first = row_col_val_triplets[0].1;
        for _ in 0..first {
            csc_colptr.push(0);
        }

        // for the first column/triplet
        csc_colptr.push(0);
        csc_rows.push(row_col_val_triplets[0].0);
        csc_vals.push(row_col_val_triplets[0].2);

        for i in 1..nnz {
            let lb = row_col_val_triplets[i - 1].1;
            let ub = row_col_val_triplets[i].1;
            for _ in lb..ub {
                csc_colptr.push(i as u64);
            }
            csc_rows.push(row_col_val_triplets[i].0 as u64);
            csc_vals.push(row_col_val_triplets[i].2);
        }

        // fill in the rest of the colptr
        let last = row_col_val_triplets[nnz - 1].1 as usize;
        for _ in last..ncol {
            csc_colptr.push(nnz as u64);
        }
        // dbg!(&csc_colptr);

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
        self: &mut Self,
        csr_cols: &Vec<u64>,
        csr_vals: &Vec<f32>,
        csr_rowptr: &Vec<u64>,
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
        self: &mut Self,
        csc_rows: &Vec<u64>,
        csc_vals: &Vec<f32>,
        csc_colptr: &Vec<u64>,
    ) -> anyhow::Result<()>;

    fn read_row_indptr(self: &mut Self) -> anyhow::Result<()>;

    fn read_column_indptr(self: &mut Self) -> anyhow::Result<()>;

    /// Backend file name
    fn get_backend_file_name(&self) -> &str;
}

//////////////////////
// helper functions //
//////////////////////

// fn iter_len<I>(iter: &I) -> usize
// where
//     I: IntoIterator,
// {
//     for j in &iter {
//     }
//     // iter.clone().into_iter().count()
// }

#[allow(dead_code)]
pub fn build_name2index_map(_names: &Vec<Box<str>>) -> HashMap<Box<str>, usize> {
    _names
        .iter()
        .enumerate()
        .map(|(r, name)| (name.clone(), r))
        .collect()
}

#[allow(dead_code)]
pub fn take_subset_indices_names(
    new_indices: &Vec<usize>,
    ntot: usize,
    old_names: Vec<Box<str>>,
) -> (HashMap<usize, usize>, Vec<Box<str>>) {
    let mut old2new: HashMap<usize, usize> = HashMap::new();
    let mut new2old = vec![];
    debug_assert!(ntot == old_names.len());
    let mut k = 0_usize;
    for idx in new_indices.iter() {
        if *idx < ntot {
            old2new.insert(*idx, k);
            new2old.push(*idx);
            k += 1;
        }
    }

    let new_names = new2old
        .iter()
        .map(|&i| old_names[i].clone())
        .collect::<Vec<Box<str>>>();

    (old2new, new_names)
}

#[allow(dead_code)]
pub fn take_subset_indices_names_if_needed(
    new_indices: Option<&Vec<usize>>,
    ntot: Option<usize>,
    old_names: Vec<Box<str>>,
) -> (HashMap<usize, usize>, Vec<Box<str>>) {
    let ntot = ntot.unwrap_or(old_names.len());
    if let Some(new_indices) = new_indices {
        take_subset_indices_names(new_indices, ntot, old_names)
    } else {
        let names = old_names;
        let identity = (0..ntot).zip(0..ntot).collect::<HashMap<usize, usize>>();
        (identity, names)
    }
}

#[allow(dead_code)]
pub fn ndarray_to_triplets(array: &Array2<f32>) -> Vec<(u64, u64, f32)> {
    let eps = 1e-6;
    array
        .indexed_iter()
        .filter(|(_, &elem)| elem.abs() > eps)
        .map(|((row, col), &value)| (row as u64, col as u64, value))
        .collect::<Vec<(u64, u64, f32)>>()
}

#[allow(dead_code)]
pub fn dmatrix_to_triplets(matrix: &DMatrix<f32>) -> Vec<(u64, u64, f32)> {
    let (nrow, _) = matrix.shape();
    let eps = 1e-6;
    matrix
        .iter() // column-major
        .enumerate()
        .filter(|(_, &elem)| elem.abs() > eps)
        .map(|(idx, &value)| {
            let row = idx % nrow;
            let col = idx / nrow;
            (row as u64, col as u64, value)
        })
        .collect::<Vec<(u64, u64, f32)>>()
}
