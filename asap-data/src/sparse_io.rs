use clap::ValueEnum;
pub use ndarray::prelude::*;
pub use std::collections::HashMap;
pub use std::ops::Range;

use crate::sparse_matrix_hdf5;
use crate::sparse_matrix_zarr;

use std::sync::{Arc, Mutex};

#[derive(ValueEnum, Clone, Debug)]
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
) -> anyhow::Result<Arc<Mutex<dyn SparseIo<IndexIter = Vec<usize>>>>> {
    let data: Arc<Mutex<dyn SparseIo<IndexIter = Vec<usize>>>> = match backend {
        SparseIoBackend::HDF5 => Arc::new(Mutex::new(sparse_matrix_hdf5::SparseMtxData::open(
            backend_file,
        )?)),
        SparseIoBackend::Zarr => Arc::new(Mutex::new(sparse_matrix_zarr::SparseMtxData::open(
            backend_file,
        )?)),
    };
    Ok(data)
}

#[allow(dead_code)]
/// Create a sparse matrix io (backend) with 10x mtx
/// * `mtx_file`: file path to the 10x mtx
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn create_sparse_matrix(
    mtx_file: &str,
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<Arc<Mutex<dyn SparseIo<IndexIter = Vec<usize>>>>> {
    let data: Arc<Mutex<dyn SparseIo<IndexIter = Vec<usize>>>> = match backend {
        SparseIoBackend::HDF5 => Arc::new(Mutex::new(
            sparse_matrix_hdf5::SparseMtxData::from_mtx_file(
                mtx_file,
                Some(backend_file),
                Some(true),
            )?,
        )),
        SparseIoBackend::Zarr => Arc::new(Mutex::new(
            sparse_matrix_zarr::SparseMtxData::from_mtx_file(
                mtx_file,
                Some(backend_file),
                Some(true),
            )?,
        )),
    };
    Ok(data)
}

#[allow(dead_code)]
pub trait SparseIo {
    type IndexIter: IntoIterator<Item = usize>;

    /// Read columns within the range and return dense `ndarray::Array2`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns(&self, columns: Self::IndexIter) -> anyhow::Result<Array2<f32>> {
        Ok(Array2::zeros((
            self.num_rows().expect("need to know the number of rows"),
            columns.into_iter().count(),
        )))
    }

    /// Read rows within the range and return dense `ndarray::Array2`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows(&self, rows: Self::IndexIter) -> anyhow::Result<Array2<f32>> {
        Ok(Array2::zeros((
            rows.into_iter().count(),
            self.num_columns()
                .expect("need to know the number of columns"),
        )))
    }

    /// Number of rows in the underlying data matrix
    fn num_rows(&self) -> Option<usize>;

    /// Number of columns in the underlying data matrix
    fn num_columns(&self) -> Option<usize>;

    /// Number of non-zero elements
    fn num_non_zeros(&self) -> Option<usize>;

    /// Set row names for the matrix
    /// * `row_name_file`: a file each line contains row name words
    fn register_row_names(&mut self, row_name_file: &str);

    /// Set column names for the matrix
    /// * `column_name_file`: a file each line contains column name words
    fn register_column_names(&mut self, column_name_file: &str);

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `name_file`: a file each line contains name words
    /// * `name_columns`: range of columns to be used for name
    /// * `name_sep`: separator for name columns
    fn register_names(
        &mut self,
        key: &str,
        name_file: &str,
        name_columns: Range<usize>,
        name_sep: &str,
    ) -> anyhow::Result<()>;

    fn row_names(&self) -> anyhow::Result<Vec<Box<str>>>;

    fn column_names(&self) -> anyhow::Result<Vec<Box<str>>>;

    /// Get back the registered names
    /// * `key`: key for the registered names
    fn retrieve_registered_names(&self, key: &str) -> anyhow::Result<Vec<Box<str>>>;

    /// Reposition rows in a new order specified by `remap`
    /// * `remap` - a hashmap of old row index to new row index
    fn remap_rows(&mut self, remap: HashMap<usize, usize>) -> anyhow::Result<()>;
}
