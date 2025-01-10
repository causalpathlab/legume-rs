use crate::sparse_matrix_hdf5;
use crate::sparse_matrix_zarr;

pub use candle_core::Tensor;
pub use clap::ValueEnum;
pub use matrix_util::traits::*;
pub use nalgebra::DMatrix;
pub use nalgebra_sparse::csr::CsrMatrix;
pub use ndarray::prelude::*;
pub use rayon::prelude::*;
pub use std::collections::HashMap;
pub use std::ops::Range;

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
/// Create a sparse matrix io (backend) with 10x mtx
/// * `mtx_file`: file path to the 10x mtx
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn create_sparse_mtx_file(
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
pub fn create_sparse_ndarray(
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
pub fn create_sparse_dmatrix(
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
    type IndexIter: IntoIterator<Item = usize>;

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

    /// Export the data to a mtx file. This will take time.
    /// * `mtx_file`: mtx file to be written
    fn to_mtx_file(&self, mtx_file: &str) -> anyhow::Result<()>;

    /// Select the columns of the data and create a new backend file
    /// * `columns`: if something, columns to be subsetted
    /// * `rows`: if something, subset the rows
    fn subset_columns_rows(
        &mut self,
        columns: Option<&Vec<usize>>,
        rows: Option<&Vec<usize>>,
    ) -> anyhow::Result<()>;

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

    /// Reposition rows in a new order specified by `remap`
    /// * `row_names_order` - a vector of row names in the new order
    fn reorder_rows(&mut self, row_names_order: &Vec<Box<str>>) -> anyhow::Result<()>;

    /// Remove backend file
    fn remove_backend_file(&self) -> anyhow::Result<()>;

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
