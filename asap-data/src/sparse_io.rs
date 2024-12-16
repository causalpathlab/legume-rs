use crate::sparse_matrix_hdf5;
use crate::sparse_matrix_zarr;
pub use clap::ValueEnum;
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
pub fn create_sparse_matrix(
    mtx_file: &str,
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    match backend {
        SparseIoBackend::HDF5 => Ok(Box::new(sparse_matrix_hdf5::SparseMtxData::from_mtx_file(
            mtx_file,
            Some(backend_file),
            Some(true),
        )?)),
        SparseIoBackend::Zarr => Ok(Box::new(sparse_matrix_zarr::SparseMtxData::from_mtx_file(
            mtx_file,
            Some(backend_file),
            Some(true),
        )?)),
    }
}

#[allow(dead_code)]
pub trait SparseIo: Sync + Send {
    type IndexIter: IntoIterator<Item = usize>;

    /// Read rows within the range and return a vector of triplets (row, column, value)
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_rows(
        &self,
        rows: Self::IndexIter,
    ) -> anyhow::Result<Vec<(usize, usize, f32)>>;

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_columns(
        &self,
        columns: Self::IndexIter,
    ) -> anyhow::Result<Vec<(usize, usize, f32)>>;

    /// Read columns within the range and return dense `ndarray::Array2`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns(&self, columns: Self::IndexIter) -> anyhow::Result<Array2<f32>>;

    /// Read rows within the range and return dense `ndarray::Array2`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows(&self, rows: Self::IndexIter) -> anyhow::Result<Array2<f32>>;

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
}

//////////////////////
// helper functions //
//////////////////////

// #[allow(dead_code)]
pub fn build_name2index_map(_names: &Vec<Box<str>>) -> HashMap<Box<str>, usize> {
    _names
        .iter()
        .enumerate()
        .map(|(r, name)| (name.clone(), r))
        .collect()
}

// #[allow(dead_code)]
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

// #[allow(dead_code)]
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
