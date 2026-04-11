#![allow(dead_code)]

use std::sync::Arc;

use crate::sparse_backend::hdf5 as sparse_matrix_hdf5;
use crate::sparse_backend::zarr as sparse_matrix_zarr;

use super::{Array2, DMatrix, SparseIo, SparseIoBackend};

/// Open a sparse matrix io (backend)
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn open_sparse_matrix(
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    match backend {
        SparseIoBackend::Zarr => Ok(Box::new(sparse_matrix_zarr::SparseMtxData::open(
            backend_file,
        )?)),
        SparseIoBackend::HDF5 => Ok(Box::new(sparse_matrix_hdf5::SparseMtxData::open(
            backend_file,
        )?)),
    }
}

/// Open a sparse matrix io (backend)
/// * `backend_file`: file path to the sparse matrix
/// * `backend`: backend type (HDF5 or Zarr)
pub fn create_sparse_from_triplets(
    triplets: &[(u64, u64, f32)],
    mtx_shape: (usize, usize, usize),
    backend_file: Option<&str>,
    backend: Option<&SparseIoBackend>,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    let mut triplets: Vec<(u64, u64, f32)> = triplets.to_vec();

    match backend {
        Some(SparseIoBackend::HDF5) => {
            let mut ret = Box::new(sparse_matrix_hdf5::SparseMtxData::new(backend_file)?);

            ret.record_mtx_shape(Some(mtx_shape))?;
            ret.record_triplets_by_col(&mut triplets)?;
            ret.record_triplets_by_row(&mut triplets)?;
            ret.read_column_indptr()?;
            ret.read_row_indptr()?;
            Ok(ret)
        }

        Some(SparseIoBackend::Zarr) | None => {
            let mut ret = Box::new(sparse_matrix_zarr::SparseMtxData::new(backend_file)?);
            ret.record_mtx_shape(Some(mtx_shape))?;
            ret.record_triplets_by_col(&mut triplets)?;
            ret.record_triplets_by_row(&mut triplets)?;
            ret.read_column_indptr()?;
            ret.read_row_indptr()?;
            Ok(ret)
        }
    }
}

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

        Some(SparseIoBackend::Zarr) | None => Ok(Box::new(
            sparse_matrix_zarr::SparseMtxData::from_mtx_file(mtx_file, backend_file, Some(true))?,
        )),
    }
}

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

        Some(SparseIoBackend::Zarr) | None => Ok(Box::new(
            sparse_matrix_zarr::SparseMtxData::from_ndarray(data, backend_file, Some(true))?,
        )),
    }
}

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

        Some(SparseIoBackend::Zarr) | None => Ok(Box::new(
            sparse_matrix_zarr::SparseMtxData::from_dmatrix(data, backend_file, Some(true))?,
        )),
    }
}

pub fn sparse_io_box_to_arc<T>(
    boxed: Box<dyn SparseIo<IndexIter = T>>,
) -> Arc<dyn SparseIo<IndexIter = T>> {
    Arc::from(boxed)
}
