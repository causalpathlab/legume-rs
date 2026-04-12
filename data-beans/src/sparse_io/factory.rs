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

/// Create a sparse backend from a borrowed triplet slice.
///
/// Clones the slice internally — prefer [`create_sparse_from_triplets_owned`]
/// when you already own the Vec, since that avoids the extra copy.
pub fn create_sparse_from_triplets(
    triplets: &[(u64, u64, f32)],
    mtx_shape: (usize, usize, usize),
    backend_file: Option<&str>,
    backend: Option<&SparseIoBackend>,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    create_sparse_from_triplets_owned(triplets.to_vec(), mtx_shape, backend_file, backend)
}

/// Create a sparse backend from an owned triplet Vec.
///
/// Moves the Vec into the backend sort/write without copying. Use this on
/// hot paths (e.g. `from-*` converters, merges) to avoid holding two full
/// copies of the triplet list simultaneously.
pub fn create_sparse_from_triplets_owned(
    mut triplets: Vec<(u64, u64, f32)>,
    mtx_shape: (usize, usize, usize),
    backend_file: Option<&str>,
    backend: Option<&SparseIoBackend>,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
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

/// Create an empty sparse backend ready for streaming CSC writes.
///
/// The returned backend has no data yet — the caller is expected to
/// drive [`SparseIo::begin_streaming_csc`], one or more
/// [`SparseIo::append_csc_slab`] calls, [`SparseIo::finalize_streaming_csc`],
/// and finally [`SparseIo::build_csr_from_csc_streaming`].
pub fn create_sparse_streaming_empty(
    backend_file: Option<&str>,
    backend: Option<&SparseIoBackend>,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    match backend {
        Some(SparseIoBackend::HDF5) => Ok(Box::new(sparse_matrix_hdf5::SparseMtxData::new(
            backend_file,
        )?)),
        Some(SparseIoBackend::Zarr) | None => Ok(Box::new(sparse_matrix_zarr::SparseMtxData::new(
            backend_file,
        )?)),
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
