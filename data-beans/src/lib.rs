pub mod convert; // data format conversion (h5/zarr → backend)
pub mod hdf5_io; // HDF5/h5ad reading helpers
pub mod qc; // functions needed to perform quality control
pub mod simulations; // simulation helpers (core, deconv, multimodal)
pub mod sparse_backend; // storage backends (zarr, hdf5)
pub mod sparse_data_visitors; // visitor
pub mod sparse_io; // traits for sparse matrix
pub mod sparse_io_stack; // tall sparse io stack of vectors
pub mod sparse_io_vector; // wide sparse io vector
pub mod sparse_util; // sparse matrix triplets utils
pub mod utilities; // shared utility functions (name matching, IO helpers)
pub mod zarr_io; // read coordinate data from zarr files

// backward-compat re-exports
pub use sparse_backend::hdf5 as sparse_matrix_hdf5;
pub use sparse_backend::zarr as sparse_matrix_zarr;
