#![allow(dead_code)]

pub use asap_data::sparse_data_visitors::*;
pub use asap_data::sparse_io::*;
pub use asap_data::sparse_io_vector::SparseIoVec;

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;
pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub struct EmptyArg {}
