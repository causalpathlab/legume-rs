#![allow(dead_code)]

use asap_data::sparse_io::*;

pub use std::sync::Arc;

pub const DEFAULT_BLOCK_SIZE: usize = 100;
pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;

pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub fn create_jobs(ntot: usize, block_size: usize) -> Vec<(usize, usize)> {
    let nblock = (ntot + block_size - 1) / block_size;

    (0..nblock)
        .map(|block| {
            let lb: usize = block * block_size;
            let ub: usize = ((block + 1) * block_size).min(ntot);
            (lb, ub)
        })
        .collect::<Vec<_>>()
}
