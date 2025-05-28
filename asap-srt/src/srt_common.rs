#![allow(dead_code)]

use asap_data::sparse_io::*;

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;
pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub use std::sync::Arc;

pub fn create_jobs(ntot: usize, block_size: usize) -> Vec<(usize, usize)> {
    let nblock = ntot.div_ceil(block_size);

    (0..nblock)
        .map(|block| {
            let lb: usize = block * block_size;
            let ub: usize = ((block + 1) * block_size).min(ntot);
            (lb, ub)
        })
        .collect::<Vec<_>>()
}
