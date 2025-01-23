use asap_data::sparse_io::*;
use nalgebra::{DMatrix, DVector};

pub const DEFAULT_BLOCK_SIZE: usize = 100;
pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

#[allow(dead_code)]
pub type Mat = DMatrix<f32>;
#[allow(dead_code)]
pub type DVec = DVector<f32>;

#[allow(dead_code)]
pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

#[allow(dead_code)]
pub fn create_jobs(block_size: usize, ntot: usize) -> Vec<(usize, usize)> {
    let nblock = (ntot + block_size - 1) / block_size;

    (0..nblock)
        .map(|block| {
            let lb: usize = block * block_size;
            let ub: usize = ((block + 1) * block_size).min(ntot);
            (lb, ub)
        })
        .collect::<Vec<_>>()
}
