#![allow(dead_code)]

use asap_data::sparse_io::*;

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

pub fn parse_comma_separated<T>(s: &str) -> Result<Vec<T>, T::Err>
where
    T: std::str::FromStr + std::fmt::Display,
{
    s.split(',').map(|x| x.parse::<T>()).collect()
}

pub fn parse_encoder_layers(s: &str) -> Result<Vec<usize>, <usize as std::str::FromStr>::Err> {
    parse_comma_separated(s)
}

pub fn row_membership_matrix(row_membership: Vec<usize>) -> Mat {
    let mtot = match row_membership.iter().max() {
        Some(&m) => m + 1,
        _ => 1,
    };

    let mut ret_dm = Mat::zeros(row_membership.len(), mtot);

    for (i, k) in row_membership.into_iter().enumerate() {
        ret_dm[(i, k)] += 1_f32;
    }

    ret_dm
}

pub struct EmptyArg {}
