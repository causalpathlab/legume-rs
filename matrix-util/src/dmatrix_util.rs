use crate::traits::*;
pub use nalgebra::{DMatrix, DVector};
pub use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_traits::Zero;
pub use rand::{thread_rng, Rng};
pub use rand_distr::{StandardNormal, Uniform};
pub use rayon::prelude::*;

#[allow(dead_code)]
/// Sample d,n matrix from U(0,1)
pub fn runif(dd: usize, nn: usize) -> DMatrix<f32> {
    let runif = Uniform::new(0_f32, 1_f32);

    let rvec = (0..(dd * nn))
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.sample(runif))
        .collect();

    DMatrix::<f32>::from_vec(dd, nn, rvec)
}

#[allow(dead_code)]
/// Sample d,n matrix from N(0,1)
pub fn rnorm(dd: usize, nn: usize) -> DMatrix<f32> {
    let rvec = (0..(dd * nn))
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.sample(StandardNormal))
        .collect();

    DMatrix::<f32>::from_vec(dd, nn, rvec)
}

#[allow(dead_code)]
/// Normalize d x m matrix X by columns
/// Y[,j] = X[,j] / max(1, norm(X[,j]))
pub fn normalize_columns(xx: &DMatrix<f32>) -> DMatrix<f32> {
    let mut ret = xx.clone();
    for mut xx_j in ret.column_iter_mut() {
        let denom = xx_j.norm().max(1.0);
        xx_j /= denom;
    }
    ret
}

#[allow(dead_code)]
/// Normalize d x m matrix X by columns
/// Y[,j] = X[,j] / max(1, norm(X[,j]))
pub fn normalize_columns_inplace(xx: &mut DMatrix<f32>) {
    for mut xx_j in xx.column_iter_mut() {
        let denom = xx_j.norm().max(1.0);
        xx_j /= denom;
    }
}

///////////////////////////
// trait implementations //
///////////////////////////

impl<T> FromTriplets for DMatrix<T>
where
    T: nalgebra::Scalar + Zero,
{
    type Mat = Self;
    type Scalar = T;

    fn from_nonzero_triplets(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(usize, usize, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat> {
        let mut data = vec![T::zero(); ncol * nrow];
        for (ii, jj, x_ij) in triplets {
            data[ii * ncol + jj] = x_ij;
        }
        Ok(DMatrix::from_row_slice(nrow, ncol, &data))
    }
}

impl<T> FromTriplets for CsrMatrix<T>
where
    T: nalgebra::Scalar + Zero + std::ops::AddAssign,
{
    type Mat = Self;
    type Scalar = T;

    fn from_nonzero_triplets(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(usize, usize, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat> {
        let mut coo = CooMatrix::<T>::new(nrow, ncol);
        for (ii, jj, x_ij) in triplets {
            coo.push(ii, jj, x_ij);
        }
        Ok(CsrMatrix::from(&coo))
    }
}
