pub use nalgebra::{DMatrix, DVector};
pub use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};

use num_traits::Float;
pub use rand::{thread_rng, Rng};
pub use rand_distr::{StandardNormal, Uniform};
pub use rayon::prelude::*;

use crate::traits::*;

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

///////////////////////////
// trait implementations //
///////////////////////////

impl<T> MatInplaceOps for CscMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    type Mat = Self;
    type Scalar = T;

    fn normalize_columns_inplace(&mut self) {
        let ncol = self.ncols();

        for j in 0..ncol {
            if let Some(x_j) = self.get_col(j) {
                let mut denom = T::zero();
                for &x_ij in x_j.values() {
                    denom += x_ij * x_ij;
                }
                denom = denom.sqrt().max(T::one());

                if let Some(mut x_j) = self.get_col_mut(j) {
                    for x_ij in x_j.values_mut() {
                        *x_ij /= denom;
                    }
                }
            }
        }
    }

    fn normalize_columns(&mut self) -> Self::Mat {
        let mut ret = self.clone();
        let ncol = ret.ncols();
        for j in 0..ncol {
            if let Some(x_j) = ret.get_col(j) {
                let mut denom = T::zero();
                for &x_ij in x_j.values() {
                    denom += x_ij * x_ij;
                }
                denom = denom.sqrt().max(T::one());

                if let Some(mut x_j) = ret.get_col_mut(j) {
                    for x_ij in x_j.values_mut() {
                        *x_ij /= denom;
                    }
                }
            }
        }
        ret
    }
}

impl<T> MatInplaceOps for DMatrix<T>
where
    T: nalgebra::RealField,
{
    type Mat = Self;
    type Scalar = T;

    fn normalize_columns_inplace(&mut self) {
        for mut xx_j in self.column_iter_mut() {
            let denom = xx_j.norm().max(T::one());
            xx_j /= denom;
        }
    }

    fn normalize_columns(&mut self) -> Self::Mat {
        let mut ret = self.clone();
        for mut xx_j in ret.column_iter_mut() {
            let denom = xx_j.norm().max(T::one());
            xx_j /= denom;
        }

        ret
    }
}

impl<T> MatTriplets for DMatrix<T>
where
    T: nalgebra::RealField + Float,
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

    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)> {
        if let Some(eps) = T::from(1e-6) {
            let nrow = self.nrows();
            let ncol = self.ncols();
            let mut triplets = vec![];
            for j in 0..ncol {
                for i in 0..nrow {
                    let x_ij = &self[(i, j)];
                    if x_ij.abs() > eps {
                        triplets.push((i, j, *x_ij));
                    }
                }
            }
            Ok((nrow, ncol, triplets))
        } else {
            anyhow::bail!("eps is not defined")
        }
    }
}

impl<T> MatTriplets for CsrMatrix<T>
where
    T: nalgebra::RealField + Float,
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
    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)> {
        if let Some(eps) = T::from(1e-6) {
            let nrow = self.nrows();
            let ncol = self.ncols();

            let mut triplets = Vec::new();

            for i in 0..nrow {
                if let Some(x_i) = self.get_row(i) {
                    let cols = x_i.col_indices();
                    let vals = x_i.values();
                    for k in 0..cols.len() {
                        let j = cols[k];
                        let x_ij = vals[k];
                        if x_ij > eps {
                            triplets.push((i, j, x_ij))
                        };
                    }
                }
            }

            Ok((nrow, ncol, triplets))
        } else {
            anyhow::bail!("eps is not defined")
        }
    }
}

impl<T> MatTriplets for CscMatrix<T>
where
    T: nalgebra::RealField + Float,
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
        Ok(CscMatrix::from(&coo))
    }
    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)> {
        if let Some(eps) = T::from(1e-6) {
            let nrow = self.nrows();
            let ncol = self.ncols();

            let mut triplets = Vec::new();
            for j in 0..ncol {
                if let Some(x_j) = self.get_col(j) {
                    let rows = x_j.row_indices();
                    let vals = x_j.values();
                    for k in 0..rows.len() {
                        let i = rows[k];
                        let x_ij = vals[k];
                        if x_ij > eps {
                            triplets.push((i, j, x_ij))
                        }
                    }
                }
            }

            Ok((nrow, ncol, triplets))
        } else {
            anyhow::bail!("eps is not defined")
        }
    }
}
