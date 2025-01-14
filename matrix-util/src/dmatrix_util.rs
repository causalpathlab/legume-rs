pub use nalgebra::{DMatrix, DVector};
pub use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};

use num_traits::{Float, Zero};
pub use rand::{thread_rng, Rng};
pub use rand_distr::{StandardNormal, Uniform};
pub use rayon::prelude::*;

use crate::traits::*;

use std::ops::AddAssign;

impl<T> CompositeOps for DMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    type Scalar = T;
    type Mat = Self;
    type Other = CscMatrix<T>;

    /// `self[:,col] += other[:,col]`
    /// * `col` - column index
    fn add_assign_column(&mut self, other: &Self::Other, j: usize) {
        debug_assert_eq!(self.nrows(), other.nrows());
        debug_assert_eq!(self.ncols(), other.ncols());
        if let Some(x_j) = other.get_col(j) {
            let vals = x_j.values();
            let rows = x_j.row_indices();
            for k in 0..vals.len() {
                let i = rows[k];
                let x_ij = vals[k];
                self[(i, j)] += x_ij;
            }
        }
    }

    /// `self += other`
    /// * `other` - `CscMatrix`
    fn add_assign(&mut self, other: &Self::Other) {
        debug_assert_eq!(self.nrows(), other.nrows());
        debug_assert_eq!(self.ncols(), other.ncols());
        for j in 0..other.ncols() {
            if let Some(x_j) = other.get_col(j) {
                let vals = x_j.values();
                let rows = x_j.row_indices();
                for k in 0..vals.len() {
                    let i = rows[k];
                    let x_ij = vals[k];
                    self[(i, j)] += x_ij;
                }
            }
        }
    }
}

// impl<T> AddAssignCscMatrix for DMatrix<T>
// where
//     T: Copy + AddAssign + Zero,
// {
//     fn add_assign(&mut self, rhs: &Self) {
//         // for temp in rhs.iter() {}
//         // for (i, j, x_ij) in rhs.iter() {
//         //     self[(i, j)] += *x_ij;
//         // }
//     }
// }

impl<T> SampleOps for DMatrix<T>
where
    T: nalgebra::RealField + Float,
{
    type Mat = Self;
    type Scalar = T;

    fn runif(nrow: usize, ncol: usize) -> Self::Mat {
        let u01 = Uniform::<f32>::new(0., 1.);

        let rvec = (0..(nrow * ncol))
            .into_par_iter()
            .map_init(thread_rng, |rng, _| {
                let x = rng.sample(u01);
                T::from(x).expect("failed to type")
            })
            .collect();

        DMatrix::<T>::from_vec(nrow, ncol, rvec)
    }

    fn rnorm(nrow: usize, ncol: usize) -> Self::Mat {
        let rvec = (0..(nrow * ncol))
            .into_par_iter()
            .map_init(thread_rng, |rng, _| {
                let x: f32 = rng.sample(StandardNormal);
                T::from(x).expect("failed to type")
            })
            .collect();

        DMatrix::<T>::from_vec(nrow, ncol, rvec)
    }
}

impl<T> MatOps for CscMatrix<T>
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

    fn normalize_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.normalize_columns_inplace();
        ret
    }

    fn scale_columns_inplace(&mut self) {
        let ncol = self.ncols();

        for j in 0..ncol {
            if let Some(x_j) = self.get_col(j) {
                let mut s0 = T::zero();
                let mut s1 = T::zero();
                let mut s2 = T::zero();

                for &x_ij in x_j.values() {
                    s0 += T::one();
                    s1 += x_ij;
                    s2 += x_ij * x_ij;
                }

                let mu = s1 / s0.max(T::one());
                let sig = (s2 / s0.max(T::one()) - mu * mu).sqrt();

                if let Some(mut x_j) = self.get_col_mut(j) {
                    if sig > T::zero() {
                        for x_ij in x_j.values_mut() {
                            *x_ij = (*x_ij - mu) / sig;
                        }
                    } else {
                        for x_ij in x_j.values_mut() {
                            *x_ij = *x_ij - mu;
                        }
                    }
                }
            }
        }
    }

    fn scale_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.scale_columns_inplace();
        ret
    }
}

impl<T> MatOps for DMatrix<T>
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

    fn normalize_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.normalize_columns_inplace();
        ret
    }

    fn scale_columns_inplace(&mut self) {
        for mut xx_j in self.column_iter_mut() {
            let mu = xx_j.mean();
            let sig = xx_j.variance().sqrt();
            xx_j.add_scalar_mut(-mu);
            if sig > T::zero() {
                xx_j /= sig;
            }
        }
    }
    fn scale_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.scale_columns_inplace();
        ret
    }
}

////////////////////////////////////////
// Input and output in triplet format //
////////////////////////////////////////

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
