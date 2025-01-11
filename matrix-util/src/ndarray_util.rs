pub use ndarray::prelude::*;
pub use rand::{thread_rng, Rng};
pub use rand_distr::{StandardNormal, Uniform};
pub use rayon::prelude::*;

use crate::traits::*;
use num_traits::{Float, FromPrimitive};

impl<T> SampleOps for ndarray::Array2<T>
where
    T: Float + FromPrimitive + Send,
{
    type Mat = Self;
    type Scalar = T;

    fn runif(dd: usize, nn: usize) -> Self::Mat {
        let u01 = Uniform::new(0_f32, 1_f32);

        let rvec: Vec<T> = (0..(dd * nn))
            .into_par_iter()
            .map_init(thread_rng, |rng, _| {
                let x = rng.sample(u01);
                T::from(x).expect("failed to type")
            })
            .collect();

        Array2::from_shape_vec((dd, nn), rvec).unwrap()
    }

    fn rnorm(dd: usize, nn: usize) -> Self::Mat {
        let rvec = (0..(dd * nn))
            .into_par_iter()
            .map_init(thread_rng, |rng, _| {
                let x: f32 = rng.sample(StandardNormal);
                T::from(x).expect("failed to type")
            })
            .collect();

        Array2::from_shape_vec((dd, nn), rvec).unwrap()
    }
}

impl<T> MatOps for ndarray::Array2<T>
where
    T: Float + FromPrimitive,
{
    type Mat = Self;
    type Scalar = T;

    fn normalize_columns(&mut self) -> Self::Mat {
        let mut xx = self.clone();
        for j in 0..xx.ncols() {
            let mut x_j = xx.column_mut(j);
            let denom = x_j.mapv(|x| x * x).sum().max(T::one()).sqrt();
            x_j.mapv_inplace(|x| x / denom);
        }
        xx
    }

    fn normalize_columns_inplace(&mut self) {
        for j in 0..self.ncols() {
            let mut x_j = self.column_mut(j);
            let denom = x_j.mapv(|x| x * x).sum().max(T::one()).sqrt();
            x_j.mapv_inplace(|x| x / denom);
        }
    }

    fn scale_columns_inplace(&mut self) {
        let mu = self.mean_axis(Axis(0)).expect("mean failed");
        let sig = self.std_axis(Axis(0), T::zero());
        let ncol = self.ncols();

        for j in 0..ncol {
            if sig[j] > T::zero() {
                self.column_mut(j).mapv_inplace(|x| (x - mu[j]) / sig[j]);
            } else {
                self.column_mut(j).mapv_inplace(|x| x - mu[j]);
            }
        }
    }
}

impl<T> MatTriplets for ndarray::Array2<T>
where
    T: Float,
{
    type Mat = Self;
    type Scalar = T;

    fn from_nonzero_triplets(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(usize, usize, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat> {
        let mut array = ndarray::Array2::<T>::zeros((nrow, ncol));
        for (ii, jj, x_ij) in triplets {
            array[(ii, jj)] = x_ij;
        }
        Ok(array)
    }

    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)> {
        if let Some(eps) = T::from(1e-6) {
            let (rows, cols) = self.dim();
            Ok((
                rows,
                cols,
                self.indexed_iter()
                    .filter_map(
                        |((i, j), &x)| {
                            if x.abs() > eps {
                                Some((i, j, x))
                            } else {
                                None
                            }
                        },
                    )
                    .collect(),
            ))
        } else {
            anyhow::bail!("eps is not defined")
        }
    }
}
