pub use ndarray::prelude::*;
pub use rand::{thread_rng, Rng};
pub use rand_distr::{StandardNormal, Uniform};
pub use rayon::prelude::*;

use crate::traits::*;
use num_traits::Float;

#[allow(dead_code)]
/// Sample d,n matrix from N(0,1)
pub fn rnorm(dd: usize, nn: usize) -> anyhow::Result<Array2<f32>> {
    let rvec = (0..(dd * nn))
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.sample(StandardNormal))
        .collect();

    let ret: Array2<f32> = Array2::from_shape_vec((dd, nn), rvec)?;
    Ok(ret)
}

#[allow(dead_code)]
/// Sample d,n matrix from U(0,1)
pub fn runif(dd: usize, nn: usize) -> anyhow::Result<Array2<f32>> {
    let runif = Uniform::new(0_f32, 1_f32);

    let rvec = (0..(dd * nn))
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.sample(runif))
        .collect();

    let ret: Array2<f32> = Array2::from_shape_vec((dd, nn), rvec)?;
    Ok(ret)
}

// #[allow(dead_code)]
// /// column-wise standardization
// /// * `xraw`: (D, N) matrix
// pub fn scale_columns(mut xraw: Array2<f32>) -> anyhow::Result<Array2<f32>> {
//     let mu = xraw
//         .mean_axis(Axis(0))
//         .ok_or(anyhow::anyhow!("mean failed"))?;
//     let sig = xraw.std_axis(Axis(0), 0.0);
//     for j in 0..xraw.ncols() {
//         xraw.column_mut(j).mapv_inplace(|x| (x - mu[j]) / sig[j]);
//     }
//     Ok(xraw)
// }

///////////////////////////
// trait implementations //
///////////////////////////

impl<T> MatInplaceOps for ndarray::Array2<T>
where
    T: Float,
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
