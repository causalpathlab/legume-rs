use crate::traits::*;
use ndarray::prelude::*;
use num_traits::{Float, FromPrimitive};
use rand::Rng;
use rand_distr::{Distribution, Gamma, StandardNormal, Uniform};
use rayon::prelude::*;

impl<T> SampleOps for ndarray::Array2<T>
where
    T: Float + FromPrimitive + Send,
{
    type Mat = Self;
    type Scalar = T;

    fn runif(dd: usize, nn: usize) -> Self::Mat {
        let u01 = Uniform::new(0_f32, 1_f32).expect("failed to create uniform distribution");

        let rvec: Vec<T> = (0..(dd * nn))
            .into_par_iter()
            .map_init(rand::rng, |rng, _| {
                let x = rng.sample(u01);
                T::from(x).expect("failed to type")
            })
            .collect();

        Array2::from_shape_vec((dd, nn), rvec).unwrap()
    }

    fn rnorm(dd: usize, nn: usize) -> Self::Mat {
        let rvec = (0..(dd * nn))
            .into_par_iter()
            .map_init(rand::rng, |rng, _| {
                let x: f32 = rng.sample(StandardNormal);
                T::from(x).expect("failed to type")
            })
            .collect();

        Array2::from_shape_vec((dd, nn), rvec).unwrap()
    }

    fn rgamma(dd: usize, nn: usize, param: (f32, f32)) -> Self::Mat {
        let (shape, scale) = param;
        let pdf = Gamma::new(shape, scale).unwrap();

        let rvec = (0..(dd * nn))
            .into_par_iter()
            .map_init(rand::rng, |rng, _| {
                let x: f32 = pdf.sample(rng);
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

    fn normalize_columns(&self) -> Self::Mat {
        let mut xx = self.clone();
        xx.normalize_columns_inplace();
        xx
    }

    fn normalize_columns_inplace(&mut self) {
        for j in 0..self.ncols() {
            let mut x_j = self.column_mut(j);
            let denom = x_j.mapv(|x| x * x).sum().max(T::one()).sqrt();
            x_j.mapv_inplace(|x| x / denom);
        }
    }

    fn scale_columns(&self) -> Self::Mat {
        let mut xx = self.clone();
        xx.scale_columns_inplace();
        xx
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
    fn centre_columns(&self) -> Self::Mat {
        let mut xx = self.clone();
        xx.centre_columns_inplace();
        xx
    }

    fn centre_columns_inplace(&mut self) {
        let mu = self.mean_axis(Axis(0)).expect("mean failed");
        let ncol = self.ncols();
        for j in 0..ncol {
            self.column_mut(j).mapv_inplace(|x| x - mu[j]);
        }
    }
}

impl<T> MatTriplets for ndarray::Array2<T>
where
    T: Float,
{
    type Mat = Self;
    type Scalar = T;

    fn from_nonzero_triplets<I>(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(I, I, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut array = ndarray::Array2::<T>::zeros((nrow, ncol));
        for (ii, jj, x_ij) in triplets {
            let ii: usize = ii.try_into().expect("failed to convert index ii");
            let jj: usize = jj.try_into().expect("failed to convert index jj");
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

impl<T> MeltOps for ndarray::Array2<T>
where
    T: Float + FromPrimitive,
{
    type Mat = Self;
    type Scalar = T;

    fn melt(&self) -> Vec<Self::Scalar> {
        let nelem = self.shape().iter().product();

        let mut val: Vec<Self::Scalar> = Vec::with_capacity(nelem);
        for (_ij, &x) in self.indexed_iter() {
            val.push(x);
        }
        val
    }

    fn melt_with_indexes(&self) -> (Vec<Self::Scalar>, Vec<Vec<usize>>) {
        let nelem = self.len();
        let mut idx: Vec<Vec<usize>> = vec![Vec::with_capacity(nelem); self.ndim()];
        let mut val: Vec<Self::Scalar> = Vec::with_capacity(nelem);
        for (ij, &x) in self.indexed_iter() {
            val.push(x);
            idx[0].push(ij.0);
            idx[1].push(ij.1);
        }
        (val, idx)
    }
}
