use crate::traits::*;
use candle_core::{Device, Tensor};
use rand::thread_rng;
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;

impl SampleOps for Tensor {
    type Mat = Self;
    type Scalar = f32;

    fn runif(nrow: usize, ncol: usize) -> Self::Mat {
        Tensor::rand(0_f32, 1_f32, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor runif")
    }

    fn rnorm(nrow: usize, ncol: usize) -> Self::Mat {
        Tensor::randn(0_f32, 1_f32, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor rnorm")
    }

    fn rgamma(nrow: usize, ncol: usize, param: (f32, f32)) -> Self::Mat {
        let (shape, scale) = param;
        let pdf = Gamma::new(shape, scale).unwrap();

        let data_vec = (0..(nrow * ncol))
            .into_par_iter()
            .map_init(thread_rng, |rng, _| pdf.sample(rng))
            .collect();

        Tensor::from_vec(data_vec, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor rgamma")
    }
}

impl MatTriplets for Tensor {
    type Mat = Self;
    type Scalar = f32;
    fn from_nonzero_triplets(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(usize, usize, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat> {
        let mut data = vec![0_f32; ncol * nrow];
        for (ii, jj, x_ij) in triplets {
            data[ii * ncol + jj] = x_ij;
        }
        Ok(Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)?)
    }

    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)> {
        if let Ok((nrow, ncol)) = self.dims2() {
            let eps = 1e-6;
            let mut ret = vec![];
            let xx: Vec<Vec<Self::Scalar>> = self.to_vec2()?;
            for (i, x_i) in xx.iter().enumerate() {
                for (j, &x_ij) in x_i.iter().enumerate() {
                    if x_ij.abs() > eps {
                        ret.push((i, j, x_ij));
                    }
                }
            }

            Ok((nrow, ncol, ret))
        } else {
            anyhow::bail!("not a 2D Tensor");
        }
    }
}
