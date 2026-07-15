use crate::rand_util::{collect_f32_seeded, entropy_seed};
use crate::traits::*;
use candle_core::{Device, Tensor};
use rand_distr::{Gamma, StandardNormal, Uniform};

impl SampleOps for Tensor {
    type Mat = Self;
    type Scalar = f32;

    fn runif(nrow: usize, ncol: usize) -> Self::Mat {
        Self::runif_seeded(nrow, ncol, entropy_seed())
    }

    fn rnorm(nrow: usize, ncol: usize) -> Self::Mat {
        Self::rnorm_seeded(nrow, ncol, entropy_seed())
    }

    fn rgamma(nrow: usize, ncol: usize, param: (f32, f32)) -> Self::Mat {
        Self::rgamma_seeded(nrow, ncol, param, entropy_seed())
    }

    fn runif_seeded(nrow: usize, ncol: usize, seed: u64) -> Self::Mat {
        let u01 = Uniform::new(0_f32, 1_f32).expect("failed to create uniform distribution");
        let data = collect_f32_seeded(nrow * ncol, u01, seed);
        Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor runif_seeded")
    }

    fn rnorm_seeded(nrow: usize, ncol: usize, seed: u64) -> Self::Mat {
        // Candle's own `Tensor::randn` on the CPU backend draws from
        // `rand::rng()` (OS entropy) and its `Device::set_seed` errors out, so
        // there is no way to seed it through candle. Generate the sample
        // host-side instead, where the seed is honored.
        let data = collect_f32_seeded(nrow * ncol, StandardNormal, seed);
        Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor rnorm_seeded")
    }

    fn rgamma_seeded(nrow: usize, ncol: usize, param: (f32, f32), seed: u64) -> Self::Mat {
        let (shape, scale) = param;
        let pdf = Gamma::new(shape, scale).unwrap();
        let data = collect_f32_seeded(nrow * ncol, pdf, seed);
        Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor rgamma_seeded")
    }
}

impl MatTriplets for Tensor {
    type Mat = Self;
    type Scalar = f32;

    fn from_nonzero_triplets<I>(
        nrow: usize,
        ncol: usize,
        triplets: &[(I, I, Self::Scalar)],
    ) -> anyhow::Result<Self::Mat>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut data = vec![0_f32; ncol * nrow];
        for &(ii, jj, x_ij) in triplets {
            let ii: usize = ii.try_into().expect("failed to convert index ii");
            let jj: usize = jj.try_into().expect("failed to convert index jj");
            data[ii * ncol + jj] = x_ij;
        }
        Ok(Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)?)
    }

    fn to_nonzero_triplets(&self) -> anyhow::Result<NRowNColTriplets<Self::Scalar>> {
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

            Ok(NRowNColTriplets {
                nrow,
                ncol,
                triplets: ret,
            })
        } else {
            anyhow::bail!("not a 2D Tensor");
        }
    }
}

// impl CandleDataLoaderOps for Tensor {
//     fn rows_to_tensor_vec(&self) -> Vec<Tensor> {
//         let mut idx_data = (0..self.dims()[0])
//             .map(|i| (i, self.narrow(0, i, 1).expect("").clone()))
//             .collect::<Vec<_>>();

//         idx_data.sort_by_key(|(i, _)| *i);
//         idx_data.into_iter().map(|(_, t)| t).collect()
//     }
// }
