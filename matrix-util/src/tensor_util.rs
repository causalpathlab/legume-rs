use crate::traits::*;
use candle_core::{Device, Tensor};

impl FromTriplets for Tensor {
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
}
