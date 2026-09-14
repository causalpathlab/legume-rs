//! Host-side parameter tables and PBG's row-wise Adagrad.

use matrix_util::rand_util::{collect_f32_seeded, mix_seed};
use rand_distr::Normal;

const INIT_STDEV: f32 = 0.1;
const ADAGRAD_EPS: f32 = 1e-10;

pub struct HierParams {
    pub h: usize,
    pub e_u: Vec<f32>,
    pub mu: Vec<f32>,
    pub b_m: Vec<f32>,
    pub r: Vec<f32>,
    pub b_g: Vec<f32>,
}

fn randn(n: usize, seed: u64) -> Vec<f32> {
    let dist = Normal::new(0.0f32, INIT_STDEV).expect("finite stdev");
    collect_f32_seeded(n, dist, seed)
}

impl HierParams {
    pub fn new(n_units: usize, n_modules: usize, n_features: usize, h: usize, seed: u64) -> Self {
        Self {
            h,
            e_u: randn(n_units * h, mix_seed(seed, 0x4855)),
            mu: randn(n_modules * h, mix_seed(seed, 0x4d55)),
            b_m: vec![0.0; n_modules],
            r: randn(n_features * h, mix_seed(seed, 0x5253)),
            b_g: vec![0.0; n_features],
        }
    }
}

pub struct RowAdagrad {
    pub acc: Vec<f32>,
    pub lr: f32,
    pub eps: f32,
}

impl RowAdagrad {
    pub fn new(n_rows: usize, lr: f32) -> Self {
        Self {
            acc: vec![0.0; n_rows],
            lr,
            eps: ADAGRAD_EPS,
        }
    }
    pub fn update(&mut self, r: usize, row: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(row.len(), grad.len());
        let g2 = grad.iter().map(|g| g * g).sum::<f32>() / grad.len().max(1) as f32;
        if g2 == 0.0 {
            return;
        }
        self.acc[r] += g2;
        let step = self.lr / (self.acc[r].sqrt() + self.eps);
        for (x, g) in row.iter_mut().zip(grad) {
            *x -= step * g;
        }
    }

    /// [`Self::update`] for a row that carries a scalar bias alongside it: the
    /// accumulator sees the mean of `grad²` over the row AND the bias, and both
    /// move by the same row step. What the callers did by concatenating the
    /// bias onto a copy of the row, without the copy.
    pub fn update_with_bias(
        &mut self,
        r: usize,
        row: &mut [f32],
        bias: &mut f32,
        grad: &[f32],
        gbias: f32,
    ) {
        debug_assert_eq!(row.len(), grad.len());
        let n = (grad.len() + 1) as f32;
        let g2 = (grad.iter().map(|g| g * g).sum::<f32>() + gbias * gbias) / n;
        if g2 == 0.0 {
            return;
        }
        self.acc[r] += g2;
        let step = self.lr / (self.acc[r].sqrt() + self.eps);
        for (x, g) in row.iter_mut().zip(grad) {
            *x -= step * g;
        }
        *bias -= step * gbias;
    }
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod params_tests;
