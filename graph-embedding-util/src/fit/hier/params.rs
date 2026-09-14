//! Host-side parameter tables and PBG's row-wise Adagrad.

use matrix_util::rand_util::mix_seed;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

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
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0f32, INIT_STDEV).expect("finite stdev");
    (0..n).map(|_| dist.sample(&mut rng)).collect()
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
    pub fn e_u_row(&self, u: usize) -> &[f32] {
        &self.e_u[u * self.h..(u + 1) * self.h]
    }
    pub fn r_row(&self, g: usize) -> &[f32] {
        &self.r[g * self.h..(g + 1) * self.h]
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
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod params_tests;
