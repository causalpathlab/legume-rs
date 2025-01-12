extern crate special;

use crate::traits::*;
use ndarray::prelude::*;

#[allow(dead_code)]
pub struct GammaMatrix {
    num_rows: usize,
    num_columns: usize,
    //////////////////////
    // hyper parameters //
    //////////////////////
    a0: f32,
    b0: f32,
    ///////////////////////////
    // sufficient statistics //
    ///////////////////////////
    a_stat: Array2<f32>,
    b_stat: Array2<f32>,
    //////////////////////////
    // estimated parameters //
    //////////////////////////
    estimated_mean: Array2<f32>,
    estimated_sd: Array2<f32>,
    estimated_log_mean: Array2<f32>,
    estimated_log_sd: Array2<f32>,
}

impl TwoStatParam for GammaMatrix {
    type Mat = Array2<f32>;
    type Scalar = f32;
    fn new(dims: (usize, usize), a: Self::Scalar, b: Self::Scalar) -> Self {
        Self {
            num_rows: dims.0,
            num_columns: dims.1,
            a0: a,
            b0: b,
            a_stat: Self::Mat::zeros(dims),
            b_stat: Self::Mat::zeros(dims),
            estimated_mean: Self::Mat::zeros(dims),
            estimated_sd: Self::Mat::zeros(dims),
            estimated_log_mean: Self::Mat::zeros(dims),
            estimated_log_sd: Self::Mat::zeros(dims),
        }
    }

    fn add_stat(&mut self, add_a: &Self::Mat, add_b: &Self::Mat) {
        self.a_stat += add_a;
        self.b_stat += add_b;
    }

    fn update_stat(&mut self, add_a: &Self::Mat, add_b: &Self::Mat) {
        self.reset_stat();
        self.add_stat(&add_a, &add_b);
    }

    fn update_stat_col(&mut self, add_a: &Self::Mat, add_b: &Self::Mat, k: usize) {
        self.a_stat.column_mut(k).zip_mut_with(add_a, |x, add_x| {
            *x = self.a0 + add_x;
        });
        self.b_stat.column_mut(k).zip_mut_with(add_b, |x, add_x| {
            *x = self.b0 + add_x;
        });
    }

    fn reset_stat(&mut self) {
        self.a_stat.fill(self.a0);
        self.b_stat.fill(self.b0);
    }

    fn nrows(&self) -> usize {
        self.num_rows
    }

    fn ncols(&self) -> usize {
        self.num_columns
    }
}
