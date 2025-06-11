#![allow(dead_code)]

extern crate special;

use crate::io::*;
use crate::traits::*;
use nalgebra::DMatrix;

#[derive(Debug)]
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
    a_stat: DMatrix<f32>,
    b_stat: DMatrix<f32>,
    //////////////////////////
    // estimated parameters //
    //////////////////////////
    estimated_mean: DMatrix<f32>,
    estimated_sd: DMatrix<f32>,
    estimated_log_mean: DMatrix<f32>,
    estimated_log_sd: DMatrix<f32>,
}

impl ParamIo for GammaMatrix {
    type Mat = DMatrix<f32>;
}

impl TwoStatParam for GammaMatrix {
    type Mat = DMatrix<f32>;
    type Scalar = f32;

    fn new(dims: (usize, usize), a: Self::Scalar, b: Self::Scalar) -> Self {
        Self {
            num_rows: dims.0,
            num_columns: dims.1,
            a0: a,
            b0: b,
            a_stat: DMatrix::from_element(dims.0, dims.1, a),
            b_stat: DMatrix::from_element(dims.0, dims.1, b),
            estimated_mean: DMatrix::zeros(dims.0, dims.1),
            estimated_sd: DMatrix::zeros(dims.0, dims.1),
            estimated_log_mean: DMatrix::zeros(dims.0, dims.1),
            estimated_log_sd: DMatrix::zeros(dims.0, dims.1),
        }
    }

    fn add_stat(&mut self, add_a: &Self::Mat, add_b: &Self::Mat) {
        self.a_stat += add_a;
        self.b_stat += add_b;
    }
    fn update_stat(&mut self, update_a: &Self::Mat, update_b: &Self::Mat) {
        self.reset_stat();
        self.add_stat(update_a, update_b);
    }
    fn reset_stat(&mut self) {
        self.a_stat.fill(self.a0);
        self.b_stat.fill(self.b0);
    }
    fn update_stat_col(&mut self, update_a: &Self::Mat, update_b: &Self::Mat, k: usize) {
        self.a_stat
            .column_mut(k)
            .copy_from(&update_a.map(|x| x + self.a0));
        self.b_stat
            .column_mut(k)
            .copy_from(&update_b.map(|x| x + self.b0));
    }

    fn nrows(&self) -> usize {
        self.num_rows
    }
    fn ncols(&self) -> usize {
        self.num_columns
    }
}

impl Inference for GammaMatrix {
    type Mat = DMatrix<f32>;
    type Scalar = f32;

    fn posterior_mean(&self) -> &Self::Mat {
        &self.estimated_mean
    }

    fn posterior_sd(&self) -> &Self::Mat {
        &self.estimated_sd
    }

    fn posterior_log_mean(&self) -> &Self::Mat {
        &self.estimated_log_mean
    }

    fn posterior_log_sd(&self) -> &Self::Mat {
        &self.estimated_log_sd
    }

    fn calibrate(&mut self) {
        self.map_calibrate_mean();
        self.map_calibrate_log_mean();
        self.map_calibrate_sd();
        self.map_calibrate_log_sd();
    }
    fn map_calibrate_mean(&mut self) {
        self.estimated_mean = self.a_stat.zip_map(&self.b_stat, |a, b| a / b);
    }
    fn map_calibrate_sd(&mut self) {
        self.estimated_sd = self.a_stat.zip_map(&self.b_stat, |a, b| a.sqrt() / b);
    }
    fn map_calibrate_log_mean(&mut self) {
        use special::Gamma;
        self.estimated_log_mean = self
            .a_stat
            .zip_map(&self.b_stat, |a, b| a.digamma() - b.ln());
    }
    fn map_calibrate_log_sd(&mut self) {
        self.estimated_log_sd = self.a_stat.map(|a| -> f32 {
            if a > 1.0 {
                1.0 / (a - 1.0).sqrt()
            } else {
                // this is actually not true
                0.0
            }
        });
    }
}
