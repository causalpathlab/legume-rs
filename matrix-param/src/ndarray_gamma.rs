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

    /// New Poisson-Gamma parameter matrix
    ///
    /// x[i,j] ~ Poisson(lambda[i,j])
    /// lambda[i,j] ~ Gamma(a0, b0)
    ///
    /// #Arguments
    /// * `dims` - dimensions of the matrix (num of rows, num of columns)
    /// * `a` - hyper parameter a0
    /// * `b` - hyper parameter b0
    ///
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

impl Inference for GammaMatrix {
    type Mat = Array2<f32>;
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
        self.estimated_mean = &self.a_stat / &self.b_stat;
    }
    fn map_calibrate_sd(&mut self) {
        self.estimated_sd = &self.a_stat.mapv(|x| x.sqrt()) / &self.b_stat;
    }
    fn map_calibrate_log_mean(&mut self) {
        use special::Gamma;
        self.estimated_log_mean =
            &self.a_stat.mapv(|a| Gamma::digamma(a)) - &self.b_stat.mapv(|b| b.ln());
    }
    fn map_calibrate_log_sd(&mut self) {
        self.estimated_log_sd = self.a_stat.mapv(|a| -> f32 {
            if a > 1.0 {
                1.0 / (a - 1.0).sqrt()
            } else {
                // this is actually not true
                0.0
            }
        });
    }
}
