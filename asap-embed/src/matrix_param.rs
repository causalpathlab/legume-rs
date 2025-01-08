extern crate special;

use nalgebra::DMatrix;

// use ndarray::prelude::*;
// use ndarray::Zip;

pub struct GammaParam {
    nrows: usize,
    ncols: usize,
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

impl GammaParam {
    pub fn mean(&self) -> &DMatrix<f32> {
        &self.estimated_mean
    }

    pub fn sd(&self) -> &DMatrix<f32> {
        &self.estimated_sd
    }

    pub fn log_mean(&self) -> &DMatrix<f32> {
        &self.estimated_log_mean
    }

    pub fn log_sd(&self) -> &DMatrix<f32> {
        &self.estimated_log_sd
    }

    pub fn reset_stat_only(&mut self) {
        self.a_stat.fill(self.a0);
        self.b_stat.fill(self.b0);
    }

    pub fn add(&mut self, add_a: &DMatrix<f32>, add_b: &DMatrix<f32>) {
        self.a_stat += add_a;
        self.b_stat += add_b;
    }

    pub fn update(&mut self, update_a: &DMatrix<f32>, update_b: &DMatrix<f32>) {
        self.reset_stat_only();
        self.add(update_a, update_b);
    }

    pub fn update_col(&mut self, update_a: &DMatrix<f32>, update_b: &DMatrix<f32>, k: usize) {
        self.a_stat
            .column_mut(k)
            .copy_from(&update_a.map(|x| x + self.a0));
        self.b_stat
            .column_mut(k)
            .copy_from(&update_b.map(|x| x + self.b0));
    }

    pub fn rows(&self) -> usize {
        self.nrows
    }

    pub fn cols(&self) -> usize {
        self.ncols
    }

    pub fn calibrate(&mut self) {
        self.map_update_mean();
        self.map_update_log_mean();
        self.map_update_sd();
        self.map_update_log_sd();
    }

    // maximum a posteriori mean = a/b
    fn map_update_mean(&mut self) {
        self.estimated_mean = self.a_stat.zip_map(&self.b_stat, |a, b| a / b);
    }

    // maximum a posteriori sqrt(a) / b
    fn map_update_sd(&mut self) {
        self.estimated_sd = self.a_stat.zip_map(&self.b_stat, |a, b| a.sqrt() / b);
    }

    // maximum a posteriori log mean = digamma(a) - ln(b)
    fn map_update_log_mean(&mut self) {
        use special::Gamma;
        self.estimated_log_mean = self
            .a_stat
            .zip_map(&self.b_stat, |a, b| a.digamma() - b.ln());
    }

    // maximum a posteriori via delta method
    // sqrt V[ln(mu)] = sqrt (V[mu] / mu)
    //                = 1/sqrt(a -1)
    // when approximated at the mode = (a - 1)/b
    fn map_update_log_sd(&mut self) {
        self.estimated_log_sd = self
            .a_stat
            .map(|a| if a > 1.0 { 1.0 / (a - 1.0).sqrt() } else { 0.0 });
    }
}
