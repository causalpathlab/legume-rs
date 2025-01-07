extern crate special;

use ndarray::prelude::*;
use ndarray::Zip;

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

impl GammaParam {
    pub fn mean(&self) -> &Array2<f32> {
        &self.estimated_mean
    }

    pub fn sd(&self) -> &Array2<f32> {
        &self.estimated_sd
    }

    pub fn log_mean(&self) -> &Array2<f32> {
        &self.estimated_log_mean
    }

    pub fn log_sd(&self) -> &Array2<f32> {
        &self.estimated_log_sd
    }

    pub fn reset_stat_only(&mut self) {
        self.a_stat.fill(self.a0);
        self.b_stat.fill(self.b0);
    }

    pub fn add(&mut self, add_a: &Array2<f32>, add_b: &Array2<f32>) {
        self.a_stat += add_a;
        self.b_stat += add_b;
    }

    pub fn update(&mut self, update_a: &Array2<f32>, update_b: &Array2<f32>) {
        self.reset_stat_only();
        self.add(update_a, update_b);
    }

    pub fn update_col(&mut self, update_a: &Array2<f32>, update_b: &Array2<f32>, k: usize) {
        self.a_stat.column_mut(k).fill(self.a0);
        self.b_stat.column_mut(k).fill(self.b0);
        Zip::from(&mut self.a_stat.column_mut(k))
            .and(&update_a.column(k))
            .for_each(|a, &delta_a| *a += delta_a);
        Zip::from(&mut self.b_stat.column_mut(k))
            .and(&update_b.column(k))
            .for_each(|b, &delta_b| *b += delta_b);
    }

    pub fn rows(&self) -> usize {
        self.nrows
    }

    pub fn cols(&self) -> usize {
        self.ncols
    }

    pub fn calibrate(&mut self) {
        self.estimated_mean = &self.a_stat / &self.b_stat;
        self._map_log_mean();
        self._map_sd();
        self._map_log_sd();
    }

    // digamma(a) - ln(b)
    fn _map_log_mean(&mut self) {
        use special::Gamma;
        Zip::from(&mut self.estimated_log_mean)
            .and(&self.a_stat)
            .and(&self.b_stat)
            .for_each(|estimated_log_mean, a, b| {
                *estimated_log_mean = a.digamma() - b.ln();
            });
    }

    // Delta method
    // sqrt V[ln(mu)] = sqrt (V[mu] / mu)
    //                = 1/sqrt(a -1)
    // when approximated at the mode = (a - 1)/b
    fn _map_log_sd(&mut self) {
        Zip::from(&mut self.estimated_log_sd)
            .and(&self.a_stat)
            .for_each(|_log_sd, a| {
                if *a > 1.0 {
                    *_log_sd = 1.0 / (a - 1.).sqrt();
                } else {
                    *_log_sd = 0.0;
                }
            });
    }

    // sqrt(a) / b
    fn _map_sd(&mut self) {
        Zip::from(&mut self.estimated_sd)
            .and(&self.a_stat)
            .and(&self.b_stat)
            .for_each(|_sd, a, b| {
                *_sd = a.sqrt() / b;
            });
    }
}
