use matrix_util::traits::{CandleDataLoaderOps, IoOps, MeltOps};

pub trait TwoStatInference: Inference + TwoStatParam {}

pub trait Inference {
    type Mat: IoOps + MeltOps + CandleDataLoaderOps;
    type Scalar: Into<f32>;

    fn posterior_mean(&self) -> &Self::Mat;
    fn posterior_sd(&self) -> &Self::Mat;
    fn posterior_log_mean(&self) -> &Self::Mat;
    fn posterior_log_sd(&self) -> &Self::Mat;
    fn posterior_sample(&self) -> anyhow::Result<Self::Mat>;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// A parameter matrix with two types of statistics
/// with hyper parameters a0 and b0
pub trait TwoStatParam {
    type Mat;
    type Scalar;

    fn new(dims: (usize, usize), a0: Self::Scalar, b0: Self::Scalar) -> Self;
    fn add_stat(&mut self, add_a: &Self::Mat, add_b: &Self::Mat);
    fn update_stat(&mut self, update_a: &Self::Mat, update_b: &Self::Mat);
    fn update_stat_col(&mut self, update_a: &Self::Mat, update_b: &Self::Mat, k: usize);
    fn reset_stat(&mut self);

    fn calibrate(&mut self);
    fn map_calibrate_mean(&mut self);
    fn map_calibrate_sd(&mut self);
    fn map_calibrate_log_mean(&mut self);
    fn map_calibrate_log_sd(&mut self);

    // fn nrows(&self) -> usize;
    // fn ncols(&self) -> usize;
    // fn len(&self) -> usize;
}
