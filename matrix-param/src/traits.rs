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
    /// Draw a fresh sample of `log λ` per element. Delta-method
    /// approximation: `log λ ≈ Normal(posterior_log_mean,
    /// posterior_log_sd²)`. Caller must have called
    /// `calibrate_with(CalibrateTarget::All)` first so log_mean / log_sd
    /// are populated.
    fn posterior_log_sample(&self) -> anyhow::Result<Self::Mat>;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Which posterior quantities to compute during calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrateTarget {
    /// Compute all: mean, sd, log_mean, log_sd
    All,
    /// Only posterior mean (a/b)
    MeanOnly,
    /// Posterior mean + log mean (digamma(a) - ln(b))
    MeanAndLogMean,
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

    /// Calibrate all posterior quantities (mean, sd, log_mean, log_sd)
    fn calibrate(&mut self) {
        self.calibrate_with(CalibrateTarget::All);
    }

    /// Calibrate only the posterior quantities specified by `target`.
    fn calibrate_with(&mut self, target: CalibrateTarget) {
        match target {
            CalibrateTarget::All => {
                self.map_calibrate_mean();
                self.map_calibrate_log_mean();
                self.map_calibrate_sd();
                self.map_calibrate_log_sd();
            }
            CalibrateTarget::MeanOnly => {
                self.map_calibrate_mean();
            }
            CalibrateTarget::MeanAndLogMean => {
                self.map_calibrate_mean();
                self.map_calibrate_log_mean();
            }
        }
    }

    fn map_calibrate_mean(&mut self);
    fn map_calibrate_sd(&mut self);
    fn map_calibrate_log_mean(&mut self);
    fn map_calibrate_log_sd(&mut self);
}
