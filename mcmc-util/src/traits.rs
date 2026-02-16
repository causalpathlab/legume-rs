use candle_core::Tensor;
use nalgebra::{DMatrix, DVector};

/// Minimal trait for types usable as ESS parameters.
/// Only requires linear combination — prior sampling is handled
/// externally via the `prior_draw` closure in `EssSampler::run`.
pub trait EssParam: Clone {
    /// Elliptical combination: `a * self + b * other`
    fn linear_combine(&self, a: f32, other: &Self, b: f32) -> Self;
}

/// Optional trait enabling summary statistics (mean, variance, quantile).
/// Requires flattening to a slice.
pub trait EssParamSummary: EssParam {
    fn as_slice(&self) -> &[f32];
    fn dim(&self) -> usize;
}

// --- nalgebra DVector<f32> ---

impl EssParam for DVector<f32> {
    fn linear_combine(&self, a: f32, other: &Self, b: f32) -> Self {
        self * a + other * b
    }
}

impl EssParamSummary for DVector<f32> {
    fn as_slice(&self) -> &[f32] {
        self.as_slice()
    }

    fn dim(&self) -> usize {
        self.nrows()
    }
}

// --- nalgebra DMatrix<f32> ---

impl EssParam for DMatrix<f32> {
    fn linear_combine(&self, a: f32, other: &Self, b: f32) -> Self {
        self * a + other * b
    }
}

impl EssParamSummary for DMatrix<f32> {
    fn as_slice(&self) -> &[f32] {
        self.as_slice()
    }

    fn dim(&self) -> usize {
        self.nrows() * self.ncols()
    }
}

// --- Vec<P> for composite parameters ---

impl<P: EssParam> EssParam for Vec<P> {
    fn linear_combine(&self, a: f32, other: &Self, b: f32) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(s, o)| s.linear_combine(a, o, b))
            .collect()
    }
}

// --- candle Tensor ---

impl EssParam for Tensor {
    fn linear_combine(&self, a: f32, other: &Self, b: f32) -> Self {
        (self * a as f64)
            .unwrap()
            .add(&(other * b as f64).unwrap())
            .unwrap()
    }
}
