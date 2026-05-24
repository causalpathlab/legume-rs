#![allow(dead_code)]

extern crate special;

use crate::io::*;
use crate::traits::*;
use nalgebra::DMatrix;
use rayon::prelude::*;

#[derive(Debug, Clone)]
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
            // `estimated_mean` is eager: the coordinate descent reads
            // `posterior_mean()` before the first calibration (relying on a
            // zero start), and it gets allocated immediately anyway.
            estimated_mean: DMatrix::zeros(dims.0, dims.1),
            // The sd / log_mean / log_sd planes are lazily allocated by
            // `map_calibrate_*` (via `calibrate_with`). An iterative fit that
            // only reads `posterior_mean()` (calibrating `MeanOnly`) never
            // pays for them — they're materialized only when output needs
            // them (a calibrate with `All` / `MeanAndLogMean`).
            estimated_sd: DMatrix::zeros(0, 0),
            estimated_log_mean: DMatrix::zeros(0, 0),
            estimated_log_sd: DMatrix::zeros(0, 0),
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

    // fn nrows(&self) -> usize {
    //     self.num_rows
    // }

    // fn ncols(&self) -> usize {
    //     self.num_columns
    // }

    // fn len(&self) -> usize {
    //     self.num_rows * self.num_columns
    // }
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

    fn posterior_sample(&self) -> anyhow::Result<Self::Mat> {
        use rand_distr::{Distribution, Gamma};
        let eps = 1e-8;

        let sampled = self
            .a_stat
            .as_slice()
            .par_iter()
            .zip(self.b_stat.as_slice().par_iter())
            .map_init(rand::rng, |rng, (&a, &b)| -> anyhow::Result<f32> {
                let shape = a + eps;
                let scale = (b + eps).recip();
                let pdf = Gamma::new(shape, scale)?;
                Ok(pdf.sample(rng))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self::Mat::from_vec(self.nrows(), self.ncols(), sampled))
    }

    fn posterior_log_sample(&self) -> anyhow::Result<Self::Mat> {
        use rand_distr::{Distribution, StandardNormal};

        let sampled: Vec<f32> = self
            .estimated_log_mean
            .as_slice()
            .par_iter()
            .zip(self.estimated_log_sd.as_slice().par_iter())
            .map_init(rand::rng, |rng, (&m, &s)| {
                let z: f32 = StandardNormal.sample(rng);
                m + s * z
            })
            .collect();

        Ok(Self::Mat::from_vec(self.nrows(), self.ncols(), sampled))
    }

    fn nrows(&self) -> usize {
        self.num_rows
    }

    fn ncols(&self) -> usize {
        self.num_columns
    }
}

/// Row-stack one plane across blocks. Returns an empty matrix (and skips
/// the work) when `enabled` is false or the plane is lazily-unallocated in
/// the first block, so empty/dropped planes never get materialized.
fn stack_field<F>(
    blocks: &[GammaMatrix],
    nrows: usize,
    ncols: usize,
    enabled: bool,
    sel: F,
) -> DMatrix<f32>
where
    F: Fn(&GammaMatrix) -> &DMatrix<f32>,
{
    if !enabled || blocks.is_empty() || sel(&blocks[0]).nrows() == 0 {
        return DMatrix::zeros(0, 0);
    }
    let mut out = DMatrix::zeros(nrows, ncols);
    let mut r0 = 0;
    for b in blocks {
        let src = sel(b);
        out.rows_mut(r0, src.nrows()).copy_from(src);
        r0 += src.nrows();
    }
    out
}

impl GammaMatrix {
    /// Drop the sufficient-stat planes (`a_stat` / `b_stat`) after
    /// calibration, keeping only the posterior estimates. Use when the
    /// consumer reads posterior means / log-means but never
    /// `posterior_sample` (which is the only reader of `a_stat`/`b_stat`).
    /// Halves the resident footprint of a calibrated parameter.
    pub fn release_stats(&mut self) {
        self.a_stat = DMatrix::zeros(0, 0);
        self.b_stat = DMatrix::zeros(0, 0);
    }

    /// Zero every `estimated_mean` entry whose corresponding `numerator` is
    /// zero, collapsing the per-column Gamma prior baseline (`a0/denom`,
    /// present at *every* unobserved cell) to exact zero. This lets a
    /// downstream triplet-ization of the mean be **sparse** — only the
    /// observed support survives. It's the lossy-but-correct choice for
    /// count-based consumers (the baseline is a regularization floor, not
    /// signal). `numerator` must match the mean's shape; only meaningful
    /// after a mean calibration.
    pub fn sparsify_mean_to_support(&mut self, numerator: &DMatrix<f32>) {
        debug_assert_eq!(self.estimated_mean.shape(), numerator.shape());
        self.estimated_mean
            .iter_mut()
            .zip(numerator.iter())
            .for_each(|(m, &n)| {
                if n == 0.0 {
                    *m = 0.0;
                }
            });
    }

    /// Row-stack per-feature-block parameters (from a gene-blocked fit)
    /// into one `[Σrowsᵢ × K]` parameter. All blocks must share the column
    /// count and hyper-params. Calibrated planes present in the first block
    /// are stacked; lazily-empty planes stay empty. `stack_stats` controls
    /// whether `a_stat`/`b_stat` are carried through — pass `false` when the
    /// output only needs posterior estimates, so the heavy sufficient-stat
    /// planes are never assembled at full width.
    pub fn vconcat(blocks: Vec<GammaMatrix>, stack_stats: bool) -> Self {
        assert!(!blocks.is_empty(), "vconcat of empty block list");
        let ncols = blocks[0].num_columns;
        let a0 = blocks[0].a0;
        let b0 = blocks[0].b0;
        let nrows: usize = blocks.iter().map(|b| b.num_rows).sum();
        let a_stat = stack_field(&blocks, nrows, ncols, stack_stats, |g| &g.a_stat);
        let b_stat = stack_field(&blocks, nrows, ncols, stack_stats, |g| &g.b_stat);
        let estimated_mean = stack_field(&blocks, nrows, ncols, true, |g| &g.estimated_mean);
        let estimated_sd = stack_field(&blocks, nrows, ncols, true, |g| &g.estimated_sd);
        let estimated_log_mean =
            stack_field(&blocks, nrows, ncols, true, |g| &g.estimated_log_mean);
        let estimated_log_sd = stack_field(&blocks, nrows, ncols, true, |g| &g.estimated_log_sd);
        Self {
            num_rows: nrows,
            num_columns: ncols,
            a0,
            b0,
            a_stat,
            b_stat,
            estimated_mean,
            estimated_sd,
            estimated_log_mean,
            estimated_log_sd,
        }
    }
}
