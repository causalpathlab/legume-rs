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
        // `sd[ln X] = sqrt(trigamma(a))` exactly, for `X ~ Gamma(a, b)` — note it
        // does not depend on the rate, which is why `b_stat` plays no part here.
        //
        // This replaced `1/sqrt(a - 1)`, the large-`a` asymptote, which was
        // wrong in the regime that dominates sparse count data: 46% high at
        // `a = 1.5`, 21% high at `a = 2.2` (a typical detected feature), and
        // agreeing only past `a ~ 100`. Below `a = 1` it has no real value at
        // all, and the old code returned 0 there — i.e. it reported PERFECT
        // certainty for a feature with no counts, whose posterior is the prior
        // and whose true `sd` is the largest in the matrix (1.283 at `a = 1`).
        // Anything reading `log_sd` as a precision was being handed the
        // inversion of the truth.
        use special::Gamma;
        self.estimated_log_sd = self.a_stat.map(|a| a.trigamma().sqrt());
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

    fn posterior_log_sample(&self, seed: u64) -> anyhow::Result<Self::Mat> {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};

        // Fixed chunk width, so which elements share an RNG is a property of
        // the data shape and not of how rayon happened to split the work.
        const CHUNK: usize = 1024;
        let m_slice = self.estimated_log_mean.as_slice();
        let s_slice = self.estimated_log_sd.as_slice();
        let mut sampled = vec![0.0f32; m_slice.len()];
        sampled
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(ci, out)| {
                let mut rng =
                    SmallRng::seed_from_u64(seed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let base = ci * CHUNK;
                for (k, o) in out.iter_mut().enumerate() {
                    let z: f32 = StandardNormal.sample(&mut rng);
                    *o = m_slice[base + k] + s_slice[base + k] * z;
                }
            });

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

    /// Drop only the rate plane `b_stat`, keeping the shape `a_stat`.
    ///
    /// `posterior_mean = a/b`, so `(mean, a)` determines `b` exactly and
    /// [`Self::posterior_sample_seeded`] reconstructs it on the fly. That makes
    /// a resamplable parameter cost ONE extra plane over a mean-only one rather
    /// than two — the difference between "keep the sufficient statistics" and
    /// "keep enough to draw", which matters at high pseudobulk counts where the
    /// planes are the bulk of the collapse's memory.
    pub fn release_rate_stat(&mut self) {
        self.b_stat = DMatrix::zeros(0, 0);
    }

    /// Whether the shape plane is resident, i.e. whether
    /// [`Self::posterior_sample_seeded`] can run. `false` after
    /// [`Self::release_stats`].
    #[must_use]
    pub fn has_shape_stat(&self) -> bool {
        self.a_stat.shape() == (self.num_rows, self.num_columns)
    }

    /// Draw a fresh `Gamma(a, rate b)` sample per element, seeded.
    ///
    /// The rate comes from `b_stat` when it is resident and from `a / mean`
    /// after [`Self::release_rate_stat`] (falling back to the prior `b0` where
    /// the mean is zero). Seeded per fixed-width CHUNK of the element stream,
    /// exactly as [`Inference::posterior_log_sample`] is, so the draw is
    /// reproducible and independent of how rayon splits the work — the
    /// unseeded [`Inference::posterior_sample`] is neither.
    ///
    /// Errors when the shape plane has been released: there is nothing to draw
    /// from, and a prior draw dressed as a posterior one would be worse than a
    /// refusal.
    pub fn posterior_sample_seeded(&self, seed: u64) -> anyhow::Result<DMatrix<f32>> {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Gamma};

        anyhow::ensure!(
            self.has_shape_stat(),
            "posterior_sample_seeded: the shape statistics were released \
             (`release_stats`), so there is nothing to draw from"
        );
        let (rows, cols) = (self.num_rows, self.num_columns);
        let a = self.a_stat.as_slice();
        let b_resident: Option<&[f32]> =
            (self.b_stat.shape() == (rows, cols)).then(|| self.b_stat.as_slice());
        if b_resident.is_none() {
            anyhow::ensure!(
                self.estimated_mean.shape() == (rows, cols),
                "posterior_sample_seeded: the rate plane was released and there is no \
                 calibrated mean to reconstruct it from"
            );
        }
        let mean = self.estimated_mean.as_slice();
        let b0 = self.b0;

        // Same shape/scale floor as the unseeded draw, and the same fixed chunk
        // width as `posterior_log_sample`: which elements share an RNG is a
        // property of the data shape, never of how rayon split the work.
        const EPS: f32 = 1e-8;
        const CHUNK: usize = 1024;
        let mut out = vec![0.0f32; rows * cols];
        out.par_chunks_mut(CHUNK).enumerate().try_for_each(
            |(ci, chunk)| -> anyhow::Result<()> {
                let mut rng =
                    SmallRng::seed_from_u64(seed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let base = ci * CHUNK;
                for (k, o) in chunk.iter_mut().enumerate() {
                    let i = base + k;
                    let rate = match b_resident {
                        Some(b) => b[i],
                        // `mean = a/b` ⇒ `b = a/mean`; a zero mean has zero shape
                        // too, and draws (essentially) zero from the prior rate.
                        None if mean[i] > 0.0 => a[i] / mean[i],
                        None => b0,
                    };
                    let pdf = Gamma::new(a[i] + EPS, (rate + EPS).recip())?;
                    *o = pdf.sample(&mut rng);
                }
                Ok(())
            },
        )?;
        Ok(DMatrix::from_vec(rows, cols, out))
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

    /// Whether the posterior at `(row, col)` carries any data beyond the
    /// prior: `a_stat > a0`. The read-only counterpart of
    /// [`Self::sparsify_mean_to_support`], for consumers that serialize the
    /// mean without owning the numerator — an unsupported entry's mean is the
    /// prior floor `a0 / (b0 + denom)`, which is regularization, not signal.
    /// Writing those floors out turns a sparse posterior dense: a carried
    /// pseudobulk reference measured 100.0% dense (34M of 34M entries) before
    /// its writer checked this.
    #[must_use]
    pub fn has_data_support(&self, row: usize, col: usize) -> bool {
        self.a_stat[(row, col)] > self.a0
    }

    /// The unregularized rate at `(row, col)`: `(a_stat − a0) / (b_stat − b0)`
    /// — data sum over data denominator, no prior in either. Zero when the
    /// entry has no data support.
    ///
    /// This is what a *serialized* posterior should usually store: paired with
    /// its denominator, it is a bijection of the sufficient statistics, so a
    /// consumer reconstructs `a_stat`/`b_stat` exactly. The posterior mean
    /// `(a0 + sum)/(b0 + n)` is the right *estimate* but the wrong *carrier* —
    /// its prior shrinkage (1.85× at `sum = 1, n = 12`) gets re-ingested as if
    /// it were data, and a second posterior forms around an already-shrunk
    /// value.
    #[must_use]
    pub fn evidence_mean(&self, row: usize, col: usize) -> f32 {
        let a = self.a_stat[(row, col)] - self.a0;
        let b = self.b_stat[(row, col)] - self.b0;
        if a > 0.0 && b > 0.0 {
            a / b
        } else {
            0.0
        }
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
