//! Per-epoch masked loader with no context window: the encoder reads the dense
//! `[N, D]` minibatch rows over every gene, zeros included.
//!
//! The window existed because the pool formed an `[N, K, H]` block, and `K`
//! sized its backward. Once the pool is one `[N, D]` product
//! ([`crate::encoder::dense_pool`]) the window buys nothing, so the top-K
//! packing, the shortlist that scored it, and the padding it needed all go.
//!
//! What is left is one draw per row per epoch, over the WHOLE gene axis:
//!
//! - the encoder sees `visible_nd`,
//! - the decoder is scored on `1 − visible_nd`,
//!
//! from the same tensor, so "the hidden set" has exactly one meaning. The draw
//! is keyed on `(epoch seed, source row)` alone — not on the batch size, the
//! shuffle, or which worker took the row — so a seeded run reproduces whatever
//! the machine does.
//!
//! The level's rows are uploaded once and a minibatch is an `index_select` of
//! them, the way the packed loader worked; only the width changed.

use crate::data::loader_util::{bootstrap_indices, upload_to_device};
use candle_core::{Device, Tensor};
use matrix_util::rand_util::mix_seed;
use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use rayon::prelude::*;

type Mat = DMatrix<f32>;

////////////////
// Mask draws //
////////////////

/// Per-row mask-rate schedule (any-order / absorbing-diffusion style).
#[derive(Clone, Copy, Debug)]
pub enum MaskSchedule {
    /// Constant mask fraction.
    Fixed,
    /// Draw the rate uniformly in `[lo, hi]` per row per epoch.
    Uniform { lo: f64, hi: f64 },
}

/// What one epoch draws for every row.
#[derive(Clone, Copy, Debug)]
pub struct MaskedDraw {
    pub schedule: MaskSchedule,
    /// The mask fraction under [`MaskSchedule::Fixed`].
    pub mask_fraction: f64,
}

/// One row's draw for the epoch: its visible mask over `[D]`.
struct RowDraw {
    visible: Vec<f32>,
}

/// Draw one row. Seeded on `(epoch_seed, row)` alone.
fn draw_row(row: usize, n_features: usize, epoch_seed: u64, draw: &MaskedDraw) -> RowDraw {
    let mut rng = SmallRng::seed_from_u64(mix_seed(epoch_seed, row as u64));
    let rate = match draw.schedule {
        MaskSchedule::Fixed => draw.mask_fraction,
        MaskSchedule::Uniform { lo, hi } => lo + (hi - lo) * rng.random::<f64>(),
    };
    // Every gene is a candidate: a zero-count gene carries information about
    // the cell, and the decoder has always been scored on it.
    let visible: Vec<f32> = (0..n_features)
        .map(|_| f32::from(u8::from(rng.random::<f64>() >= rate)))
        .collect();
    RowDraw { visible }
}

////////////////////////////
// Device-resident level  //
////////////////////////////

/// One level's dense rows, resident on the device.
///
/// `input` is what the encoder reads (the epoch's Poisson draw where thinning
/// is on, exactly as the packed loader was rebuilt per epoch); `target` is the
/// batch-free row the decoder is scored against.
pub struct DenseMaskedLevel {
    n_features: usize,
    p: usize,
    input_pd: Tensor,
    null_pd: Option<Tensor>,
    target_pd: Tensor,
    /// `[1, D]` per-gene mean rate — the count-rate divisor, shared by rows.
    mean_1d: Tensor,
    dev: Device,
}

/// One minibatch: the dense rows plus this epoch's mask.
pub struct DenseMaskedMinibatch {
    /// `[N]` u32 source row of each minibatch row.
    pub row_ids: Tensor,
    /// `[N, D]` encoder input values.
    pub x_nd: Tensor,
    /// `[N, D]` per-row batch null, when the level has one.
    pub x0_nd: Option<Tensor>,
    /// `[N, D]` 1 where the encoder may look.
    pub visible_nd: Tensor,
    /// `[N, D]` decoder target counts.
    pub target_nd: Tensor,
}

impl DenseMaskedLevel {
    /// Upload one level's rows. `input`, `target` are `[P, D]`; `null` is the
    /// per-row batch null at the same shape; `mean` is the per-gene rate `[D]`.
    pub fn from_mats(
        input: &Mat,
        null: Option<&Mat>,
        target: &Mat,
        mean: &[f32],
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (p, d) = (input.nrows(), input.ncols());
        anyhow::ensure!(
            target.nrows() == p && target.ncols() == d,
            "target rows {}×{} do not match the input's {p}×{d}",
            target.nrows(),
            target.ncols()
        );
        if let Some(n) = null {
            anyhow::ensure!(
                n.nrows() == p && n.ncols() == d,
                "batch null is {}×{}, expected {p}×{d}",
                n.nrows(),
                n.ncols()
            );
        }
        anyhow::ensure!(
            mean.len() == d,
            "per-gene mean has {} entries, expected {d}",
            mean.len()
        );
        Ok(Self {
            n_features: d,
            p,
            input_pd: upload_to_device(input, dev)?,
            null_pd: null.map(|n| upload_to_device(n, dev)).transpose()?,
            target_pd: upload_to_device(target, dev)?,
            mean_1d: Tensor::from_vec(mean.to_vec(), (1, d), dev)?,
            dev: dev.clone(),
        })
    }

    #[must_use]
    pub fn num_data(&self) -> usize {
        self.p
    }

    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// `[1, D]` per-gene mean rate, the encoder's count-rate divisor.
    #[must_use]
    pub fn feature_mean_1d(&self) -> &Tensor {
        &self.mean_1d
    }

    /// A fresh epoch: every row once in a random order, the last minibatch
    /// padded by resampling, with this epoch's draws for every row.
    pub fn begin_epoch(
        &self,
        epoch_seed: u64,
        draw: &MaskedDraw,
        batch_size: usize,
    ) -> anyhow::Result<DenseMaskedEpoch<'_>> {
        anyhow::ensure!(self.p > 0, "begin_epoch on an empty level");
        anyhow::ensure!(batch_size > 0, "batch_size must be > 0");
        let nbatch = self.p.div_ceil(batch_size);
        let ntot = nbatch * batch_size;
        let mut order: Vec<u32> = (0..self.p as u32).collect();
        order.shuffle(&mut rand::rng());
        order.extend(bootstrap_indices::<u32>(self.p, ntot - self.p));
        Ok(DenseMaskedEpoch {
            level: self,
            order,
            batch_size,
            epoch_seed,
            draw: *draw,
        })
    }

    /// One minibatch of exactly `n` rows, cycling over the level — for the GPU
    /// memory probe, which must see the shapes a real step retains.
    pub fn probe_minibatch(
        &self,
        n: usize,
        epoch_seed: u64,
        draw: &MaskedDraw,
    ) -> anyhow::Result<DenseMaskedMinibatch> {
        anyhow::ensure!(self.p > 0, "probe_minibatch on an empty level");
        let order: Vec<u32> = (0..n).map(|i| (i % self.p) as u32).collect();
        DenseMaskedEpoch {
            level: self,
            order,
            batch_size: n,
            epoch_seed,
            draw: *draw,
        }
        .batch(0)
    }
}

/// One epoch's row order plus the draw it replays. Minibatches are built on
/// demand: the dense rows are `[N, D]`, so materializing every batch up front
/// would hold the whole level a second time for no gain.
pub struct DenseMaskedEpoch<'a> {
    level: &'a DenseMaskedLevel,
    order: Vec<u32>,
    batch_size: usize,
    epoch_seed: u64,
    draw: MaskedDraw,
}

impl DenseMaskedEpoch<'_> {
    #[must_use]
    pub fn n_batches(&self) -> usize {
        self.order.len().div_ceil(self.batch_size)
    }

    /// Build minibatch `b`: `index_select` the resident rows and draw the mask
    /// for exactly those source rows.
    pub fn batch(&self, b: usize) -> anyhow::Result<DenseMaskedMinibatch> {
        let lv = self.level;
        let start = b * self.batch_size;
        anyhow::ensure!(start < self.order.len(), "batch {b} is past the epoch");
        let len = self.batch_size.min(self.order.len() - start);
        let ids = &self.order[start..start + len];
        let d = lv.n_features;

        // Keyed on the SOURCE row, so a row's mask is the same whichever batch
        // it lands in and whichever worker draws it.
        let draws: Vec<RowDraw> = ids
            .par_iter()
            .map(|&r| draw_row(r as usize, d, self.epoch_seed, &self.draw))
            .collect();

        let mut vis = vec![0f32; len * d];
        for (row, rd) in draws.iter().enumerate() {
            vis[row * d..(row + 1) * d].copy_from_slice(&rd.visible);
        }
        let cpu = Device::Cpu;
        let visible_nd = Tensor::from_vec(vis, (len, d), &cpu)?.to_device(&lv.dev)?;

        let row_ids = Tensor::from_vec(ids.to_vec(), len, &lv.dev)?;
        let sel = |t: &Tensor| t.index_select(&row_ids, 0);
        let x_nd = sel(&lv.input_pd)?;
        let x0_nd = lv.null_pd.as_ref().map(sel).transpose()?;
        let target_nd = sel(&lv.target_pd)?;

        Ok(DenseMaskedMinibatch {
            row_ids,
            x_nd,
            x0_nd,
            visible_nd,
            target_nd,
        })
    }
}

#[cfg(test)]
#[path = "masked_dense_tests.rs"]
mod masked_dense_tests;
