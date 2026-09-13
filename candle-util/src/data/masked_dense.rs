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
//! - the decoder is scored at `hidden_ids`, which is where `visible_nd` is 0,
//!
//! from one draw, so "the hidden set" has exactly one meaning. The count is
//! FIXED at `round(rate · D)` rather than one Bernoulli per gene, so the hidden
//! set is a `[N, d_h]` block the decoder's head can gather by instead of a
//! ragged list it has to mask over. The draw is keyed on `(epoch seed, source
//! row)` alone — not on the batch size, the shuffle, or which worker took the
//! row — so a seeded run reproduces whatever the machine does.
//!
//! The level's rows are uploaded once and a minibatch is an `index_select` of
//! them, the way the packed loader worked; only the width changed. The upload
//! takes the pseudobulk posterior in its native `[D, P]` layout and transposes
//! it nowhere — see [`crate::data::loader_util::upload_columns_as_rows`].

use crate::data::loader_util::{bootstrap_indices, upload_columns_as_rows};
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

/// One row's draw for the epoch: the ids it hides, ascending and without
/// repeats.
///
/// The visible mask is not carried alongside them. It is the SAME draw read
/// the other way — `visible[g] == 0` exactly at `hidden` — so it is derived
/// from the id block once it is on the device (see [`visible_from_hidden`]),
/// not filled `D`-long per row on the host and uploaded again every step.
type RowDraw = Vec<u32>;

/// How many genes a row of `n_features` hides at `rate`: `round(rate · D)`.
///
/// The count is FIXED rather than binomial, which is what lets a minibatch
/// carry the hidden set as a `[N, d_h]` id block instead of a ragged list — the
/// decoder then evaluates its likelihood only where it is scored.
///
/// The rate must lie in the OPEN interval (0, 1) and the count it implies must
/// leave the encoder something to read and the decoder something to answer for.
/// The CLI guarantees the interval — `MaskedTopicArgs::validate` refuses 0 and
/// 1 by name, for `--mask-fraction` and for the uniform schedule's bounds — so
/// what stands here is a debug assertion, not a clamp: a clamp answers a
/// degenerate rate with a one-gene draw the caller never asked for, and does it
/// silently.
fn hidden_count(n_features: usize, rate: f64) -> usize {
    debug_assert!(
        n_features >= 2,
        "a {n_features}-gene axis cannot be split into a visible and a hidden part"
    );
    debug_assert!(
        rate > 0.0 && rate < 1.0,
        "mask rate {rate} is outside the open interval (0, 1); the CLI refuses this"
    );
    // Rounding a VALID rate can still land on an edge of a small axis: 0.4 over
    // two genes rounds to one, over one gene to zero, and 0.9 over five rounds
    // to all five. That is a property of rounding, not a degenerate request,
    // so it is bounded here in release builds too: at least one gene hidden,
    // at least one left visible.
    let m = (rate * n_features as f64).round() as usize;
    m.clamp(1, n_features - 1)
}

/// `m` distinct ids from `0..d`, by Floyd's algorithm: `m` draws, not a
/// `d`-long permutation, so the cost is the size of the sample rather than the
/// size of the gene axis.
fn floyd_sample(d: usize, m: usize, rng: &mut SmallRng) -> Vec<u32> {
    let mut seen = std::collections::HashSet::with_capacity(m);
    let mut out = Vec::with_capacity(m);
    for j in (d - m)..d {
        let t = rng.random_range(0..=j);
        let pick = if seen.insert(t) {
            t
        } else {
            seen.insert(j);
            j
        };
        out.push(pick as u32);
    }
    out.sort_unstable();
    out
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
    floyd_sample(n_features, hidden_count(n_features, rate), &mut rng)
}

/// `[N, D]` visible mask from `hidden_ids [N, d_h]`: ones with a zero scattered
/// at every hidden id, built ON the device the ids already live on.
///
/// `scatter` OVERWRITES, which is what a repeat needs: the `Uniform` schedule
/// pads a short row with its own last hidden id, and a `scatter_add` of `−1`
/// would drive that gene to `−1` instead of leaving it hidden. The ones and the
/// zeros are freshly built constants, so nothing here joins the gradient graph
/// — the mask multiplies the encoder's scores, it is not learned.
fn visible_from_hidden(hidden_ids: &Tensor, n_features: usize) -> candle_core::Result<Tensor> {
    let (n, dh) = hidden_ids.dims2()?;
    let dev = hidden_ids.device();
    let ones = Tensor::ones((n, n_features), candle_core::DType::F32, dev)?;
    let zeros = Tensor::zeros((n, dh), candle_core::DType::F32, dev)?;
    ones.scatter(hidden_ids, &zeros, 1)
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
    /// `[N, d_h]` u32 hidden gene ids, ascending within a row — the same draw
    /// as `visible_nd`, in the form the decoder's head gathers by.
    ///
    /// Under [`MaskSchedule::Fixed`] every row hides the same `d_h` and the
    /// block is exact as it stands. Under [`MaskSchedule::Uniform`] the count
    /// is per-row, so the block is as wide as the widest row and a short row's
    /// tail repeats its own last hidden id — a real hidden gene, so a gather is
    /// in range — with `hidden_weight` zero there.
    pub hidden_ids: Tensor,
    /// `[N, d_h]` 1 on a drawn slot, 0 on a pad. `None` when every row in the
    /// batch hides exactly `d_h`, which is the whole of the fixed schedule: the
    /// head then multiplies by nothing.
    pub hidden_weight: Option<Tensor>,
    /// `[N, D]` decoder target counts.
    pub target_nd: Tensor,
}

impl DenseMaskedLevel {
    /// Upload one level's rows. `input`, `target` are **`[D, P]`** — genes down,
    /// pseudobulk samples across, the layout the collapsed posterior is sampled
    /// in; `null` is the batch null at the same shape; `mean` is the per-gene
    /// rate `[D]`. Each becomes the `[P, D]` resident tensor with no transpose
    /// on either side of the upload.
    pub fn from_mats(
        input_dp: &Mat,
        null_dp: Option<&Mat>,
        target_dp: &Mat,
        mean: &[f32],
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (d, p) = (input_dp.nrows(), input_dp.ncols());
        anyhow::ensure!(
            target_dp.nrows() == d && target_dp.ncols() == p,
            "target rows {}×{} do not match the input's {d}×{p} (genes × samples)",
            target_dp.nrows(),
            target_dp.ncols()
        );
        if let Some(n) = null_dp {
            anyhow::ensure!(
                n.nrows() == d && n.ncols() == p,
                "batch null is {}×{}, expected {d}×{p} (genes × samples)",
                n.nrows(),
                n.ncols()
            );
        }
        anyhow::ensure!(
            mean.len() == d,
            "per-gene mean has {} entries, expected {d}",
            mean.len()
        );
        // With no batch-adjusted target the caller hands the SAME matrix in as
        // both — the identity the Poisson-thinning branch already tests for —
        // and uploading it twice holds two resident `[P, D]` copies of one set
        // of numbers. `Tensor` is `Arc`-backed, so the reuse is a pointer copy.
        let input_pd = upload_columns_as_rows(input_dp, dev)?;
        let target_pd = if std::ptr::eq(input_dp, target_dp) {
            input_pd.clone()
        } else {
            upload_columns_as_rows(target_dp, dev)?
        };
        Ok(Self {
            n_features: d,
            p,
            input_pd,
            null_pd: null_dp
                .map(|n| upload_columns_as_rows(n, dev))
                .transpose()?,
            target_pd,
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

        // The draw as an id block. Ragged only under `Uniform`, where the short
        // rows are padded with their own last hidden id and weighted 0.
        let cpu = Device::Cpu;
        let dh = draws.iter().map(Vec::len).max().unwrap_or(1);
        let ragged = draws.iter().any(|rd| rd.len() != dh);
        let mut hid = vec![0u32; len * dh];
        let mut w = vec![0f32; len * dh];
        for (row, rd) in draws.iter().enumerate() {
            let n_hid = rd.len();
            let slot = &mut hid[row * dh..(row + 1) * dh];
            slot[..n_hid].copy_from_slice(rd);
            slot[n_hid..].fill(rd[n_hid - 1]);
            w[row * dh..row * dh + n_hid].fill(1.0);
        }
        let hidden_ids = Tensor::from_vec(hid, (len, dh), &cpu)?.to_device(&lv.dev)?;
        // …and the visible mask FROM that block, on the device. The `[N, D]`
        // host buffer this used to fill and upload every step said nothing the
        // `[N, d_h]` ids do not already say.
        let visible_nd = visible_from_hidden(&hidden_ids, d)?;
        let hidden_weight = if ragged {
            Some(Tensor::from_vec(w, (len, dh), &cpu)?.to_device(&lv.dev)?)
        } else {
            None
        };

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
            hidden_ids,
            hidden_weight,
            target_nd,
        })
    }
}

#[cfg(test)]
#[path = "masked_dense_tests.rs"]
mod masked_dense_tests;
