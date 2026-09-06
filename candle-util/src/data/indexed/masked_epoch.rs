//! Per-epoch masked loader: every draw the masked trainer needs, made once
//! per epoch on the host and kept on the device as one shuffled block.
//!
//! The packed context of a pseudobulk row — its top-K gene ids, values, batch
//! null and gene mean — is the same bytes every epoch; only the row order and
//! the mask change. So the packed tables are uploaded ONCE per level and every
//! epoch is a single `index_select` of the shuffled row ids plus `narrow`
//! views per minibatch, the way the dense loader works. The context mask, the
//! query set, and the target counts at the context and query genes are drawn
//! per SOURCE ROW from the epoch seed, so a row's mask does not depend on the
//! batch size or the shuffle, and each row sees a fresh mask every epoch
//! (dynamic masking: a row is visited once per epoch, so this is the same
//! distribution as a fresh mask per step, drawn where it costs nothing).
//!
//! The `[P, D]` target matrix stays on the host: what the device gets is the
//! target gathered at the K context slots and the Q query genes of each row.

use super::pack::pack_at_indices;
use super::{IndexedInMemoryData, IndexedMinibatchData, IndexedSample};
use crate::data::loader_util::bootstrap_indices;
use crate::value_transform::anscombe_lite;
use candle_core::{Device, Tensor};
use matrix_util::rand_util::mix_seed;
use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;

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
    /// `Some(extra)`: build a query set per row — every masked real context
    /// gene plus `extra` genes drawn uniformly outside the row's whole
    /// context, zeros included. `None`: no query set.
    pub query_extra: Option<usize>,
}

/// One row's draws for the epoch.
struct RowDraw {
    /// `[K]` 1 on a visible real slot, 0 on masked and pad slots.
    visible: Vec<f32>,
    /// Query gene ids (masked context genes first, then extras).
    queries: Vec<u32>,
}

/// Draw one row: the mask over its real slots and, when asked, its query set.
/// Seeded on `(epoch_seed, row)` alone.
fn draw_row(
    sample: &IndexedSample,
    row: usize,
    k: usize,
    n_features: usize,
    epoch_seed: u64,
    draw: &MaskedDraw,
) -> RowDraw {
    let mut rng = SmallRng::seed_from_u64(mix_seed(epoch_seed, row as u64));
    let rate = match draw.schedule {
        MaskSchedule::Fixed => draw.mask_fraction,
        MaskSchedule::Uniform { lo, hi } => lo + (hi - lo) * rng.random::<f64>(),
    };
    let real = sample.indices.len().min(k);
    let mut visible = vec![0f32; k];
    let mut queries = Vec::new();
    for ((&gene, &value), vis) in sample.indices[..real]
        .iter()
        .zip(&sample.values[..real])
        .zip(visible.iter_mut())
    {
        // A real slot carries a positive value; a zero here is a pad, which is
        // neither visible nor a target.
        if value <= 0.0 {
            continue;
        }
        if rng.random::<f64>() < rate {
            queries.push(gene);
        } else {
            *vis = 1.0;
        }
    }
    match draw.query_extra {
        Some(extra) => {
            let context: HashSet<u32> = sample.indices[..real]
                .iter()
                .zip(&sample.values[..real])
                .filter(|(_, &v)| v > 0.0)
                .map(|(&g, _)| g)
                .collect();
            let want = extra.min(n_features.saturating_sub(context.len()));
            let mut drawn: HashSet<u32> = HashSet::with_capacity(want);
            let mut tries = 0usize;
            while drawn.len() < want && tries < 64 * want.max(1) {
                tries += 1;
                let g = rng.random_range(0..n_features) as u32;
                if !context.contains(&g) {
                    drawn.insert(g);
                }
            }
            let mut extras: Vec<u32> = drawn.into_iter().collect();
            extras.sort_unstable();
            queries.extend(extras);
        }
        None => queries.clear(),
    }
    RowDraw { visible, queries }
}

////////////////////////////
// Device-resident level  //
////////////////////////////

/// One level's packed context, resident on the device, plus the host samples
/// the per-epoch draws read.
pub struct MaskedLevelData {
    samples: Vec<IndexedSample>,
    n_features: usize,
    k: usize,
    indices_pk: Tensor,
    values_pk: Tensor,
    null_pk: Option<Tensor>,
    mean_pk: Option<Tensor>,
    /// The encoder's Anscombe gate, computed once: `anscombe_lite(values, null, mean)`.
    gate_pk: Tensor,
    /// Target counts at the context slots, 0 on pads.
    target_at_context_pk: Tensor,
    dev: Device,
}

/// Query genes of one minibatch.
pub struct QueryBatch {
    /// `[N, Q]` u32, gene 0 on pads.
    pub ids: Tensor,
    /// `[N, Q]` 1 on a real query, 0 on a pad.
    pub weight: Tensor,
    /// `[N, Q]` target counts at the query genes, 0 on pads.
    pub target: Tensor,
}

/// One minibatch: the packed context views plus this epoch's draws.
pub struct MaskedMinibatch {
    pub base: IndexedMinibatchData,
    /// `[N, K]` the encoder's gate per slot.
    pub gate: Tensor,
    /// `[N, K]` 1 on a visible real slot.
    pub visible: Tensor,
    /// `[N, K]` target counts at the context slots.
    pub target_at_context: Tensor,
    pub query: Option<QueryBatch>,
}

/// One epoch's minibatches and their host-side counts.
pub struct MaskedEpoch {
    pub batches: Vec<MaskedMinibatch>,
    /// Per batch: number of real queries (Σ weight).
    pub n_queries: Vec<f32>,
}

impl IndexedInMemoryData {
    /// Upload this level's packed context once. `target` is the level's
    /// `[P, D]` target rows (batch-free where a batch-aware collapse ran).
    pub fn to_device_resident(
        &self,
        target: &Mat,
        dev: &Device,
    ) -> anyhow::Result<MaskedLevelData> {
        let p = self.num_data();
        let k = self.input_context_size;
        anyhow::ensure!(
            target.nrows() == p && target.ncols() == self.n_input_features,
            "target rows {}×{} do not match the loader's {} samples × {} features",
            target.nrows(),
            target.ncols(),
            p,
            self.n_input_features
        );
        let all: Vec<usize> = (0..p).collect();
        let host = self.build_minibatch(&all, &Device::Cpu)?;
        let gate = anscombe_lite(
            &host.input_values,
            host.input_values_null.as_ref(),
            host.input_values_mean.as_ref(),
        )?;
        let target_at_context = pack_at_indices(
            &self.input_samples,
            &all,
            k,
            &Device::Cpu,
            |_, kk, si, feat| {
                if self.input_samples[si].values[kk] > 0.0 {
                    target[(si, feat as usize)]
                } else {
                    0.0
                }
            },
        )?;
        let up = |t: &Tensor| t.to_device(dev);
        Ok(MaskedLevelData {
            samples: self.input_samples.clone(),
            n_features: self.n_input_features,
            k,
            indices_pk: up(&host.input_indices)?,
            values_pk: up(&host.input_values)?,
            null_pk: host.input_values_null.as_ref().map(up).transpose()?,
            mean_pk: host.input_values_mean.as_ref().map(up).transpose()?,
            gate_pk: up(&gate)?,
            target_at_context_pk: up(&target_at_context)?,
            dev: dev.clone(),
        })
    }
}

impl MaskedLevelData {
    #[must_use]
    pub fn num_data(&self) -> usize {
        self.samples.len()
    }

    /// A fresh epoch: every row once in a random order, the last minibatch
    /// padded by resampling, with this epoch's draws for every row. (The
    /// host-side loader draws a bootstrap cover instead — some rows twice,
    /// some not at all — which is why "a fresh mask per epoch" is only the
    /// same distribution as "a fresh mask per step" once every row is visited
    /// once.)
    pub fn begin_epoch(
        &self,
        target: &Mat,
        epoch_seed: u64,
        draw: &MaskedDraw,
        batch_size: usize,
    ) -> anyhow::Result<MaskedEpoch> {
        let n = self.num_data();
        anyhow::ensure!(n > 0, "begin_epoch on an empty level");
        anyhow::ensure!(batch_size > 0, "batch_size must be > 0");
        let nbatch = n.div_ceil(batch_size);
        let ntot = nbatch * batch_size;
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.shuffle(&mut rand::rng());
        idx.extend(bootstrap_indices::<u32>(n, ntot - n));
        self.epoch_from_indices(target, &idx, batch_size, epoch_seed, draw)
    }

    /// One minibatch of exactly `n` rows, cycling over the level, with draws —
    /// for the GPU memory probe, which must see the shapes a real step retains.
    pub fn probe_minibatch(
        &self,
        target: &Mat,
        n: usize,
        epoch_seed: u64,
        draw: &MaskedDraw,
    ) -> anyhow::Result<MaskedMinibatch> {
        let p = self.num_data();
        anyhow::ensure!(p > 0, "probe_minibatch on an empty level");
        let idx: Vec<u32> = (0..n).map(|i| (i % p) as u32).collect();
        let mut ep = self.epoch_from_indices(target, &idx, n, epoch_seed, draw)?;
        Ok(ep.batches.remove(0))
    }

    fn epoch_from_indices(
        &self,
        target: &Mat,
        idx: &[u32],
        batch_size: usize,
        epoch_seed: u64,
        draw: &MaskedDraw,
    ) -> anyhow::Result<MaskedEpoch> {
        let p = self.num_data();
        let k = self.k;
        anyhow::ensure!(
            target.nrows() == p && target.ncols() == self.n_features,
            "target rows do not match the level"
        );
        // Per-row draws over every source row, in parallel, keyed on the row.
        let rows: Vec<RowDraw> = (0..p)
            .into_par_iter()
            .map(|r| draw_row(&self.samples[r], r, k, self.n_features, epoch_seed, draw))
            .collect();
        let mut visible = vec![0f32; p * k];
        for (r, rd) in rows.iter().enumerate() {
            visible[r * k..(r + 1) * k].copy_from_slice(&rd.visible);
        }
        let cpu = Device::Cpu;
        let visible_pk = Tensor::from_vec(visible, (p, k), &cpu)?.to_device(&self.dev)?;

        let query_pq = match draw.query_extra {
            Some(_) => {
                let q = rows
                    .iter()
                    .map(|rd| rd.queries.len())
                    .max()
                    .unwrap_or(0)
                    .max(1);
                let mut ids = vec![0u32; p * q];
                let mut weight = vec![0f32; p * q];
                let mut tgt = vec![0f32; p * q];
                // One row per worker: the target reads jump across columns,
                // so this is the loader's cache-bound loop.
                ids.par_chunks_mut(q)
                    .zip(weight.par_chunks_mut(q))
                    .zip(tgt.par_chunks_mut(q))
                    .zip(rows.par_iter())
                    .enumerate()
                    .for_each(|(r, (((id, w), t), rd))| {
                        for (j, &g) in rd.queries.iter().enumerate() {
                            id[j] = g;
                            w[j] = 1.0;
                            t[j] = target[(r, g as usize)];
                        }
                    });
                Some((
                    Tensor::from_vec(ids, (p, q), &cpu)?.to_device(&self.dev)?,
                    Tensor::from_vec(weight, (p, q), &cpu)?.to_device(&self.dev)?,
                    Tensor::from_vec(tgt, (p, q), &cpu)?.to_device(&self.dev)?,
                ))
            }
            None => None,
        };

        // One shuffled block per table; minibatches are views into it.
        let ntot = idx.len();
        let idx_t = Tensor::from_vec(idx.to_vec(), ntot, &self.dev)?;
        let sel = |t: &Tensor| t.index_select(&idx_t, 0);
        let s_indices = sel(&self.indices_pk)?;
        let s_values = sel(&self.values_pk)?;
        let s_null = self.null_pk.as_ref().map(sel).transpose()?;
        let s_mean = self.mean_pk.as_ref().map(sel).transpose()?;
        let s_gate = sel(&self.gate_pk)?;
        let s_tac = sel(&self.target_at_context_pk)?;
        let s_vis = sel(&visible_pk)?;
        let s_query = match query_pq.as_ref() {
            Some((i, w, t)) => Some((sel(i)?, sel(w)?, sel(t)?)),
            None => None,
        };

        let nbatch = ntot.div_ceil(batch_size);
        let mut batches = Vec::with_capacity(nbatch);
        let mut n_queries = Vec::with_capacity(nbatch);
        for b in 0..nbatch {
            let start = b * batch_size;
            let len = batch_size.min(ntot - start);
            let cut = |t: &Tensor| t.narrow(0, start, len);
            let rows_b = &idx[start..start + len];
            n_queries.push(
                rows_b
                    .iter()
                    .map(|&r| rows[r as usize].queries.len() as f32)
                    .sum(),
            );
            let query = match s_query.as_ref() {
                Some((i, w, t)) => Some(QueryBatch {
                    ids: cut(i)?,
                    weight: cut(w)?,
                    target: cut(t)?,
                }),
                None => None,
            };
            batches.push(MaskedMinibatch {
                base: IndexedMinibatchData {
                    row_ids: cut(&idx_t)?,
                    input_indices: cut(&s_indices)?,
                    input_values: cut(&s_values)?,
                    input_values_null: s_null.as_ref().map(cut).transpose()?,
                    input_values_mean: s_mean.as_ref().map(cut).transpose()?,
                },
                gate: cut(&s_gate)?,
                visible: cut(&s_vis)?,
                target_at_context: cut(&s_tac)?,
                query,
            });
        }
        Ok(MaskedEpoch { batches, n_queries })
    }
}

#[cfg(test)]
#[path = "masked_epoch_tests.rs"]
mod masked_epoch_tests;
