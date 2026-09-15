//! Phase 2 by **distilled pooled-gene encoders**: instead of re-fitting every
//! cell against the frozen dictionary by block SGD ([`super::block_sgd`]),
//! train a small encoder to reproduce the phase-1 pseudobulk embeddings from
//! their members' counts, then encode every cell in one forward pass.
//!
//! One encoder per COUNT track of the feature axis ([`tracks`]), each pooling
//! only its own track's rows; a cell's placement is the mean over the tracks it
//! has counts on. A one-track axis — `senna bge` — has exactly one, and that
//! path is the previous single-dictionary code op for op.
//!
//! # Why this is the right object
//!
//! With the feature side frozen, the phase-2 gradient is `(μ − N)·Ẽ`. Its data
//! half `N·Ẽ` is the count-weighted sum of the gene embeddings — exactly what
//! [`candle_util::encoder::PooledGeneEncoder`] pools — so the encoder carries the
//! sufficient statistic of the data term and only the partition term needs the
//! FC nonlinearity. The block SGD never converged on real fits (every block at
//! its step cap), so the encoder is compared against an under-converged solve,
//! not an exact one.
//!
//! # Training pairs
//!
//! Every level's fitted table `e_pb [n_pb × H]` is the target; the input is a
//! **random subset of the pseudobulk's member cells**, their batch-folded counts
//! averaged. The subset size is log-uniform in `[1, |members|]`, so a size-one
//! subset is a single cell: the shift from pseudobulk depth to cell depth is in
//! the training set by construction, not hoped away. A seeded tenth of the
//! pseudobulks per level is held out and scored on its full membership.
//!
//! # Then the cells themselves
//!
//! The tables are a few hundred anchors, so a trunk distilled on them places
//! twenty thousand cells as a smooth function of those anchors: one sheet per
//! lineage where a per-cell fit resolves states. The distilled trunk is
//! therefore only the warm start. [`refine`] then trains it on the cells'
//! own likelihood — the phase-2 objective itself, the full log-partition over
//! every gene with the intercept profiled out, `N_c·lse_f(s_cf) − Σ_f n_cf·s_cf`
//! plus the ridge, SUMMED over every track present (a track with no encoder
//! still constrains the placement) — one `[B,H]·[H,D]` matmul per track per
//! step, so no negatives are sampled and nothing is approximated. The block
//! SGD solved that objective per cell from a cold start and never converged;
//! here every cell's gradient improves one small map shared by every cell like
//! it.
//!
//! # What the cell gets
//!
//! `θ_c` from the refined encoder is the warm start, and the block SGD
//! ([`super::block_sgd::polish_cells`]) finishes it on the exact per-cell
//! objective under a short step cap: a shared map of any width places a cell
//! as a function of its neighbours, and the last fraction of the likelihood is
//! per cell. Started near the optimum of a convex problem, that costs a
//! quarter of the cold budget. The intercept comes out of the same solve. The
//! gauge fold is the shared tail in [`super::cells`].
//!
//! # One estimator, both halves
//!
//! The trained trunks leave the fit as [`CellEncoders`], which the run persists
//! — one safetensors per track, the one-track file unchanged — and `senna
//! predict` reloads, so a query cell is placed by the same map the run's own
//! cells were: the invariant [`super::FrozenProjector`] states for the SGD
//! path.

use super::block_sgd::{self, Phase2Input, Phase2Out};
use super::{cell_edges, CellBatchFold, FrozenProjection};
use crate::fit::config::TrackSpec;
use crate::progress::new_progress_bar;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_util::encoder::{PooledGeneEncoder, PooledGeneEncoderArgs};
use candle_util::feature_embedding::FeatureEmbedding;
use log::info;
use matrix_util::rand_util::mix_seed;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::borrow::Cow;

mod tracks;

pub(crate) use tracks::{split_rows_by_track, TrackRows};
pub use tracks::{CellEncoders, TrackEncoder};

///////////////
// Constants //
///////////////

/// Width of the trunk between the pool and the head. One hidden layer: the
/// pairs number in the low thousands, so a deep stack would fit the tables'
/// noise rather than the map.
const TRUNK_WIDTH: usize = 128;
/// Passes over the training pseudobulks; each pass redraws every subset. On
/// real fits the held-out error stops moving by the twentieth pass — the
/// tables are noisy targets and the map is small — so the budget is short.
const EPOCHS: usize = 40;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f64 = 1e-4;
const GRAD_CLIP: f64 = 5.0;
/// Pseudobulks per optimizer step.
const ROWS_PER_STEP: usize = 128;
/// Passes over every cell in the likelihood refinement.
const REFINE_EPOCHS: usize = 10;
/// Cells per refinement step: the dense block is `REFINE_CELLS_PER_STEP × D`.
const REFINE_CELLS_PER_STEP: usize = 256;
/// A warm start, so a fraction of the distillation rate.
const REFINE_LEARNING_RATE: f64 = 3e-4;
const REFINE_REPORT_EVERY: usize = 2;
/// Share of each level's pseudobulks held out of training and scored on their
/// full membership.
const HOLDOUT_FRACTION: f64 = 0.1;
const REPORT_EVERY: usize = 20;
/// Sub-stream tag for a NON-base track's distillation seed. Track 0 keeps the
/// fit's own seed, so a one-track (`senna bge`) run draws exactly the stream it
/// drew before tracks existed; every further track mixes this tag with its own
/// index so no two tracks share an init.
const TRACK_SEED_TAG: u64 = 0x5452_4143_4b00;
/// The var-name prefix the trunk is saved under.
const VAR_PREFIX: &str = "cell_enc";
/// The per-gene mean's tensor name inside the saved file.
const MEAN_TENSOR: &str = "cell_enc.feature_mean";

//////////////////
// Host structs //
//////////////////

/// One cell's batch-folded sparse counts.
#[derive(Clone)]
pub(crate) struct FoldedRow {
    pub feats: Vec<u32>,
    pub counts: Vec<f32>,
}

impl FoldedRow {
    pub(crate) fn new(feats: Vec<u32>, counts: Vec<f32>) -> Self {
        Self { feats, counts }
    }

    fn as_slices(&self) -> (&[u32], &[f32]) {
        (&self.feats, &self.counts)
    }
}

/// A level's targets: the fitted table and, per pseudobulk, its members as
/// positions into the folded rows.
pub(crate) struct DistillTargets<'a> {
    pub e_pb: &'a DMatrix<f32>,
    pub groups: &'a [Vec<usize>],
}

/// One collapse level as the caller holds it.
pub(crate) struct DistillLevel<'a> {
    /// `[n_pb × H]` phase-1 table.
    pub e_pb: &'a DMatrix<f32>,
    /// Global cell id → pseudobulk.
    pub cell_to_pb: &'a [usize],
}

pub(crate) struct DistillSpec<'a> {
    pub levels: &'a [DistillLevel<'a>],
    pub seed: u64,
}

/// The frozen dictionary on the device: the table for the pool, its
/// transpose for the intercept, and the bias row.
///
/// `Clone` is a refcount bump on both tensors and on the shared
/// [`FeatureEmbedding`] (candle's `Tensor` is `Arc` inside), so a track's
/// dictionary can be handed to its encoder and still be scored against by the
/// refinement without a second upload.
#[derive(Clone)]
pub(crate) struct FrozenDict {
    features: std::sync::Arc<FeatureEmbedding>,
    e_hd: Tensor,
    b_1d: Tensor,
    h: usize,
    d: usize,
}

impl FrozenDict {
    pub(crate) fn new(
        feat: &[f32],
        b_feat: &[f32],
        h: usize,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let d = b_feat.len();
        anyhow::ensure!(
            feat.len() == d * h,
            "dictionary has {} entries, expected {d} × {h}",
            feat.len()
        );
        let e_dh = Tensor::from_slice(feat, (d, h), dev)?;
        Self::assemble(e_dh, Tensor::from_slice(b_feat, (1, d), dev)?, h, d)
    }

    /// The dictionary restricted to `rows` (global feature rows): `D =
    /// rows.len()` and result row `i` is global row `rows[i]`.
    ///
    /// Over the whole axis in order this is exactly [`Self::new`] — asserted by
    /// `for_rows_over_all_rows_equals_new`. `new` still builds its tensors
    /// straight from the caller's slices rather than delegating here, because a
    /// one-track fit must not pay for a gather it cannot use.
    pub(crate) fn for_rows(
        feat: &[f32],
        b_feat: &[f32],
        h: usize,
        rows: &[u32],
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let d_full = b_feat.len();
        anyhow::ensure!(
            feat.len() == d_full * h,
            "dictionary has {} entries, expected {d_full} × {h}",
            feat.len()
        );
        let d = rows.len();
        let mut e = Vec::with_capacity(d * h);
        let mut b = Vec::with_capacity(d);
        for &r in rows {
            let r = r as usize;
            anyhow::ensure!(
                r < d_full,
                "track row {r} is past the feature axis ({d_full} rows)"
            );
            e.extend_from_slice(&feat[r * h..(r + 1) * h]);
            b.push(b_feat[r]);
        }
        Self::assemble(
            Tensor::from_vec(e, (d, h), dev)?,
            Tensor::from_vec(b, (1, d), dev)?,
            h,
            d,
        )
    }

    fn assemble(e_dh: Tensor, b_1d: Tensor, h: usize, d: usize) -> anyhow::Result<Self> {
        Ok(Self {
            e_hd: e_dh.t()?.contiguous()?,
            features: FeatureEmbedding::fixed(e_dh),
            b_1d,
            h,
            d,
        })
    }

    /// Rows of this dictionary.
    pub(crate) fn d(&self) -> usize {
        self.d
    }
}

/// What the distillation reports back, for the log and the tests.
pub(crate) struct DistillStats {
    pub held_out_mse: f32,
    pub held_out_target_var: f32,
    pub held_out_cosine: f32,
    pub n_train_pairs: usize,
    pub n_held_out: usize,
}

//////////////////
// Host helpers //
//////////////////

/// Add the members' folded counts into `dst` (dense over the genes) and divide
/// by their number: the mean row. Empty members leave `dst` untouched.
pub(crate) fn aggregate_into(rows: &[FoldedRow], members: &[usize], dst: &mut [f32]) {
    if members.is_empty() {
        return;
    }
    for &i in members {
        for (&f, &c) in rows[i].feats.iter().zip(&rows[i].counts) {
            dst[f as usize] += c;
        }
    }
    let n = members.len() as f32;
    dst.iter_mut().for_each(|v| *v /= n);
}

/// Per-gene mean of the folded counts over every row — the encoder's
/// count-rate divisor, in the same units as its inputs.
pub(crate) fn gene_mean(rows: &[FoldedRow], d: usize) -> Vec<f32> {
    let sum = rows
        .par_iter()
        .fold(
            || vec![0f64; d],
            |mut acc, r| {
                for (&f, &c) in r.feats.iter().zip(&r.counts) {
                    acc[f as usize] += f64::from(c);
                }
                acc
            },
        )
        .reduce(
            || vec![0f64; d],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );
    let n = rows.len().max(1) as f64;
    sum.iter().map(|v| (v / n) as f32).collect()
}

/// Members (positions into the folded rows) of each pseudobulk, from the
/// global cell → pseudobulk map and the rows' global ids.
pub(crate) fn members_by_pb(
    cell_to_pb: &[usize],
    row_cells: &[u32],
    n_pb: usize,
) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); n_pb];
    for (pos, &cell) in row_cells.iter().enumerate() {
        if let Some(&p) = cell_to_pb.get(cell as usize) {
            if p < n_pb {
                groups[p].push(pos);
            }
        }
    }
    groups
}

/// Seeded split of `0..n` into (train, held out), holding out `frac` rounded
/// down — nothing when that rounds to zero.
pub(crate) fn split_holdout(n: usize, frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut ids: Vec<usize> = (0..n).collect();
    ids.shuffle(&mut StdRng::seed_from_u64(seed));
    let n_held = ((n as f64) * frac).floor() as usize;
    let mut held = ids[..n_held].to_vec();
    let mut train = ids[n_held..].to_vec();
    train.sort_unstable();
    held.sort_unstable();
    (train, held)
}

/// Re-draw the trunk's linear weights and the query from `seed`; the
/// batch-norm's affine and running statistics keep their defaults.
fn seed_trunk(varmap: &VarMap, seed: u64) -> anyhow::Result<()> {
    Ok(candle_util::nn::seed_uniform_vars(varmap, seed, |name| {
        name.contains("bn_z")
    })?)
}

/// A log-uniform subset size in `[1, n]`.
fn subset_size(n: usize, rng: &mut StdRng) -> usize {
    if n <= 1 {
        return n.max(1);
    }
    let u: f64 = rng.random::<f64>() * (n as f64).ln();
    (u.exp().round() as usize).clamp(1, n)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// Scatter sparse `(feats, counts)` nodes into a dense `[n × d]` buffer, in
/// parallel over the rows, returning each row's total count.
fn densify(nodes: &[(&[u32], &[f32])], d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![0f32; nodes.len() * d];
    let totals: Vec<f32> = x
        .par_chunks_mut(d)
        .zip(nodes.par_iter())
        .map(|(dst, &(feats, counts))| {
            for (&f, &c) in feats.iter().zip(counts) {
                dst[f as usize] = c;
            }
            counts.iter().sum()
        })
        .collect();
    (x, totals)
}

/// Scatter sparse `(id, feature ids, counts)` nodes into a dense `[n × d]`
/// buffer of ONE track, dropping every edge that is not on it. `local_of_row`
/// is global feature row → position within the track, with `u32::MAX` for a row
/// of another track. Returns each row's total count ON THIS TRACK — zero when
/// the node has none, which is what marks the track absent for that node.
fn densify_mapped(
    nodes: &[(u32, &[u32], &[f32])],
    local_of_row: &[u32],
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![0f32; nodes.len() * d];
    let totals: Vec<f32> = x
        .par_chunks_mut(d)
        .zip(nodes.par_iter())
        .map(|(dst, &(_, feats, counts))| {
            let mut total = 0f32;
            for (&f, &c) in feats.iter().zip(counts) {
                let local = local_of_row[f as usize];
                if local != u32::MAX {
                    dst[local as usize] = c;
                    total += c;
                }
            }
            total
        })
        .collect();
    (x, totals)
}

//////////////////////
// The cell encoder //
//////////////////////

/// A trained pooled-gene trunk on a frozen dictionary: the map a run's cells
/// were placed by, kept so a query is placed by the same one.
///
/// Persisted as ONE safetensors file — the trunk's parameters plus the per-gene
/// mean it divides by; the dictionary is the run's own output — and rebuilt by
/// [`Self::load`] on that dictionary.
pub struct CellEncoder {
    encoder: PooledGeneEncoder,
    varmap: VarMap,
    dict: FrozenDict,
    /// `[1, D]`, the count-rate divisor of the trunk's gate.
    mean_1d: Tensor,
}

impl CellEncoder {
    /// An untrained trunk on `dict`, its vars registered under [`VAR_PREFIX`].
    fn build(dict: FrozenDict, mean_1d: &[f32], dev: &Device) -> anyhow::Result<Self> {
        anyhow::ensure!(
            mean_1d.len() == dict.d,
            "per-gene mean has {} entries, the dictionary {}",
            mean_1d.len(),
            dict.d
        );
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let encoder = PooledGeneEncoder::new(
            std::sync::Arc::clone(&dict.features),
            PooledGeneEncoderArgs {
                layers: &[TRUNK_WIDTH],
                out_dim: dict.h,
                attn_pool: true,
                in_dim_extra: 0,
            },
            &varmap,
            vb.pp(VAR_PREFIX),
        )?;
        let mean_1d = Tensor::from_slice(mean_1d, (1, dict.d), dev)?;
        Ok(Self {
            encoder,
            varmap,
            dict,
            mean_1d,
        })
    }

    /// Rebuild a saved trunk on the run's dictionary (`feat` row-major
    /// `[D × h]`, `b_feat [D]`) from the file [`Self::save`] wrote.
    pub fn load(
        feat: &[f32],
        b_feat: &[f32],
        h: usize,
        path: &str,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        Self::load_on_dict(FrozenDict::new(feat, b_feat, h, dev)?, path, dev)
    }

    /// [`Self::load`] restricted to one track's feature rows — the same file,
    /// rebuilt on the dictionary rows that trunk reads.
    pub fn load_on_rows(
        feat: &[f32],
        b_feat: &[f32],
        h: usize,
        rows: &[u32],
        path: &str,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        Self::load_on_dict(FrozenDict::for_rows(feat, b_feat, h, rows, dev)?, path, dev)
    }

    fn load_on_dict(dict: FrozenDict, path: &str, dev: &Device) -> anyhow::Result<Self> {
        let mean_1d: Vec<f32> = candle_util::candle_core::safetensors::load(path, dev)?
            .remove(MEAN_TENSOR)
            .ok_or_else(|| anyhow::anyhow!("{path}: no `{MEAN_TENSOR}` tensor"))?
            .flatten_all()?
            .to_vec1()?;
        let mut this = Self::build(dict, &mean_1d, dev)?;
        // Matches by name and ignores the mean tensor, which is not a var.
        this.varmap.load(path)?;
        Ok(this)
    }

    /// Write the trunk's parameters and the per-gene mean to one safetensors.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let mut tensors: std::collections::HashMap<String, Tensor> = self
            .varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
            .collect();
        tensors.insert(MEAN_TENSOR.to_string(), self.mean_1d.flatten_all()?);
        candle_util::candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    /// Subtract `shift [h]` from every output: `encode(x) ← encode(x) − shift`,
    /// exactly, by moving it into the head's bias. The gauge fix removes the
    /// population mean `θ̄` from the run's cells and folds `⟨e_f, θ̄⟩` into
    /// `b_feat`; the persisted encoder has to place a query in that same frame,
    /// or a query lands `+θ̄` from the run's own cells and its scores against the
    /// re-gauged `b_feat` are off by `⟨e_f, θ̄⟩`.
    pub(crate) fn shift_output(&self, shift: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(
            shift.len() == self.dict.h,
            "shift has {} entries, h = {}",
            shift.len(),
            self.dict.h
        );
        let name = format!("{VAR_PREFIX}.nn.enc.z.mean.bias");
        let vars = self.varmap.data().lock().unwrap();
        let bias = vars
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("encoder head bias `{name}` missing"))?;
        let shift_t = Tensor::from_slice(shift, self.dict.h, bias.device())?;
        bias.set(&(bias.as_tensor() - shift_t)?)?;
        Ok(())
    }

    /// The per-gene mean the trunk divides by, on the dictionary's axis.
    pub fn feature_mean(&self) -> anyhow::Result<Vec<f32>> {
        Ok(self.mean_1d.flatten_all()?.to_vec1()?)
    }

    /// Nodes to hand one [`Self::encode_edges`] call: one dense encoding block,
    /// sized by the same activation budget as the SGD's blocks.
    pub fn group_nodes(&self) -> usize {
        block_sgd::block_cells(self.dict.d)
    }

    /// Encode sparse nodes in dense blocks → `(θ [n × h] row-major, intercept [n])`.
    fn encode(&self, nodes: &[(&[u32], &[f32])]) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let (h, d, dev) = (self.dict.h, self.dict.d, self.mean_1d.device());
        let mut latent = vec![0f32; nodes.len() * h];
        let mut intercept = vec![0f32; nodes.len()];
        for (b, block) in nodes.chunks(self.group_nodes()).enumerate() {
            let start = b * self.group_nodes();
            let (x, totals) = densify(block, d);
            let x = Tensor::from_vec(x, (block.len(), d), dev)?;
            let theta = self
                .encoder
                .forward(&x, None, Some(&self.mean_1d), None, false)?;
            let c = null_intercept(&self.dict, &theta, &totals)?;
            latent[start * h..(start + block.len()) * h]
                .copy_from_slice(&theta.flatten_all()?.to_vec1::<f32>()?);
            intercept[start..start + block.len()].copy_from_slice(&c);
        }
        Ok((latent, intercept))
    }

    /// Place nodes given as `(id, feature ids, counts)` on the dictionary's axis
    /// — the predict-time entry, the sibling of [`super::FrozenProjector::project`]
    /// with the same node tuple. Ids are positions in the returned θ.
    pub fn encode_edges(
        &self,
        nodes: &[(u32, &[u32], &[f32])],
    ) -> anyhow::Result<FrozenProjection> {
        let slices: Vec<(&[u32], &[f32])> = nodes.iter().map(|&(_, f, c)| (f, c)).collect();
        let (theta, b_node) = self.encode(&slices)?;
        Ok(FrozenProjection { theta, b_node })
    }
}

//////////////////
// Distillation //
//////////////////

/// Train a trunk onto the levels' tables; `rows` are the folded rows every
/// level's groups index into.
pub(crate) fn distill(
    dict: FrozenDict,
    mean_1d: &[f32],
    levels: &[DistillTargets<'_>],
    rows: &[FoldedRow],
    seed: u64,
    dev: &Device,
) -> anyhow::Result<(CellEncoder, DistillStats)> {
    let (h, d) = (dict.h, dict.d);
    let cell_encoder = CellEncoder::build(dict, mean_1d, dev)?;
    seed_trunk(&cell_encoder.varmap, seed)?;
    let (encoder, mean_t) = (&cell_encoder.encoder, &cell_encoder.mean_1d);
    let mut adam = AdamW::new(
        cell_encoder.varmap.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            weight_decay: WEIGHT_DECAY,
            ..Default::default()
        },
    )?;

    // Each level's table row-major, so a target row is one contiguous copy.
    let tables: Vec<Vec<f32>> = levels
        .iter()
        .map(|lv| lv.e_pb.transpose().as_slice().to_vec())
        .collect();
    let target_row = |l: usize, p: usize| &tables[l][p * h..(p + 1) * h];

    // (level, pb) pairs that train, and those held out, per level.
    let mut train_pairs: Vec<(usize, usize)> = Vec::new();
    let mut held_pairs: Vec<(usize, usize)> = Vec::new();
    for (l, lv) in levels.iter().enumerate() {
        anyhow::ensure!(
            lv.e_pb.ncols() == h,
            "level {l}: the pseudobulk table has {} columns, the dictionary {h}",
            lv.e_pb.ncols()
        );
        let populated: Vec<usize> = (0..lv.groups.len().min(lv.e_pb.nrows()))
            .filter(|&p| !lv.groups[p].is_empty())
            .collect();
        let (train, held) =
            split_holdout(populated.len(), HOLDOUT_FRACTION, mix_seed(seed, l as u64));
        train_pairs.extend(train.iter().map(|&i| (l, populated[i])));
        held_pairs.extend(held.iter().map(|&i| (l, populated[i])));
    }
    anyhow::ensure!(
        !train_pairs.is_empty(),
        "no populated pseudobulk to distil from"
    );

    // The held-out set, uploaded once: the full membership as input, the
    // table row as target.
    let n_held = held_pairs.len();
    let held_x = {
        let mut x = vec![0f32; n_held * d];
        x.par_chunks_mut(d)
            .zip(held_pairs.par_iter())
            .for_each(|(dst, &(l, p))| aggregate_into(rows, &levels[l].groups[p], dst));
        Tensor::from_vec(x, (n_held, d), dev)?
    };
    let held_y = {
        let y: Vec<f32> = held_pairs
            .iter()
            .flat_map(|&(l, p)| target_row(l, p).iter().copied())
            .collect();
        Tensor::from_vec(y, (n_held, h), dev)?
    };
    let held_target_var = if n_held > 0 {
        held_y
            .broadcast_sub(&held_y.mean_keepdim(0)?)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?
    } else {
        f32::NAN
    };
    // Held-out MSE by the same tensor arithmetic as the training loss, plus the
    // mean row-wise cosine.
    let score_held = || -> anyhow::Result<(f32, f32)> {
        if n_held == 0 {
            return Ok((f32::NAN, f32::NAN));
        }
        let z = encoder.forward(&held_x, None, Some(mean_t), None, false)?;
        let mse = (&z - &held_y)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let (z, y): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (z.to_vec2()?, held_y.to_vec2()?);
        let cos = z.iter().zip(&y).map(|(p, t)| cosine(p, t)).sum::<f32>() / n_held as f32;
        Ok((mse, cos))
    };

    info!(
        "Phase 2 (encoder) — distilling a pooled-gene encoder (trunk {TRUNK_WIDTH}, H={h}) onto \
         {} pseudobulk rows over {} level(s); {n_held} held out; {EPOCHS} epochs of \
         {ROWS_PER_STEP}-row steps, lr {LEARNING_RATE}",
        train_pairs.len(),
        levels.len(),
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let bar = new_progress_bar(EPOCHS as u64);
    for epoch in 0..EPOCHS {
        train_pairs.shuffle(&mut rng);
        // Summed on the device; read once per report.
        let mut loss_sum = Tensor::zeros((), DType::F32, dev)?;
        let mut n_steps = 0usize;
        for chunk in train_pairs.chunks(ROWS_PER_STEP) {
            let n = chunk.len();
            // The subsets are drawn sequentially, so the stream is seeded; the
            // dense rows are built in place, in parallel — this host work, not
            // the device, is what bounds a step.
            let drawn: Vec<Vec<usize>> = chunk
                .iter()
                .map(|&(l, p)| {
                    let members = &levels[l].groups[p];
                    let s = subset_size(members.len(), &mut rng);
                    rand::seq::index::sample(&mut rng, members.len(), s)
                        .into_iter()
                        .map(|i| members[i])
                        .collect()
                })
                .collect();
            let mut x = vec![0f32; n * d];
            x.par_chunks_mut(d)
                .zip(drawn.par_iter())
                .for_each(|(dst, members)| aggregate_into(rows, members, dst));
            let y: Vec<f32> = chunk
                .iter()
                .flat_map(|&(l, p)| target_row(l, p).iter().copied())
                .collect();
            let x = Tensor::from_vec(x, (n, d), dev)?;
            let y = Tensor::from_vec(y, (n, h), dev)?;
            let z = encoder.forward(&x, None, Some(mean_t), None, true)?;
            let loss = (z - y)?.sqr()?.mean_all()?;
            candle_util::grad_clip::clipped_backward_step(&mut adam, &loss, GRAD_CLIP)?;
            loss_sum = (loss_sum + loss.detach())?;
            n_steps += 1;
        }
        bar.inc(1);
        // Interim reports only; the final score is the caller's summary line.
        if (epoch + 1).is_multiple_of(REPORT_EVERY) && epoch + 1 < EPOCHS {
            let (mse, cos) = score_held()?;
            info!(
                "Phase 2 (encoder) — epoch {}: train MSE {:.4}, held-out MSE {mse:.4} (target var \
                 {held_target_var:.4}), held-out cosine {cos:.3}",
                epoch + 1,
                loss_sum.to_scalar::<f32>()? / n_steps.max(1) as f32,
            );
        }
    }
    bar.finish_and_clear();
    let (held_out_mse, held_out_cosine) = score_held()?;
    let stats = DistillStats {
        held_out_mse,
        held_out_target_var: held_target_var,
        held_out_cosine,
        n_train_pairs: train_pairs.len(),
        n_held_out: n_held,
    };
    Ok((cell_encoder, stats))
}

////////////////
// Refinement //
////////////////

/// What the refinement reports back: the per-count negative log-likelihood
/// over every cell before the first step and after the last.
pub(crate) struct RefineStats {
    pub nll_per_count_before: f32,
    pub nll_per_count_after: f32,
    pub n_cells: usize,
}

/// The multinomial NLL with the intercept profiled out, per row.
fn multinomial_nll(x: &Tensor, s: &Tensor, totals: &Tensor) -> anyhow::Result<Tensor> {
    Ok(candle_util::loss::multinomial_nll_profiled(x, s, totals)?)
}

/// Each listed track's dense block for the cells at `idx` (positions into the
/// folded rows), with that block's row totals. `which` selects the tracks and
/// fixes the order of the result.
///
/// A track the cell has no counts on comes back as an all-zero block with a
/// zero total, which is exactly what makes its likelihood term vanish.
fn track_blocks(
    by_track: &[Cow<'_, [FoldedRow]>],
    dicts: &[FrozenDict],
    which: &[usize],
    idx: &[usize],
    dev: &Device,
) -> anyhow::Result<(Vec<Tensor>, Vec<Vec<f32>>)> {
    let mut xs = Vec::with_capacity(which.len());
    let mut totals = Vec::with_capacity(which.len());
    for &t in which {
        let d_t = dicts[t].d;
        let slices: Vec<(&[u32], &[f32])> =
            idx.iter().map(|&i| by_track[t][i].as_slices()).collect();
        let (x, tot) = densify(&slices, d_t);
        xs.push(Tensor::from_vec(x, (idx.len(), d_t), dev)?);
        totals.push(tot);
    }
    Ok((xs, totals))
}

/// `Σ_t multinomial_nll(x^t, θ·E^tᵀ + b^t, N^t)` over EVERY track, `[n]`.
///
/// An absent track contributes exactly `0`: its block is all zeros and its
/// total is zero, so `N·lse − Σ n·s` is `0 − 0`. A track with no encoder still
/// enters here — it has a dictionary and counts, so it constrains `θ` even
/// though nothing reads it.
fn summed_nll(
    theta: &Tensor,
    dicts: &[FrozenDict],
    xs: &[Tensor],
    totals: &[Vec<f32>],
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let mut acc: Option<Tensor> = None;
    for (t, dict) in dicts.iter().enumerate() {
        let s = theta.matmul(&dict.e_hd)?.broadcast_add(&dict.b_1d)?;
        let n_t = Tensor::from_slice(&totals[t], totals[t].len(), dev)?;
        let nll = multinomial_nll(&xs[t], &s, &n_t)?;
        acc = Some(match acc {
            None => nll,
            Some(a) => (a + nll)?,
        });
    }
    acc.ok_or_else(|| anyhow::anyhow!("no track to score"))
}

/// Train the trunks on the cells' own likelihood, starting from the distilled
/// maps. Every cell is visited once per pass in a seeded order; the ridge is
/// the phase-2 prior `(λ/2)‖θ‖²`, the same one the block SGD carries.
///
/// The loss sums the multinomial NLL over **every** track present — including a
/// track with no encoder — against the one combined `θ`, so the gradient flows
/// back into every count track's trunk through one optimizer over the
/// concatenated var lists.
///
/// With ONE track this is the previous single-dictionary refinement, op for op:
/// the sum over tracks is the one term, and [`CellEncoders::theta_block`]
/// short-circuits to that trunk's forward.
pub(crate) fn refine(
    encs: &CellEncoders,
    dicts: &[FrozenDict],
    by_track: &[Cow<'_, [FoldedRow]>],
    lambda: f64,
    seed: u64,
    dev: &Device,
) -> anyhow::Result<RefineStats> {
    anyhow::ensure!(
        !dicts.is_empty() && dicts.len() == by_track.len(),
        "refine: {} dictionaries for {} track(s) of folded rows",
        dicts.len(),
        by_track.len()
    );
    let n_cells = by_track[0].len();
    anyhow::ensure!(
        by_track.iter().all(|t| t.len() == n_cells),
        "refine: the tracks disagree on the cell count"
    );
    let all_tracks: Vec<usize> = (0..dicts.len()).collect();
    // Where each encoder's track sits in `dicts` / `by_track`.
    let enc_track: Vec<usize> = encs.iter().iter().map(|te| te.track as usize).collect();
    anyhow::ensure!(
        enc_track.iter().all(|&t| t < dicts.len()),
        "refine: an encoder names a track this axis does not have"
    );
    let group = encs.group_nodes();
    let order_all: Vec<usize> = (0..n_cells).collect();

    // The whole-population per-count NLL, in evaluation mode; the block size is
    // the encode block so the dense buffers are bounded the same way.
    let score_all = || -> anyhow::Result<f32> {
        let (mut nll, mut total) = (0f64, 0f64);
        for block in order_all.chunks(group) {
            let (xs, totals) = track_blocks(by_track, dicts, &all_tracks, block, dev)?;
            let enc_xs: Vec<Tensor> = enc_track.iter().map(|&t| xs[t].clone()).collect();
            let enc_tot: Vec<&[f32]> = enc_track.iter().map(|&t| totals[t].as_slice()).collect();
            let theta = encs.theta_block(&enc_xs, &enc_tot, false)?;
            let s = summed_nll(&theta, dicts, &xs, &totals, dev)?;
            nll += f64::from(s.sum_all()?.to_scalar::<f32>()?);
            total += totals
                .iter()
                .flat_map(|t| t.iter())
                .map(|&v| f64::from(v))
                .sum::<f64>();
        }
        Ok((nll / total.max(1.0)) as f32)
    };
    let nll_per_count_before = score_all()?;
    info!(
        "Phase 2 (encoder) — refining {} trunk(s) on {n_cells} cells' likelihood over {} track(s): \
         {REFINE_EPOCHS} epochs of {REFINE_CELLS_PER_STEP}-cell steps, lr {REFINE_LEARNING_RATE}, \
         ridge λ={lambda}; NLL/count before {nll_per_count_before:.4}",
        enc_track.len(),
        dicts.len(),
    );

    // One optimizer over every trunk's vars: the combined θ is one object and a
    // cell's gradient has to reach each track that placed it.
    let mut vars = Vec::new();
    for te in encs.iter() {
        vars.extend(te.encoder.varmap.all_vars());
    }
    let mut adam = AdamW::new(
        vars,
        ParamsAdamW {
            lr: REFINE_LEARNING_RATE,
            weight_decay: WEIGHT_DECAY,
            ..Default::default()
        },
    )?;
    let mut order: Vec<usize> = (0..n_cells).collect();
    let mut rng = StdRng::seed_from_u64(mix_seed(seed, 0x5245_4649_4e45));
    let half_lambda = lambda / 2.0;
    let bar = new_progress_bar(REFINE_EPOCHS as u64);
    for epoch in 0..REFINE_EPOCHS {
        order.shuffle(&mut rng);
        let mut loss_sum = Tensor::zeros((), DType::F32, dev)?;
        let mut n_steps = 0usize;
        for chunk in order.chunks(REFINE_CELLS_PER_STEP) {
            let (xs, totals) = track_blocks(by_track, dicts, &all_tracks, chunk, dev)?;
            let enc_xs: Vec<Tensor> = enc_track.iter().map(|&t| xs[t].clone()).collect();
            let enc_tot: Vec<&[f32]> = enc_track.iter().map(|&t| totals[t].as_slice()).collect();
            let theta = encs.theta_block(&enc_xs, &enc_tot, true)?;
            let nll = summed_nll(&theta, dicts, &xs, &totals, dev)?;
            let ridge = theta.sqr()?.sum(1)?.affine(half_lambda, 0.0)?;
            let loss = (nll + ridge)?.mean_all()?;
            candle_util::grad_clip::clipped_backward_step(&mut adam, &loss, GRAD_CLIP)?;
            loss_sum = (loss_sum + loss.detach())?;
            n_steps += 1;
        }
        bar.inc(1);
        // The running training loss is the checkpoint reading; the whole-set
        // NLL is scored once before and once after, not every other epoch.
        if (epoch + 1).is_multiple_of(REFINE_REPORT_EVERY) && epoch + 1 < REFINE_EPOCHS {
            info!(
                "Phase 2 (encoder) — refine epoch {}: mean per-cell loss {:.2}",
                epoch + 1,
                loss_sum.to_scalar::<f32>()? / n_steps.max(1) as f32,
            );
        }
    }
    bar.finish_and_clear();
    let nll_per_count_after = score_all()?;
    Ok(RefineStats {
        nll_per_count_before,
        nll_per_count_after,
        n_cells,
    })
}

///////////////
// Intercept //
///////////////

/// `c_n = ln(total_n) − logsumexp_f(θ_n·e_f + b_f)` — the exact conditional MLE
/// of the per-cell intercept given `θ`. A cell with no counts gets the score
/// clamp's floor.
pub(crate) fn null_intercept(
    dict: &FrozenDict,
    theta_nh: &Tensor,
    totals: &[f32],
) -> anyhow::Result<Vec<f32>> {
    let s = theta_nh.matmul(&dict.e_hd)?.broadcast_add(&dict.b_1d)?; // [N, D]
    let lse: Vec<f32> = s.log_sum_exp(1)?.to_vec1()?;
    let clamp = crate::cell_projection::SCORE_CLAMP as f32;
    Ok(totals
        .iter()
        .zip(&lse)
        .map(|(&n, &z)| {
            if n > 0.0 {
                (n.ln() - z).clamp(-clamp, clamp)
            } else {
                -clamp
            }
        })
        .collect())
}

/////////////////
// Entry point //
/////////////////

/// Project every cell through the distilled encoders. Same inputs and output as
/// [`block_sgd::project_cells`], plus the trained maps.
///
/// One encoder per COUNT track, each pooling only its own track's rows; the
/// cell's warm start is the mean over the tracks it has counts on, refined on
/// the summed likelihood of EVERY track, and finished by the per-track polish.
///
/// # The one-track path
///
/// `senna bge` has a single count track over the whole axis, and that path is
/// the previous code op for op: `FrozenDict::new` (no gather),
/// [`split_rows_by_track`] borrowing the folded rows, `spec.seed` itself as the
/// distillation seed, a single `encoder.forward` inside the refinement, and the
/// single trunk's own `encode` for the warm start.
pub(crate) fn project_cells(
    input: &Phase2Input,
    cells: &[(u32, &[u32], &[f32])],
    batch_fold: Option<CellBatchFold>,
    spec: &DistillSpec<'_>,
    tracks: &TrackSpec,
) -> anyhow::Result<(Phase2Out, CellEncoders)> {
    let (h, dev) = (input.h, input.dev);
    let d = input.b_feat.len();
    let one_track = tracks.n_tracks() == 1;

    // Every cell's folded row, once.
    let rows: Vec<FoldedRow> = cells
        .par_iter()
        .map(|&(cell, feats, counts)| {
            let (f, c): (Vec<u32>, Vec<f32>) = cell_edges(cell, feats, counts, batch_fold).unzip();
            FoldedRow::new(f, c)
        })
        .collect();
    let row_cells: Vec<u32> = cells.iter().map(|&(c, _, _)| c).collect();

    // The axis, per track: the rows, the dictionary restricted to them, and
    // each cell's counts relabelled to the track's local ids. At T = 1 the one
    // track IS the whole axis — `FrozenDict::new` uploads the caller's slices
    // with no gather and the split borrows `rows` — so nothing is copied.
    let mut track_rows = TrackRows::all(tracks);
    let dicts: Vec<FrozenDict> = if one_track {
        vec![FrozenDict::new(input.feat, input.b_feat, h, dev)?]
    } else {
        track_rows
            .iter()
            .map(|tr| FrozenDict::for_rows(input.feat, input.b_feat, h, &tr.rows, dev))
            .collect::<anyhow::Result<_>>()?
    };
    let by_track = split_rows_by_track(&rows, tracks);

    // The distillation targets are the same pseudobulks for every track: a
    // track changes which counts are read, not which cells a pseudobulk holds.
    let groups: Vec<Vec<Vec<usize>>> = spec
        .levels
        .iter()
        .map(|lv| members_by_pb(lv.cell_to_pb, &row_cells, lv.e_pb.nrows()))
        .collect();
    let targets: Vec<DistillTargets<'_>> = spec
        .levels
        .iter()
        .zip(&groups)
        .map(|(lv, g)| DistillTargets {
            e_pb: lv.e_pb,
            groups: g,
        })
        .collect();

    let count_tracks = tracks.count_tracks();
    let mut encoders: Vec<TrackEncoder> = Vec::with_capacity(count_tracks.len());
    for &t in &count_tracks {
        let rows_t = std::mem::take(&mut track_rows[t].rows);
        let track = track_rows[t].track;
        let mean_1d = gene_mean(&by_track[t], dicts[t].d());
        // Track 0 keeps the fit's own stream, so a one-track run draws exactly
        // what it drew before; a further track gets its own sub-stream.
        let seed_t = if t == 0 {
            spec.seed
        } else {
            mix_seed(spec.seed, TRACK_SEED_TAG + t as u64)
        };
        let name = tracks.tracks[t].name.clone();
        let (encoder, stats) = distill(
            dicts[t].clone(),
            &mean_1d,
            &targets,
            &by_track[t],
            seed_t,
            dev,
        )?;
        info!(
            "Phase 2 (encoder) — track `{name}` distilled on {} pairs; held-out ({}) MSE {:.4} \
             against target variance {:.4}, cosine {:.3}",
            stats.n_train_pairs,
            stats.n_held_out,
            stats.held_out_mse,
            stats.held_out_target_var,
            stats.held_out_cosine
        );
        encoders.push(TrackEncoder {
            track,
            name,
            rows: rows_t,
            encoder,
        });
    }
    let encs = CellEncoders::new(encoders, d);

    let refined = refine(&encs, &dicts, &by_track, input.lambda, spec.seed, dev)?;
    info!(
        "Phase 2 (encoder) — refined on {} cells; NLL/count {:.4} → {:.4}",
        refined.n_cells, refined.nll_per_count_before, refined.nll_per_count_after
    );

    // ONE TRACK, not one encoder: an axis with a single count track beside
    // non-count ones also has a single encoder, but that encoder's dictionary
    // is narrower than the axis, so its input must be the SPLIT rows.
    let latent = if one_track {
        let enc = encs
            .single()
            .ok_or_else(|| anyhow::anyhow!("a one-track axis must have exactly one encoder"))?;
        let slices: Vec<(&[u32], &[f32])> = rows.iter().map(FoldedRow::as_slices).collect();
        enc.encode(&slices)?.0
    } else {
        encode_combined(&encs, &dicts, &by_track, h, dev)?
    };
    info!(
        "Phase 2 (encoder) — {} cell(s) encoded in blocks of {}; polishing each from there",
        rows.len(),
        encs.group_nodes()
    );
    // The encoders' placement is the warm start; the block SGD finishes each
    // cell on the exact objective. `predict` walks the same two steps.
    let pass = block_sgd::polish_cells(input, cells, batch_fold, &latent, tracks)?;

    let out = block_sgd::finish(input, cells, pass);
    Ok((out, encs))
}

/// `θ [n × h]` for every cell by the combine rule, from the already-split
/// per-track rows — the multi-encoder sibling of [`CellEncoder::encode`].
fn encode_combined(
    encs: &CellEncoders,
    dicts: &[FrozenDict],
    by_track: &[Cow<'_, [FoldedRow]>],
    h: usize,
    dev: &Device,
) -> anyhow::Result<Vec<f32>> {
    let enc_track: Vec<usize> = encs.iter().iter().map(|te| te.track as usize).collect();
    let n_cells = by_track[0].len();
    let order: Vec<usize> = (0..n_cells).collect();
    let group = encs.group_nodes();
    let mut latent = vec![0f32; n_cells * h];
    for (b, block) in order.chunks(group).enumerate() {
        let start = b * group;
        let (xs, totals) = track_blocks(by_track, dicts, &enc_track, block, dev)?;
        let refs: Vec<&[f32]> = totals.iter().map(Vec::as_slice).collect();
        let theta = encs.theta_block(&xs, &refs, false)?;
        latent[start * h..(start + block.len()) * h]
            .copy_from_slice(&theta.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok(latent)
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod encoder_tests;
