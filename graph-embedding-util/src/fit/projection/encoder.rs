//! Phase 2 for the plain (bge) model by a **distilled pooled-gene encoder**:
//! instead of re-fitting every cell against the frozen dictionary by block SGD
//! ([`super::block_sgd`]), train a small encoder to reproduce the phase-1
//! pseudobulk embeddings from their members' counts, then encode every cell in
//! one forward pass.
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
//! The target for a pseudobulk is its **MAP placement** under the frozen
//! dictionary — its members' folded counts summed and solved by the block SGD
//! ([`super::block_sgd::project_cells`], every level in one short pass) — not
//! its phase-1 row. The phase-1 rows are trained by their own sampled edges
//! and land only loosely where the likelihood puts them (measured: cosine
//! ≈ 0.4 to the MAP row, the residual full-rank), so an encoder distilled on
//! them learns noise; the MAP rows are a deterministic function of the counts
//! the encoder reads. The input is a
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
//! plus the ridge — one `[B,H]·[H,D]` matmul per step, so no negatives are
//! sampled and nothing is approximated. The block SGD solved that objective
//! per cell from a cold start and never converged; here every cell's gradient
//! improves one small map shared by every cell like it.
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
//! The trained trunk leaves the fit as a [`CellEncoder`], which the run persists
//! and `senna predict` reloads, so a query cell is placed by the same map the
//! run's own cells were — the invariant [`super::FrozenProjector`] states for
//! the SGD path.

use super::block_sgd::{self, Phase2Input, Phase2Out};
use super::{cell_edges, CellBatchFold, FrozenProjection};
use crate::progress::new_progress_bar;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_util::encoder::{PooledGeneEncoder, PooledGeneEncoderArgs};
use candle_util::feature_embedding::FeatureEmbedding;
use log::info;
use matrix_util::rand_util::{mix_seed, name_seed};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

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
/// The var-name prefix the trunk is saved under.
const VAR_PREFIX: &str = "cell_enc";
/// The per-gene mean's tensor name inside the saved file.
const MEAN_TENSOR: &str = "cell_enc.feature_mean";

//////////////////
// Host structs //
//////////////////

/// One cell's batch-folded sparse counts.
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
    /// `[n_pb × H]` phase-1 table — read for its row count only; the targets
    /// are re-solved by [`map_targets`].
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
        Ok(Self {
            e_hd: e_dh.t()?.contiguous()?,
            features: FeatureEmbedding::fixed(e_dh),
            b_1d: Tensor::from_slice(b_feat, (1, d), dev)?,
            h,
            d,
        })
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

/// Re-draw the trunk's linear weights and the query — uniform in `±1/√fan_in`,
/// biases at zero — each from its own name-keyed sub-stream of `seed`, so a
/// run replays and adding a var never shifts another's draw. The batch-norm's
/// affine and running statistics keep their defaults.
fn seed_trunk(varmap: &VarMap, seed: u64) -> anyhow::Result<()> {
    let tbl = varmap.data().lock().unwrap();
    for (name, var) in tbl.iter() {
        if name.contains("bn_z") {
            continue;
        }
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let draw: Vec<f32> = if name.ends_with(".bias") {
            vec![0f32; n]
        } else {
            let bound = (1.0 / *dims.last().unwrap_or(&1) as f64).sqrt();
            let mut rng = StdRng::seed_from_u64(name_seed(seed, name));
            (0..n)
                .map(|_| ((rng.random::<f64>() * 2.0 - 1.0) * bound) as f32)
                .collect()
        };
        var.set(&Tensor::from_vec(draw, dims.as_slice(), var.device())?)?;
    }
    Ok(())
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
        let mean_1d: Vec<f32> = candle_util::candle_core::safetensors::load(path, dev)?
            .remove(MEAN_TENSOR)
            .ok_or_else(|| anyhow::anyhow!("{path}: no `{MEAN_TENSOR}` tensor"))?
            .flatten_all()?
            .to_vec1()?;
        let mut this = Self::build(FrozenDict::new(feat, b_feat, h, dev)?, &mean_1d, dev)?;
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

/////////////////
// MAP targets //
/////////////////

/// Solve every pseudobulk of every level for its MAP placement under the
/// frozen dictionary: members' folded rows summed, one cold block-SGD pass
/// over all levels together, no gauge fix (the targets stay in the
/// dictionary's own frame, which is the frame the encoder reproduces). A
/// pseudobulk with no member rows keeps a zero row.
pub(crate) fn map_targets(
    input: &Phase2Input,
    rows: &[FoldedRow],
    groups: &[Vec<Vec<usize>>],
) -> anyhow::Result<Vec<DMatrix<f32>>> {
    let (h, d) = (input.h, input.b_feat.len());
    // Flatten (level, pb) → one node each, in level-major order.
    let mut feats_all: Vec<Vec<u32>> = Vec::new();
    let mut counts_all: Vec<Vec<f32>> = Vec::new();
    for level in groups {
        for members in level {
            let mut dense = vec![0f32; d];
            for &i in members {
                for (&f, &c) in rows[i].feats.iter().zip(&rows[i].counts) {
                    dense[f as usize] += c;
                }
            }
            let (f, c): (Vec<u32>, Vec<f32>) = dense
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0.0)
                .map(|(f, &c)| (f as u32, c))
                .unzip();
            feats_all.push(f);
            counts_all.push(c);
        }
    }
    let n_pb = feats_all.len();
    let nodes: Vec<(u32, &[u32], &[f32])> = (0..n_pb)
        .map(|i| (i as u32, feats_all[i].as_slice(), counts_all[i].as_slice()))
        .collect();
    let pb_input = Phase2Input {
        n_cells: n_pb,
        label: "Phase 2 (pb targets)",
        gauge_fix: false,
        joint: false,
        ..*input
    };
    let out = block_sgd::project_cells(&pb_input, &nodes, None, None)?;
    let mut tables = Vec::with_capacity(groups.len());
    let mut at = 0usize;
    for level in groups {
        let n = level.len();
        tables.push(DMatrix::<f32>::from_row_slice(n, h, &out.theta[at * h..(at + n) * h]));
        at += n;
    }
    Ok(tables)
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

/// `N_c·logsumexp_f(s_cf) − Σ_f n_cf·s_cf` per row, for a dense block `x [n, D]`
/// and its scores `s [n, D]`: the multinomial negative log-likelihood with the
/// intercept profiled out, up to the count-only constant.
fn multinomial_nll(x: &Tensor, s: &Tensor, totals: &Tensor) -> anyhow::Result<Tensor> {
    let lse = s.log_sum_exp(1)?; // [n]
    let data = (x * s)?.sum(1)?; // [n]
    Ok(((totals * lse)? - data)?)
}

/// Train the trunk on the cells' own likelihood, starting from the distilled
/// map. Every cell is visited once per pass in a seeded order; the ridge is
/// the phase-2 prior `(λ/2)‖θ‖²`, the same one the block SGD carries.
pub(crate) fn refine(
    cell_encoder: &CellEncoder,
    rows: &[FoldedRow],
    lambda: f64,
    seed: u64,
    dev: &Device,
) -> anyhow::Result<RefineStats> {
    let d = cell_encoder.dict.d;
    let dict = &cell_encoder.dict;
    let mean_t = &cell_encoder.mean_1d;
    let encoder = &cell_encoder.encoder;
    let n_cells = rows.len();

    // The whole-population per-count NLL, in evaluation mode; the block size
    // is the encode block so the dense buffer is bounded the same way.
    let score_all = || -> anyhow::Result<f32> {
        let (mut nll, mut total) = (0f64, 0f64);
        for block in rows.chunks(cell_encoder.group_nodes()) {
            let slices: Vec<(&[u32], &[f32])> = block.iter().map(FoldedRow::as_slices).collect();
            let (x, totals) = densify(&slices, d);
            let x = Tensor::from_vec(x, (block.len(), d), dev)?;
            let theta = encoder.forward(&x, None, Some(mean_t), None, false)?;
            let s = theta.matmul(&dict.e_hd)?.broadcast_add(&dict.b_1d)?;
            let t = Tensor::from_slice(&totals, block.len(), dev)?;
            nll += f64::from(multinomial_nll(&x, &s, &t)?.sum_all()?.to_scalar::<f32>()?);
            total += totals.iter().map(|&v| f64::from(v)).sum::<f64>();
        }
        Ok((nll / total.max(1.0)) as f32)
    };
    let nll_per_count_before = score_all()?;
    info!(
        "Phase 2 (encoder) — refining the trunk on {n_cells} cells' likelihood: {REFINE_EPOCHS}          epochs of {REFINE_CELLS_PER_STEP}-cell steps, lr {REFINE_LEARNING_RATE}, ridge λ={lambda};          NLL/count before {nll_per_count_before:.4}"
    );

    let mut adam = AdamW::new(
        cell_encoder.varmap.all_vars(),
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
            let slices: Vec<(&[u32], &[f32])> = chunk.iter().map(|&i| rows[i].as_slices()).collect();
            let (x, totals) = densify(&slices, d);
            let x = Tensor::from_vec(x, (chunk.len(), d), dev)?;
            let t = Tensor::from_slice(&totals, chunk.len(), dev)?;
            let theta = encoder.forward(&x, None, Some(mean_t), None, true)?;
            let s = theta.matmul(&dict.e_hd)?.broadcast_add(&dict.b_1d)?;
            let nll = multinomial_nll(&x, &s, &t)?;
            let ridge = theta.sqr()?.sum(1)?.affine(half_lambda, 0.0)?;
            let loss = (nll + ridge)?.mean_all()?;
            candle_util::grad_clip::clipped_backward_step(&mut adam, &loss, GRAD_CLIP)?;
            loss_sum = (loss_sum + loss.detach())?;
            n_steps += 1;
        }
        bar.inc(1);
        if (epoch + 1).is_multiple_of(REFINE_REPORT_EVERY) && epoch + 1 < REFINE_EPOCHS {
            info!(
                "Phase 2 (encoder) — refine epoch {}: mean per-cell loss {:.2}, NLL/count {:.4}",
                epoch + 1,
                loss_sum.to_scalar::<f32>()? / n_steps.max(1) as f32,
                score_all()?
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

/// Project every cell through the distilled encoder. Same inputs and output as
/// [`block_sgd::project_cells`] on the plain path, plus the trained encoder.
pub(crate) fn project_cells(
    input: &Phase2Input,
    cells: &[(u32, &[u32], &[f32])],
    batch_fold: Option<CellBatchFold>,
    spec: &DistillSpec<'_>,
) -> anyhow::Result<(Phase2Out, CellEncoder)> {
    let (h, dev) = (input.h, input.dev);
    let d = input.b_feat.len();
    let dict = FrozenDict::new(input.feat, input.b_feat, h, dev)?;

    // Every cell's folded row, once.
    let rows: Vec<FoldedRow> = cells
        .par_iter()
        .map(|&(cell, feats, counts)| {
            let (f, c): (Vec<u32>, Vec<f32>) = cell_edges(cell, feats, counts, batch_fold).unzip();
            FoldedRow::new(f, c)
        })
        .collect();
    let row_cells: Vec<u32> = cells.iter().map(|&(c, _, _)| c).collect();
    let mean_1d = gene_mean(&rows, d);

    let groups: Vec<Vec<Vec<usize>>> = spec
        .levels
        .iter()
        .map(|lv| members_by_pb(lv.cell_to_pb, &row_cells, lv.e_pb.nrows()))
        .collect();
    let tables = map_targets(input, &rows, &groups)?;
    let targets: Vec<DistillTargets<'_>> = tables
        .iter()
        .zip(&groups)
        .map(|(t, g)| DistillTargets {
            e_pb: t,
            groups: g,
        })
        .collect();
    let (encoder, stats) = distill(dict, &mean_1d, &targets, &rows, spec.seed, dev)?;
    info!(
        "Phase 2 (encoder) — distilled on {} pairs; held-out ({}) MSE {:.4} against target \
         variance {:.4}, cosine {:.3}",
        stats.n_train_pairs,
        stats.n_held_out,
        stats.held_out_mse,
        stats.held_out_target_var,
        stats.held_out_cosine
    );

    let refined = refine(&encoder, &rows, input.lambda, spec.seed, dev)?;
    info!(
        "Phase 2 (encoder) — refined on {} cells; NLL/count {:.4} → {:.4}",
        refined.n_cells,
        refined.nll_per_count_before,
        refined.nll_per_count_after
    );

    let slices: Vec<(&[u32], &[f32])> = rows.iter().map(FoldedRow::as_slices).collect();
    let (latent, _) = encoder.encode(&slices)?;
    info!(
        "Phase 2 (encoder) — {} cell(s) encoded in blocks of {}; polishing each from there",
        rows.len(),
        encoder.group_nodes()
    );
    // The encoder's placement is the warm start; the block SGD finishes each
    // cell on the exact objective. `predict` walks the same two steps.
    let pass = block_sgd::polish_cells(input, cells, batch_fold, &latent)?;

    let out = block_sgd::finish(input, cells, pass, None);
    Ok((out, encoder))
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod encoder_tests;
