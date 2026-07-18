//! Phase-2 alternative — **frozen-feature NCE cell projection**.
//!
//! Instead of the analytical per-cell Poisson-MAP ([`crate::cell_projection`],
//! an O(m·h²) IRLS solve per cell on the CPU), this recovers each cell's
//! embedding by *training* it against the FROZEN feature dictionary with the
//! same NCE loss used for the pb-level calibration
//! ([`crate::loss::nce_loss_identity`]).
//!
//! Given a fixed feature side, cells don't couple, so cells are processed in
//! independent **blocks** — each a short AdamW run over a fresh `e_cell` block
//! `Var` on the GPU. The whole projection is therefore GPU-batched with no
//! per-cell linear solve: the CPU bottleneck (which is ~50× worse at h=128
//! because the Hessian is O(m·h²)) disappears.
//!
//! bge only for now (free feature model, `factor = None`); the gem β-sharing /
//! velocity path stays on the analytical solve, which also emits `δ`.
//!
//! Batch correction matches the analytical path: each cell's counts are divided
//! by its finest-pb `μ_residual` fold-factor (via [`cell_edges`]) before they
//! drive the block samplers, so the stochastic and analytical projections
//! de-batch the same way.

use crate::fit::projection::{cell_edges, collect_sampler_cells, l2_direction, CellBatchDivisor};
use crate::loss::{nce_loss_identity, EdgeBatch, NceObjective, PerBatchStratifiedCellSampler};
use crate::model::JointEmbedModel;
use candle_util::candle_core::{DType, Device, Result, Tensor, Var};
use candle_util::candle_nn::{AdamW, Optimizer, VarMap};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Hyperparameters for the frozen-feature NCE projection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NceProjectionOpts {
    pub objective: NceObjective,
    /// Cells per independent block (one AdamW run each).
    pub block_size: usize,
    /// AdamW passes per block. One epoch = one pass over the block's cells
    /// (`ceil(block_len / batch_size)` steps); with `batch_size ≥ block_size`
    /// that is one step, so `epochs` ≈ step count. Density-independent.
    pub epochs: usize,
    /// Positive edges per step.
    pub batch_size: usize,
    /// Negatives per positive.
    pub n_negatives: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

/// bge phase-2 projection budget. `batch ≥ block` ⇒ 1 epoch = 1 step, so
/// `epochs × (batch / block)` ≈ 600 gradient samples/cell against the fixed
/// dictionary — enough to converge a direction.
const DEFAULT_BLOCK_SIZE: usize = 2048;
const DEFAULT_EPOCHS: usize = 100;
const DEFAULT_BATCH_SIZE: usize = 12288;
const DEFAULT_LEARNING_RATE: f64 = 0.05;

impl NceProjectionOpts {
    /// The bge phase-2 defaults (see the `DEFAULT_*` consts above).
    pub(crate) fn bge_default(objective: NceObjective, n_negatives: usize, seed: u64) -> Self {
        Self {
            objective,
            block_size: DEFAULT_BLOCK_SIZE,
            epochs: DEFAULT_EPOCHS,
            batch_size: DEFAULT_BATCH_SIZE,
            n_negatives,
            learning_rate: DEFAULT_LEARNING_RATE,
            seed,
        }
    }
}

/// Fit `e_cell` for one block of cells against the frozen `(e_feat, b_feat)`,
/// returning the raw `[block_len × h]` row-major embedding (caller
/// L2-normalizes for the bge direction store).
fn fit_cell_block(
    e_feat: &Tensor,
    b_feat: &Tensor,
    block: &[(&[u32], &[f32])],
    h: usize,
    opts: &NceProjectionOpts,
    block_seed: u64,
    dev: &Device,
    // Parallelize the per-step batch construction across cores. On when blocks
    // run sequentially (GPU): the CPU sampler feeds the busy GPU. Off when blocks
    // themselves run in parallel (CPU): the parallelism is already at the block
    // level, so nesting would only oversubscribe.
    parallel_sampler: bool,
) -> Result<Vec<f32>> {
    let n = block.len();

    // Fresh cell-side vars: small-random directions, zero library-size bias.
    // Only these train; the feature side is `detach`ed so no gradient flows to
    // (or is computed for) the frozen dictionary.
    let e_cell = Var::from_tensor(&Tensor::randn(0f32, 0.01f32, (n, h), dev)?)?;
    let b_cell = Var::from_tensor(&Tensor::zeros(n, DType::F32, dev)?)?;
    let model = JointEmbedModel {
        e_feat: e_feat.detach(),
        e_cell: e_cell.as_tensor().clone(),
        b_feat: b_feat.detach(),
        b_cell: b_cell.as_tensor().clone(),
        factor: None,
        embedding_dim: h,
    };
    let mut opt = AdamW::new_lr(vec![e_cell.clone(), b_cell.clone()], opts.learning_rate)?;

    // Block-local samplers (mirror the per-batch stratified cell sampler):
    // pick a cell ∝ its degree, then a feature within it ∝ count; negatives from
    // the block's feature marginal.
    let cell_w: Vec<f32> = block
        .iter()
        .map(|(_, c)| c.iter().sum::<f32>().max(1e-6))
        .collect();
    let cell_picker = WeightedIndex::new(&cell_w).expect("block cell weights positive");
    let feat_pickers: Vec<WeightedIndex<f32>> = block
        .iter()
        .map(|(_, c)| {
            WeightedIndex::new(c.iter().map(|&x| x.max(1e-6))).expect("block feat weights positive")
        })
        .collect();
    let mut neg_map: FxHashMap<u32, f32> = FxHashMap::default();
    for (feats, counts) in block {
        for (&f, &c) in feats.iter().zip(counts.iter()) {
            *neg_map.entry(f).or_insert(0.0) += c;
        }
    }
    let neg_feats: Vec<u32> = neg_map.keys().copied().collect();
    let neg_w: Vec<f32> = neg_feats.iter().map(|f| neg_map[f]).collect();
    let neg_picker = WeightedIndex::new(&neg_w).expect("block neg weights positive");

    let bs = opts.batch_size.max(1);
    let k = opts.n_negatives;
    // One epoch = one pass over the block's cells; density-independent.
    let steps = (opts.epochs * n.div_ceil(bs)).max(1);
    // The per-step batch construction (`~bs·(2+k)` weighted draws) dominates the
    // wall-clock at bs=8192, so build it in parallel when asked: each chunk fills
    // its own slice with a chunk-seeded RNG (deterministic, scheduling-independent).
    // `chunk == bs` (parallel off) yields a single chunk → serial.
    let chunk = if parallel_sampler {
        bs.div_ceil(rayon::current_num_threads()).max(1)
    } else {
        bs
    };
    let edge_weights = vec![1.0f32; bs];

    for step in 0..steps {
        let mut coarse_cells = vec![0u32; bs];
        let mut fine_feats = vec![0u32; bs];
        let mut negs = vec![0u32; bs * k];
        coarse_cells
            .par_chunks_mut(chunk)
            .zip(fine_feats.par_chunks_mut(chunk))
            .zip(negs.par_chunks_mut(chunk * k))
            .enumerate()
            .for_each(|(ci, ((cc, ff), nn))| {
                let mut rng = StdRng::seed_from_u64(
                    block_seed
                        ^ (step as u64).wrapping_mul(0x9E37_79B9)
                        ^ (ci as u64).wrapping_mul(0x85EB_CA77),
                );
                for j in 0..cc.len() {
                    let c = cell_picker.sample(&mut rng);
                    cc[j] = c as u32; // local block index → row of e_cell
                    let (feats, _) = block[c];
                    ff[j] = feats[feat_pickers[c].sample(&mut rng)];
                    for t in 0..k {
                        nn[j * k + t] = neg_feats[neg_picker.sample(&mut rng)];
                    }
                }
            });
        let batch = EdgeBatch {
            coarse_cells,
            fine_feats,
            neg_feats: negs,
            edge_weights: edge_weights.clone(),
            n_negatives: k,
        };
        let loss = nce_loss_identity(&model, batch, None, opts.objective, dev)?;
        opt.backward_step(&loss)?;
    }

    e_cell.as_tensor().flatten_all()?.to_vec1::<f32>()
}

/// bge phase-2 via frozen-feature NCE. Streams the cells in blocks, fits each
/// block's `e_cell` against the frozen dictionary, stores the L2 direction (bge
/// convention), and publishes `e_cell` into the varmap. Returns the per-cell MAP
/// norm (`‖e_cell‖`, the empty-droplet QC signal), matching
/// [`super::projection::project_cells_phase2`]'s first return value.
pub(crate) fn project_cells_frozen_feature(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    opts: &NceProjectionOpts,
    batch_divisor: Option<CellBatchDivisor>,
    dev: &Device,
) -> Result<Vec<f32>> {
    let h = model.embedding_dim;
    let e_feat = model.e_feat.clone();
    let b_feat = model.b_feat.clone();
    let cells = collect_sampler_cells(cell_samplers);
    let block_size = opts.block_size.max(1);
    let chunks: Vec<&[(u32, &[u32], &[f32])]> = cells.chunks(block_size).collect();
    // Blocks are independent given the frozen dictionary. On CPU, run them in
    // parallel (rayon) with a serial in-block sampler; on GPU (one device), run
    // them sequentially with a parallel in-block sampler feeding the device.
    let parallel_blocks = dev.is_cpu();
    info!(
        "Phase 2 — frozen-feature NCE projection ({} cells, {} blocks of ≤{}, {} epochs/block, {:?}, {})",
        cells.len(),
        chunks.len(),
        block_size,
        opts.epochs,
        opts.objective,
        if parallel_blocks {
            "parallel blocks (cpu)"
        } else {
            "sequential blocks (device)"
        }
    );

    let fit = |bi: usize, chunk: &[(u32, &[u32], &[f32])]| -> Result<Vec<f32>> {
        // Batch-correct each cell's counts (features unchanged) before they drive
        // the samplers — the same μ_residual divide the analytical path applies.
        let counts: Vec<Vec<f32>> = chunk
            .iter()
            .map(|&(cell, feats, cnt)| {
                cell_edges(cell, feats, cnt, batch_divisor)
                    .into_iter()
                    .map(|(_, c)| c)
                    .collect()
            })
            .collect();
        let block: Vec<(&[u32], &[f32])> = chunk
            .iter()
            .zip(counts.iter())
            .map(|(&(_, f, _), c)| (f, c.as_slice()))
            .collect();
        let block_seed = opts.seed ^ (bi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        fit_cell_block(&e_feat, &b_feat, &block, h, opts, block_seed, dev, !parallel_blocks)
    };
    let raws: Vec<Vec<f32>> = if parallel_blocks {
        chunks
            .par_iter()
            .enumerate()
            .map(|(bi, ch)| fit(bi, ch))
            .collect::<Result<Vec<_>>>()?
    } else {
        chunks
            .iter()
            .enumerate()
            .map(|(bi, ch)| fit(bi, ch))
            .collect::<Result<Vec<_>>>()?
    };

    let mut e_out = vec![0f32; n_cells * h];
    let mut cell_nrms = vec![0f32; n_cells];
    for (chunk, raw) in chunks.iter().zip(raws.iter()) {
        for (li, &(cell, _, _)) in chunk.iter().enumerate() {
            let row = &raw[li * h..(li + 1) * h];
            cell_nrms[cell as usize] = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let dir = l2_direction(row);
            let s = cell as usize * h;
            e_out[s..s + h].copy_from_slice(&dir);
        }
    }

    // Publish e_cell (mirror project_cells_phase2's varmap write).
    let e_t = Tensor::from_vec(e_out, (n_cells, h), dev)?;
    {
        let vars = varmap.data().lock().unwrap();
        if let Some(v) = vars.get("e_cell") {
            v.set(&e_t)?;
        }
    }
    model.e_cell = e_t;
    Ok(cell_nrms)
}

#[cfg(test)]
#[path = "projection_nce_tests.rs"]
mod tests;
