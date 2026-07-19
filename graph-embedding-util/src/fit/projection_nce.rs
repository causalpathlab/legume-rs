//! Phase-2 alternative — **frozen-feature NCE cell projection**.
//!
//! Instead of the analytical per-cell Poisson-MAP ([`crate::cell_projection`],
//! an O(m·h²) IRLS solve per cell on the CPU), this recovers each cell's
//! embedding by *training* it against the FROZEN feature dictionary with the
//! same NCE loss used for the pb-level calibration
//! ([`crate::loss::nce_loss_identity`]).
//!
//! Given a fixed feature side, cells don't couple, so cells are processed in
//! independent **blocks** — each a short AdamW run over a fresh block `Var`,
//! GPU-batched (sequential blocks + parallel sampler) or CPU-parallel (rayon
//! over independent blocks). No per-cell linear solve, so the CPU bottleneck
//! (~50× worse at h=128 because the Hessian is O(m·h²)) disappears.
//!
//! - **bge** (free feature model, `unspliced_rows = None`): one θ pass over all
//!   edges; stores the L2 direction; no velocity.
//! - **gem** (β-sharing, `unspliced_rows = Some(mask)`): TWO passes per block —
//!   θ from the SPLICED edges, then δ from the UNSPLICED edges scored at `θ+δ`
//!   (θ frozen) — the stochastic analogue of `solve_node_splice`. Stores θ raw
//!   and emits the per-cell velocity increment `δ`.
//!
//! Init is seeded host-side (`block_seed`) so the projection is bit-reproducible;
//! the parallel sampler uses a fixed chunk count so draws are thread-independent.
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

/// Fit one block's cell-side against the frozen `(e_feat, b_feat)` via NCE,
/// returning the raw `[block_len × h]` row-major trained embedding.
///
/// With `base = None` the trained Var is the identity θ (scored cell-side = θ).
/// With `base = Some(θ_block)` (`[block_len × h]`, frozen) the trained Var is the
/// increment δ and the scored cell-side is `θ + δ` — the stochastic analogue of
/// [`super::projection`]'s `solve_cell_increment`. The returned value is always
/// the trained Var (θ or δ). Cells with no edges (empty spliced/unspliced set)
/// never enter the samplers and their output row stays zero.
#[allow(clippy::too_many_arguments)]
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
    // Frozen cell-side offset (`[block_len × h]`). `Some` ⇒ train δ against `θ+δ`.
    base: Option<&Tensor>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = block.len();

    // Fresh cell-side var (θ, or δ when `base` is set): small-random init, zero
    // library-size bias. Only these train; the feature side is `detach`ed so no
    // gradient flows to (or is computed for) the frozen dictionary. The init is
    // drawn HOST-side from a `block_seed`-seeded RNG (candle's CPU `randn` is
    // unseedable), so the projection is bit-reproducible across runs.
    let mut init_rng = StdRng::seed_from_u64(block_seed ^ 0x1017_1717_1017_1717);
    let init_distr = rand_distr::Normal::new(0.0f32, 0.01f32).expect("valid init normal");
    let e_init: Vec<f32> = (0..n * h).map(|_| init_distr.sample(&mut init_rng)).collect();
    let e_cell = Var::from_tensor(&Tensor::from_vec(e_init, (n, h), dev)?)?;
    let b_cell = Var::from_tensor(&Tensor::zeros(n, DType::F32, dev)?)?;
    let mut model = JointEmbedModel {
        e_feat: e_feat.detach(),
        // θ path (`base = None`): this Var-aliasing clone tracks the in-place AdamW
        // update. δ path (`base = Some`): overwritten each step with a fresh
        // `base + var` (a derived node that must be rebuilt to see the update — see
        // the loop), so this initial value is only a placeholder.
        e_cell: e_cell.as_tensor().clone(),
        b_feat: b_feat.detach(),
        b_cell: b_cell.as_tensor().clone(),
        factor: None,
        embedding_dim: h,
    };
    let mut opt = AdamW::new_lr(vec![e_cell.clone(), b_cell.clone()], opts.learning_rate)?;

    // Cells with ≥1 edge; empty cells never enter the samplers (mirrors the
    // analytic θ=0 / δ=0 on an empty identity/velocity set) — `WeightedIndex`
    // also panics on an empty weight list.
    let active: Vec<usize> = (0..n).filter(|&i| !block[i].0.is_empty()).collect();
    if !active.is_empty() {
        // Block-local samplers (mirror the per-batch stratified cell sampler):
        // pick a cell ∝ its degree, then a feature within it ∝ count; negatives
        // from the block's feature marginal. Indexed over `active`.
        let cell_w: Vec<f32> = active
            .iter()
            .map(|&i| block[i].1.iter().sum::<f32>().max(1e-6))
            .collect();
        let cell_picker = WeightedIndex::new(&cell_w).expect("block cell weights positive");
        let feat_pickers: Vec<WeightedIndex<f32>> = active
            .iter()
            .map(|&i| {
                WeightedIndex::new(block[i].1.iter().map(|&x| x.max(1e-6)))
                    .expect("block feat weights positive")
            })
            .collect();
        let mut neg_map: FxHashMap<u32, f32> = FxHashMap::default();
        for &i in &active {
            let (feats, counts) = block[i];
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
        // The per-step batch construction (`~bs·(2+k)` weighted draws) dominates
        // the wall-clock, so build it in parallel when asked: each chunk fills its
        // own slice with a chunk-seeded RNG. The chunk COUNT is fixed (not tied to
        // the thread count) so the per-chunk seeds — and thus the sampled edges —
        // are reproducible across hosts / `RAYON_NUM_THREADS`; rayon still spreads
        // the chunks over cores. `!parallel_sampler` (blocks already parallel) ⇒
        // one serial chunk.
        const SAMPLER_CHUNKS: usize = 64;
        let chunk = if parallel_sampler {
            bs.div_ceil(SAMPLER_CHUNKS).max(1)
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
                        let a = cell_picker.sample(&mut rng); // index into `active`
                        let c = active[a];
                        cc[j] = c as u32; // block-local index → row of e_cell / base
                        ff[j] = block[c].0[feat_pickers[a].sample(&mut rng)];
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
            // Refresh θ+δ so the scored cell-side tracks the in-place AdamW update;
            // the `None` path shares the Var's storage and needs no refresh.
            if let Some(bt) = base {
                model.e_cell = bt.add(e_cell.as_tensor())?;
            }
            let loss = nce_loss_identity(&model, batch, None, opts.objective, dev)?;
            opt.backward_step(&loss)?;
        }
    }

    let mut out = e_cell.as_tensor().flatten_all()?.to_vec1::<f32>()?;
    // Empty cells never trained: zero their embedding row; their bias stays at the
    // zero init.
    let b_out = b_cell.as_tensor().to_vec1::<f32>()?;
    for i in 0..n {
        if block[i].0.is_empty() {
            out[i * h..(i + 1) * h].fill(0.0);
        }
    }
    Ok((out, b_out))
}

/// Phase-2 via frozen-feature NCE. Streams the cells in blocks, fits each block's
/// cell-side against the frozen dictionary, publishes `e_cell` into the varmap,
/// and returns `(cell_nrms, velocity)` — matching
/// [`super::projection::project_cells_phase2`].
///
/// - `unspliced_rows = None` (bge): a single θ pass over all edges; `velocity` is
///   `None`. Byte-identical to the previous behaviour.
/// - `unspliced_rows = Some(mask)` (gem β-sharing): a TWO-pass per block — θ from
///   the cell's spliced edges, then δ from its unspliced edges with the scored
///   cell-side `θ+δ` (θ frozen) — the stochastic analogue of `solve_node_splice`.
///   `velocity` is the `[n_cells × h]` δ buffer (the δ pass runs only when the
///   mask actually has unspliced rows). gem stores θ raw (magnitude is the
///   activity/QC signal); bge stores the L2 direction. δ is always raw.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_cells_frozen_feature(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    opts: &NceProjectionOpts,
    batch_divisor: Option<CellBatchDivisor>,
    unspliced_rows: Option<&[bool]>,
    dev: &Device,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let h = model.embedding_dim;
    let e_feat = model.e_feat.clone();
    let b_feat = model.b_feat.clone();
    let cells = collect_sampler_cells(cell_samplers);
    let block_size = opts.block_size.max(1);
    let chunks: Vec<&[(u32, &[u32], &[f32])]> = cells.chunks(block_size).collect();
    // gem (β-sharing) stores θ raw (magnitude = activity/QC); bge stores the
    // depth-robust L2 direction.
    let store_raw = unspliced_rows.is_some();
    // Run the δ (velocity-increment) pass only if there actually are unspliced rows
    // — a spliced-only gem run skips it (no empty δ pass, no `velocity_increment`).
    let two_pass = unspliced_rows.is_some_and(|u| u.iter().any(|&b| b));
    // Blocks are independent given the frozen dictionary. On CPU, run them in
    // parallel (rayon) with a serial in-block sampler; on GPU (one device), run
    // them sequentially with a parallel in-block sampler feeding the device.
    let parallel_blocks = dev.is_cpu();
    info!(
        "Phase 2 — frozen-feature NCE projection ({} cells, {} blocks of ≤{}, {} epochs/block, {:?}, {}{})",
        cells.len(),
        chunks.len(),
        block_size,
        opts.epochs,
        opts.objective,
        if parallel_blocks { "parallel blocks (cpu)" } else { "sequential blocks (device)" },
        if two_pass { ", θ+δ (spliced/unspliced)" } else { "" }
    );

    #[allow(clippy::type_complexity)]
    let fit = |bi: usize,
               chunk: &[(u32, &[u32], &[f32])]|
     -> Result<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> {
        let nb = chunk.len();
        // Batch-correct each cell's counts (features unchanged) — the same
        // μ_residual divide the analytical path applies — then split each cell's
        // edges into spliced / unspliced by the mask (all-spliced when `None`).
        // `unspliced` stays empty on the bge (single-pass) path.
        let mut spliced: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(nb);
        let mut unspliced: Vec<(Vec<u32>, Vec<f32>)> =
            Vec::with_capacity(if two_pass { nb } else { 0 });
        for &(cell, feats, cnt) in chunk {
            let (mut sf, mut sc, mut uf, mut uc) = (vec![], vec![], vec![], vec![]);
            for (f, c) in cell_edges(cell, feats, cnt, batch_divisor) {
                if unspliced_rows.is_some_and(|u| u[f as usize]) {
                    uf.push(f);
                    uc.push(c);
                } else {
                    sf.push(f);
                    sc.push(c);
                }
            }
            spliced.push((sf, sc));
            if two_pass {
                unspliced.push((uf, uc));
            }
        }
        fn as_blocks(v: &[(Vec<u32>, Vec<f32>)]) -> Vec<(&[u32], &[f32])> {
            v.iter().map(|(f, c)| (f.as_slice(), c.as_slice())).collect()
        }
        let seed = opts.seed ^ (bi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (theta, b_theta) = fit_cell_block(
            &e_feat, &b_feat, &as_blocks(&spliced), h, opts, seed, dev, !parallel_blocks, None,
        )?;
        let delta = if two_pass {
            let theta_t = Tensor::from_slice(&theta, (nb, h), dev)?;
            let (mut d, _b_delta) = fit_cell_block(
                &e_feat,
                &b_feat,
                &as_blocks(&unspliced),
                h,
                opts,
                seed ^ 0xDE17_A000_DE17_A000,
                dev,
                !parallel_blocks,
                Some(&theta_t),
            )?;
            // A cell with no spliced edges has θ=0 (no identity to increment), so
            // force δ=0 — matching the analytic `solve_node_splice`, which returns
            // an empty δ whenever ‖θ‖ ≤ 1e-8.
            for (i, (sf, _)) in spliced.iter().enumerate() {
                if sf.is_empty() {
                    d[i * h..(i + 1) * h].fill(0.0);
                }
            }
            Some(d)
        } else {
            None
        };
        Ok((theta, b_theta, delta))
    };
    let raws: Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> = if parallel_blocks {
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
    let mut b_out = vec![0f32; n_cells];
    let mut cell_nrms = vec![0f32; n_cells];
    let mut vel_out = two_pass.then(|| vec![0f32; n_cells * h]);
    for (chunk, (theta, b_theta, delta)) in chunks.iter().zip(raws.iter()) {
        for (li, &(cell, _, _)) in chunk.iter().enumerate() {
            let row = &theta[li * h..(li + 1) * h];
            cell_nrms[cell as usize] = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            b_out[cell as usize] = b_theta[li];
            let s = cell as usize * h;
            if store_raw {
                e_out[s..s + h].copy_from_slice(row);
            } else {
                e_out[s..s + h].copy_from_slice(&l2_direction(row));
            }
            if let (Some(vel), Some(d)) = (vel_out.as_mut(), delta.as_ref()) {
                vel[s..s + h].copy_from_slice(&d[li * h..(li + 1) * h]);
            }
        }
    }

    // Publish e_cell + the per-cell library-size intercept b_cell (mirror
    // project_cells_phase2's varmap write; the δ pass's intercept is discarded).
    let e_t = Tensor::from_vec(e_out, (n_cells, h), dev)?;
    let b_t = Tensor::from_vec(b_out, n_cells, dev)?;
    {
        let vars = varmap.data().lock().unwrap();
        if let Some(v) = vars.get("e_cell") {
            v.set(&e_t)?;
        }
        if let Some(v) = vars.get("b_cell") {
            v.set(&b_t)?;
        }
    }
    model.e_cell = e_t;
    model.b_cell = b_t;
    Ok((cell_nrms, vel_out))
}

#[cfg(test)]
#[path = "projection_nce_tests.rs"]
mod tests;
