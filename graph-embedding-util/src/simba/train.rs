//! The PBG training loop: one fused softmax step per single-relation batch,
//! RowAdagrad on both tables, stochastic weight decay, and a fixed
//! evaluation hold-out scored with the same loss.
//!
//! One step, for a batch of `k` chunks of `c` positives with `u` uniform
//! negatives per chunk (`L` cells, `R` genes, `[k, c, D]`):
//!
//! ```text
//!   pos       = Σ_d L·R                                  [k, c]
//!   rhs_bat   = L·Rᵀ + mask,  lhs_bat = R·Lᵀ + mask      [k, c, c]
//!   rhs_uni   = L·URᵀ,        lhs_uni = R·ULᵀ            [k, c, u]
//!   ℓ         = softmax_nce(pos, [rhs_bat, rhs_uni]) + softmax_nce(pos, [lhs_bat, lhs_uni])
//!   loss      = Σ ℓ · row_w          (row_w = relation weight, 0 on pad rows)
//! ```
//!
//! `mask` is PBG's `−1e9` on the diagonal (a positive never competes with
//! itself) and on every pad column (a padded last chunk contributes no
//! negatives), so the fused block reproduces `prepare_negatives` exactly.

use super::batch::{EpochBatcher, PaddedBatch};
use super::graph::{auto_wd, EdgeList, RelationTable};
use super::row_adagrad::RowAdagrad;
use super::{SimbaConfig, INIT_STDEV, MASK_NEG};
use crate::loss::softmax_nce;
use crate::progress::new_progress_bar;
use candle_util::candle_core::{DType, Device, Result, Tensor, Var};
use matrix_util::rand_util::name_seed;
use matrix_util::traits::SampleOps;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use std::ops::Range;
use std::sync::atomic::Ordering;

/// Per-epoch record: losses are per edge, weight decay excluded (PBG's
/// `Stats.loss`).
#[derive(Clone, Debug)]
pub struct EpochStats {
    pub epoch: usize,
    pub train_loss: f64,
    pub eval_loss: Option<f64>,
    /// Batches that drew the weight-decay term this epoch.
    pub wd_hits: usize,
}

pub struct TrainOutput {
    /// `[N, D]` on the CPU, detached.
    pub e_cell: Tensor,
    /// `[G, D]` on the CPU, detached.
    pub e_gene: Tensor,
    pub epochs: Vec<EpochStats>,
    pub relations: RelationTable,
    /// The weight decay actually used (auto or pinned).
    pub wd: f64,
    pub n_train_edges: usize,
    pub n_eval_edges: usize,
}

/// The score blocks of one batch, before the loss.
pub(crate) struct ScoreBlocks {
    pub pos: Tensor,
    pub rhs_bat: Tensor,
    pub lhs_bat: Tensor,
    pub rhs_uni: Option<Tensor>,
    pub lhs_uni: Option<Tensor>,
}

pub(crate) struct SimbaModel {
    pub e_cell: Var,
    pub e_gene: Var,
    /// `−1e9` on the diagonal, `[1, c, c]`, built once.
    diag_neg: Tensor,
}

impl SimbaModel {
    /// Fresh tables, `N(0, INIT_STDEV)` per coordinate, seeded per table.
    pub(crate) fn new(
        n_cells: usize,
        n_genes: usize,
        dim: usize,
        c: usize,
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        let init = |name: &str, rows: usize| -> Result<Var> {
            let t = Tensor::rnorm_seeded(rows, dim, name_seed(seed, name))
                .affine(INIT_STDEV, 0.0)?
                .to_device(dev)?
                .contiguous()?;
            Var::from_tensor(&t)
        };
        Self::assemble(init("e_cell", n_cells)?, init("e_gene", n_genes)?, c, dev)
    }

    /// Tables supplied by the caller (tests).
    #[cfg(test)]
    pub(crate) fn from_tables(e_cell: &Tensor, e_gene: &Tensor, c: usize) -> Result<Self> {
        let dev = e_cell.device().clone();
        Self::assemble(
            Var::from_tensor(&e_cell.contiguous()?)?,
            Var::from_tensor(&e_gene.contiguous()?)?,
            c,
            &dev,
        )
    }

    fn assemble(e_cell: Var, e_gene: Var, c: usize, dev: &Device) -> Result<Self> {
        let diag_neg = Tensor::eye(c.max(1), DType::F32, dev)?
            .affine(MASK_NEG, 0.0)?
            .unsqueeze(0)?;
        Ok(Self {
            e_cell,
            e_gene,
            diag_neg,
        })
    }

    pub(crate) fn score_blocks(&self, b: &PaddedBatch, dev: &Device) -> Result<ScoreBlocks> {
        let (k, c, u) = (b.k, b.c, b.u);
        let d = self.e_cell.dim(1)?;
        let p = k * c;
        let lhs = Tensor::from_slice(&b.lhs, p, dev)?;
        let rhs = Tensor::from_slice(&b.rhs, p, dev)?;
        let l = self
            .e_cell
            .as_tensor()
            .index_select(&lhs, 0)?
            .reshape((k, c, d))?;
        let r = self
            .e_gene
            .as_tensor()
            .index_select(&rhs, 0)?
            .reshape((k, c, d))?;
        let pos = (&l * &r)?.sum(2)?; // [k, c]
                                      // `(1 − valid) · MASK_NEG` on pad columns, plus the cached diagonal.
        let pad = Tensor::from_slice(&b.col_valid, (k, 1, c), dev)?.affine(-MASK_NEG, MASK_NEG)?;
        let mask = (self.diag_neg.broadcast_as((k, c, c))? + pad.broadcast_as((k, c, c))?)?;
        let rhs_bat = (l.matmul(&r.t()?)? + &mask)?;
        let lhs_bat = (r.matmul(&l.t()?)? + &mask)?;
        let (rhs_uni, lhs_uni) = if u > 0 {
            let ul_idx = Tensor::from_slice(&b.uni_lhs, k * u, dev)?;
            let ur_idx = Tensor::from_slice(&b.uni_rhs, k * u, dev)?;
            let ul = self
                .e_cell
                .as_tensor()
                .index_select(&ul_idx, 0)?
                .reshape((k, u, d))?;
            let ur = self
                .e_gene
                .as_tensor()
                .index_select(&ur_idx, 0)?
                .reshape((k, u, d))?;
            (Some(l.matmul(&ur.t()?)?), Some(r.matmul(&ul.t()?)?))
        } else {
            (None, None)
        };
        Ok(ScoreBlocks {
            pos,
            rhs_bat,
            lhs_bat,
            rhs_uni,
            lhs_uni,
        })
    }

    /// PBG's batch loss: relation-weighted sum over positives of the lhs- and
    /// rhs-corrupted softmax losses. Weight decay is added by the caller.
    pub(crate) fn batch_loss(&self, b: &PaddedBatch, dev: &Device) -> Result<Tensor> {
        let s = self.score_blocks(b, dev)?;
        let p = b.k * b.c;
        let pos = s.pos.reshape(p)?;
        let mut rhs_negs = vec![s.rhs_bat.reshape((p, b.c))?];
        let mut lhs_negs = vec![s.lhs_bat.reshape((p, b.c))?];
        if let Some(t) = s.rhs_uni {
            rhs_negs.push(t.reshape((p, b.u))?);
        }
        if let Some(t) = s.lhs_uni {
            lhs_negs.push(t.reshape((p, b.u))?);
        }
        let per_row = (softmax_nce(&pos, &rhs_negs)? + softmax_nce(&pos, &lhs_negs)?)?;
        let row_w = Tensor::from_slice(&b.row_w, p, dev)?;
        (per_row * row_w)?.sum_all()
    }

    /// `Σ‖E_cell‖² + Σ‖E_gene‖²` — PBG's `l2_norm()` over the entity tables.
    pub(crate) fn frob_sq(&self) -> Result<Tensor> {
        gram_trace(self.e_cell.as_tensor())? + gram_trace(self.e_gene.as_tensor())?
    }
}

/// `trace(XᵀX)` through one `[D, D]` gemm, so no `[N, D]` square is retained
/// for backward (the trick `loss::embedding_ridge` uses).
fn gram_trace(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(1)?;
    let gram = x.t()?.matmul(x)?;
    let eye = Tensor::eye(d, gram.dtype(), gram.device())?;
    (gram * eye)?.sum_all()
}

/// Train both tables on `edges` (consumed: it is shuffled in place).
pub fn train(mut edges: EdgeList, cfg: &SimbaConfig) -> anyhow::Result<TrainOutput> {
    let dev = &cfg.device;
    let n = edges.len();
    anyhow::ensure!(n > 0, "simba: no edges to train on");
    anyhow::ensure!(cfg.dim > 0, "simba: --embedding-dim must be positive");
    anyhow::ensure!(
        cfg.num_batch_negs > 0,
        "simba: --num-batch-negs must be positive"
    );
    anyhow::ensure!(cfg.batch_size > 0, "simba: --batch-size must be positive");
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    edges.shuffle_range(0..n, &mut rng);
    // PBG: `int(num_edges * eval_fraction)`; ours is a fixed tail.
    let n_eval = (n as f64 * cfg.eval_fraction.clamp(0.0, 0.5)) as usize;
    let n_train = n - n_eval;
    anyhow::ensure!(
        n_train > 0,
        "simba: no training edges left after the hold-out"
    );
    let wd = cfg.wd.unwrap_or_else(|| auto_wd(n));
    let wd_prob = if cfg.wd_interval > 0 {
        1.0 / cfg.wd_interval as f64
    } else {
        0.0
    };
    let wd_scale = wd * cfg.wd_interval as f64;
    let rel = RelationTable::from_levels(&edges.levels_present());
    log::info!(
        "simba train: {} train + {} eval edges, {} relations (weights {:?}), dim {}, {} epochs, lr {}, batch {}, negs {}+{}, wd {} every ~{} batches",
        n_train, n_eval, rel.len(), rel.weights, cfg.dim, cfg.epochs, cfg.lr, cfg.batch_size,
        cfg.num_batch_negs, cfg.num_uniform_negs, wd, cfg.wd_interval
    );

    let model = SimbaModel::new(
        edges.n_cells,
        edges.n_genes,
        cfg.dim,
        cfg.num_batch_negs,
        cfg.seed,
        dev,
    )?;
    let mut opt_cell = RowAdagrad::new(edges.n_cells, cfg.lr, dev)?;
    let mut opt_gene = RowAdagrad::new(edges.n_genes, cfg.lr, dev)?;
    let stop = crate::stop::stop_flag();

    let mut epochs = Vec::with_capacity(cfg.epochs);
    for epoch in 0..cfg.epochs {
        edges.shuffle_range(0..n_train, &mut rng);
        let mut batcher = EpochBatcher::new(&edges, 0..n_train, &rel, cfg.batch_size);
        let mut acc = Tensor::zeros((), DType::F32, dev)?;
        let mut hits = 0usize;
        let mut seen = 0usize;
        let mut interrupted = false;
        let bar = new_progress_bar(n_train as u64);
        bar.set_message(format!("simba epoch {}/{}", epoch + 1, cfg.epochs));
        while let Some(b) = batcher.next_batch(
            &edges,
            &rel,
            cfg.num_batch_negs,
            cfg.num_uniform_negs,
            &mut rng,
        ) {
            let loss = model.batch_loss(&b, dev)?;
            let total = if wd > 0.0 && rng.random::<f64>() < wd_prob {
                hits += 1;
                (&loss + model.frob_sq()?.affine(wd_scale, 0.0)?)?
            } else {
                loss.clone()
            };
            let grads = total.backward()?;
            if let Some(g) = grads.get(&model.e_cell) {
                opt_cell.step(&model.e_cell, g)?;
            }
            if let Some(g) = grads.get(&model.e_gene) {
                opt_gene.step(&model.e_gene, g)?;
            }
            acc = (acc + loss.detach())?.detach();
            seen += b.n_real;
            bar.inc(b.n_real as u64);
            if stop.load(Ordering::Relaxed) {
                interrupted = true;
                break;
            }
        }
        bar.finish_and_clear();
        // Per edge actually seen, so an interrupted epoch is not under-reported.
        let train_loss = f64::from(acc.to_scalar::<f32>()?) / seen.max(1) as f64;
        let eval_loss = if n_eval > 0 {
            Some(eval_loss(
                &model,
                &edges,
                n_train..n,
                &rel,
                cfg,
                dev,
                &mut rng,
            )?)
        } else {
            None
        };
        match eval_loss {
            Some(e) => log::info!(
                "simba epoch {}/{}: train loss {train_loss:.4}/edge, eval loss {e:.4}/edge, wd hits {hits}",
                epoch + 1,
                cfg.epochs
            ),
            None => log::info!(
                "simba epoch {}/{}: train loss {train_loss:.4}/edge, wd hits {hits}",
                epoch + 1,
                cfg.epochs
            ),
        }
        epochs.push(EpochStats {
            epoch,
            train_loss,
            eval_loss,
            wd_hits: hits,
        });
        if interrupted {
            log::warn!("simba: interrupted after epoch {}", epoch + 1);
            break;
        }
    }
    let cpu = Device::Cpu;
    Ok(TrainOutput {
        e_cell: model.e_cell.as_tensor().detach().to_device(&cpu)?,
        e_gene: model.e_gene.as_tensor().detach().to_device(&cpu)?,
        epochs,
        relations: rel,
        wd,
        n_train_edges: n_train,
        n_eval_edges: n_eval,
    })
}

/// Mean per-edge loss over the held-out `range`, same negatives, no update.
fn eval_loss(
    model: &SimbaModel,
    edges: &EdgeList,
    range: Range<usize>,
    rel: &RelationTable,
    cfg: &SimbaConfig,
    dev: &Device,
    rng: &mut StdRng,
) -> anyhow::Result<f64> {
    let n = range.len().max(1);
    let mut batcher = EpochBatcher::new(edges, range, rel, cfg.batch_size);
    let mut acc = Tensor::zeros((), DType::F32, dev)?;
    while let Some(b) =
        batcher.next_batch(edges, rel, cfg.num_batch_negs, cfg.num_uniform_negs, rng)
    {
        acc = (acc + model.batch_loss(&b, dev)?.detach())?.detach();
    }
    Ok(f64::from(acc.to_scalar::<f32>()?) / n as f64)
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod train_tests;
