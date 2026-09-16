//! The PBG training loop over a typed graph: one fused softmax step per
//! single-relation batch, RowAdagrad on the flat table, stochastic weight
//! decay, and a per-relation evaluation hold-out scored with the same loss.

use super::batch::{EpochBatcher, PaddedBatch};
use super::graph::{NodeTypeTable, RelationTable, TypedEdgeList};
use super::model::FneModel;
use super::row_adagrad::RowAdagrad;
use super::{EpochStats, FneConfig};
use crate::progress::new_progress_bar;
use crate::simba::auto_wd;
use candle_util::candle_core::{DType, Device, Tensor};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use std::ops::Range;
use std::sync::atomic::Ordering;

/// Edge bookkeeping of one relation after the hold-out split, and its
/// per-edge losses in the last epoch trained — the readout that tells a
/// multi-relation run which relation is learning.
#[derive(Clone, Debug, PartialEq)]
pub struct RelationStats {
    pub n_edges: usize,
    pub n_train: usize,
    pub n_eval: usize,
    /// Mean per-edge train loss of the last epoch (weight decay excluded).
    pub train_loss: f64,
    /// Mean per-edge eval loss of the last epoch, when anything was held out.
    pub eval_loss: Option<f64>,
}

/// Per-relation loss sums kept on the device: one add per batch, one
/// sync per relation per epoch.
struct RelationAcc {
    sums: Vec<Tensor>,
    seen: Vec<usize>,
}

impl RelationAcc {
    fn new(n_rel: usize, dev: &Device) -> anyhow::Result<Self> {
        let sums = (0..n_rel)
            .map(|_| Tensor::zeros((), DType::F32, dev))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            sums,
            seen: vec![0; n_rel],
        })
    }

    fn add(&mut self, rel: usize, loss: &Tensor, n_real: usize) -> anyhow::Result<()> {
        self.sums[rel] = (&self.sums[rel] + loss.detach())?.detach();
        self.seen[rel] += n_real;
        Ok(())
    }

    fn total_seen(&self) -> usize {
        self.seen.iter().sum()
    }

    /// `(Σ over relations, per-relation mean per edge)` in one device
    /// sync; a relation with nothing seen reports NaN.
    fn finish(&self) -> anyhow::Result<(f64, Vec<f64>)> {
        let sums = Tensor::stack(&self.sums, 0)?.to_vec1::<f32>()?;
        let total: f64 = sums.iter().map(|&v| f64::from(v)).sum();
        let per = sums
            .iter()
            .zip(&self.seen)
            .map(|(&v, &n)| {
                if n > 0 {
                    f64::from(v) / n as f64
                } else {
                    f64::NAN
                }
            })
            .collect();
        Ok((total, per))
    }
}

/// Hand every batch of `blocks` to `step` and accumulate the per-relation
/// losses it returns — the one loop both training and evaluation drain
/// through, so their batching can never drift apart. Returns the
/// accumulator and whether the stop flag interrupted the pass.
#[allow(clippy::too_many_arguments)]
fn drain_blocks(
    blocks: &[Range<usize>],
    edges: &TypedEdgeList,
    types: &NodeTypeTable,
    rels: &RelationTable,
    cfg: &FneConfig,
    dev: &Device,
    rng: &mut StdRng,
    bar: Option<&indicatif::ProgressBar>,
    mut step: impl FnMut(&PaddedBatch) -> anyhow::Result<Tensor>,
) -> anyhow::Result<(RelationAcc, bool)> {
    let stop = crate::stop::stop_flag();
    let mut batcher = EpochBatcher::new(blocks, cfg.batch_size);
    let mut acc = RelationAcc::new(rels.len(), dev)?;
    while let Some(b) = batcher.next_batch(
        edges,
        types,
        rels,
        cfg.num_batch_negs,
        cfg.num_uniform_negs,
        rng,
    ) {
        let loss = step(&b)?;
        acc.add(b.rel, &loss, b.n_real)?;
        if let Some(bar) = bar {
            bar.inc(b.n_real as u64);
        }
        if bar.is_some() && stop.load(Ordering::Relaxed) {
            return Ok((acc, true));
        }
    }
    Ok((acc, false))
}

pub struct FneOutput {
    /// `[N_total, D]` on the CPU, detached, node types stacked in table order.
    pub embedding: Tensor,
    pub node_types: NodeTypeTable,
    pub relations: RelationTable,
    pub per_relation: Vec<RelationStats>,
    pub epochs: Vec<EpochStats>,
    /// The weight decay actually used (auto or pinned).
    pub wd: f64,
    pub n_edges: usize,
    pub n_train_edges: usize,
    pub n_eval_edges: usize,
}

/// Split every relation's block into a train prefix and an eval suffix:
/// `n_eval = min(n − 1, max(floor, ⌊n · f⌋))`, and 0 when `f` is 0, so no
/// relation is ever emptied of training edges by the hold-out.
pub(crate) fn split_relation(n: usize, eval_fraction: f64, eval_min: usize) -> (usize, usize) {
    if n == 0 {
        return (0, 0);
    }
    let f = eval_fraction.clamp(0.0, 0.5);
    let n_eval = if f <= 0.0 {
        0
    } else {
        ((n as f64 * f) as usize).max(eval_min).min(n - 1)
    };
    (n - n_eval, n_eval)
}

/// Train the table on `edges` (consumed: grouped by relation and shuffled
/// in place).
pub fn train(
    mut edges: TypedEdgeList,
    types: NodeTypeTable,
    rels: RelationTable,
    cfg: &FneConfig,
) -> anyhow::Result<FneOutput> {
    let dev = &cfg.device;
    let n = edges.len();
    anyhow::ensure!(n > 0, "fne: no edges to train on");
    anyhow::ensure!(cfg.dim > 0, "fne: the embedding dimension must be positive");
    anyhow::ensure!(
        cfg.num_batch_negs > 0,
        "fne: the number of batch negatives must be positive"
    );
    anyhow::ensure!(cfg.batch_size > 0, "fne: the batch size must be positive");
    edges.validate(&types, &rels)?;

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let blocks = edges.group_by_relation(rels.len());
    let mut train_blocks: Vec<Range<usize>> = Vec::with_capacity(rels.len());
    let mut eval_blocks: Vec<Range<usize>> = Vec::with_capacity(rels.len());
    let mut per_relation = Vec::with_capacity(rels.len());
    for b in &blocks {
        edges.shuffle_range(b.clone(), &mut rng);
        let (n_train, n_eval) =
            split_relation(b.len(), cfg.eval_fraction, cfg.eval_min_per_relation);
        train_blocks.push(b.start..b.start + n_train);
        eval_blocks.push(b.start + n_train..b.end);
        per_relation.push(RelationStats {
            n_edges: b.len(),
            n_train,
            n_eval,
            train_loss: f64::NAN,
            eval_loss: None,
        });
    }
    let n_train: usize = per_relation.iter().map(|s| s.n_train).sum();
    let n_eval: usize = per_relation.iter().map(|s| s.n_eval).sum();
    anyhow::ensure!(
        n_train > 0,
        "fne: no training edges left after the hold-out"
    );

    let wd = cfg.wd.unwrap_or_else(|| auto_wd(n));
    let wd_prob = if cfg.wd_interval > 0 {
        1.0 / cfg.wd_interval as f64
    } else {
        0.0
    };
    let wd_scale = wd * cfg.wd_interval as f64;
    log::info!(
        "fne train: {} nodes in {} types, {} relations, {} train + {} eval edges, dim {}, {} epochs, lr {}, batch {}, negs {}+{}, wd {} every ~{} batches",
        types.n_total(), types.len(), rels.len(), n_train, n_eval, cfg.dim, cfg.epochs, cfg.lr,
        cfg.batch_size, cfg.num_batch_negs, cfg.num_uniform_negs, wd, cfg.wd_interval
    );
    for (r, s) in per_relation.iter().enumerate() {
        let rel = rels.get(r);
        log::info!(
            "fne relation `{}` ({} → {}, weight {}): {} edges, {} train, {} eval",
            rel.name,
            types.name(rel.lhs_type as usize),
            types.name(rel.rhs_type as usize),
            rel.weight,
            s.n_edges,
            s.n_train,
            s.n_eval
        );
    }

    let model = FneModel::new(&types, cfg.dim, cfg.num_batch_negs, cfg.seed, dev)?;
    let mut opt = RowAdagrad::new(types.n_total(), cfg.lr, dev)?;

    let mut epochs = Vec::with_capacity(cfg.epochs);
    for epoch in 0..cfg.epochs {
        for b in &train_blocks {
            edges.shuffle_range(b.clone(), &mut rng);
        }
        let mut hits = 0usize;
        let bar = new_progress_bar(n_train as u64);
        bar.set_message(format!("fne epoch {}/{}", epoch + 1, cfg.epochs));
        // The weight-decay draw needs its own RNG stream: the batcher
        // borrows `rng` for the whole pass.
        let mut wd_rng = StdRng::seed_from_u64(cfg.seed ^ (epoch as u64 + 1).rotate_left(32));
        let (acc, interrupted) = drain_blocks(
            &train_blocks,
            &edges,
            &types,
            &rels,
            cfg,
            dev,
            &mut rng,
            Some(&bar),
            |b| {
                let loss = model.batch_loss(b, dev)?;
                let total = if wd > 0.0 && wd_rng.random::<f64>() < wd_prob {
                    hits += 1;
                    (&loss + model.frob_sq()?.affine(wd_scale, 0.0)?)?
                } else {
                    loss.clone()
                };
                let grads = total.backward()?;
                if let Some(g) = grads.get(&model.e) {
                    opt.step(&model.e, g)?;
                }
                Ok(loss)
            },
        )?;
        bar.finish_and_clear();
        // Per edge actually seen, so an interrupted epoch is not under-reported.
        let (train_sum, train_per_rel) = acc.finish()?;
        let train_loss = train_sum / acc.total_seen().max(1) as f64;
        let eval = if n_eval > 0 {
            let (acc, _) = drain_blocks(
                &eval_blocks,
                &edges,
                &types,
                &rels,
                cfg,
                dev,
                &mut rng,
                None,
                |b| Ok(model.batch_loss(b, dev)?),
            )?;
            let (sum, per) = acc.finish()?;
            Some((sum / n_eval as f64, per))
        } else {
            None
        };
        let eval_loss = eval.as_ref().map(|(total, _)| *total);
        for (r, s) in per_relation.iter_mut().enumerate() {
            s.train_loss = train_per_rel[r];
            s.eval_loss = eval
                .as_ref()
                .map(|(_, per)| per[r])
                .filter(|v| v.is_finite());
            log::debug!(
                "fne epoch {}/{} relation `{}`: train loss {:.4}/edge{}",
                epoch + 1,
                cfg.epochs,
                rels.get(r).name,
                s.train_loss,
                s.eval_loss
                    .map_or(String::new(), |e| format!(", eval loss {e:.4}/edge"))
            );
        }
        match eval_loss {
            Some(e) => log::info!(
                "fne epoch {}/{}: train loss {train_loss:.4}/edge, eval loss {e:.4}/edge, wd hits {hits}",
                epoch + 1,
                cfg.epochs
            ),
            None => log::info!(
                "fne epoch {}/{}: train loss {train_loss:.4}/edge, wd hits {hits}",
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
            log::warn!("fne: interrupted after epoch {}", epoch + 1);
            break;
        }
    }
    let cpu = Device::Cpu;
    Ok(FneOutput {
        embedding: model.e.as_tensor().detach().to_device(&cpu)?,
        node_types: types,
        relations: rels,
        per_relation,
        epochs,
        wd,
        n_edges: n,
        n_train_edges: n_train,
        n_eval_edges: n_eval,
    })
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod train_tests;
