//! Build the frozen contrastive index from a real count backend + a fitted
//! (MAP) cell side, so the posterior samplers run on actual data — not just the
//! synthetic fixtures.
//!
//! For the feature/gate side, the anchor is a **gene** and the frozen other side
//! is the **cell** embedding. [`build_gene_index`] streams the count backend once
//! (the same slab / `for_each_triplet` path `fit`'s cell sampler uses), buckets
//! nonzeros by gene into per-gene `(cell, count)` edges, and draws one **frozen
//! negative slate** of cells shared across genes (Trap 1 — the slate must not move
//! between sweeps). `partition_scale = n_cells / |slate|` folds the sampled rate
//! sum back up to the full Poisson normalizer; pass `n_partition = 0` for the
//! exact all-cells partition (small data only).

use super::lnpdf::{FrozenSide, NodeTerm};
use crate::cell_projection::SCORE_CLAMP;
use crate::data::UnifiedData;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Owned frozen index: the fixed other side (cells) plus per-anchor (gene)
/// observed edges and the shared negative slate. Views into it are handed to the
/// samplers via [`Self::frozen_side`] / [`Self::node_terms`].
pub struct ContrastiveIndex {
    /// Frozen other-side embeddings `[n_other × h]` row-major (the MAP cell side).
    pub other_e: Vec<f32>,
    /// Frozen other-side biases `[n_other]`.
    pub other_b: Vec<f32>,
    pub h: usize,
    /// Per-anchor observed `(other-index, count)` edges (`pos[g]` = gene `g`'s cells).
    pub pos: Vec<Vec<(u32, f32)>>,
    /// Per-anchor fixed bias `b_g` (held at the MAP for the gate sweep).
    pub anchor_b: Vec<f32>,
    /// Frozen negative slate of other-indices (cells), shared across anchors.
    pub partition: Vec<u32>,
    /// `n_other / |partition|` — folds the sampled slate up to the full sum.
    pub partition_scale: f64,
    /// Optional `[n_anchors × h]` row-major frozen directions, one per anchor,
    /// handed to each [`NodeTerm`] as its `offset` (see that field). `None` for
    /// the plain case where the sampler explores an absolute loading.
    pub anchor_offset: Option<Vec<f32>>,
}

impl ContrastiveIndex {
    /// The frozen other side as a borrowing [`FrozenSide`].
    #[must_use]
    pub fn frozen_side(&self) -> FrozenSide<'_> {
        FrozenSide {
            e: &self.other_e,
            b: &self.other_b,
            h: self.h,
        }
    }

    /// One [`NodeTerm`] per anchor (all sharing the frozen slate), each carrying
    /// its [`Self::anchor_offset`] row when one is set.
    #[must_use]
    pub fn node_terms(&self) -> Vec<NodeTerm<'_>> {
        self.pos
            .iter()
            .enumerate()
            .map(|(a, pos)| NodeTerm {
                pos,
                partition: &self.partition,
                partition_scale: self.partition_scale,
                offset: self
                    .anchor_offset
                    .as_ref()
                    .map(|off| &off[a * self.h..(a + 1) * self.h]),
            })
            .collect()
    }

    /// Number of anchors (genes).
    #[must_use]
    pub fn n_anchors(&self) -> usize {
        self.pos.len()
    }

    /// Anchor indices to sample: all of them when `n == 0` or `n >= n_anchors`,
    /// else the `n` carrying the most observed count mass, returned in anchor
    /// order so the output table keeps the axis's own ordering.
    ///
    /// Ranking by total count puts a capped budget where the posterior is
    /// informative: an anchor with no counts has nothing for the likelihood to
    /// move off the prior.
    #[must_use]
    pub fn top_anchors_by_count(&self, n: usize) -> Vec<usize> {
        let n_anchors = self.n_anchors();
        if n == 0 || n >= n_anchors {
            return (0..n_anchors).collect();
        }
        let total: Vec<f32> = self
            .pos
            .iter()
            .map(|p| p.iter().map(|&(_, c)| c).sum())
            .collect();
        let mut order: Vec<usize> = (0..n_anchors).collect();
        // Partial select — only the top `n` need be ordered, and `n` is typically
        // far smaller than the feature axis.
        order.select_nth_unstable_by(n, |&a, &b| {
            total[b]
                .partial_cmp(&total[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(n);
        order.sort_unstable();
        order
    }

    /// Re-fit each anchor's bias so the **null** model reproduces that anchor's
    /// observed total count, and report how far the fitted model had to move.
    ///
    /// **Not used by the gate / hyper path**, which runs the profile likelihood
    /// [`super::lnpdf::multinomial_ll`] and has no intercept to calibrate. This
    /// remains for a caller that genuinely holds `b_a` fixed under
    /// [`super::lnpdf::poisson_ll`].
    ///
    /// The frozen `anchor_b` arrives from an **NCE**-trained fit, where a bias
    /// absorbs a feature's marginal frequency only up to the objective's own
    /// normalization — it is not a Poisson log-rate. The samplers here *are*
    /// Poisson, and they hold the bias fixed while sampling the loading, so an
    /// uncalibrated intercept does not stay a harmless offset: with the rate
    /// term too small, `Σ_pos n·s` dominates the likelihood and every anchor is
    /// pushed toward the same count-weighted mean direction of the frozen side —
    /// a posterior that collapses onto one or two dims regardless of the data.
    ///
    /// At `θ = 0` the rate sum is `exp(b_a) · scale · Σ_{o ∈ partition} exp(b_o)`,
    /// so matching it to `T_a = Σ_pos n` is closed-form:
    ///
    /// ```text
    ///   b_a*  =  ln(T_a)  −  ln( scale · Σ_o exp(⟨offset_a, e_o⟩ + b_o) )
    /// ```
    ///
    /// Without [`Self::anchor_offset`] that right-hand term is the same for every
    /// anchor, so it is one pass over the frozen side plus `O(1)` per anchor. With
    /// an offset the null is "frozen direction only" rather than "nothing", so the
    /// sum is per-anchor and runs under `rayon`. An anchor with no counts has no
    /// rate to match and is parked at `-SCORE_CLAMP` (rate ≈ 0), which is what a
    /// never-observed feature should predict.
    ///
    /// This makes the null model exact per anchor, which is also what the
    /// spike-and-slab comparison in [`super::hyper_ss`] needs: its `ℓ(0)` is the
    /// baseline every inclusion decision is measured against.
    pub fn calibrate_anchor_bias(&mut self) -> BiasCalibration {
        // Without an offset the partition sum is the SAME for every anchor, so it
        // is computed once. With one, `θ = 0` still leaves `⟨offset_a, e_o⟩` in
        // the score, so the sum is per-anchor — the null being calibrated is
        // "frozen direction only", not "nothing".
        let shared_log_part = self.log_partition_sum(None);
        let per_anchor: Option<Vec<f64>> = self.anchor_offset.as_ref().map(|off| {
            (0..self.pos.len())
                .into_par_iter()
                .map(|a| self.log_partition_sum(Some(&off[a * self.h..(a + 1) * self.h])))
                .collect()
        });

        // The shift statistics describe anchors that HAVE a rate to match. An
        // empty anchor is parked at the clamp by definition, so folding it in
        // would report the clamp rather than the calibration: on a full gene
        // annotation over half the rows can be empty, which drags the median to
        // ±SCORE_CLAMP and hides what the real anchors actually did.
        let mut abs: Vec<f32> = Vec::with_capacity(self.pos.len());
        let (mut n_empty, mut n_clamped) = (0usize, 0usize);
        for (a, pos) in self.pos.iter().enumerate() {
            let total: f64 = pos.iter().map(|&(_, n)| f64::from(n)).sum();
            let log_part = per_anchor.as_ref().map_or(shared_log_part, |v| v[a]);
            let b_new = if total > 0.0 {
                let raw = total.ln() - log_part;
                if raw.abs() > SCORE_CLAMP {
                    n_clamped += 1;
                }
                let b = raw.clamp(-SCORE_CLAMP, SCORE_CLAMP) as f32;
                abs.push((b - self.anchor_b[a]).abs());
                b
            } else {
                n_empty += 1;
                -SCORE_CLAMP as f32
            };
            self.anchor_b[a] = b_new;
        }

        abs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        BiasCalibration {
            median_abs_shift: abs.get(abs.len() / 2).copied().unwrap_or(0.0),
            max_abs_shift: abs.last().copied().unwrap_or(0.0),
            n_empty,
            n_clamped,
        }
    }

    /// `ln( scale · Σ_{o ∈ partition} exp(⟨offset, e_o⟩ + b_o) )` — the rate an
    /// anchor predicts at `θ = 0`, in logs.
    ///
    /// Max-shifted and accumulated in `f64`: the exponents span the whole bias
    /// range, so a naive sum loses its low bits exactly where the slate is
    /// heterogeneous — the same widening `cell_projection` uses.
    fn log_partition_sum(&self, offset: Option<&[f32]>) -> f64 {
        let expo = |o: u32| -> f64 {
            let b = f64::from(self.other_b[o as usize]);
            match offset {
                None => b,
                Some(off) => {
                    let e_o = &self.other_e[o as usize * self.h..(o as usize + 1) * self.h];
                    b + off
                        .iter()
                        .zip(e_o)
                        .map(|(f, e)| f64::from(*f) * f64::from(*e))
                        .sum::<f64>()
                }
            }
        };
        let m = self
            .partition
            .iter()
            .map(|&o| expo(o))
            .fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return 0.0;
        }
        let s: f64 = self.partition.iter().map(|&o| (expo(o) - m).exp()).sum();
        m + (self.partition_scale * s).max(f64::MIN_POSITIVE).ln()
    }
}

/// How far [`ContrastiveIndex::calibrate_anchor_bias`] had to move the frozen
/// biases — i.e. how far the trained model was from being Poisson-calibrated.
/// A shift of a few nats is routine (different objective); tens of nats means
/// the frozen intercepts carried essentially no rate information.
#[derive(Clone, Copy, Debug)]
pub struct BiasCalibration {
    /// Median `|b* − b|` in nats over anchors that had counts to match.
    pub median_abs_shift: f32,
    /// Largest `|b* − b|` over the same anchors.
    pub max_abs_shift: f32,
    /// Anchors with no observed counts (parked at a ≈0 rate). Excluded from the
    /// shift statistics — see [`ContrastiveIndex::calibrate_anchor_bias`].
    pub n_empty: usize,
    /// Anchors whose required intercept exceeded `SCORE_CLAMP` and was truncated.
    /// Non-zero means the frozen model's rate scale is so far off that the null
    /// cannot be matched exactly, and those anchors' baselines are only
    /// approximate — a real caveat on their inclusion calls, not a rounding nit.
    pub n_clamped: usize,
}

/// How feature ROWS map onto sampling ANCHORS, for a model whose estimand is not
/// one-per-row.
///
/// `senna bge` embeds each row freely, so a row *is* an anchor and no grouping is
/// needed. `faba gem` shares one `β_g` across a gene's spliced and unspliced rows
/// (β-sharing), so its anchor is the **gene** — and its two gates read different
/// tracks: identity `β_g` from the spliced rows, velocity `δ_g` from the unspliced
/// ones. Both are expressed here by mapping the wanted rows to their gene and
/// dropping the rest, so the bucketing pass is written once rather than re-rolled
/// per caller.
pub struct RowGrouping<'a> {
    /// Feature row → anchor id, length `n_features`. `u32::MAX` drops the row,
    /// which is how a track is selected.
    pub row_to_anchor: &'a [u32],
    /// Number of anchors (rows may map to fewer anchors than there are rows).
    pub n_anchors: usize,
}

/// The fitted MAP side the samplers hold frozen: the cell embedding + biases,
/// pulled off the model once.
///
/// Bundled because these five values travel together through every builder and
/// sampler entry point, and three of them are same-typed `&[f32]` — a transposed
/// pair would compile and silently produce a wrong posterior.
pub struct FrozenMap {
    /// `[n_cells × h]` row-major MAP cell embedding.
    pub e_cell: Vec<f32>,
    /// `[n_cells]` MAP cell biases.
    pub b_cell: Vec<f32>,
    /// `[n_features]` MAP feature biases (a starting point; see [`build_index`]).
    pub b_feat: Vec<f32>,
    pub h: usize,
}

impl FrozenMap {
    /// Pull the frozen side off a fitted model, moving it to CPU.
    pub fn from_model(model: &crate::model::JointEmbedModel) -> anyhow::Result<Self> {
        let cpu = candle_util::candle_core::Device::Cpu;
        let to_vec = |t: &candle_util::candle_core::Tensor| -> anyhow::Result<Vec<f32>> {
            Ok(t.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?)
        };
        Ok(Self {
            e_cell: to_vec(&model.e_cell)?,
            b_cell: to_vec(&model.b_cell)?,
            b_feat: to_vec(&model.b_feat)?,
            h: model.e_cell.dim(1)?,
        })
    }
}

/// Stream the count backend and build a **calibrated** contrastive index against
/// the frozen side.
///
/// `grouping` is `None` for one anchor per feature row (`senna bge`), or a
/// [`RowGrouping`] to pool rows into genes and drop the ones a track does not read
/// (`faba gem`). `offset` carries a per-anchor frozen direction for a second,
/// dependent effect (gem's velocity gate holds `β_g` while sampling `δ_g`). When
/// `n_partition > 0` a frozen slate of that many other-side rows is drawn once
/// (seeded by `seed`) and its scale set to `n_other / n_partition`;
/// `n_partition == 0` uses every row exactly (`scale = 1`).
///
/// The per-anchor intercept `anchor_b` is carried but **not** used by the gate or
/// hyper samplers: they run the profile likelihood
/// ([`super::lnpdf::multinomial_ll`]), which maximizes the intercept out in closed
/// form. That is why no calibration step appears here — see
/// [`ContrastiveIndex::calibrate_anchor_bias`] for the fixed-intercept Poisson
/// case that still needs one.
pub fn build_index(
    unified: &UnifiedData,
    frozen: &FrozenMap,
    n_partition: usize,
    seed: u64,
    grouping: Option<RowGrouping<'_>>,
    offset: Option<&[f32]>,
) -> anyhow::Result<ContrastiveIndex> {
    let mut idx = build_inner(unified, frozen, n_partition, seed, grouping)?;
    idx.anchor_offset = offset.map(<[f32]>::to_vec);
    Ok(idx)
}

/// Anchors with no observed counts at all — their likelihood is flat, so the
/// posterior is the prior. Worth reporting: on a full gene annotation this can be
/// over half the axis.
impl ContrastiveIndex {
    #[must_use]
    pub fn n_empty_anchors(&self) -> usize {
        self.pos.iter().filter(|p| p.is_empty()).count()
    }
}

fn build_inner(
    unified: &UnifiedData,
    frozen: &FrozenMap,
    n_partition: usize,
    seed: u64,
    grouping: Option<RowGrouping<'_>>,
) -> anyhow::Result<ContrastiveIndex> {
    let (e_cell, b_cell, b_feat, h) = (
        frozen.e_cell.as_slice(),
        frozen.b_cell.as_slice(),
        frozen.b_feat.as_slice(),
        frozen.h,
    );
    let data = unified.count_backend();
    let n_cells = data.num_columns();
    let n_features = unified.n_features();
    let backend_rows = data.num_rows();

    // backend row → unified feature id (u32::MAX ⇒ dropped by a subset).
    let mut backend_to_unified = vec![u32::MAX; backend_rows];
    for (uid, &brow) in unified.feature_to_backend_row.iter().enumerate() {
        if brow < backend_rows {
            backend_to_unified[brow] = uid as u32;
        }
    }

    // Slab width ~8M edges (mirrors `fit`'s cell sampler); fall back to a fixed
    // cell-count slab when nnz can't be reported.
    let chunk = match data.num_non_zeros() {
        Ok(nnz) if nnz > 0 => {
            let avg = (nnz / n_cells.max(1)).max(1);
            (8_000_000 / avg).clamp(1, n_cells.max(1))
        }
        _ => (1usize << 14).min(n_cells.max(1)),
    };

    // Resolve the row→anchor map once; the ungrouped case is the identity.
    if let Some(g) = &grouping {
        anyhow::ensure!(
            g.row_to_anchor.len() == n_features,
            "row_to_anchor has {} entries but the feature axis has {n_features}",
            g.row_to_anchor.len()
        );
    }
    let n_anchors = grouping.as_ref().map_or(n_features, |g| g.n_anchors);
    let anchor_of = |uid: usize| -> u32 {
        grouping
            .as_ref()
            .map_or(uid as u32, |g| g.row_to_anchor[uid])
    };

    // Bucket nonzeros by anchor. Passing `0..n_cells` makes `out_col` the global
    // cell id directly. A grouped anchor accumulates every mapped row's edges, so
    // one cell can appear more than once — the Poisson data term is a sum over
    // edges, so repeated `(cell, count)` entries add up exactly as pooling should.
    let mut pos: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_anchors];
    data.for_each_triplet(0..n_cells, chunk, |brow, out_col, v| {
        if v == 0.0 {
            return;
        }
        let uid = backend_to_unified[brow as usize];
        if uid == u32::MAX {
            return;
        }
        let a = anchor_of(uid as usize);
        if a == u32::MAX {
            return; // row dropped by the grouping (e.g. the other splice track)
        }
        pos[a as usize].push((out_col as u32, v));
    })?;

    // Per-anchor starting bias: the first row that maps to it. Refit by
    // `calibrate_anchor_bias`, which is what makes the pooled intercept meaningful.
    let anchor_b = match &grouping {
        None => b_feat.to_vec(),
        Some(g) => {
            let mut b = vec![0.0f32; n_anchors];
            let mut seen = vec![false; n_anchors];
            for (uid, &a) in g.row_to_anchor.iter().enumerate() {
                if a != u32::MAX && !seen[a as usize] {
                    b[a as usize] = b_feat[uid];
                    seen[a as usize] = true;
                }
            }
            b
        }
    };

    // Frozen negative slate (Trap 1): drawn once, shared across genes.
    let (partition, partition_scale) = if n_partition == 0 || n_partition >= n_cells {
        ((0..n_cells as u32).collect(), 1.0)
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut all: Vec<u32> = (0..n_cells as u32).collect();
        all.partial_shuffle(&mut rng, n_partition);
        all.truncate(n_partition);
        (all, n_cells as f64 / n_partition as f64)
    };

    Ok(ContrastiveIndex {
        other_e: e_cell.to_vec(),
        other_b: b_cell.to_vec(),
        h,
        pos,
        anchor_b,
        partition,
        partition_scale,
        anchor_offset: None,
    })
}

#[cfg(test)]
#[path = "index_tests.rs"]
mod index_tests;
