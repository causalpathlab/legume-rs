//! Per-pair Poisson-MAP projection onto cage's frozen gene embedding.
//!
//! cage trains cells and genes into one shared `D`-dim space by predicting
//! spatial adjacency. What the rest of pinto consumes, though, is a *pair*
//! latent: `lc` / `dsvd` cluster cell pairs into link communities and derive a
//! cell's propensity from the mix its incident edges carry. This module
//! produces that pair latent the same way `senna bge` / `senna gem` produce a
//! cell latent in phase 2 — freeze the feature side, re-estimate each node
//! against its own counts.
//!
//! The node here is the pair, and its observation is the POOLED count
//! `n_uv,g = x_gu + x_gv`: Poisson-exact (a sum of Poissons is Poisson), and it
//! puts a pair that straddles a boundary *between* the two programs it pools
//! rather than on either one — which is what earns interface edges their own
//! cluster without an explicit difference channel.
//!
//! ```text
//! s_uv,g = ⟨e_g, e_uv⟩ + b_g + β_uv
//! μ_uv,g = exp(s_uv,g)
//! L      = Σ_{g ∈ G} μ_uv,g  −  Σ_g n_uv,g · s_uv,g  +  (λ/2)‖e_uv‖²
//! ```
//!
//! - `e_g` is cage's trained `e_feat` row, **frozen**. That is what makes the
//!   problem identified: the basis is pinned by the dictionary, so the `O(D)`
//!   rotation freedom that sank the free-edge-embedding experiment cannot
//!   exist here, and the objective (Poisson NLL + ridge) is strictly convex in
//!   `e_uv`.
//! - `b_g` is the **empirical** log gene abundance, not cage's `b_feat`. The
//!   trained gene bias came out of a logistic NCE and is a graph-popularity
//!   term, not a log-rate; this likelihood needs a log-rate. Fixing `b_g` to
//!   data also closes the gauge freedom geu's phase 2 has to correct for
//!   (`θ ← θ − v`, `b_g ← b_g + ⟨e_g, v⟩` leaves every score identical *only*
//!   when `b_g` is free to absorb the shift) — so no gauge fix is needed here,
//!   and `θ = 0` means "composition equal to the population average" rather
//!   than an arbitrary corner of the space.
//! - `β_uv` is a free scalar per pair. It absorbs the pooled library size, so
//!   `e_uv` carries composition only — the job `b_cell` does in bge/gem phase 2
//!   ("always fitted … keeping `e_c` depth-corrected"). With `β_uv` free the
//!   Poisson MAP *is* the multinomial MAP over the pair's gene composition.
//!
//! # How it is solved
//!
//! The encoder ([`encoder`]) amortizes it: a shared trunk reads each
//! endpoint's sufficient statistic — the counts enter the objective only
//! through `Σ_g n_g e_g`, `N` and `Σ_g n_g b_g`, three sparse sums — into a
//! cell code and a small gated mixture turns two codes into the pair code,
//! trained on this same likelihood and run once over every pair on the
//! device. Per node the
//! problem is `D+1` parameters and strictly convex, so the exact solver is
//! Newton with the partition summed over the active axis
//! ([`newton_polish`]): the check the encoder is held to on a seeded sample
//! after every fit, and the finisher for the few placements whose certificate
//! puts them far from the optimum.
//!
//! Two things keep a solve cheap:
//!
//! 1. **`β_uv` is profiled out, not descended.** Given `e_uv`, the optimal
//!    intercept is closed-form (`Σ_g μ_g = N_uv`), so it is solved exactly each
//!    step and never enters the Newton system — and by the envelope theorem
//!    the gradient of the profile objective is just the full gradient evaluated
//!    there, so this costs nothing in correctness. What is left is the
//!    multinomial gradient
//!    `N_uv · (predicted composition mean − observed composition mean) + λ e_uv`
//!    and its Hessian `N_uv · Cov_p(e) + λI`, which need no partition *value*,
//!    only normalized weights — so nothing overflows and the ridge is the only
//!    thing setting scale.
//! 2. **The data term is a constant.** `Σ_g n_uv,g · e_g` is linear in the
//!    parameters, so it is computed once per node and never re-derived inside
//!    the loop.
//!
//! # Why not geu's block SGD
//!
//! `graph-embedding-util`'s `fit::projection::block_sgd` already generalizes
//! "frozen dictionary, per-node Poisson-MAP" over two node types (cells,
//! pseudobulks) and sums the partition exactly, as one side of a matmul it is
//! computing anyway. A pair is a third node type, so the obvious question is
//! why it isn't fed to that engine. Three reasons, in order of how binding they
//! are:
//!
//! - Its entry point is `pub(crate)`, and its batch divisor is wired to bge /
//!   gem's pseudobulk hierarchy (the per-batch gene fold `δ` indexed by each cell's batch), which a
//!   spatial pair — batch-divided per endpoint, before pooling — does not have.
//!   Reaching it means widening another crate's API and generalizing that
//!   abstraction for one caller.
//! - It is candle/`Device`-coupled, and each pair is `D+1` parameters — the
//!   arithmetic is nowhere near GEMM-shaped per node.
//! - The profiled intercept above removes the reason to form the partition
//!   *value* at all: the gradient and the Hessian need normalized weights, and
//!   a `D × D` Newton step per node is what a strictly convex `D+1`-parameter
//!   problem calls for — a saving a block-SGD engine has no way to express.
//!
//! If a second caller ever wants this, the right move is to lift the solver,
//! not to widen the engine — the per-node loop below has no pair-specific
//! arithmetic in it.

use crate::util::common::*;
use crate::util::gene_axis::GeneAxis;
use candle_util::candle_core::Device;
use matrix_util::utils::{generate_minibatch_intervals, quantiles};

/// Clamp on the linear predictor before `exp`. f32 overflows at 88; the same
/// bound geu puts on every Poisson fit in the workspace.
const SCORE_CLAMP: f32 = 30.0;

pub(crate) mod encoder;
mod scoring;
pub use encoder::PairEncoderSpec;
pub use scoring::PairScore;

/// A placement whose certificate — the Newton decrement, the excess in nats
/// the local quadratic model puts on it over the optimum — exceeds this is
/// finished exactly: a row this far out is one the shared map extrapolated
/// on.
const RESCUE_GAP_NATS: f32 = 16.0;
/// Newton steps at most for such a row; the solve stops early once its
/// gradient is small.
const RESCUE_STEPS: usize = 32;
/// Newton steps at most for a solve from the origin. The objective is
/// strictly convex and the step is line-searched, so a solve settles well
/// inside this; the cap only bounds a degenerate row.
const SOLVE_STEPS: usize = 64;
/// A Newton step longer than this, in `e_uv` units, is a long way from the
/// quadratic regime, so it is line-searched: shrunk by halves until it lowers
/// the objective by at least [`ARMIJO`] of what the linear model promised.
/// Shorter steps are taken whole — they are where Newton converges
/// quadratically, and an objective difference that small is inside f32
/// rounding.
const NEWTON_LONG_STEP: f32 = 1.0;
const ARMIJO: f32 = 1e-4;
const MAX_BACKTRACKS: usize = 20;
/// Below this gradient norm, relative to the depth, the Newton solve is done.
const NEWTON_GRAD_TOL: f32 = 1e-4;

/// Where the encoder that places the pairs comes from.
#[derive(Debug, Clone, Copy)]
pub enum PairSolver<'a> {
    /// Fit the encoder on this run's pairs, place everything with it, and
    /// save it to `save_to`.
    TrainEncoder {
        spec: &'a PairEncoderSpec,
        dev: &'a Device,
        save_to: &'a str,
    },
    /// Place everything with an encoder saved by an earlier run.
    LoadEncoder { path: &'a str, dev: &'a Device },
}

/// Knobs for [`project_pairs`]: where the encoder comes from, plus how the
/// pair axis is walked. The objective's one parameter, the ridge, travels
/// with the encoder — set in [`PairEncoderSpec`] when it is trained, read
/// back from the file when it is loaded.
#[derive(Debug, Clone)]
pub struct PairProjectionArgs<'a> {
    pub solver: PairSolver<'a>,
    /// Seed; each pair derives its own stream so the fit is reproducible
    /// regardless of how rayon schedules the work.
    pub seed: u64,
    /// Pairs per encoder block, and cells per read block.
    pub pair_block: usize,
    /// Feature names the agreement correlations are computed over, from
    /// `--eval-features`. `None` leaves them `NaN`: correlating over the whole
    /// active axis means a sort per pair, and pairs outnumber cells by an order
    /// of magnitude, so it is opt-in rather than a silent cost.
    pub eval_features: Option<Vec<Box<str>>>,
    /// Score every pair's held-out likelihood ([`PairScore`]). Off, `scores`
    /// comes back empty: the exhaustive pass per pair is only worth paying
    /// where the numbers are written out.
    pub score_pairs: bool,
}

/// Per-endpoint batch division, applied to each cell's counts *before* they are
/// pooled — the same per-batch fold divide `senna bge` applies in phase 2, so a
/// pair's latent reflects de-batched composition. Without it a multi-batch run
/// clusters edges by batch, since every edge is within-batch by construction.
#[derive(Copy, Clone)]
pub struct PairBatchDivisor<'a> {
    /// `[n_genes × n_batches]` multiplicative batch effect `δ`.
    pub delta: &'a Mat,
    /// Cell → its column of `delta`.
    pub batch_of_cell: &'a [u32],
}

/// Every cell's own placement, in column order: the MAP of its doubled
/// profile `2x_c`, whose composition is `x_c`'s — the self-pair, so a cell
/// sits where the same map puts a pair of two copies of it.
pub struct CellLatent {
    /// `[n_cells × D]`.
    pub latent: Mat,
    /// Per-cell intercept, `[n_cells]`.
    pub bias: Vec<f32>,
}

/// What the projection hands back, in `edges` order.
pub struct PairLatent {
    /// `[n_pairs × D]` pair embedding `e_uv`.
    pub latent: Mat,
    /// Fitted per-pair intercept `β_uv`, `[n_pairs]`. Not consumed downstream
    /// (clustering is on the composition), but it is the pair's log pooled
    /// depth and worth keeping for diagnostics.
    pub bias: Vec<f32>,
    /// Held-out predictive score per pair, against the model's own abundance
    /// null; empty unless [`PairProjectionArgs::score_pairs`].
    pub scores: Vec<PairScore>,
    pub cells: CellLatent,
}

/// The frozen side of the projection, flattened once for the inner loop.
///
/// `e_feat` is `[G × D]` column-major (nalgebra), so a per-gene row read is
/// strided — every solve would walk the matrix against the cache. Rows are
/// copied out row-major once here instead, restricted to genes that carry any
/// count at all (a gene with zero total appears in no pair's profile and
/// contributes `exp(-∞) = 0` to the partition).
pub struct PairDictionary {
    /// Row-major `[n_active × D]`.
    feat: Vec<f32>,
    /// Empirical log gene abundance, `[n_active]`.
    b: Vec<f32>,
    /// Global gene id → active-list position, `u32::MAX` when inactive.
    local_of_gene: Vec<u32>,
    d: usize,
    /// `ln Σ_g exp(b_g)` — the log-partition at `e_uv = 0`. Stored in log space
    /// because every use is a log-space one, and it is fixed for the whole run.
    log_z: f32,
}

impl PairDictionary {
    /// Build the frozen side from cage's `[G × D]` gene embedding and the
    /// per-gene count totals over all cells (`n_cells` turns those totals into
    /// the per-cell mean the log-rate offset needs).
    pub fn new(e_feat: &Mat, gene_totals: &[f64], n_cells: usize) -> anyhow::Result<Self> {
        let n_genes = gene_totals.len();
        let d = e_feat.ncols();
        anyhow::ensure!(
            e_feat.nrows() == n_genes,
            "pair projection: e_feat has {} rows, expected {n_genes}",
            e_feat.nrows()
        );
        anyhow::ensure!(d > 0, "pair projection: empty embedding dimension");
        anyhow::ensure!(n_cells > 0, "pair projection: no cells");

        let active: Vec<usize> = (0..n_genes).filter(|&g| gene_totals[g] > 0.0).collect();
        anyhow::ensure!(
            !active.is_empty(),
            "pair projection: every gene has zero total count"
        );

        let mut local_of_gene = vec![u32::MAX; n_genes];
        let mut feat = Vec::with_capacity(active.len() * d);
        let mut b = Vec::with_capacity(active.len());
        for (local, &g) in active.iter().enumerate() {
            local_of_gene[g] = local as u32;
            for j in 0..d {
                feat.push(e_feat[(g, j)]);
            }
            // Mean count per cell, on the log scale the Poisson rate lives on.
            // The pooled-pair factor of two is constant across genes and is
            // absorbed by `β_uv`.
            let m = gene_totals[g] / n_cells as f64;
            b.push(m.ln() as f32);
        }

        // `Σ_g exp(b_g)` without ever calling `exp`: `b_g` IS `ln(total_g/n)`,
        // so the summands are the gene means themselves and the sum is the mean
        // library size. Accumulating those in f64 is exact where
        // `Σ exp(ln(mean))` would round-trip every term through two
        // transcendentals — and no max-subtraction is needed, since the naive
        // form's failure mode (overflow on a large `b_g`) cannot arise from a
        // sum of per-cell mean counts.
        let log_z = {
            let mean_lib: f64 =
                active.iter().map(|&g| gene_totals[g]).sum::<f64>() / n_cells as f64;
            anyhow::ensure!(
                mean_lib > 0.0 && mean_lib.is_finite(),
                "pair projection: mean library size is {mean_lib}, expected a positive finite value"
            );
            mean_lib.ln() as f32
        };
        Ok(Self {
            feat,
            b,
            local_of_gene,
            d,
            log_z,
        })
    }

    /// Number of genes carrying counts — the axis the partition runs over.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.b.len()
    }

    /// Map a `(global gene id, count)` profile onto the active-list positions the
    /// solver and the scorer both index by. Genes with no counts anywhere are dropped:
    /// they carry no information and are not on the partition axis.
    #[must_use]
    fn to_local(&self, obs: &[(u32, f32)]) -> Vec<(u32, f32)> {
        obs.iter()
            .filter_map(|&(g, n)| {
                let l = *self.local_of_gene.get(g as usize)?;
                (l != u32::MAX && n > 0.0).then_some((l, n))
            })
            .collect()
    }

    /// The statistic a local profile enters the objective through:
    /// `(Σ_g n_g e_g, Σ_g n_g, Σ_g n_g b_g)`.
    pub(crate) fn statistic(&self, genes: &[u32], counts: &[f32]) -> (Vec<f32>, f32, f32) {
        let d = self.d;
        let mut sums = vec![0f32; d];
        let (mut total, mut offset) = (0f32, 0f32);
        for (&g, &n) in genes.iter().zip(counts) {
            let row = &self.feat[g as usize * d..(g as usize + 1) * d];
            for (s, &e) in sums.iter_mut().zip(row) {
                *s += n * e;
            }
            total += n;
            offset += n * self.b[g as usize];
        }
        (sums, total, offset)
    }

    /// The clamped log-rates `⟨e_g, θ⟩ + b_g` over the active axis.
    fn log_rates(&self, theta: &[f32]) -> Vec<f32> {
        let d = self.d;
        self.feat
            .chunks_exact(d)
            .zip(&self.b)
            .map(|(row, &b)| {
                let dot: f32 = row.iter().zip(theta).map(|(&e, &t)| e * t).sum();
                (dot + b).clamp(-SCORE_CLAMP, SCORE_CLAMP)
            })
            .collect()
    }

    /// The exact multinomial NLL of a local profile at `θ`, up to the ridge:
    /// the objective the solver, the finisher and the encoder all minimise.
    pub(crate) fn nll(&self, obs: &[(u32, f32)], theta: &[f32]) -> f32 {
        let rates = self.log_rates(theta);
        let lse = scoring::log_sum_exp(rates.iter().copied());
        let (mut total, mut data) = (0f32, 0f32);
        for &(g, n) in obs {
            total += n;
            data += n * rates[g as usize];
        }
        total * lse - data
    }

    /// Solve one pair exactly from its `(global gene id, pooled count)`
    /// profile. Returns `(e_uv, β_uv, certificate)`.
    #[cfg(test)]
    #[must_use]
    pub fn solve(&self, obs: &[(u32, f32)], ridge: f32) -> (Vec<f32>, f32, f32) {
        solve_exact(&self.to_local(obs), self, ridge)
    }

    /// Finish one pair from `init` by Newton steps on the exact objective.
    /// Returns `(e_uv, β_uv, certificate)`.
    #[cfg(test)]
    #[must_use]
    pub fn polish(
        &self,
        obs: &[(u32, f32)],
        ridge: f32,
        init: &[f32],
        max_steps: usize,
    ) -> (Vec<f32>, f32, f32) {
        newton_polish(&self.to_local(obs), self, ridge, init, max_steps)
    }
}

/// Project every cell pair — and every cell — onto cage's frozen gene embedding.
///
/// `gene_totals` is per GENE, already folded off the row axis — both the
/// partition and each pair's profile live there, because `e_feat` is per gene
/// and a channelized matrix's two rows are one gene's pooled count rather than
/// two categories of the multinomial. It is passed in rather than computed here
/// because it is a whole-matrix streaming pass and the caller has already made
/// it for the splice report.
///
/// `e_feat` is cage's trained `[G × D]` gene embedding, used as-is: the
/// selection gate is already expressed in its values, so re-applying `pip` here
/// would shrink the same selection twice. Genes the gate drove to `‖e_g‖ ≈ 0`
/// contribute a constant `exp(b_g + β)` to the partition and therefore cannot
/// pull on `e_uv` — no special-casing needed.
///
/// Every cell is read once, in contiguous column blocks, into the corpus the
/// encoder trains on; every pair is then the merge of two rows and every cell
/// its own doubled row.
pub fn project_pairs(
    data: &SparseIoVec,
    edges: &[(u32, u32)],
    e_feat: &Mat,
    batch: Option<PairBatchDivisor<'_>>,
    args: &PairProjectionArgs<'_>,
    axis: &GeneAxis,
    gene_totals: &[f64],
) -> anyhow::Result<PairLatent> {
    let n_genes = axis.n_genes();
    let n_cells = data.num_columns();
    let d = e_feat.ncols();
    anyhow::ensure!(
        axis.n_rows() == data.num_rows(),
        "pair projection: gene axis has {} rows, data has {}",
        axis.n_rows(),
        data.num_rows()
    );
    anyhow::ensure!(
        e_feat.nrows() == n_genes,
        "pair projection: e_feat has {} rows, data has {n_genes} genes",
        e_feat.nrows()
    );
    anyhow::ensure!(d > 0, "pair projection: empty embedding dimension");
    anyhow::ensure!(
        gene_totals.len() == n_genes,
        "pair projection: {} gene totals, expected {n_genes}",
        gene_totals.len()
    );

    let dict = PairDictionary::new(e_feat, gene_totals, n_cells)?;
    let scored_positions: Option<Vec<u32>> = match args.eval_features.as_ref() {
        Some(names) => {
            let positions = dict.eval_positions(axis.gene_names(), names);
            anyhow::ensure!(
                !positions.is_empty(),
                "--eval-features matched no gene that carries counts in this sample"
            );
            info!(
                "Agreement axis: {} of {} named features carry counts here",
                positions.len(),
                names.len()
            );
            Some(positions)
        }
        None => None,
    };
    // Resolved once for the whole run: the membership bitmap and the null's
    // log-partition are the same for every pair.
    let eval_axis = dict.eval_axis(scored_positions);
    if dict.n_active() < n_genes {
        info!(
            "Pair projection: {} of {n_genes} genes carry counts; the rest sit out the partition",
            dict.n_active(),
        );
    }
    let corpus = build_corpus(data, &dict, batch, axis, args.pair_block)?;

    let encoded = match args.solver {
        PairSolver::TrainEncoder { spec, dev, save_to } => {
            let enc = encoder::PairEncoder::build(&dict, &corpus, spec, args.seed, dev)?;
            let stats = enc.train(&corpus, edges, spec, args.seed)?;
            info!(
                "Pair encoder: {} steps; held-out NLL/count after {:.4}",
                stats.steps, stats.nll_per_count
            );
            enc.save(save_to)?;
            info!("Wrote {save_to}");
            project_with_encoder(&enc, &dict, &corpus, edges, args)?
        }
        PairSolver::LoadEncoder { path, dev } => {
            let enc = encoder::PairEncoder::load(&dict, path, dev)?;
            info!(
                "Pair encoder: loaded {path} (L={}, K={}, λ={})",
                enc.trunk_width(),
                enc.n_experts(),
                enc.ridge()
            );
            project_with_encoder(&enc, &dict, &corpus, edges, args)?
        }
    };

    let scores = if args.score_pairs {
        edges
            .par_iter()
            .enumerate()
            .map(|(i, &(u, v))| {
                let obs = corpus[u as usize].pooled(&corpus[v as usize]);
                let z: Vec<f32> = encoded.pairs.latent.row(i).iter().copied().collect();
                dict.score_local(&obs, &z, &eval_axis)
            })
            .collect()
    } else {
        Vec::new()
    };

    let encoder::Encoded { pairs, cells } = encoded;
    Ok(PairLatent {
        latent: pairs.latent,
        bias: pairs.bias,
        scores,
        cells: CellLatent {
            latent: cells.latent,
            bias: cells.bias,
        },
    })
}

////////////////
// The corpus //
////////////////

/// Every cell's active-axis row, read once in contiguous column blocks.
fn build_corpus(
    data: &SparseIoVec,
    dict: &PairDictionary,
    batch: Option<PairBatchDivisor<'_>>,
    axis: &GeneAxis,
    block: usize,
) -> anyhow::Result<Vec<encoder::CellRow>> {
    let n_cells = data.num_columns();
    let mut corpus: Vec<encoder::CellRow> = Vec::with_capacity(n_cells);
    let bar = new_progress_bar(n_cells as u64).with_message("reading cells");
    for (lb, ub) in generate_minibatch_intervals(n_cells, axis.n_genes(), Some(block.max(1))) {
        let slab = data.read_columns_csc(lb..ub)?;
        let (offsets, rows, vals) = (slab.col_offsets(), slab.row_indices(), slab.values());
        let block_rows: Vec<encoder::CellRow> = (lb..ub)
            .into_par_iter()
            .map(|c| {
                let col = c - lb;
                let (s, e) = (offsets[col], offsets[col + 1]);
                let counts = endpoint_counts(&rows[s..e], &vals[s..e], c as u32, batch);
                encoder::CellRow::from_profile(dict, &axis.pool_profile(counts))
            })
            .collect();
        corpus.extend(block_rows);
        bar.inc((ub - lb) as u64);
    }
    bar.finish_and_clear();
    let nnz: usize = corpus.iter().map(|r| r.genes.len()).sum();
    info!(
        "Pair corpus: {n_cells} cells, {nnz} counts on the {}-gene active axis ({:.1} MB)",
        dict.n_active(),
        (nnz * 8) as f64 / 1e6
    );
    Ok(corpus)
}

/// One cell's `(row, count)` profile from its CSC column, batch-divided when
/// the run has batches — the one place a cell's counts are read.
fn endpoint_counts(
    rows: &[usize],
    vals: &[f32],
    cell: u32,
    batch: Option<PairBatchDivisor<'_>>,
) -> Vec<(u32, f32)> {
    let mut vals = vals.to_vec();
    if let Some(bd) = batch {
        let b = bd.batch_of_cell[cell as usize] as usize;
        adjust_by_poisson_ratio(&mut vals, |k| bd.delta[(rows[k], b)]);
    }
    rows.iter()
        .zip(vals)
        .filter(|(_, x)| *x > 0.0)
        .map(|(&r, x)| (r as u32, x))
        .collect()
}

/////////////////////
// The encoder arm //
/////////////////////

/// Every pair and every cell through the encoder, then the check against the
/// exact solve and the finishing of the rows the certificate puts far out.
///
/// The check is always reported: it is the one number that says whether the
/// encoder earned its place on this run.
fn project_with_encoder(
    enc: &encoder::PairEncoder,
    dict: &PairDictionary,
    corpus: &[encoder::CellRow],
    edges: &[(u32, u32)],
    args: &PairProjectionArgs<'_>,
) -> anyhow::Result<encoder::Encoded> {
    let mut encoded = enc.encode_all(corpus, edges, args.pair_block, encoder::CELL_BLOCK)?;

    // The amortization gap: how far the shared map sits from the per-pair
    // optimum, on a seeded sample.
    // The ridge is the encoder's: the one it was fitted under, whether this
    // run trained it or loaded it.
    let ridge = enc.ridge();
    let check = encoder::ExactCheck::new(dict, corpus, edges, ridge, args.seed);
    let report = |what: &str, latent: &Mat| {
        let gap = check.compare(dict, latent);
        info!(
            "{what} vs the converged MAP on {} pairs: mean cosine {:.3}, NLL ratio {:.5}; \
             ‖z‖ median/max {:.2}/{:.2} against {:.2}/{:.2}",
            check.n_pairs(),
            gap.mean_cosine,
            gap.nll_ratio,
            gap.norm_encoder.0,
            gap.norm_encoder.1,
            gap.norm_exact.0,
            gap.norm_exact.1
        );
    };
    report("Pair encoder", &encoded.pairs.latent);

    // Every placement carries a certificate; the rows it puts far out — the
    // rare inputs a shared map extrapolates on — are finished exactly.
    let spread = |gap: &[f32]| quantiles(gap, &[0.5, 0.99, 1.0]);
    let (p, c) = (spread(&encoded.pairs.gap), spread(&encoded.cells.gap));
    info!(
        "Placement certificate (nats above the optimum), median/99%/max: pairs {:.3}/{:.2}/{:.1}, \
         cells {:.3}/{:.2}/{:.1}",
        p[0], p[1], p[2], c[0], c[1], c[2]
    );
    let n_pairs = finish_rows(dict, &mut encoded.pairs, ridge, 0.0, |e| {
        let (u, v) = edges[e];
        corpus[u as usize].pooled(&corpus[v as usize])
    });
    // The cell's intercept is its own depth, not the doubled one the solve sees.
    let n_cells = finish_rows(
        dict,
        &mut encoded.cells,
        ridge,
        -std::f32::consts::LN_2,
        |c| corpus[c].doubled(),
    );
    if n_pairs + n_cells > 0 {
        info!(
            "Finished {n_pairs} pairs and {n_cells} cells exactly (up to {RESCUE_STEPS} Newton \
             steps each): placed more than {RESCUE_GAP_NATS} nats above the optimum"
        );
        report("After finishing", &encoded.pairs.latent);
    }
    Ok(encoded)
}

/// Finish the rows of `placement` whose certificate exceeds
/// [`RESCUE_GAP_NATS`] by Newton steps on the exact objective from the
/// encoder's placement, and re-certify them. `profile_of` gives a row's
/// local profile; `bias_shift` moves the solved intercept onto the row's own
/// depth. Returns how many rows were finished.
fn finish_rows(
    dict: &PairDictionary,
    placement: &mut encoder::Placement,
    ridge: f32,
    bias_shift: f32,
    profile_of: impl Fn(usize) -> Vec<(u32, f32)> + Sync,
) -> usize {
    let ids: Vec<usize> = (0..placement.len())
        .filter(|&i| placement.gap[i] > RESCUE_GAP_NATS)
        .collect();
    let fits: Vec<(Vec<f32>, f32, f32)> = ids
        .par_iter()
        .map(|&i| {
            let init: Vec<f32> = placement.latent.row(i).iter().copied().collect();
            let (theta, beta, gap) =
                newton_polish(&profile_of(i), dict, ridge, &init, RESCUE_STEPS);
            (theta, beta + bias_shift, gap)
        })
        .collect();
    for (&i, (theta, beta, gap)) in ids.iter().zip(fits) {
        placement.set(i, &theta, beta, gap);
    }
    ids.len()
}

/////////////////
// The solvers //
/////////////////

/// One node's objective, set up once: the data half of the gradient (constant
/// in the parameters) and a composition pass over the active axis shared by
/// the Newton step, the objective it line-searches and the certificate.
struct PairProblem<'a> {
    dict: &'a PairDictionary,
    /// `Σ_g n_g e_g / N`: the observed composition mean.
    obs_mean: Vec<f32>,
    total: f32,
    log_total: f32,
    /// The active axis's normalised softmax weights, after [`Self::composition`].
    weights: Vec<f32>,
    /// `Σ_g w_g e_g`: the predicted composition mean, after [`Self::composition`].
    pred_mean: Vec<f32>,
}

impl<'a> PairProblem<'a> {
    /// `None` for a profile with no mass: the likelihood says nothing about
    /// it, and the origin is where the ridge puts it.
    fn new(obs: &[(u32, f32)], dict: &'a PairDictionary) -> Option<Self> {
        let d = dict.d;
        let total: f32 = obs.iter().map(|&(_, n)| n).sum();
        if obs.is_empty() || !total.is_finite() || total <= 0.0 {
            return None;
        }
        let mut obs_mean = vec![0f32; d];
        for &(g, n) in obs {
            let row = &dict.feat[g as usize * d..(g as usize + 1) * d];
            for (o, &e) in obs_mean.iter_mut().zip(row) {
                *o += n * e;
            }
        }
        for o in obs_mean.iter_mut() {
            *o /= total;
        }
        Some(Self {
            dict,
            obs_mean,
            total,
            log_total: total.ln(),
            weights: vec![0f32; dict.n_active()],
            pred_mean: vec![0f32; d],
        })
    }

    /// The composition the current `θ` predicts — normalised weights over the
    /// active axis and their mean dictionary row — and the log-partition
    /// `lse_g(⟨e_g, θ⟩ + b_g)`, from which the intercept that matches the
    /// node's mass is `β = ln N − lse` in closed form. `None` once the weights
    /// are not finite.
    fn composition(&mut self, theta: &[f32]) -> Option<f32> {
        let d = self.dict.d;
        let mut max_score = f32::NEG_INFINITY;
        for ((w, row), &b) in self
            .weights
            .iter_mut()
            .zip(self.dict.feat.chunks_exact(d))
            .zip(&self.dict.b)
        {
            let a: f32 = row.iter().zip(theta).map(|(&e, &t)| e * t).sum();
            *w = (a + b).clamp(-SCORE_CLAMP, SCORE_CLAMP);
            max_score = max_score.max(*w);
        }
        let mut w_sum = 0f32;
        for w in self.weights.iter_mut() {
            *w = (*w - max_score).exp();
            w_sum += *w;
        }
        if !w_sum.is_finite() || w_sum <= 0.0 {
            return None;
        }
        self.pred_mean.fill(0.0);
        for (w, row) in self.weights.iter_mut().zip(self.dict.feat.chunks_exact(d)) {
            *w /= w_sum;
            for (p, &e) in self.pred_mean.iter_mut().zip(row) {
                *p += *w * e;
            }
        }
        // Kept in log space: no partition value is ever exponentiated at full
        // scale.
        Some(max_score + w_sum.ln())
    }

    /// The gradient at the current composition: the multinomial's
    /// `N·(predicted − observed mean)` plus the ridge.
    fn gradient(&self, theta: &[f32], ridge: f32) -> Vec<f32> {
        self.pred_mean
            .iter()
            .zip(&self.obs_mean)
            .zip(theta)
            .map(|((&p, &o), &t)| self.total * (p - o) + ridge * t)
            .collect()
    }

    /// The profiled objective at `θ` given its log-partition, up to the
    /// constant `Σ_g n_g b_g`: `N·lse − N·⟨m, θ⟩ + (λ/2)‖θ‖²`.
    fn objective(&self, theta: &[f32], ridge: f32, lse: f32) -> f32 {
        let fit: f32 = self.obs_mean.iter().zip(theta).map(|(&m, &t)| m * t).sum();
        let sq: f32 = theta.iter().map(|&t| t * t).sum();
        self.total * (lse - fit) + 0.5 * ridge * sq
    }
}

/// One node's exact optimum from the origin. Returns `(e_uv, β_uv,
/// certificate)`; a node with no counts gets the origin.
fn solve_exact(obs: &[(u32, f32)], dict: &PairDictionary, ridge: f32) -> (Vec<f32>, f32, f32) {
    newton_polish(obs, dict, ridge, &vec![0f32; dict.d], SOLVE_STEPS)
}

/// Newton on one node's `e_uv` from `init`, with the partition summed exactly
/// and `β_uv` profiled out each step. The objective is strictly convex with
/// Hessian `N·Cov_p(e) + λI`, a `D × D` solve per step: from a warm start it
/// settles in a few steps, and from the origin a long step is line-searched
/// so it cannot overshoot. Returns `(e_uv, β_uv, certificate)`, the
/// certificate being the Newton decrement `½ ∇ᵀH⁻¹∇` at the returned
/// placement — the excess likelihood in nats the local quadratic model puts
/// on it over the optimum.
fn newton_polish(
    obs: &[(u32, f32)],
    dict: &PairDictionary,
    ridge: f32,
    init: &[f32],
    max_steps: usize,
) -> (Vec<f32>, f32, f32) {
    let d = dict.d;
    let Some(mut problem) = PairProblem::new(obs, dict) else {
        return (vec![0f32; d], 0.0, 0.0);
    };
    let mut theta = init.to_vec();
    let mut beta = problem.log_total - dict.log_z;
    let mut decrement = f32::INFINITY;
    let Some(mut lse) = problem.composition(&theta) else {
        return (theta, beta, decrement);
    };
    let mut hess = nalgebra::DMatrix::<f32>::zeros(d, d);

    // One pass more than the steps: the last only profiles `β` and the
    // certificate at the final `θ`. The composition is always the one at
    // `theta` here — every accepted step leaves it there.
    let max_steps = max_steps.max(1);
    for it in 0..=max_steps {
        beta = problem.log_total - lse;
        let grad = nalgebra::DVector::<f32>::from_vec(problem.gradient(&theta, ridge));
        // `N·(E[e eᵀ] − p̄ p̄ᵀ) + λI`, symmetric, positive definite.
        hess.fill(0.0);
        for (&w, row) in problem.weights.iter().zip(dict.feat.chunks_exact(d)) {
            for i in 0..d {
                let wi = w * row[i];
                for j in 0..=i {
                    hess[(i, j)] += wi * row[j];
                }
            }
        }
        for i in 0..d {
            for j in 0..=i {
                let v =
                    problem.total * (hess[(i, j)] - problem.pred_mean[i] * problem.pred_mean[j]);
                hess[(i, j)] = v;
                hess[(j, i)] = v;
            }
            hess[(i, i)] += ridge;
        }
        let Some(chol) = hess.clone().cholesky() else {
            break;
        };
        let step = chol.solve(&grad);
        decrement = 0.5 * grad.dot(&step);
        if it == max_steps || grad.norm() < NEWTON_GRAD_TOL * problem.total.max(1.0) {
            break;
        }
        let trial_at = |t: f32| -> Vec<f32> {
            theta
                .iter()
                .zip(step.iter())
                .map(|(&th, &s)| th - t * s)
                .collect()
        };
        // A short step is the quadratic regime: take it whole. A long one is
        // shrunk until it lowers the objective by a fraction of the decrease
        // the linear model promised (`∇ᵀstep`, positive since `H ≻ 0`).
        let accepted = if step.norm() <= NEWTON_LONG_STEP {
            let trial = trial_at(1.0);
            problem.composition(&trial).map(|l| (trial, l))
        } else {
            let f_here = problem.objective(&theta, ridge, lse);
            let promised = grad.dot(&step);
            let mut t = NEWTON_LONG_STEP / step.norm();
            let mut found = None;
            for _ in 0..MAX_BACKTRACKS {
                let trial = trial_at(t);
                if let Some(l) = problem.composition(&trial) {
                    if problem.objective(&trial, ridge, l) <= f_here - ARMIJO * t * promised {
                        found = Some((trial, l));
                        break;
                    }
                }
                t *= 0.5;
            }
            found
        };
        // Nothing along the step lowers the objective: `theta` is the optimum
        // to working precision, and `beta` and the decrement are already its.
        let Some((trial, l)) = accepted else {
            break;
        };
        theta = trial;
        lse = l;
    }
    (theta, beta, decrement)
}
