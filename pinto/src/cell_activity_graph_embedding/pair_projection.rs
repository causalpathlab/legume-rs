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
//! Each pair is an independent `D+1`-parameter problem, so the solve is rayon-
//! parallel over pairs (the outermost loop, and the only one — the per-pair
//! work is scalar) with its own Adam per pair.
//!
//! Two things make each step cheap:
//!
//! 1. **`β_uv` is profiled out, not descended.** Given `e_uv`, the optimal
//!    intercept is closed-form (`Σ_g μ_g = N_uv`), so it is solved exactly each
//!    step and never enters Adam — and by the envelope theorem the gradient of
//!    the profile objective is just the full gradient evaluated there, so this
//!    costs nothing in correctness. What is left is the multinomial gradient
//!    `N_uv · (predicted composition mean − observed composition mean) + λ e_uv`,
//!    which needs no partition *value*, only a normalized weight — so nothing
//!    overflows and the ridge is the only thing setting scale.
//! 2. **The partition is sampled.** Summing `Σ_{g ∈ G} μ_g` exactly costs
//!    `E × steps × G × D` flops (~3.4 TFLOP on a Visium run); instead each step
//!    draws [`PairProjectionArgs::gene_sample`] genes ∝ `exp(b_g)`, the
//!    empirical abundance. Because `exp(b_g)/q_g` is then constant, the
//!    importance weights cancel and the estimator is exact at `e_uv = 0`,
//!    leaving only the `⟨e_g, e_uv⟩` deviation to carry variance.
//!
//! The data term `Σ_g n_uv,g · e_g` is linear in the parameters, so it is a
//! constant per pair — computed once, never re-derived inside the loop.
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
//!   gem's pseudobulk hierarchy (`μ_residual` indexed by `cell_to_pb`), which a
//!   spatial pair — batch-divided per endpoint, before pooling — does not have.
//!   Reaching it means widening another crate's API and generalizing that
//!   abstraction for one caller.
//! - It is candle/`Device`-coupled. This step runs after cage's training loop
//!   has released the device, on CPU, and each pair is `D+1` parameters — the
//!   arithmetic is nowhere near GEMM-shaped per node.
//! - The profiled intercept above removes the reason to form the partition at
//!   all: the gradient needs a normalized weight, not a partition *value*. That
//!   is what makes a sampled estimate sufficient here, and it is a saving the
//!   exact-partition engine has no way to express.
//!
//! If a second caller ever wants this, the right move is to lift the solver,
//! not to widen the engine — the per-node loop below has no pair-specific
//! arithmetic in it.

use crate::util::common::*;
use crate::util::gene_axis::GeneAxis;
use matrix_util::utils::generate_minibatch_intervals;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

/// Clamp on the linear predictor before `exp`. f32 overflows at 88; the same
/// bound geu puts on every Poisson fit in the workspace.
const SCORE_CLAMP: f32 = 30.0;

/// Target per-step movement of the linear predictor `s`, used to auto-scale the
/// learning rate. Adam's per-coordinate step is ≈ `lr`, so `Δs ≈ lr · D · rms(e)`
/// and `lr = TARGET_DELTA_S / (D · rms(e))`. Without this the rate would have to
/// be re-tuned for every dictionary scale — `‖e_g‖` is not a fixed quantity.
const TARGET_DELTA_S: f32 = 0.05;

/// Learning-rate floor as a fraction of the initial rate, decayed linearly over
/// a pair's steps so the tail settles instead of dithering around the optimum.
const LR_FLOOR_FRAC: f32 = 0.05;

const ADAM_B1: f32 = 0.9;
const ADAM_B2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Knobs for one node's solve — everything [`PairDictionary::project`] reads,
/// and nothing it doesn't. Kept apart from [`PairProjectionArgs`] so the solve
/// boundary doesn't take orchestration parameters (seed, block sizes) it has no
/// use for.
#[derive(Debug, Clone)]
pub struct ProjectionArgs {
    /// Ridge `λ` on `e_uv` (never on `β_uv`, which must stay free to absorb
    /// depth).
    pub ridge: f32,
    /// Adam steps per pair.
    pub steps: usize,
    /// Genes drawn per step to estimate the log-partition. `0` sums every gene
    /// exactly — correct, and affordable only on small feature axes.
    pub gene_sample: usize,
}

impl Default for ProjectionArgs {
    fn default() -> Self {
        Self {
            ridge: 1.0,
            steps: 300,
            gene_sample: 512,
        }
    }
}

/// Knobs for [`project_pairs`]: the per-node solve plus how the pair axis is
/// walked.
#[derive(Debug, Clone)]
pub struct PairProjectionArgs {
    /// Passed through to every per-pair solve.
    pub projection: ProjectionArgs,
    /// Seed; each pair derives its own stream so the fit is reproducible
    /// regardless of how rayon schedules the work.
    pub seed: u64,
    /// Pairs per read block. Bounds the count slab held at once: a block reads
    /// the columns of its ≤ `2 × pair_block` distinct endpoints.
    pub pair_block: usize,
}

impl Default for PairProjectionArgs {
    fn default() -> Self {
        Self {
            projection: ProjectionArgs::default(),
            seed: 42,
            pair_block: 8192,
        }
    }
}

/// Per-endpoint batch division, applied to each cell's counts *before* they are
/// pooled — the same `μ_residual` divide `senna bge` applies in phase 2, so a
/// pair's latent reflects de-batched composition. Without it a multi-batch run
/// clusters edges by batch, since every edge is within-batch by construction.
#[derive(Copy, Clone)]
pub struct PairBatchDivisor<'a> {
    /// `[n_genes × n_batches]` multiplicative batch effect `δ`.
    pub delta: &'a Mat,
    /// Cell → its column of `delta`.
    pub batch_of_cell: &'a [u32],
}

/// What the projection hands back, in `edges` order.
pub struct PairLatent {
    /// `[n_pairs × D]` pair embedding `e_uv`.
    pub latent: Mat,
    /// Fitted per-pair intercept `β_uv`, `[n_pairs]`. Not consumed downstream
    /// (clustering is on the composition), but it is the pair's log pooled
    /// depth and worth keeping for diagnostics.
    pub bias: Vec<f32>,
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
    /// Draws genes ∝ `exp(b_g)`, making the importance weights cancel.
    proposal: WeightedIndex<f32>,
    /// Auto-scaled initial learning rate.
    lr0: f32,
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
        let mut weights = Vec::with_capacity(active.len());
        for (local, &g) in active.iter().enumerate() {
            local_of_gene[g] = local as u32;
            for j in 0..d {
                feat.push(e_feat[(g, j)]);
            }
            // Mean count per cell, on the log scale the Poisson rate lives on.
            // The pooled-pair factor of two is constant across genes and is
            // absorbed by `β_uv`.
            b.push((gene_totals[g] / n_cells as f64).ln() as f32);
            weights.push(gene_totals[g] as f32);
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
        let proposal = WeightedIndex::new(&weights)
            .map_err(|e| anyhow::anyhow!("pair projection: gene proposal: {e}"))?;

        let rms = {
            let ss: f64 = feat.iter().map(|&x| (x as f64) * (x as f64)).sum();
            ((ss / feat.len().max(1) as f64).sqrt() as f32).max(1e-6)
        };
        let lr0 = TARGET_DELTA_S / (d as f32 * rms);

        Ok(Self {
            feat,
            b,
            local_of_gene,
            d,
            log_z,
            proposal,
            lr0,
        })
    }

    /// Number of genes carrying counts — the axis the partition runs over.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.b.len()
    }

    /// Project one pair from its `(global gene id, pooled count)` profile.
    /// Genes with no counts anywhere are dropped (they carry no information and
    /// are not on the partition axis). Returns `(e_uv, β_uv)`.
    #[must_use]
    pub fn project(
        &self,
        obs: &[(u32, f32)],
        args: &ProjectionArgs,
        rng: &mut SmallRng,
    ) -> (Vec<f32>, f32) {
        let local: Vec<(u32, f32)> = obs
            .iter()
            .filter_map(|&(g, n)| {
                let l = *self.local_of_gene.get(g as usize)?;
                (l != u32::MAX && n > 0.0).then_some((l, n))
            })
            .collect();
        solve_pair(&local, self, args, rng)
    }
}

/// Project every cell pair onto cage's frozen gene embedding.
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
pub fn project_pairs(
    data: &SparseIoVec,
    edges: &[(u32, u32)],
    e_feat: &Mat,
    batch: Option<PairBatchDivisor<'_>>,
    args: &PairProjectionArgs,
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
    let n_pairs = edges.len();
    if n_pairs == 0 {
        return Ok(PairLatent {
            latent: Mat::zeros(0, d),
            bias: Vec::new(),
        });
    }

    let dict = PairDictionary::new(e_feat, gene_totals, n_cells)?;
    if dict.n_active() < n_genes {
        info!(
            "Pair projection: {} of {n_genes} genes carry counts; the rest sit out the partition",
            dict.n_active(),
        );
    }
    info!(
        "Pair projection: {} pairs × {} genes → {}-dim, ridge λ={}, {} Adam steps, \
         partition from {} sampled genes",
        n_pairs,
        dict.n_active(),
        d,
        args.projection.ridge,
        args.projection.steps,
        if args.projection.gene_sample == 0 {
            dict.n_active()
        } else {
            args.projection.gene_sample
        },
    );

    let mut latent = Mat::zeros(n_pairs, d);
    let mut bias = vec![0f32; n_pairs];

    let bar = new_progress_bar(n_pairs as u64).with_message("pair projection");
    for (lb, ub) in generate_minibatch_intervals(n_pairs, n_genes, Some(args.pair_block.max(1))) {
        let chunk = &edges[lb..ub];

        // One scattered column read per block: the endpoints of this block's
        // pairs, deduped (adjacent pairs share cells, so this is well under
        // `2 × pair_block`).
        let mut cells: Vec<usize> = chunk
            .iter()
            .flat_map(|&(u, v)| [u as usize, v as usize])
            .collect();
        cells.sort_unstable();
        cells.dedup();
        let slab = data.read_columns_csc(cells.iter().copied())?;
        // The slab's own CSC arrays, borrowed once for the whole block: a
        // column is a slice of these, so no per-pair view or copy is needed.
        let (col_offsets, slab_rows, slab_vals) =
            (slab.col_offsets(), slab.row_indices(), slab.values());
        let col_of: HashMap<usize, usize> = cells
            .iter()
            .enumerate()
            .map(|(local, &glob)| (glob, local))
            .collect();

        let solved: Vec<(Vec<f32>, f32)> = chunk
            .par_iter()
            .enumerate()
            .map(|(i, &(u, v))| {
                let obs = axis.pool_profile(pooled_profile(
                    SlabCols {
                        offsets: col_offsets,
                        rows: slab_rows,
                        vals: slab_vals,
                        col_of: &col_of,
                    },
                    u,
                    v,
                    batch,
                ));
                // Per-pair stream keyed on the global pair id, so the fit does
                // not depend on rayon's scheduling.
                let mut rng = SmallRng::seed_from_u64(
                    args.seed ^ ((lb + i) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                dict.project(&obs, &args.projection, &mut rng)
            })
            .collect();
        bar.inc(chunk.len() as u64);

        for (i, (theta, beta)) in solved.into_iter().enumerate() {
            for (j, &t) in theta.iter().enumerate() {
                latent[(lb + i, j)] = t;
            }
            bias[lb + i] = beta;
        }
    }
    bar.finish_and_clear();

    Ok(PairLatent { latent, bias })
}

/// One block's count slab, as the CSC arrays themselves plus the cell → column
/// map. Passing the arrays (rather than the matrix) lets a column be a borrowed
/// slice of them: `CscMatrix::col` hands back a view that owns the borrow, so
/// slices taken from it cannot outlive the view.
#[derive(Copy, Clone)]
struct SlabCols<'a> {
    offsets: &'a [usize],
    rows: &'a [usize],
    vals: &'a [f32],
    col_of: &'a HashMap<usize, usize>,
}

/// Pooled `(row, count)` profile for one pair, sorted by row index.
///
/// Still the ROW axis: the endpoint merge is a linear walk of two CSC columns,
/// which are sorted by row, and folding rows onto genes here would break that
/// ordering mid-merge. The caller applies [`GeneAxis::pool_profile`] to the
/// result instead.
///
/// The two endpoint columns are already sorted by row index, so this is a
/// linear merge. Batch division happens per endpoint *before* pooling — the two
/// cells may sit in different batches, and dividing after the sum would apply
/// one batch's fold factor to the other's counts.
fn pooled_profile(
    slab: SlabCols<'_>,
    u: u32,
    v: u32,
    batch: Option<PairBatchDivisor<'_>>,
) -> Vec<(u32, f32)> {
    // Both slices are borrowed straight out of the slab; only the batch-divided
    // path needs an owned copy of the values, and nothing ever needs to own the
    // row indices.
    let endpoint = |cell: u32| -> (&[usize], std::borrow::Cow<'_, [f32]>) {
        let Some(&col) = slab.col_of.get(&(cell as usize)) else {
            return (&[], std::borrow::Cow::Borrowed(&[]));
        };
        let (s, e) = (slab.offsets[col], slab.offsets[col + 1]);
        let rows = &slab.rows[s..e];
        let Some(bd) = batch else {
            return (rows, std::borrow::Cow::Borrowed(&slab.vals[s..e]));
        };
        let mut vals = slab.vals[s..e].to_vec();
        let b = bd.batch_of_cell[cell as usize] as usize;
        adjust_by_poisson_ratio(&mut vals, |k| bd.delta[(rows[k], b)]);
        (rows, std::borrow::Cow::Owned(vals))
    };

    let (lr, lv) = endpoint(u);
    let (rr, rv) = endpoint(v);

    let mut out: Vec<(u32, f32)> = Vec::with_capacity(lr.len() + rr.len());
    let mut push = |g: usize, x: f32| {
        if x > 0.0 {
            out.push((g as u32, x));
        }
    };
    let (mut i, mut j) = (0usize, 0usize);
    while i < lr.len() && j < rr.len() {
        match lr[i].cmp(&rr[j]) {
            std::cmp::Ordering::Less => {
                push(lr[i], lv[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                push(rr[j], rv[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                push(lr[i], lv[i] + rv[j]);
                i += 1;
                j += 1;
            }
        }
    }
    while i < lr.len() {
        push(lr[i], lv[i]);
        i += 1;
    }
    while j < rr.len() {
        push(rr[j], rv[j]);
        j += 1;
    }
    out
}

/// Adam on one pair's `e_uv`, with `β_uv` profiled out each step.
///
/// Returns `(e_uv, β_uv)`. A pair with no pooled counts gets the origin: the
/// likelihood says nothing about it, and the origin is where the ridge puts it.
fn solve_pair(
    obs: &[(u32, f32)],
    dict: &PairDictionary,
    args: &ProjectionArgs,
    rng: &mut SmallRng,
) -> (Vec<f32>, f32) {
    let d = dict.d;
    let total: f32 = obs.iter().map(|&(_, n)| n).sum();
    if obs.is_empty() || !total.is_finite() || total <= 0.0 {
        return (vec![0f32; d], 0.0);
    }

    // Observed composition mean `Σ_g n_g e_g / N` — the data half of the
    // gradient, constant in the parameters, so it is formed once.
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

    let n_active = dict.b.len();
    let sample = if args.gene_sample == 0 {
        n_active
    } else {
        args.gene_sample.min(n_active)
    };
    let exhaustive = sample == n_active;

    let mut theta = vec![0f32; d];
    let mut m = vec![0f32; d];
    let mut v = vec![0f32; d];

    // Everything the step loop would otherwise re-derive. `log_total` and
    // `log_scale` are fixed for this pair; `log_z` is fixed for the whole run.
    // The exhaustive gene list never changes either — under that mode the
    // "sample" IS the whole active axis, so it is filled once here.
    let log_total = total.ln();
    let log_scale = if exhaustive {
        0.0
    } else {
        dict.log_z - (sample as f32).ln()
    };
    // At `θ = 0` the partition is exactly `z · exp(β)`, so this initialization
    // already matches the pair's total mass; every later step only corrects it.
    let mut beta = log_total - dict.log_z;

    let mut scores = vec![0f32; sample];
    let mut genes = vec![0u32; sample];
    let mut pred_mean = vec![0f32; d];
    if exhaustive {
        for (s, g) in genes.iter_mut().enumerate() {
            *g = s as u32;
        }
    }

    // Adam's bias-correction terms are `β₁ᵗ` / `β₂ᵗ`, i.e. one multiply apart
    // between steps — carried forward rather than re-raised to the power each
    // step.
    let (mut b1t, mut b2t) = (1f32, 1f32);
    let steps = args.steps.max(1);

    for step in 0..steps {
        // ── Sample the partition ──────────────────────────────────────────
        if !exhaustive {
            for g in genes.iter_mut() {
                *g = dict.proposal.sample(rng) as u32;
            }
        }
        // `exp(b_g)/q_g` is constant under this proposal, so the importance
        // weights cancel and only `⟨e_g, θ⟩` varies. Exhaustive sampling has no
        // proposal to cancel, so it carries `b_g` explicitly.
        let mut max_score = f32::NEG_INFINITY;
        for (s, &g) in genes.iter().enumerate() {
            let g = g as usize;
            let row = &dict.feat[g * d..(g + 1) * d];
            let mut a: f32 = row.iter().zip(&theta).map(|(&e, &t)| e * t).sum();
            if exhaustive {
                a += dict.b[g];
            }
            a = a.clamp(-SCORE_CLAMP, SCORE_CLAMP);
            scores[s] = a;
            max_score = max_score.max(a);
        }

        // ── Predicted composition mean (self-normalized) ──────────────────
        let mut w_sum = 0f32;
        pred_mean.fill(0.0);
        for (&score, &g) in scores.iter().zip(genes.iter()) {
            let w = (score - max_score).exp();
            w_sum += w;
            let row = &dict.feat[g as usize * d..(g as usize + 1) * d];
            for (p, &e) in pred_mean.iter_mut().zip(row) {
                *p += w * e;
            }
        }
        if !w_sum.is_finite() || w_sum <= 0.0 {
            break;
        }
        for p in pred_mean.iter_mut() {
            *p /= w_sum;
        }

        // ── β is closed-form given θ: Σ_g μ_g = N_uv ──────────────────────
        // `Σ_g exp(⟨e_g,θ⟩ + b_g) ≈ (z/S)·Σ_s exp(⟨e_s,θ⟩)` under the
        // abundance proposal (`log_scale` carries the cancelled weights; it is
        // zero when the sum is exhaustive and exact), so `β = ln N − ln(that)`.
        // Kept in log space; no partition value is ever exponentiated at full
        // scale.
        beta = log_total - (log_scale + max_score + w_sum.ln());

        // ── Adam on θ ─────────────────────────────────────────────────────
        // Multinomial gradient: predicted minus observed composition, scaled by
        // the pair's mass, plus the ridge.
        let frac = step as f32 / steps as f32;
        let lr = dict.lr0 * (1.0 - (1.0 - LR_FLOOR_FRAC) * frac);
        b1t *= ADAM_B1;
        b2t *= ADAM_B2;
        let (bc1, bc2) = (1.0 - b1t, 1.0 - b2t);
        for j in 0..d {
            let grad = total * (pred_mean[j] - obs_mean[j]) + args.ridge * theta[j];
            m[j] = ADAM_B1 * m[j] + (1.0 - ADAM_B1) * grad;
            v[j] = ADAM_B2 * v[j] + (1.0 - ADAM_B2) * grad * grad;
            let m_hat = m[j] / bc1;
            let v_hat = v[j] / bc2;
            theta[j] -= lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
        }
    }

    (theta, beta)
}
