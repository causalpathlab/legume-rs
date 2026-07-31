//! **Tier 2** — per-dimension block Gibbs: independent per-dim inclusion with
//! per-dim hyperparameters `(σ₀h², π₀h)`, the `hyper.rs` doc's "per-dim variances
//! under a shared prior".
//!
//! # Why this exists
//!
//! The samplers this replaced both asked a *whole-gene* question. A SuSiE
//! single-effect gate is a softmax over dims, so exactly one dim wins and the
//! per-dim PIPs **compete**; a spike-and-slab on the gene is coarser still — one
//! Bernoulli per gene ("does this gene load anything"), with two global scalars
//! `(σ₀², π₀)` shared across all `H` dims. Both were retired rather than kept for
//! comparison: they were unreachable from any CLI, and a second sampler
//! contradicting "MCMC happens at the pb level" is an invitation to wire it back up.
//!
//! Selection is really a **per-dim** question — "which genes load dim `h`" — and
//! that is what this samples:
//!
//! ```text
//!   θ_gh = z_gh · β_gh,   β_gh ~ N(0, σ₀h²),   z_gh ~ Bernoulli(1 − π₀h)
//! ```
//!
//! Two consequences, and they are the point:
//!
//! 1. **PIPs no longer compete.** A gene loading many dims can read `PIP ≈ 1` on
//!    all of them, instead of having one softmax's worth of mass to divide. That
//!    is the whole argument: a set-valued truth cannot be reported by a model with
//!    one unit of mass to spend. Measured on planted data, a three-dim truth gets
//!    3.00 of 3 dims named here against 0.00 of 3 from the gate — not a tuning
//!    gap, since one unit split three ways cannot reach 0.5 anywhere.
//! 2. **Per-dim hypers instead of two global scalars.** One shared `π₀` forced to
//!    describe all `H` dims at once collapses to "include everything" (measured
//!    `π₀ = 0.0044` on BMMC). Each `(σ₀h², π₀h)` here pools over *all genes* on
//!    its own dim, so it is well identified — thousands of observations per
//!    parameter — and a coherent gene module can reinforce itself on its own dim
//!    without competing against every other dim's sparsity.
//!
//! # This is a MODELLING CHANGE, not a bug fix
//!
//! Independent per-dim inclusion is **not** the softmax single-effect the trained
//! variational gate uses, so this deliberately stops being "the exact posterior of
//! the deployed gate" — which the retired single-effect gate was. The question it
//! answers instead is "which dims does this gene use", which is the one the
//! downstream selection and annotation calls actually ask.
//!
//! NOTE the trained gate is now a per-(gene, dim) Bernoulli spike-and-slab
//! (`crate::model::FeatureGateSpec`) — the SAME generative model this block samples,
//! optimized variationally instead. So `σ(S)` from an SGD fit and the `pip` this
//! block reports are one estimand estimated two ways, and comparing them is the
//! intended check on the variational approximation. (Before the gate went Bernoulli
//! it was a softmax over genes within a dim, a simplex, which no rescaling connects
//! to a product of Bernoullis; measured rank correlation between the two was 0.02.)
//!
//! # Blocking, and the mixing trade
//!
//! What this file owns is the **statistical scaffolding**: the sweep loop, the per-dim
//! hyper draws, the accumulators, the diagnostics and the interrupt. The sampling
//! itself is a column pass in [`super::column`] — for each dim, every anchor at once —
//! and this drives it one tile of anchors at a time so each tile's score block stays
//! cache-resident across the whole dim scan.
//!
//! Given the frozen cell side, genes are conditionally independent, which is what makes
//! that transpose valid: a single dim's conditional factorizes over anchors. Two things
//! follow, and they are load-bearing:
//!
//!   * dims stay **sequential** within a tile, so this is a proper systematic-scan
//!     Gibbs — dim `h` conditions on the already updated dims `< h`;
//!   * the hypers update once per sweep, after every tile has reported, from additive
//!     sufficient statistics. Standard blocked Gibbs, and unaffected by how the anchor
//!     axis happened to be cut.
//!
//! Tiling is for locality, never for mixing, and the answer must not depend on the tile
//! width — every anchor's randomness is keyed by its global index. `column`'s own tests
//! assert that directly.
//!
//! The honest cost: a coordinate-wise scan mixes **worse** than a joint-over-dims block
//! when the dims are correlated, and they are — the dims are coupled through the cell
//! Gram `Σ = Eᶜᵀ Eᶜ`, which is not diagonal. We take that trade because the *selection*
//! estimand is what was wrong, not the mixing, and because a joint move over dims
//! cannot produce a per-coordinate inclusion indicator at all. The standard fix if it
//! bites is to draw in a whitened basis; the per-dim readout is what must stay in the
//! original basis, not the draw.
//!
//! # Identifiability caveat — read the PIPs as a set
//!
//! With the cell side frozen the rotational gauge is gone *within a fit*, so
//! per-dim quantities are well defined there. They are **not** comparable across
//! fits: two seeds give cross-seed ARI 0.0365 on `argmax|E_feat|` against a
//! same-seed floor of 1.0000, because the basis rotates between runs even though
//! the geometry does not. Never compare a dim index from one fit to another.
//!
//! The residual within-fit risk is **collinearity**: if two dims of `e_cell` are
//! near-parallel, "which of them does gene `g` load" is not identifiable, and
//! independent per-dim marginals split the mass between them and can look
//! confidently wrong on *both*. This is not hypothetical — measured max VIF on
//! real 12k BMMC fits at `H=16` is **29–37**, far past the conventional 5. Check
//! [`super::frozen_diag`], which reports it per run and warns at 5. When it is
//! high, read a gene's row as a profile rather than a winner: the mass a dim
//! carries may belong to its collinear partner. A single-effect gate would offer a
//! set-valued credible set instead, but only by sampling a per-gene
//! parameterization the trainer no longer uses.

use super::column;
use super::diagnostics::{scalar_diagnostics, ChainDiag};
use super::hyper::{sample_pi0, HalfCauchyVar, StickBreaking};

use super::lnpdf::{FrozenSide, NodeTerm};
use super::score::{ProfiledPoisson, SlateSlab};
use crate::progress::new_progress_bar;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Default stick-breaking concentration: `α = 1`, i.e. a feature loads ONE dim in
/// expectation. Chosen as the sparsest defensible prior rather than fitted — measured
/// dims-per-feature on BM1 was 1.89 under the unordered Beta, so `α = 1` leans toward
/// sparsity and lets the likelihood argue upward. `Beta(1, 1)` is uniform on each
/// stick, so it is also the least committal choice of stick distribution.
pub const DEFAULT_STICK_ALPHA: f64 = 1.0;

/// Configuration for [`dim_block`].
pub struct DimBlockConfig {
    pub n_sweeps: usize,
    pub burnin: usize,
    /// ESS transitions per `(gene, dim)` per sweep. The scan already costs `H`
    /// likelihood passes per gene per sweep, so `1` here buys `H` transitions per
    /// gene per sweep — it is not the same `1` a whole-gene sampler would mean.
    pub transitions_per_dim: usize,
    pub seed: u64,
    /// Half-Cauchy prior scale `A` on each dim's slab SD.
    pub half_cauchy_scale: f64,
    /// Beta hyperprior `(a, b)` on every `π₀h`. Shared across dims — it is the
    /// prior, not the parameter, that is shared, which is what makes this a
    /// partial-pooled per-dim model rather than `H` unrelated fits.
    pub beta_a: f64,
    pub beta_b: f64,
    /// `Some(α)` selects the TRUNCATED IBP — the DEFAULT: inclusion rates come from
    /// stick-breaking
    /// (`v_j ~ Beta(α,1)`, `π_h = ∏_{j≤h} v_j`) instead of an independent `Beta(a,b)`
    /// per dim, so they are forced to DECREASE with the dim index and surplus dims are
    /// squeezed off rather than each re-estimating the same rate. `None` keeps the
    /// independent per-dim Beta. See [`StickBreaking`].
    pub stick_alpha: Option<f64>,
    /// Optional `[n_anchors × h]` row-major warm start for `β`, e.g. the phase-1
    /// SGD MAP. `None` cold-starts at zero.
    ///
    /// **`z` is always cold**, warm start or not — and note this is now a CHOICE
    /// rather than a necessity. The trained gate is the same Bernoulli spike-and-slab
    /// (`crate::model::FeatureGateSpec`), so its `σ(S)` could in principle seed `z`.
    /// It deliberately does not: warm-starting the indicator from a variational
    /// optimum is what `posterior::run`'s exclusivity argument rules out, since a
    /// mean-field `q(z)` cannot represent the correlated selection this block exists
    /// to explore. A warm `β` gives the chain a sensible scale; the data still has to
    /// earn each dim.
    pub init_beta: Option<Vec<f32>>,
    /// Progress-bar label, so a run with several blocks says which one is moving.
    pub label: Box<str>,
    /// Draw a sweep progress bar. Off when this is the INNER block of a larger
    /// driver that owns the bar — an alternating sweep calls this twice per outer
    /// sweep, and two one-tick bars per sweep would flood the shared
    /// `MULTI_PROGRESS` instead of reporting anything.
    pub show_progress: bool,
    /// Optional `[n_anchors × h]` row-major veto: `false` forces `z = 0` for that
    /// `(anchor, dim)` regardless of the likelihood. `None` lets every coordinate
    /// be drawn.
    ///
    /// This is how a **nested** gate is expressed. `faba gem`'s velocity `δ_g` is a
    /// deviation from the identity loading `β_g`, so "this gene moves along a dim
    /// its identity does not load" is a state the model should not visit. Vetoing
    /// it also breaks the symmetry that otherwise lets two independent
    /// spike-and-slabs split inclusion mass between `(z_β=1, z_δ=0)` and
    /// `(z_β=0, z_δ=1)` on a gene where only their sum is identified.
    ///
    /// Vetoed coordinates are excluded from the per-dim `π₀h` reduction: they were
    /// never eligible, so counting them as observed nulls would report the
    /// constraint back as if it were measured sparsity.
    pub z_allowed: Option<Vec<bool>>,
    /// Optional `[n_anchors × h]` row-major incoming inclusion state.
    ///
    /// **Required for an alternating sampler.** A block driven one sweep at a time
    /// must resume from the `z` the previous sweep left (see
    /// [`DimBlockResult::final_z`]); starting cold every time means the `z[d]`
    /// branch below never runs, every `β` is redrawn from its prior, and
    /// [`Self::init_beta`] is silently discarded. `None` cold-starts every
    /// coordinate off, which is right only for a chain that owns all its sweeps.
    pub init_z: Option<Vec<bool>>,
    /// Sample the inclusion indicators. `false` pins every `z` on and skips the
    /// draw entirely, giving a plain per-dim Gaussian block.
    ///
    /// This is a structural statement, not a prior setting. Pushing `π₀ → 0`
    /// through the Beta hyperprior cannot express it: `log_prior_odds` is floored
    /// at `ln(1e12) ≈ 27.6` nats, which a pseudobulk anchor's likelihood
    /// difference (routinely thousands of nats) swamps, so coordinates would still
    /// be sampled off.
    pub selection: bool,
}

impl DimBlockConfig {
    #[must_use]
    pub fn new(n_sweeps: usize, burnin: usize, seed: u64) -> Self {
        Self {
            n_sweeps,
            burnin,
            transitions_per_dim: 1,
            seed,
            half_cauchy_scale: 1.0,
            beta_a: 9.0,
            beta_b: 1.0,
            stick_alpha: Some(DEFAULT_STICK_ALPHA),
            init_beta: None,
            label: "mcmc".into(),
            show_progress: true,
            z_allowed: None,
            init_z: None,
            selection: true,
        }
    }

    /// Resume the inclusion state from a previous sweep — see [`Self::init_z`].
    #[must_use]
    pub fn with_init_z(mut self, z: Vec<bool>) -> Self {
        self.init_z = Some(z);
        self
    }

    /// Turn the spike-and-slab off: every coordinate stays included and the block
    /// is a plain per-dim Gaussian update. See [`Self::selection`].
    #[must_use]
    pub fn without_selection(mut self) -> Self {
        self.selection = false;
        // Clear the inclusion prior with it. Nothing reads it when `selection` is off —
        // both `dim_block_multi` and `column`'s draw branch on `selection` first — but
        // leaving it set means two configs in one run disagree about which prior is in
        // effect, which is the state a reader has to reason about. One switch, not two.
        self.stick_alpha = None;
        self
    }

    /// Restrict which `(anchor, dim)` coordinates may be included — see
    /// [`Self::z_allowed`].
    #[must_use]
    pub fn with_z_allowed(mut self, allowed: Vec<bool>) -> Self {
        self.z_allowed = Some(allowed);
        self
    }

    /// Silence the sweep bar — for an inner block whose driver owns the bar.
    #[must_use]
    pub fn quiet(mut self) -> Self {
        self.show_progress = false;
        self
    }

    /// Warm-start `β` from `[n_anchors × h]` row-major values — see
    /// [`Self::init_beta`] for why `z` is not warm-started with it.
    #[must_use]
    pub fn with_init_beta(mut self, beta: Vec<f32>) -> Self {
        self.init_beta = Some(beta);
        self
    }

    #[must_use]
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.into();
        self
    }
}

/// Result of a per-dim block run. The `[n_genes × H]` tables are **row-major**,
/// gene-major — `pip[g * h + d]` — matching the `{out}.{tag}_pip.parquet` layout.
pub struct DimBlockResult {
    /// Per-`(gene, dim)` posterior inclusion probability `P(z_gh = 1)`. Unlike the
    /// gate's softmax PIP these do **not** sum to 1 across a gene's row.
    pub pip: Vec<f32>,
    /// Per-`(gene, dim)` posterior mean loading `E[β_gh]`.
    pub mean_beta: Vec<f32>,
    /// Posterior-mean slab variance per dim `[H]`.
    pub sigma2: Vec<f64>,
    /// Posterior-mean null mass per dim `[H]`.
    pub pi0: Vec<f64>,
    /// Mixing diagnostics per dim `[H]`.
    pub sigma_diag: Vec<ChainDiag>,
    pub pi0_diag: Vec<ChainDiag>,
    pub h: usize,
    /// Sweeps actually retained (less than `n_sweeps − burnin` only if SIGINT hit).
    pub n_kept: usize,
    /// `[n_anchors × h]` inclusion state after the LAST sweep — a single draw, not
    /// a posterior summary. It exists so an outer alternating sampler can condition
    /// the next block on it (see [`DimBlockConfig::z_allowed`]); for reporting,
    /// use [`Self::pip`].
    pub final_z: Vec<bool>,
    /// Slice transitions that exhausted their bracket and fell back to the current
    /// value, summed over every `(anchor, dim, sweep)`.
    ///
    /// This is the sampler's analogue of a rejected move, and it is reported rather
    /// than swallowed for the same reason a truncated result set is: a run where most
    /// coordinates stall still produces a full table of plausible numbers. Compare it
    /// against [`Self::n_transitions`].
    pub fallbacks: usize,
    /// Slice transitions ATTEMPTED, the denominator for [`Self::fallbacks`] — only the
    /// coordinates that actually ran a slice move, not every `(anchor, dim)`. An excluded
    /// coordinate draws from the prior with no bracket and cannot fall back.
    pub n_transitions: usize,
    /// Batched likelihood evaluations — the run's actual cost, in the unit the column
    /// pass charges for.
    pub n_evals: usize,
    /// `[n_anchors × n_terms]` profiled intercepts from the LAST sweep, anchor-major.
    ///
    /// A single draw's worth, like [`Self::final_z`], and for the same reason: an outer
    /// alternating sampler needs the intercept that goes WITH the loadings it is about
    /// to condition on. Loadings and biases have to describe one model — pairing a
    /// sampled embedding with an intercept fitted under a different state is exactly the
    /// mismatch that made frozen intercepts need recalibration.
    pub b_profiled: Vec<f32>,
    /// Likelihood terms per anchor — the row stride of [`Self::b_profiled`].
    ///
    /// Carried so no caller has to know it. gem's β block runs over two splice tracks and
    /// so has two intercepts per gene; every other block has one. Read it through
    /// [`Self::intercept`] rather than indexing.
    pub n_terms: usize,
}

impl DimBlockResult {
    /// Gene `g`'s per-dim PIP row `[H]`.
    #[must_use]
    pub fn pip_row(&self, g: usize) -> &[f32] {
        &self.pip[g * self.h..(g + 1) * self.h]
    }

    /// Gene `g`'s per-dim posterior-mean loading row `[H]`.
    #[must_use]
    pub fn beta_row(&self, g: usize) -> &[f32] {
        &self.mean_beta[g * self.h..(g + 1) * self.h]
    }

    /// Anchor `g`'s profiled intercept for likelihood term `t`.
    ///
    /// The point of the accessor is that the stride lives here and nowhere else. A caller
    /// that indexed `b_profiled` directly would have to hard-code the term count, and a
    /// block that later gained or lost a term would silently read the wrong element rather
    /// than fail to compile.
    ///
    /// # Panics
    ///
    /// If `t >= n_terms` — that is a caller confusing one block's term layout for
    /// another's, which should be loud.
    #[must_use]
    pub fn intercept(&self, g: usize, t: usize) -> f32 {
        assert!(
            t < self.n_terms,
            "term {t} requested from a block with {} term(s)",
            self.n_terms
        );
        self.b_profiled[g * self.n_terms + t]
    }
}

/// Run the per-dim block Gibbs. `nodes[g]` is gene `g`'s likelihood terms against
/// the frozen `side`; the per-gene intercept is profiled out by the score itself
/// (see [`super::score::ProfiledPoisson`]), so none is supplied.
///
/// SIGINT is polled at the sweep boundary: the accumulators are already valid
/// posterior means over the sweeps completed so far, so an interrupted run
/// returns a coarser answer rather than a wrong or truncated one. Check
/// [`DimBlockResult::n_kept`] to see how many sweeps it actually got.
/// Single-term convenience wrapper over [`dim_block_multi`] — one likelihood term
/// per anchor, which is every caller that is not modelling two tracks.
#[must_use]
pub fn dim_block(nodes: &[NodeTerm], side: &FrozenSide, cfg: &DimBlockConfig) -> DimBlockResult {
    let anchors: Vec<Vec<NodeTerm>> = nodes.iter().map(|n| vec![*n]).collect();
    dim_block_multi(&anchors, side, cfg)
}

/// Per-dim block Gibbs where each anchor's likelihood is a **sum** of terms.
///
/// More than one term is needed when an anchor's data splits into blocks that see
/// different frozen offsets. `faba gem`'s identity loading `β_g` is the motivating
/// case: a spliced row scores `⟨β_g, e_p⟩` and an unspliced row scores
/// `⟨β_g + δ_g, e_p⟩`, so β's conditional is
///
/// ```text
///   ℓ(β) = ℓ_spliced(β; offset 0) + ℓ_unspliced(β; offset δ)
/// ```
///
/// and a single [`NodeTerm`] — which carries one offset for all its edges — cannot
/// express it. Dropping the unspliced half instead (the obvious shortcut) makes
/// the block something other than `p(β | δ, ·)`, and the surrounding blocks then
/// stop being conditionals of a common joint, which costs the chain its stationary
/// distribution.
///
/// Summing **profile** log-likelihoods is legitimate here precisely because the
/// two tracks are different feature ROWS with independent per-row biases
/// (`b_{g,spliced}` vs `b_{g,unspliced}`): each term's intercept profiles out on
/// its own, so the sum of the profiles is the profile of the joint. It would NOT
/// be legitimate for two terms sharing one intercept.
#[must_use]
pub fn dim_block_multi(
    anchors: &[Vec<NodeTerm>],
    side: &FrozenSide,
    cfg: &DimBlockConfig,
) -> DimBlockResult {
    let nodes = anchors;
    let h = side.h;
    let n_genes = nodes.len();

    // Live state, FLAT `[n_genes × h]`: the slab effects `β` and the inclusion
    // indicators `z`. Both are carried, because `z` genuinely gates — the effective
    // loading is `z ⊙ β`, so a coordinate that is off contributes nothing to what the
    // other dims condition on. `z` always cold-starts every dim off, so the data has
    // to earn each one rather than starting from "all included"; `β` may be
    // warm-started (see `DimBlockConfig::init_beta`).
    let mut beta: Vec<f32> = match cfg.init_beta.as_deref() {
        None => vec![0.0f32; n_genes * h],
        Some(v) => {
            // A silently mis-shaped warm start would produce a confident wrong answer
            // rather than an obvious failure, so this is a hard check even though the
            // function is infallible otherwise.
            assert!(
                v.len() == n_genes * h,
                "init_beta is {} floats but {n_genes} anchors × {h} dims = {}",
                v.len(),
                n_genes * h
            );
            v.to_vec()
        }
    };
    let mut zed: Vec<bool> = match cfg.init_z.as_deref() {
        // `selection == false` pins every coordinate on; there is nothing to resume.
        _ if !cfg.selection => vec![true; n_genes * h],
        None => vec![false; n_genes * h],
        Some(v) => {
            assert!(
                v.len() == n_genes * h,
                "init_z is {} flags but {n_genes} anchors × {h} dims = {}",
                v.len(),
                n_genes * h
            );
            v.to_vec()
        }
    };

    // Anchors are processed in TILES so each tile's `[b × k]` score block stays
    // cache-resident across the whole dim loop; see `super::column`.
    let n_terms = nodes.first().map_or(1, Vec::len);
    let slate_k = nodes
        .first()
        .and_then(|terms| terms.first())
        .map_or(1, |t| t.partition.len());
    let tile_b = column::tile_size(slate_k, n_terms);
    let tiles: Vec<(usize, usize)> = (0..n_genes)
        .step_by(tile_b)
        .map(|lo| (lo, (lo + tile_b).min(n_genes)))
        .collect();

    let mut hv: Vec<HalfCauchyVar> = (0..h)
        .map(|_| HalfCauchyVar::new(cfg.half_cauchy_scale))
        .collect();
    let mut sigma2 = vec![cfg.half_cauchy_scale * cfg.half_cauchy_scale; h];
    // Either an independent `Beta(a,b)` per dim, or the truncated-IBP sticks. Both
    // produce the same thing — a length-`h` vector of exclusion rates — which is the
    // model's ONLY view of the prior, so nothing downstream changes with the choice.
    let mut sticks = cfg.stick_alpha.map(|a| StickBreaking::new(a, h));
    let mut pi0 = match &sticks {
        Some(sb) => sb.pi0(),
        None => vec![cfg.beta_a / (cfg.beta_a + cfg.beta_b); h],
    };
    let mut hyper_rng = StdRng::seed_from_u64(cfg.seed ^ 0x7A11_0C0D);

    let mut sigma2_chain: Vec<Vec<f64>> = vec![Vec::new(); h];
    let mut pi0_chain: Vec<Vec<f64>> = vec![Vec::new(); h];
    let mut pip_acc = vec![0.0f64; n_genes * h];
    let mut beta_acc = vec![0.0f64; n_genes * h];
    // One transposed slate view per TERM, built once and shared read-only across
    // anchors. Terms may carry different slates, so this is per index rather than
    // one global slab.
    let slabs: Vec<SlateSlab> = nodes
        .first()
        .map(|terms| {
            terms
                .iter()
                .map(|n| SlateSlab::new(n.partition, side))
                .collect()
        })
        .unwrap_or_default();

    let mut n_kept = 0usize;
    let mut fallbacks = 0usize;
    let mut n_evals = 0usize;
    let mut n_transitions = 0usize;
    let mut b_profiled = vec![0.0f32; n_genes * n_terms];
    let stop = crate::stop::stop_flag();

    // The whole sampler is this one loop, and on a real dictionary it runs for
    // minutes with nothing on stdout. Ticked from the serial outer loop, so it
    // never contends with the rayon map inside.
    let pb_bar = cfg.show_progress.then(|| {
        new_progress_bar(cfg.n_sweeps as u64).with_message(format!("{} sweeps", cfg.label))
    });

    for sweep in 0..cfg.n_sweeps {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let sd: Vec<f32> = sigma2
            .iter()
            .map(|v| (v.max(1e-12)).sqrt() as f32)
            .collect();
        let log_prior_odds: Vec<f64> = pi0
            .iter()
            .map(|p| ((1.0 - p).max(1e-12) / p.max(1e-12)).ln())
            .collect();

        let veto = cfg.z_allowed.as_deref();
        let selection = cfg.selection;
        // One tile at a time, dims INSIDE. `super::column` explains why: given the
        // frozen side a dim's conditional factorizes over anchors, so a tile's score
        // block can be carried across the whole dim loop by rank-1 updates and stays
        // cache-resident while it is.
        let draws: Vec<(usize, usize, column::TileDraw)> = tiles
            .par_iter()
            .map(|&(lo, hi)| {
                let cargs = column::ColumnArgs {
                    side,
                    sd: &sd,
                    log_prior_odds: &log_prior_odds,
                    veto,
                    selection,
                    transitions: cfg.transitions_per_dim,
                    seed: cfg.seed,
                    sweep,
                    h,
                };
                let draw = column::sample_tile(
                    &ProfiledPoisson,
                    &cargs,
                    &nodes[lo..hi],
                    &slabs,
                    lo,
                    &beta[lo * h..hi * h],
                    &zed[lo * h..hi * h],
                );
                (lo, hi, draw)
            })
            .collect();

        // Reduce per DIM: each column's included effects give that dim's slab
        // variance, and its null count gives that dim's sparsity.
        //
        // `n_eligible` counts only coordinates the veto actually let the data
        // decide. A vetoed coordinate is off by construction, so folding it into
        // the Bernoulli trials would report the CONSTRAINT back as measured
        // sparsity — with a nested gate over a dim where the parent included 2% of
        // genes, `π₀` would read ≈0.98 and then tighten the prior odds against the
        // coordinates that were genuinely eligible.
        let mut sum_sq = vec![0.0f64; h];
        let mut n_incl = vec![0usize; h];
        let mut n_eligible = vec![0usize; h];
        let keep = sweep >= cfg.burnin;
        for (lo, hi, draw) in draws {
            fallbacks += draw.fallbacks;
            n_evals += draw.rounds;
            n_transitions += draw.transitions;
            for i in 0..(hi - lo) {
                let g = lo + i;
                for d in 0..h {
                    let v = f64::from(draw.beta[i * h + d]);
                    if veto.is_none_or(|mask| mask[g * h + d]) {
                        n_eligible[d] += 1;
                    }
                    if draw.z[i * h + d] {
                        sum_sq[d] += v * v;
                        n_incl[d] += 1;
                        if keep {
                            pip_acc[g * h + d] += 1.0;
                            // The EFFECTIVE loading is `z·β`, so an excluded draw
                            // contributes 0 — the posterior mean is over what the model
                            // actually uses, not over the latent slab value.
                            beta_acc[g * h + d] += v;
                        }
                    }
                }
            }
            beta[lo * h..hi * h].copy_from_slice(&draw.beta);
            zed[lo * h..hi * h].copy_from_slice(&draw.z);
            b_profiled[lo * n_terms..hi * n_terms].copy_from_slice(&draw.b_profiled);
        }

        for d in 0..h {
            sigma2[d] = hv[d].sample(sum_sq[d], n_incl[d].max(1), &mut hyper_rng);
        }
        // Only eligible coordinates are Bernoulli trials — see the note above. With
        // selection off there is nothing to infer, so `π₀` is left at its prior mean
        // rather than being "estimated" from a vector of all-ones.
        if selection {
            match &mut sticks {
                // Truncated IBP: ONE joint update, because the sticks are coupled —
                // `v_j` constrains every dim from `j` on, which is exactly the coupling
                // an independent per-dim draw lacks.
                Some(sb) => {
                    sb.sample(&n_incl, &n_eligible, &mut hyper_rng);
                    pi0 = sb.pi0();
                }
                None => {
                    for d in 0..h {
                        pi0[d] = sample_pi0(
                            n_eligible[d].saturating_sub(n_incl[d]),
                            n_eligible[d],
                            cfg.beta_a,
                            cfg.beta_b,
                            &mut hyper_rng,
                        );
                    }
                }
            }
        }

        if keep {
            for d in 0..h {
                sigma2_chain[d].push(sigma2[d]);
                pi0_chain[d].push(pi0[d]);
            }
            n_kept += 1;
        }
        if let Some(bar) = &pb_bar {
            bar.inc(1);
        }
    }
    // After the loop, not inside it — the SIGINT `break` above exits here, so this
    // is the one place that clears the bar on both the normal and interrupted path.
    if let Some(bar) = &pb_bar {
        bar.finish_and_clear();
    }

    let inv = 1.0 / n_kept.max(1) as f64;
    DimBlockResult {
        pip: pip_acc.iter().map(|&a| (a * inv) as f32).collect(),
        mean_beta: beta_acc.iter().map(|&a| (a * inv) as f32).collect(),
        sigma2: sigma2_chain
            .iter()
            .map(|c| c.iter().sum::<f64>() * inv)
            .collect(),
        pi0: pi0_chain
            .iter()
            .map(|c| c.iter().sum::<f64>() * inv)
            .collect(),
        sigma_diag: sigma2_chain.iter().map(|c| scalar_diagnostics(c)).collect(),
        pi0_diag: pi0_chain.iter().map(|c| scalar_diagnostics(c)).collect(),
        h,
        n_kept,
        final_z: zed,
        fallbacks,
        n_transitions,
        n_evals,
        b_profiled,
        n_terms,
    }
}

#[cfg(test)]
#[path = "dim_block_tests.rs"]
mod dim_block_tests;
