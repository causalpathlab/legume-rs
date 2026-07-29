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
//! NOTE the trained gate has since moved to normalizing over GENES within a dim
//! (`crate::model::SoftmaxGateSpec`), so neither sampler here is its exact
//! posterior — but this one, being per-dim, is the closer match.
//!
//! # Blocking, and the mixing trade
//!
//! Given the frozen cell side, genes are conditionally independent, so the outer
//! `rayon par_iter` is over **genes** and the dim scan runs *inside* each gene's
//! closure. That ordering is deliberate:
//!
//!   * it is a proper systematic-scan Gibbs — dim `h` conditions on the already
//!     updated dims `< h` of the same gene;
//!   * it amortizes the per-gene scratch buffers over the whole scan instead of
//!     reallocating per `(gene, dim)`;
//!   * the hypers update once per sweep, after every gene — standard blocked Gibbs.
//!
//! The honest cost: a coordinate-wise scan mixes **worse** than a joint-over-dims
//! block when the dims are correlated, and they are — the dims are coupled through
//! the cell Gram `Σ = Eᶜᵀ Eᶜ`, which is not diagonal. We take that trade because the
//! *selection* estimand is what was wrong, not the mixing. The standard fix if it
//! bites is to draw in a whitened basis; the per-dim readout is what must stay in
//! the original basis, not the draw.
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

use super::diagnostics::{scalar_diagnostics, ChainDiag};
use super::hyper::{gene_rng, sample_pi0, sigmoid, HalfCauchyVar};
use super::lnpdf::{multinomial_ll, FrozenSide, NodeTerm};
use crate::cell_projection::SCORE_CLAMP;
use crate::progress::new_progress_bar;
use mcmc_util::engine::elliptical_slice_step;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

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
    /// Optional `[n_anchors × h]` row-major warm start for `β`, e.g. the phase-1
    /// SGD MAP. `None` cold-starts at zero.
    ///
    /// **`z` is always cold**, warm start or not. The trained gate is a softmax
    /// *over genes within a dim* (`crate::model::SoftmaxGateSpec`), which is a
    /// different object from this per-dim spike-and-slab — thresholding one into
    /// the other would invent a correspondence neither parameterization implies.
    /// A warm `β` gives the chain a sensible scale; the data still has to earn
    /// each dim.
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
}

/// Per-gene sweep state handed back from the parallel map.
struct GeneDraw {
    beta: DVector<f32>,
    z: Vec<bool>,
}

/// The frozen side restricted to one term's negative slate, **transposed** so a
/// single dim's column is contiguous.
///
/// The dim scan changes exactly one coordinate at a time, so a score only ever
/// shifts by `Δ · e_o[d]`. Holding `e_o[d]` contiguous across the slate turns the
/// rate normalizer from `|slate|` dot products of length `h` into one pass of
/// `|slate|` multiply-adds — the difference between a per-anchor cost that scales
/// with `h²` and one that scales with `h`.
///
/// Built once per block and shared read-only across anchors: `h × |slate|` floats,
/// 128 KB at `h = 32`, `|slate| = 1024`.
struct SlateSlab {
    /// `col[d * k + j] = e[slate[j] * h + d]`.
    col: Vec<f32>,
    /// `b_o` per slate entry.
    b: Vec<f32>,
    k: usize,
}

impl SlateSlab {
    fn new(partition: &[u32], side: &FrozenSide) -> Self {
        let (h, k) = (side.h, partition.len());
        let mut col = vec![0f32; h * k];
        let mut b = vec![0f32; k];
        for (j, &o) in partition.iter().enumerate() {
            let row = &side.e[o as usize * h..(o as usize + 1) * h];
            for (d, v) in row.iter().enumerate() {
                col[d * k + j] = *v;
            }
            b[j] = side.b[o as usize];
        }
        Self { col, b, k }
    }

    #[inline]
    fn dim(&self, d: usize) -> &[f32] {
        &self.col[d * self.k..(d + 1) * self.k]
    }
}

/// `s[j] = ⟨v, e_o⟩ + b_o` for an anchor's current effective loading `v` — its
/// frozen offset plus every INCLUDED dim. One `|slate| × h` pass per term per
/// sweep; every dim afterwards is `O(|slate|)`.
fn seed_scores(
    node: &NodeTerm,
    slab: &SlateSlab,
    th: &DVector<f32>,
    z: &[bool],
    h: usize,
) -> Vec<f64> {
    let mut s: Vec<f64> = slab.b.iter().map(|b| f64::from(*b)).collect();
    for d in 0..h {
        let w = node.offset.map_or(0.0, |b| b[d]) + if z[d] { th[d] } else { 0.0 };
        if w == 0.0 {
            continue;
        }
        for (sj, cj) in s.iter_mut().zip(slab.dim(d)) {
            *sj += f64::from(w) * f64::from(*cj);
        }
    }
    s
}

/// The profile log-likelihood at coordinate `d` set to `x`, given running scores
/// `s` that already exclude that coordinate.
///
/// Identical in value to [`multinomial_ll`], not an approximation of it:
/// [`SCORE_CLAMP`] is applied here, at the point the score is exponentiated,
/// which is exactly where `score()` applies it. The saving is that the other
/// `h − 1` coordinates are already folded into `s`, so the normalizer costs one
/// pass over the slate instead of `|slate|` dot products of length `h`.
///
/// Falls back to the full walk when the loading leaves the moment's safe radius —
/// the same guard [`multinomial_ll`] applies, for the same reason.
fn incremental_ll(
    node: &NodeTerm,
    off: &[f32],
    x: f32,
    d: usize,
    slab: &SlateSlab,
    s: &[f64],
    side: &FrozenSide,
) -> f32 {
    let Some(mom) = node.moment else {
        // No precomputed moment: the data term needs the edge walk anyway, so
        // there is nothing to gain here.
        return multinomial_ll_at(node, off, x, d, side);
    };
    if mom.total == 0.0 {
        return 0.0; // no counts ⇒ flat in `e_a`
    }
    // Outside the safe radius the collapsed data term is not the clamped one.
    let mut nrm2 = x * x;
    for (kk, o) in off.iter().enumerate() {
        if kk != d {
            nrm2 += o * o;
        }
    }
    if nrm2.sqrt() > mom.safe_radius {
        return multinomial_ll_at(node, off, x, d, side);
    }

    // Data term: ⟨off + x·e_d, m⟩ plus the `Σ n·b_o` the collapse omits, so this
    // agrees with `multinomial_ll_at` in absolute value and the radius guard's
    // choice of form is not observable in a `z` logit or a slice threshold.
    let mut data = mom.bias_dot;
    for (kk, (o, m)) in off.iter().zip(&mom.m).enumerate() {
        if kk != d {
            data += f64::from(*o) * f64::from(*m);
        }
    }
    data += f64::from(x) * f64::from(mom.m[d]);

    // Normalizer: one pass, clamping each score exactly as `score()` does.
    let col = slab.dim(d);
    let mut m_max = f64::NEG_INFINITY;
    for (sj, cj) in s.iter().zip(col) {
        let v = (sj + f64::from(x) * f64::from(*cj)).clamp(-SCORE_CLAMP, SCORE_CLAMP);
        if v > m_max {
            m_max = v;
        }
    }
    if !m_max.is_finite() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for (sj, cj) in s.iter().zip(col) {
        let v = (sj + f64::from(x) * f64::from(*cj)).clamp(-SCORE_CLAMP, SCORE_CLAMP);
        acc += (v - m_max).exp();
    }
    (data - mom.total * (m_max + acc.max(f64::MIN_POSITIVE).ln())) as f32
}

/// Reference path: rebuild the full loading and call [`multinomial_ll`].
fn multinomial_ll_at(node: &NodeTerm, off: &[f32], x: f32, d: usize, side: &FrozenSide) -> f32 {
    let mut e = vec![0f32; side.h];
    e[d] = x;
    let n = NodeTerm {
        offset: Some(off),
        ..*node
    };
    multinomial_ll(&e, &n, side)
}

/// Run the per-dim block Gibbs. `nodes[g]` is gene `g`'s likelihood terms against
/// the frozen `side`; the per-gene intercept is profiled out by
/// [`multinomial_ll`], so none is supplied.
///
/// SIGINT is polled at the sweep boundary: the accumulators are already valid
/// posterior means over the sweeps completed so far, so an interrupted run
/// returns a coarser answer rather than a wrong or truncated one. Check
/// [`DimBlockResult::n_kept`] to see how many sweeps it actually got.
/// Single-term convenience wrapper over [`dim_block_multi`] — one likelihood term
/// per anchor, which is every caller that is not modelling two tracks.
#[must_use]
pub fn dim_block(nodes: &[NodeTerm], side: &FrozenSide, cfg: &DimBlockConfig) -> DimBlockResult {
    let anchors: Vec<Vec<NodeTerm>> = nodes.iter().map(|n| vec![*n]) .collect();
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
    let k = cfg.transitions_per_dim.max(1);

    // Live per-gene state: the slab effects `β_g` and the inclusion indicators
    // `z_g`. BOTH are carried, because `z` genuinely gates — the effective loading
    // is `z ⊙ β`, so a coordinate that is off contributes nothing to the offset the
    // other dims condition on. `z` always cold-starts every dim off, so the data
    // has to earn each one rather than starting from "all included"; `β` may be
    // warm-started from a MAP fit (see `DimBlockConfig::init_beta`).
    let mut beta: Vec<DVector<f32>> = match cfg.init_beta.as_deref() {
        None => (0..n_genes).map(|_| DVector::zeros(h)).collect(),
        Some(v) => {
            // A silently mis-shaped warm start would produce a confident wrong
            // answer rather than an obvious failure, so this is a hard check even
            // though the function is infallible otherwise.
            assert!(
                v.len() == n_genes * h,
                "init_beta is {} floats but {n_genes} anchors × {h} dims = {}",
                v.len(),
                n_genes * h
            );
            (0..n_genes)
                .map(|g| DVector::from_column_slice(&v[g * h..(g + 1) * h]))
                .collect()
        }
    };
    let mut zed: Vec<Vec<bool>> = match cfg.init_z.as_deref() {
        // `selection == false` pins every coordinate on; there is nothing to resume.
        _ if !cfg.selection => (0..n_genes).map(|_| vec![true; h]).collect(),
        None => (0..n_genes).map(|_| vec![false; h]).collect(),
        Some(v) => {
            assert!(
                v.len() == n_genes * h,
                "init_z is {} flags but {n_genes} anchors × {h} dims = {}",
                v.len(),
                n_genes * h
            );
            (0..n_genes).map(|g| v[g * h..(g + 1) * h].to_vec()).collect()
        }
    };
    let zeros_h = DVector::<f32>::zeros(h);

    let mut hv: Vec<HalfCauchyVar> = (0..h)
        .map(|_| HalfCauchyVar::new(cfg.half_cauchy_scale))
        .collect();
    let mut sigma2 = vec![cfg.half_cauchy_scale * cfg.half_cauchy_scale; h];
    let mut pi0 = vec![cfg.beta_a / (cfg.beta_a + cfg.beta_b); h];
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
    let stop = crate::stop::stop_flag();

    // The whole sampler is this one loop, and on a real dictionary it runs for
    // minutes with nothing on stdout. Ticked from the serial outer loop, so it
    // never contends with the rayon map inside.
    let pb_bar = cfg
        .show_progress
        .then(|| new_progress_bar(cfg.n_sweeps as u64).with_message(format!("{} sweeps", cfg.label)));

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
        let draws: Vec<GeneDraw> = (0..n_genes)
            .into_par_iter()
            .map(|g| {
                let mut rng = gene_rng(cfg.seed, g, sweep);
                let node = &nodes[g];
                let mut th = beta[g].clone();
                let mut z = zed[g].clone();

                // Scratch, allocated once per anchor per sweep and rewritten per
                // dim — the reason anchors are the parallel axis. One offset
                // buffer per TERM, since terms differ in their frozen base.
                let mut offs: Vec<Vec<f32>> = vec![vec![0.0f32; h]; node.len()];
                let mut cur = DVector::<f32>::zeros(h);
                let mut nu = DVector::<f32>::zeros(h);

                // Running slate scores, one buffer per term: `s[j] = ⟨v, e_o⟩ + b_o`
                // for the anchor's CURRENT effective loading `v` (its frozen offset
                // plus every INCLUDED dim). Seeding costs one `|slate| × h` pass per
                // term per sweep; after that each dim is `O(|slate|)`.
                let mut srun: Vec<Vec<f64>> = node
                    .iter()
                    .enumerate()
                    .map(|(t, n)| seed_scores(n, &slabs[t], &th, &z, h))
                    .collect();

                for d in 0..h {
                    // The frozen direction for this coordinate: the term's own
                    // offset (gem's velocity track holds β_g here) PLUS the
                    // EFFECTIVE loading of every other dim, `z_k·β_k`. A dim that
                    // is off contributes nothing — that is what makes this a
                    // spike-and-slab rather than a slab with a label attached.
                    for (t, off) in offs.iter_mut().enumerate() {
                        for (kk, o) in off.iter_mut().enumerate() {
                            let base = node[t].offset.map_or(0.0, |b| b[kk]);
                            *o = base + if kk == d || !z[kk] { 0.0 } else { th[kk] };
                        }
                    }

                    // Peel dim `d` out of every term's running score, so `s_wo[t][j]`
                    // is the score with this coordinate contributing nothing. Adding
                    // `x · e_o[d]` back is then the whole cost of evaluating at `x`.
                    let peeled = if z[d] { th[d] } else { 0.0 };
                    for (t, s) in srun.iter_mut().enumerate() {
                        if peeled != 0.0 {
                            let colr = slabs[t].dim(d);
                            for (sj, cj) in s.iter_mut().zip(colr) {
                                *sj -= f64::from(peeled) * f64::from(*cj);
                            }
                        }
                    }

                    let lnpdf = |x: &DVector<f32>| -> f32 {
                        node.iter()
                            .zip(&offs)
                            .enumerate()
                            .map(|(t, (n, off))| {
                                incremental_ll(n, off, x[d], d, &slabs[t], &srun[t], side)
                            })
                            .sum()
                    };

                    // ℓ with dim `d` OFF. Needed for the `z` draw, and it is also the
                    // likelihood the ESS would start from, so it is not extra work.
                    let ll_off = lnpdf(&zeros_h);

                    // β_gd | z_gd. `cur` and `nu` are zero off coordinate `d`, so the
                    // ESS ellipse moves ONLY that coordinate — a 1-D slice through the
                    // joint, with the rest carried in `off`.
                    cur.fill(0.0);
                    cur[d] = th[d];
                    let ll_on = if z[d] {
                        // On: fit it against the likelihood.
                        let mut ll = lnpdf(&cur);
                        nu.fill(0.0);
                        for _ in 0..k {
                            let g_std: f64 = StandardNormal.sample(&mut rng);
                            nu[d] = g_std as f32 * sd[d];
                            let (nc, nl) = elliptical_slice_step(&cur, &nu, &lnpdf, ll, &mut rng);
                            cur = nc;
                            ll = nl;
                        }
                        ll
                    } else {
                        // Off: the likelihood does not see `β_gd` at all, so its exact
                        // conditional IS the prior. Drawing it here rather than keeping
                        // the stale fit is what makes the `z` step below a valid Gibbs
                        // move — and it is why no Occam/BIC correction is needed. A
                        // fitted `β` always beats 0, so comparing against one would bias
                        // every coordinate ON; a prior draw usually does not, and the
                        // penalty emerges from the chain visiting these states.
                        let g_std: f64 = StandardNormal.sample(&mut rng);
                        cur[d] = g_std as f32 * sd[d];
                        lnpdf(&cur)
                    };
                    th[d] = cur[d];

                    // z_gd | β_gd — an exact Gibbs draw from the joint, needing no
                    // marginal likelihood: `p(z|β,y) ∝ p(y|z·β)·p(z)`. A vetoed
                    // coordinate is pinned off instead: it is excluded by the
                    // model, not merely unfavoured by the data.
                    z[d] = if !selection {
                        true
                    } else {
                        match veto {
                            Some(mask) if !mask[g * h + d] => false,
                            _ => {
                                let logit =
                                    log_prior_odds[d] + f64::from(ll_on) - f64::from(ll_off);
                                rng.random::<f64>() < sigmoid(logit)
                            }
                        }
                    };

                    // Fold the coordinate's NEW effective value back in, so the
                    // running scores describe the state the next dim conditions on.
                    let restored = if z[d] { th[d] } else { 0.0 };
                    if restored != 0.0 {
                        for (t, s) in srun.iter_mut().enumerate() {
                            let colr = slabs[t].dim(d);
                            for (sj, cj) in s.iter_mut().zip(colr) {
                                *sj += f64::from(restored) * f64::from(*cj);
                            }
                        }
                    }
                }
                GeneDraw { beta: th, z }
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
        for (g, draw) in draws.into_iter().enumerate() {
            for d in 0..h {
                let v = f64::from(draw.beta[d]);
                if veto.is_none_or(|mask| mask[g * h + d]) {
                    n_eligible[d] += 1;
                }
                if draw.z[d] {
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
            beta[g] = draw.beta;
            zed[g] = draw.z;
        }

        for d in 0..h {
            sigma2[d] = hv[d].sample(sum_sq[d], n_incl[d].max(1), &mut hyper_rng);
            // Only eligible coordinates are Bernoulli trials — see the note above.
            // With selection off there is nothing to infer, so `π₀` is left at its
            // prior mean rather than being "estimated" from a vector of all-ones.
            if selection {
                pi0[d] = sample_pi0(
                    n_eligible[d].saturating_sub(n_incl[d]),
                    n_eligible[d],
                    cfg.beta_a,
                    cfg.beta_b,
                    &mut hyper_rng,
                );
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
        final_z: zed.into_iter().flatten().collect(),
    }
}

#[cfg(test)]
#[path = "dim_block_tests.rs"]
mod dim_block_tests;
