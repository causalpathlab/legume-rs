//! What the sampler needs of a score, and the scores that satisfy it.
//!
//! # Why a seam at all
//!
//! The sampler is a **likelihood oracle**. Elliptical slice sampling constrains the
//! *prior* — Gaussian, which the slab `N(0, σ₀d²)` is — and reads the likelihood
//! only through `ll(θ) > threshold`; the `z` draw reads it only through
//! `ll_on − ll_off`. Nothing in the transition kernel knows the score is
//! bilinear-Poisson. What couples them is three algebraic identities:
//!
//! | identity | what it buys |
//! |---|---|
//! | `s_j = ⟨v, e_j⟩ + b_j` affine in `v` | scores maintainable by a rank-1 update per dim |
//! | data term linear in the scores | [`super::lnpdf::AnchorMoment`]'s `Σ_pos n·⟨v,e_o⟩ = ⟨v,m⟩` |
//! | elementwise vs log-of-a-sum reduction | whether the slate term needs a max shift |
//!
//! A score that has those properties should get the fast paths; one that does not
//! must still work. So they are **declared**, not assumed — [`AnchorScore`] carries
//! one required evaluation plus capability predicates, and an undeclared capability
//! costs speed, never correctness.
//!
//! # Why the evaluation is batched over a column
//!
//! Given the frozen other side, anchors are conditionally independent, so the
//! conditional for one embedding dim factorizes over anchors. The sampler exploits
//! that by moving a whole column of anchors at once — `n` independent 1-D
//! transitions, evaluated as one dense pass rather than `n` scalar walks. So the
//! natural shape here is *one dim, many anchors*: [`AnchorScore::ll_column`].
//!
//! This is batching of the **evaluation**, not of the move. Each anchor keeps its
//! own angle, bracket and acceptance. A single joint move over `R^n` would need the
//! summed log-likelihood of every anchor to clear one slice threshold, and its
//! acceptance region shrinks exponentially in `n`.
//!
//! # Precision
//!
//! Scores are stored `f32` — that is what keeps a tile cache-resident, and what a
//! dense backend would need — but every reduction over the slate accumulates in
//! `f64` after a max shift. A logsumexp over a thousand terms is exactly where a
//! naive `f32` sum loses its low bits, and the quantity being reduced is the one the
//! sampler compares against a threshold.

use super::lnpdf::{multinomial_ll, FrozenSide, NodeTerm};
use crate::cell_projection::SCORE_CLAMP;

/// The frozen side restricted to one term's negative slate, **transposed** so a
/// single dim's column is contiguous.
///
/// A column pass changes exactly one coordinate at a time, so a score only ever
/// shifts by `Δ · e_o[d]`. Holding `e_o[d]` contiguous across the slate is what turns
/// the rate normalizer from `|slate|` dot products of length `h` into one pass of
/// `|slate|` multiply-adds — a per-anchor cost that scales with `h` rather than `h²`.
///
/// Built once per block and shared read-only across every tile: `h × |slate|` floats,
/// 128 KB at `h = 32`, `|slate| = 1024`, so it stays resident alongside a tile.
pub struct SlateSlab {
    /// `col[d * k + j] = e[slate[j] * h + d]`.
    col: Vec<f32>,
    /// `b_o` per slate entry.
    b: Vec<f32>,
    k: usize,
}

impl SlateSlab {
    /// Transpose `side`'s slate rows into dim-major order.
    #[must_use]
    pub fn new(partition: &[u32], side: &FrozenSide) -> Self {
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

    /// Dim `d`'s contiguous column over the slate, `[k]`.
    #[inline]
    #[must_use]
    pub fn dim(&self, d: usize) -> &[f32] {
        &self.col[d * self.k..(d + 1) * self.k]
    }

    /// Per-slate-entry biases `b_o`, `[k]`.
    #[inline]
    #[must_use]
    pub fn biases(&self) -> &[f32] {
        &self.b
    }

    /// Slate width.
    #[inline]
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }
}

/// Everything a column evaluation reads, built once per tile per sweep.
///
/// All fields are `[b …]`-shaped over the anchors of ONE tile, not over the whole
/// axis, and all are immutable: the peel/restore that advances the dim loop is the
/// caller's business, so a score cannot corrupt the state it is handed.
///
/// "Peeled" throughout means *dim `d`'s sampled contribution removed*. A frozen
/// [`Self::offset`] is **not** peeled — it is not being sampled — so its dim-`d`
/// component stays folded into [`Self::s`], [`Self::data`] and [`Self::sumsq`], and
/// an evaluation at `x` adds only the sampled part back.
pub struct ColumnCtx<'a> {
    /// `[b × k]` row-major slate scores, peeled: `s[i,j] = Σ_{k≠d} v_ik·e_j[k] + b_j`
    /// with any offset folded in.
    ///
    /// **EMPTY when the score declined [`AnchorScore::affine_in_anchor`]** — the block is
    /// only meaningful for a score whose per-entry value is affine in the loading, so it is
    /// not built at all otherwise. Reach it through [`Self::scores`], which panics rather
    /// than returning nothing.
    pub s: &'a [f32],
    /// `[b × h]` effective loading `z ⊙ β` with dim `d` **zeroed**. Only the exact
    /// fallback needs it; the fast path works from `s` alone.
    pub v: &'a [f32],
    /// `[b]` data term, peeled: `Σ_{k≠d} (v+off)_ik · m_i[k] + Σ_pos n·b_o`.
    ///
    /// `f64` because the `Σ n·b_o` term grows with an anchor's edge count — a
    /// pseudobulk anchored against the feature axis has tens of thousands of edges —
    /// and this is `[b]`-sized, so the precision is free. Only the `[b × k]` score
    /// block is `f32`, and only because keeping a tile cache-resident is the point.
    pub data: &'a [f64],
    /// `[b]` squared radius, peeled: `Σ_{k≠d} (v+off)_ik²`. The other half of the
    /// safe-radius guard; the dim-`d` term is added per evaluation.
    pub sumsq: &'a [f32],
    /// `[b × h]` count-weighted moments `m_i = Σ_pos n·e_o`.
    pub m: &'a [f32],
    /// `[b]` total observed count per anchor, `T_i = Σ_pos n`.
    pub total: &'a [f64],
    /// `[b]` largest `‖v + off‖` for which no observed edge's score can saturate
    /// [`SCORE_CLAMP`]. Past it the collapsed data term stops differing from the
    /// walk by a constant, so the walk is taken instead.
    pub safe_radius: &'a [f32],
    /// `[b × h]` frozen offset — gem's velocity track carries `β_g` here so `δ_g`
    /// is sampled as a deviation from it. `None` is the plain case.
    pub offset: Option<&'a [f32]>,
    /// `[b]` observed `(other-index, count)` edges per anchor, for the exact walk.
    ///
    /// Borrowed, not owned: these are the caller's edge lists and nothing here mutates
    /// them. Copying instead would duplicate the whole observed data of one side per tile
    /// per sweep — on a pb block whose anchors each touch tens of thousands of edges that
    /// is the entire count matrix, memcpy'd, every sweep.
    pub pos: &'a [&'a [(u32, f32)]],
    /// Other-side indices summed in the normalizer.
    pub partition: &'a [u32],
    /// `n_other / |partition|`, folding a sampled slate up to the full sum.
    pub partition_scale: f64,
    /// The frozen other side the scores were built against.
    pub side: &'a FrozenSide<'a>,
    /// Slate width, `partition.len()`.
    pub k: usize,
    /// Embedding dimension.
    pub h: usize,
}

impl ColumnCtx<'_> {
    /// Anchor `i`'s peeled slate scores, `[k]`.
    ///
    /// # Panics
    ///
    /// If the score declined [`AnchorScore::affine_in_anchor`], in which case the block was
    /// never built. That is a score reading state it told the caller it would not need, and
    /// it should fail loudly rather than be handed a plausible empty slice.
    #[inline]
    #[must_use]
    pub fn scores(&self, i: usize) -> &[f32] {
        assert!(
            !self.s.is_empty(),
            "this score declared `affine_in_anchor() == false`, so no slate score block was \
             built — compute from `v` and the edges instead"
        );
        &self.s[i * self.k..(i + 1) * self.k]
    }

    /// Anchor `i`'s effective loading with dim `d` zeroed, `[h]`.
    #[inline]
    #[must_use]
    pub fn loading(&self, i: usize) -> &[f32] {
        &self.v[i * self.h..(i + 1) * self.h]
    }

    /// Anchor `i`'s frozen offset at dim `d`, or `0` when there is none.
    #[inline]
    #[must_use]
    pub fn offset_at(&self, i: usize, d: usize) -> f32 {
        self.offset.map_or(0.0, |o| o[i * self.h + d])
    }

    /// Anchor `i`'s moment component at dim `d`.
    #[inline]
    #[must_use]
    pub fn moment_at(&self, i: usize, d: usize) -> f32 {
        self.m[i * self.h + d]
    }
}

/// A score the per-dim column sampler can evaluate.
///
/// Implementors own the *whole* reduction, not a per-slate-entry hook. That is
/// deliberate: the profiled Poisson's `−T·ln Σ_j exp(s_j)` is a log **of a sum** and
/// needs a max shift across the slate, while a logistic's `Σ_j log σ(−s_j)` is
/// elementwise. One per-entry hook cannot express both without one of them paying
/// for machinery it does not use.
pub trait AnchorScore: Sync {
    /// Short name for logs, so a run says which score and which path it took.
    fn label(&self) -> &'static str;

    /// Evaluate one column: `out[i]` is the log-likelihood of anchor `active[i]`
    /// with dim `d` set to `x[i]`.
    ///
    /// `active` indexes into the TILE (`0..b`), not the global anchor axis, and
    /// `x` / `out` are parallel to it. `c_d` is the slate's dim-`d` column, `[k]`.
    ///
    /// The contract is that this is a pure function of `(ctx, d, c_d, x)` — the
    /// sampler calls it many times per dim with different `x` and expects no hidden
    /// state to have moved.
    fn ll_column(
        &self,
        ctx: &ColumnCtx<'_>,
        d: usize,
        c_d: &[f32],
        x: &[f32],
        active: &[u32],
        out: &mut [f32],
    );

    /// Is `s_j` affine in the anchor's loading, so [`ColumnCtx::s`] can be carried across
    /// the dim loop by a rank-1 update?
    ///
    /// **This is honoured, not merely recorded.** Declaring `false` makes the caller skip
    /// building the score block entirely — the dominant setup cost, one `O(b·k·h)` pass per
    /// tile per sweep — and leaves [`ColumnCtx::s`] EMPTY. A score that declines this must
    /// therefore compute from [`ColumnCtx::v`] and the raw edges, and must not call
    /// [`ColumnCtx::scores`], which will panic on an empty block rather than hand back
    /// silence. `column_tests`'s `a_non_bilinear_likelihood_is_sampled_correctly` is a
    /// worked example.
    ///
    /// Declaring it wrongly in the other direction — `true` for a score that is not affine
    /// — is the dangerous case: the block would be carried by an update its algebra does
    /// not license, and nothing would fail loudly.
    fn affine_in_anchor(&self) -> bool {
        false
    }

    /// Is the data term linear in the scores, so an anchor's observed edges collapse
    /// to [`super::lnpdf::AnchorMoment`]'s single dot product?
    ///
    /// `false` means the data term walks the edge list on every evaluation. That is
    /// `O(|pos_i|)`, not `O(|pos_i|·h)`, so it is cheap — the expensive half is the
    /// normalizer, and this predicate does not affect it.
    fn data_term_is_linear(&self) -> bool {
        false
    }
}

/////////////////////////////////////////////////////
// The profiled Poisson — the default, and the one //
// every other score is validated against         //
/////////////////////////////////////////////////////

/// Poisson likelihood with the anchor intercept **profiled out**, evaluated over a
/// frozen slate. The estimand [`multinomial_ll`] documents, in column form.
///
/// ```text
///   ll_i(v) = ⟨v + off, m_i⟩  −  T_i · ln Σ_{j ∈ slate} exp(s_ij)
/// ```
///
/// Both fast paths apply: `s_j` is affine in `v`, and the data term is linear in the
/// scores. So a dim change is a rank-1 update of `s` and the data term is one axpy —
/// which together are why a column pass costs `O(b·k)` per evaluation rather than
/// `O(b·k·h)`.
///
/// Outside an anchor's [`ColumnCtx::safe_radius`] the collapse is not exact — the
/// clamp is nonlinear in `v` — so that anchor falls back to the audited scalar walk
/// in [`multinomial_ll`]. Inside, the two agree to `f32` rounding; that equivalence
/// is asserted pointwise in `score_tests.rs` and is what makes this rewrite
/// auditable.
pub struct ProfiledPoisson;

impl AnchorScore for ProfiledPoisson {
    fn label(&self) -> &'static str {
        "profiled-poisson"
    }

    fn affine_in_anchor(&self) -> bool {
        true
    }

    fn data_term_is_linear(&self) -> bool {
        true
    }

    fn ll_column(
        &self,
        ctx: &ColumnCtx<'_>,
        d: usize,
        c_d: &[f32],
        x: &[f32],
        active: &[u32],
        out: &mut [f32],
    ) {
        debug_assert_eq!(c_d.len(), ctx.k);
        debug_assert_eq!(x.len(), active.len());
        debug_assert_eq!(out.len(), active.len());

        // One scratch loading, reused across the batch. Only the fallback path
        // touches it, and that path is rare by construction.
        let mut full = vec![0f32; ctx.h];

        for (slot, (&i, &xi)) in out.iter_mut().zip(active.iter().zip(x)) {
            let i = i as usize;
            let total = ctx.total[i];
            if total == 0.0 {
                // No counts ⇒ the likelihood is flat in `v`, so the conditional is
                // the prior and every candidate scores the same.
                *slot = 0.0;
                continue;
            }

            // Radius guard. `sumsq` already holds the other coordinates, offset
            // folded; dim `d` contributes the sampled value PLUS its frozen offset.
            let off_d = ctx.offset_at(i, d);
            let v_d = xi + off_d;
            // `sumsq` is carried by subtract-then-add across the dim loop, so f32
            // cancellation can leave it very slightly NEGATIVE — reachable whenever the
            // peeled coordinate was the anchor's only nonzero one, which is the common case
            // early in a run while `z` is still mostly off. `sqrt` of that is NaN, and
            // `NaN > radius` is FALSE, so the guard would quietly wave the fast path
            // through instead of falling back. Clamp rather than trust the arithmetic.
            let r2 = (ctx.sumsq[i].max(0.0) + v_d * v_d).max(0.0);
            if r2.sqrt() > ctx.safe_radius[i] {
                *slot = self.walk(ctx, i, d, xi, &mut full);
                continue;
            }

            // Data term: the peeled part plus this coordinate's contribution. Note
            // it is the SAMPLED value that is added — the offset's share is already
            // inside `ctx.data`, which also carries the `Σ n·b_o` the `⟨v, m⟩`
            // collapse omits, so this branch equals `Self::walk` rather than
            // differing from it by a constant.
            let data = ctx.data[i] + f64::from(xi) * f64::from(ctx.moment_at(i, d));

            // Normalizer: ONE pass over the slate, clamping each score exactly where
            // `lnpdf::score` clamps it.
            //
            // A logsumexp normally needs a first pass to find the row max. Here it does
            // not, and the reason is the clamp: every term is already confined to
            // `[-SCORE_CLAMP, SCORE_CLAMP]`, so shifting by the constant `SCORE_CLAMP`
            // puts every exponent in `[e^-60, 1]` — nowhere near f64's range at either
            // end. A fixed shift is as stable as the data-dependent one and halves the
            // innermost loop of the whole sampler.
            let s = ctx.scores(i);
            let xd = f64::from(xi);
            let mut acc = 0.0f64;
            for (sj, cj) in s.iter().zip(c_d) {
                let v = (f64::from(*sj) + xd * f64::from(*cj)).clamp(-SCORE_CLAMP, SCORE_CLAMP);
                acc += (v - SCORE_CLAMP).exp();
            }
            if !acc.is_finite() {
                *slot = 0.0;
                continue;
            }
            let log_part = SCORE_CLAMP + acc.max(f64::MIN_POSITIVE).ln();
            *slot = (data - total * log_part) as f32;
        }
    }
}

impl ProfiledPoisson {
    /// Exact fallback for an anchor whose loading has left its safe radius: rebuild
    /// the full vector and hand it to the audited scalar path.
    ///
    /// `multinomial_ll` without a moment walks every edge and clamps per edge, which
    /// is precisely the form the collapse stops matching out here.
    fn walk(&self, ctx: &ColumnCtx<'_>, i: usize, d: usize, xi: f32, full: &mut [f32]) -> f32 {
        full.copy_from_slice(ctx.loading(i));
        full[d] = xi;
        let node = NodeTerm {
            offset: ctx.offset.map(|o| &o[i * ctx.h..(i + 1) * ctx.h]),
            ..NodeTerm::new(ctx.pos[i], ctx.partition, ctx.partition_scale)
        };
        multinomial_ll(full, &node, ctx.side)
    }
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
