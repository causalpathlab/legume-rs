//! Per-node Poisson log-likelihood against a **frozen** other side.
//!
//! For an anchor node `a` (a cell in the cell sweep, a feature in the feature
//! sweep) with parameter `θ_a = [e_a ; b_a]` (length `H+1`) and the other side
//! held fixed as `{ (e_o, b_o) }`, the score is `s_ao = ⟨e_a, e_o⟩ + b_a + b_o`
//! and the Poisson log-likelihood is
//!
//! ```text
//!   Σ_{o ∈ pos}  n_ao · s_ao   −   scale · Σ_{o ∈ partition}  exp(s_ao)
//! ```
//!
//! `pos` are the observed `(o, count)` edges; `partition` is the set of other-side
//! rows summed in the rate normalizer (the whole other side for the exact
//! small-scale case, or a frozen sampled slate with `scale = |pool|/K` at scale).
//! The Gaussian prior on `θ_a` is supplied by the sampler (ESS draws the ellipse
//! from it), so this function is the **likelihood only**.
//!
//! Every linear predictor is clamped at [`SCORE_CLAMP`] before `exp`, matching
//! `crate::cell_projection::SCORE_CLAMP` (f32 `exp` overflows at ~88; the shared
//! bound keeps the whole crate's Poisson fits consistent). Accumulation is in
//! `f64` — the same widening `cell_projection` uses — so the sum stays honest
//! when many small terms are added.

use crate::cell_projection::SCORE_CLAMP;
use nalgebra::DVector;

/// The frozen other side of the bilinear score: row-major embeddings `e`
/// (`[n_other × h]`) and per-row biases `b` (`[n_other]`).
pub struct FrozenSide<'a> {
    pub e: &'a [f32],
    pub b: &'a [f32],
    pub h: usize,
}

impl FrozenSide<'_> {
    /// Number of frozen other-side rows.
    #[must_use]
    pub fn n(&self) -> usize {
        self.b.len()
    }

    /// Row `o`'s embedding slice `[h]`.
    #[inline]
    fn row(&self, o: u32) -> &[f32] {
        let o = o as usize;
        &self.e[o * self.h..(o + 1) * self.h]
    }
}

/// Precomputed sufficient statistics that collapse an anchor's **data** term to
/// a single dot product.
///
/// [`multinomial_ll`]'s data term is `Σ_pos n·⟨e_a + off, e_o⟩`, which is exactly
/// `⟨e_a + off, Σ_pos n·e_o⟩`. The inner sum does not depend on `e_a`, so it can
/// be formed once per anchor instead of re-walking every edge on every likelihood
/// evaluation — turning an `O(|pos| · h)` term into `O(h)`. It is the same
/// quantity [`multinomial_ll`]'s own gradient note calls `m_a`.
///
/// This matters most when an anchor's edge list is the whole other side. A
/// pseudobulk anchored against the feature axis has tens of thousands of observed
/// edges, and re-walking them per dim per sweep dominates everything else.
///
/// **Difference from the edge walk, and the bound that keeps it safe.** The
/// collapsed form drops the constant `Σ_pos n·b_o`, which cancels in both
/// consumers (an elliptical-slice threshold comparison, and the `z` draw's
/// `ll_on − ll_off`), so that part is free.
///
/// It also cannot apply the per-edge [`SCORE_CLAMP`], and that part is **not**
/// free. `SCORE_CLAMP` is 30 — a modelling bound, not an `exp` overflow guard at
/// ~88 — and `clamp` is nonlinear in `e_a`, so once any score saturates the two
/// forms stop differing by a constant. Worse, the partition term stays clamped:
/// an unclamped data term against a capped normalizer is unbounded above in
/// `e_a`, and a slice sampler will happily walk out along it.
///
/// So [`Self::safe_radius`] records how far `e_a + offset` may reach before ANY
/// score could saturate, and [`multinomial_ll`] falls back to the exact clamped
/// edge walk outside it. Inside the radius the collapse is exact; outside, it is
/// simply not used.
pub struct AnchorMoment {
    /// `[h]` count-weighted sum of the observed other-side rows, `Σ_pos n·e_o`.
    pub m: Vec<f32>,
    /// `Σ_pos n` — the anchor's total observed count.
    pub total: f64,
    /// Largest `‖e_a + offset‖` for which no observed edge's score can reach
    /// `±SCORE_CLAMP`, so the collapsed data term is exactly the clamped one.
    ///
    /// From `|s| ≤ ‖e_a + off‖·max‖e_o‖ + max|b_o|` over this anchor's edges.
    pub safe_radius: f32,
}

impl AnchorMoment {
    /// Form the moment for one anchor's edges against `side`.
    #[must_use]
    pub fn new(pos: &[(u32, f32)], side: &FrozenSide) -> Self {
        let mut m = vec![0.0f32; side.h];
        let mut total = 0.0f64;
        let mut max_row_norm = 0.0f32;
        let mut max_abs_b = 0.0f32;
        for &(o, n) in pos {
            let e_o = side.row(o);
            let mut nrm2 = 0.0f32;
            for (acc, v) in m.iter_mut().zip(e_o) {
                *acc += n * v;
                nrm2 += v * v;
            }
            max_row_norm = max_row_norm.max(nrm2.sqrt());
            max_abs_b = max_abs_b.max(side.b[o as usize].abs());
            total += f64::from(n);
        }
        let headroom = SCORE_CLAMP as f32 - max_abs_b;
        let safe_radius = if max_row_norm > 0.0 && headroom > 0.0 {
            headroom / max_row_norm
        } else {
            // A degenerate side (or a bias already at the clamp) gives the fast
            // path no safe region at all; always take the exact walk.
            0.0
        };
        Self {
            m,
            total,
            safe_radius,
        }
    }
}

/// One anchor node's likelihood terms against the frozen side.
#[derive(Clone, Copy)]
pub struct NodeTerm<'a> {
    /// Observed `(other-index, count)` edges — the data term.
    pub pos: &'a [(u32, f32)],
    /// Other-indices summed in the rate normalizer (all others, or a frozen slate).
    pub partition: &'a [u32],
    /// `|pool| / K` — folds a sampled `partition` back up to the full-sum scale.
    /// `1.0` when `partition` is the whole other side (the exact case).
    pub partition_scale: f64,
    /// Fixed `[h]` vector added to the sampled `e_a` before the dot product, so
    /// the sampler explores a **deviation from a frozen direction** rather than an
    /// absolute loading: `⟨e_a + offset, e_o⟩`.
    ///
    /// This is what lets a second, dependent effect be sampled against a first one
    /// held at its MAP. `faba gem`'s velocity gate is the motivating case — an
    /// unspliced row scores `⟨β_g + δ_g, e_c⟩`, so sampling `δ_g` means carrying
    /// `β_g` as the offset — but nothing here is gem-specific. `None` is the plain
    /// case, and costs nothing.
    pub offset: Option<&'a [f32]>,
    /// Precomputed data-term statistics. `Some` makes [`multinomial_ll`] skip the
    /// edge walk entirely — see [`AnchorMoment`] for what that changes (nothing
    /// that reaches an inference) and when it is worth it. `None` walks `pos`.
    pub moment: Option<&'a AnchorMoment>,
}

impl<'a> NodeTerm<'a> {
    /// A node with no frozen offset — the common case.
    #[must_use]
    pub fn new(pos: &'a [(u32, f32)], partition: &'a [u32], partition_scale: f64) -> Self {
        Self {
            pos,
            partition,
            partition_scale,
            offset: None,
            moment: None,
        }
    }

    /// Attach precomputed data-term statistics; see [`AnchorMoment`].
    #[must_use]
    pub fn with_moment(mut self, moment: &'a AnchorMoment) -> Self {
        self.moment = Some(moment);
        self
    }
}

/// `s_ao = ⟨e_a + offset, e_o⟩ + b_a + b_o`, clamped to `±SCORE_CLAMP`. `e_a` /
/// `b_a` come from `θ_a`; `e_o` / `b_o` from the frozen side; `offset` is the
/// anchor's frozen direction (`None` ⇒ zero).
#[inline]
fn score(e_a: &[f32], b_a: f64, o: u32, side: &FrozenSide, offset: Option<&[f32]>) -> f64 {
    let e_o = side.row(o);
    let dot: f64 = match offset {
        None => e_a
            .iter()
            .zip(e_o)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum(),
        Some(off) => e_a
            .iter()
            .zip(off)
            .zip(e_o)
            .map(|((a, f), b)| (f64::from(*a) + f64::from(*f)) * f64::from(*b))
            .sum(),
    };
    (dot + b_a + f64::from(side.b[o as usize])).clamp(-SCORE_CLAMP, SCORE_CLAMP)
}

/// Per-node Poisson log-likelihood with the embedding `e_a` and bias `b_a` passed
/// separately (see the module doc). Split this way so a caller can either sample the
/// bias alongside the loading ([`poisson_lnpdf`], which packs `θ = [e ; b]`) or hold
/// it fixed and sample only the `H`-dim loading.
#[must_use]
pub fn poisson_ll(e_a: &[f32], b_a: f64, node: &NodeTerm, side: &FrozenSide) -> f32 {
    debug_assert_eq!(e_a.len(), side.h);
    let mut ll = 0.0f64;
    for &(o, n) in node.pos {
        ll += f64::from(n) * score(e_a, b_a, o, side, node.offset);
    }
    let mut part = 0.0f64;
    for &o in node.partition {
        part += score(e_a, b_a, o, side, node.offset).exp();
    }
    ll -= node.partition_scale * part;
    ll as f32
}

/// Per-node Poisson log-likelihood (see the module doc). `theta` is `[e_a ; b_a]`
/// of length `h + 1` — the bias is the last coordinate and is sampled with the rest.
#[must_use]
pub fn poisson_lnpdf(theta: &DVector<f32>, node: &NodeTerm, side: &FrozenSide) -> f32 {
    let h = side.h;
    debug_assert_eq!(theta.len(), h + 1);
    poisson_ll(&theta.as_slice()[..h], f64::from(theta[h]), node, side)
}

/// Per-node **profile** log-likelihood: the Poisson with its intercept `b_a`
/// analytically maximized out, which is exactly the multinomial (softmax)
/// likelihood over the other side.
///
/// # Why this and not [`poisson_ll`]
///
/// `b_a` is a pure nuisance parameter — it sets the anchor's overall rate, not
/// which direction it loads — and it has a closed-form optimum:
///
/// ```text
///   ∂ℓ/∂b_a = T_a − exp(b_a)·A(θ) = 0   ⇒   exp(b_a*) = T_a / A(θ)
///   A(θ) = Σ_o exp(⟨e_a, e_o⟩ + b_o),   T_a = Σ_pos n
/// ```
///
/// Substituting it back collapses the two-parameter Poisson to
///
/// ```text
///   ℓ_p(e_a) = Σ_pos n·s_o  −  T_a · ln Σ_{o ∈ partition} exp(s_o) + const
/// ```
///
/// with `s_o = ⟨e_a, e_o⟩ + b_o` — no `b_a` anywhere. Three consequences, and
/// they are the reason this is the default for the gate:
///
/// 1. **No intercept to get wrong.** Holding `b_a` fixed at a value fitted under
///    a *different* objective (the trainer is NCE, not Poisson) is what made the
///    frozen intercepts need recalibration at all.
/// 2. **The score is centred by construction.** `∇ℓ_p(0) = m_a − T_a·μ₀`, where
///    `m_a = Σ_pos n·e_o` and `μ₀` is the `softmax(b_o)`-weighted mean of the
///    frozen side. The anchor is compared against the *global mean direction*
///    rather than against the origin, so the shared count-weighted direction —
///    which otherwise pulls every anchor onto the same one or two dims — cancels.
///    That collapse is the failure this function exists to remove.
/// 3. It is the same estimand phase-1 trains under `NceObjective::Softmax`.
///
/// `partition_scale` folds a sampled slate up to the full sum inside the `ln`, so
/// it contributes `ln(scale)` — constant in `e_a`, hence dropped. Accumulated in
/// `f64` and max-shifted, since `logsumexp` over a whole slate is exactly where a
/// naive sum loses its low bits.
#[must_use]
pub fn multinomial_ll(e_a: &[f32], node: &NodeTerm, side: &FrozenSide) -> f32 {
    debug_assert_eq!(e_a.len(), side.h);
    // The intercept is profiled out, so any constant here would cancel; 0 keeps
    // `score` shared with the Poisson path.
    // The collapsed form is exact only while no observed score can saturate
    // `SCORE_CLAMP`; past that the clamp is nonlinear in `e_a` and the fast path
    // stops agreeing with the walk by a constant. Checking costs one norm.
    let moment = node.moment.filter(|mom| {
        let r2: f32 = match node.offset {
            None => e_a.iter().map(|v| v * v).sum(),
            Some(off) => e_a.iter().zip(off).map(|(a, f)| (a + f) * (a + f)).sum(),
        };
        r2.sqrt() <= mom.safe_radius
    });
    let (data, total) = match moment {
        // Collapsed form: `Σ_pos n·⟨e_a + off, e_o⟩ = ⟨e_a + off, m⟩`. See
        // `AnchorMoment` for why dropping the `Σ n·b_o` constant leaves every
        // downstream comparison unchanged.
        Some(mom) => {
            debug_assert_eq!(mom.m.len(), side.h);
            let dot: f64 = match node.offset {
                None => e_a
                    .iter()
                    .zip(&mom.m)
                    .map(|(a, m)| f64::from(*a) * f64::from(*m))
                    .sum(),
                Some(off) => e_a
                    .iter()
                    .zip(off)
                    .zip(&mom.m)
                    .map(|((a, f), m)| (f64::from(*a) + f64::from(*f)) * f64::from(*m))
                    .sum(),
            };
            (dot, mom.total)
        }
        None => {
            let mut data = 0.0f64;
            let mut total = 0.0f64;
            for &(o, n) in node.pos {
                data += f64::from(n) * score(e_a, 0.0, o, side, node.offset);
                total += f64::from(n);
            }
            (data, total)
        }
    };
    if total == 0.0 {
        return 0.0; // no counts ⇒ the likelihood is flat in `e_a`
    }
    let mut m = f64::NEG_INFINITY;
    for &o in node.partition {
        m = m.max(score(e_a, 0.0, o, side, node.offset));
    }
    let mut s = 0.0f64;
    for &o in node.partition {
        s += (score(e_a, 0.0, o, side, node.offset) - m).exp();
    }
    (data - total * (m + s.max(f64::MIN_POSITIVE).ln())) as f32
}

#[cfg(test)]
#[path = "lnpdf_tests.rs"]
mod lnpdf_tests;
