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
        }
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
/// separately (see the module doc). Shared core for both the full-`θ` sweep
/// ([`poisson_lnpdf`], which samples the bias too) and the gate
/// ([`super::gate`], which fixes `b_a` at the MAP and samples only the `H`-dim
/// gated loading).
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
    let mut data = 0.0f64;
    let mut total = 0.0f64;
    for &(o, n) in node.pos {
        data += f64::from(n) * score(e_a, 0.0, o, side, node.offset);
        total += f64::from(n);
    }
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
