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
}

/// `s_ao = ⟨e_a, e_o⟩ + b_a + b_o`, clamped to `±SCORE_CLAMP`. `e_a` / `b_a` come
/// from `θ_a`; `e_o` / `b_o` from the frozen side.
#[inline]
fn score(e_a: &[f32], b_a: f64, o: u32, side: &FrozenSide) -> f64 {
    let e_o = side.row(o);
    let dot: f64 = e_a
        .iter()
        .zip(e_o)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
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
        ll += f64::from(n) * score(e_a, b_a, o, side);
    }
    let mut part = 0.0f64;
    for &o in node.partition {
        part += score(e_a, b_a, o, side).exp();
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
