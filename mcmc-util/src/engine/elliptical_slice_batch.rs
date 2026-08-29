//! `n` **independent** scalar elliptical-slice transitions, evaluated in batch.
//!
//! # Independent, not joint
//!
//! This is `n` separate transitions that happen to be evaluated together. Each item
//! keeps its own slice threshold, its own bracket and its own acceptance; the only
//! thing shared is that `lnpdf` is called **once per shrinkage round** over the items
//! still searching, so a dense backend sees one array operation instead of `n` scalar
//! walks.
//!
//! It is emphatically **not** one move in `Rⁿ`. A joint move would need the summed
//! log-likelihood of every item to clear a single threshold, and its acceptance region
//! shrinks exponentially in `n` — at the sizes this exists for (tens of thousands of
//! anchors) it would never move. The batching is licensed by the items being
//! conditionally independent, which is a property of the caller's model, not of this
//! function.
//!
//! # Why one RNG per item
//!
//! [`elliptical_slice_batch`] takes `rngs: &mut [R]`, not a single stream. Drawing
//! every item's randomness from one shared stream would make the result depend on how
//! the caller happened to group items into batches — so the same seed would give
//! different answers at different batch sizes, and reproducibility would quietly be a
//! function of a performance knob. With one stream per item, keyed by the caller from
//! `(seed, item, sweep)`, the result is invariant to grouping.
//!
//! # Guarantees carried over from the scalar kernel
//!
//! Same `MAX_BRACKET_ITERS` cap and `BRACKET_MIN_WIDTH` floor as
//! [`super::elliptical_slice_step`], applied **per item**, with the same
//! fall-back-to-current behaviour — well defined because `cur_lnpdf > hh` holds by
//! construction from `hh = ln(U) + cur_lnpdf` with `U ∈ (0,1)`. Fallbacks are
//! *counted* and returned rather than swallowed: they are this kernel's analogue of a
//! rejected move, and an active-set loop is exactly where such a count is easy to
//! lose.

use rand::{Rng, RngExt};
use std::f32::consts::PI;

use super::elliptical_slice::{BRACKET_MIN_WIDTH, MAX_BRACKET_ITERS};

/// Result of one batched sweep of scalar ESS transitions.
pub struct BatchStep {
    /// The new value per item — either an accepted ellipse point or, on a bracket
    /// fallback, the item's current value unchanged.
    pub value: Vec<f32>,
    /// The log-likelihood at [`Self::value`], so the caller need not re-evaluate.
    pub lnpdf: Vec<f32>,
    /// How many items exhausted their bracket and fell back. Report it — a silently
    /// growing fallback count is a stalled sampler that still returns numbers.
    pub fallbacks: usize,
    /// How many times `lnpdf` was called, i.e. shrinkage rounds. One per round over
    /// a decaying active set, so this bounds the batch's cost.
    pub rounds: usize,
}

/// Run one scalar ESS transition for every item.
///
/// - `cur`: each item's current value.
/// - `prior_draw`: each item's `ν`, drawn from ITS prior — for a spike-and-slab slab
///   that is `N(0, σ₀d²)`, so the caller owns the per-dim scale.
/// - `cur_lnpdf`: the log-likelihood at `cur`, cached by the caller.
/// - `rngs`: one stream per item; see the module doc for why this is not one stream.
/// - `lnpdf`: `lnpdf(x, active, out)` fills `out[i]` with the log-likelihood of item
///   `active[i]` at value `x[i]`. Called once per round; `active` shrinks as items
///   accept, and `x` / `out` are parallel to it.
///
/// # Panics
///
/// If `cur`, `prior_draw`, `cur_lnpdf` and `rngs` are not all the same length — a
/// mismatch there would silently sample the wrong item's conditional.
pub fn elliptical_slice_batch<R: Rng>(
    cur: &[f32],
    prior_draw: &[f32],
    cur_lnpdf: &[f32],
    rngs: &mut [R],
    lnpdf: &mut impl FnMut(&[f32], &[u32], &mut [f32]),
) -> BatchStep {
    let n = cur.len();
    assert_eq!(prior_draw.len(), n, "prior_draw must be one per item");
    assert_eq!(cur_lnpdf.len(), n, "cur_lnpdf must be one per item");
    assert_eq!(rngs.len(), n, "rngs must be one per item");

    let mut value = cur.to_vec();
    let mut out_lnpdf = cur_lnpdf.to_vec();

    // Per-item slice threshold and bracket. `hh` is fixed for the whole transition;
    // the bracket is what shrinks.
    let mut hh = vec![0.0f32; n];
    let mut angle = vec![0.0f32; n];
    let mut lo = vec![0.0f32; n];
    let mut hi = vec![0.0f32; n];
    for i in 0..n {
        let u: f32 = rngs[i].random();
        hh[i] = u.ln() + cur_lnpdf[i];
        let phi: f32 = rngs[i].random_range(0.0..2.0 * PI);
        angle[i] = phi;
        lo[i] = phi - 2.0 * PI;
        hi[i] = phi;
    }

    let mut active: Vec<u32> = (0..n as u32).collect();
    let mut x = vec![0.0f32; n];
    let mut ll = vec![0.0f32; n];
    let mut fallbacks = 0usize;
    let mut rounds = 0usize;

    for _ in 0..MAX_BRACKET_ITERS {
        if active.is_empty() {
            break;
        }
        // Build this round's candidates for the items still searching.
        x.clear();
        for &i in &active {
            let i = i as usize;
            let a = angle[i];
            x.push(cur[i] * a.cos() + prior_draw[i] * a.sin());
        }
        ll.resize(active.len(), 0.0);
        lnpdf(&x, &active, &mut ll[..active.len()]);
        rounds += 1;

        // Retire what cleared its threshold; shrink the rest.
        let mut still = Vec::with_capacity(active.len());
        for (slot, &i) in active.iter().enumerate() {
            let idx = i as usize;
            if ll[slot] > hh[idx] {
                value[idx] = x[slot];
                out_lnpdf[idx] = ll[slot];
                continue;
            }
            if angle[idx] < 0.0 {
                lo[idx] = angle[idx];
            } else {
                hi[idx] = angle[idx];
            }
            if hi[idx] - lo[idx] < BRACKET_MIN_WIDTH {
                // Collapsed bracket: the proposal is numerically indistinguishable
                // from current, which is in the slice by construction.
                fallbacks += 1;
                continue;
            }
            angle[idx] = rngs[idx].random_range(lo[idx]..hi[idx]);
            still.push(i);
        }
        active = still;
    }

    // Anything still searching hit the iteration cap. Same fallback, same reason.
    fallbacks += active.len();

    BatchStep {
        value,
        lnpdf: out_lnpdf,
        fallbacks,
        rounds,
    }
}

#[cfg(test)]
#[path = "elliptical_slice_batch_tests.rs"]
mod elliptical_slice_batch_tests;
