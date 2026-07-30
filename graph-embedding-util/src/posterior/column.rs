//! The column pass: for each embedding dim, sample **every** anchor.
//!
//! # The transpose, and why
//!
//! The obvious layout is one task per anchor with the dim scan inside. On a real
//! dictionary that is tens of thousands of scattered scalar loops — per-anchor tasks,
//! per-dim branching, `f64` accumulators over a few hundred elements each. The work is
//! right and the shape is wrong.
//!
//! Given the frozen other side, anchors are conditionally independent, so a single
//! dim's conditional **factorizes over anchors**. That licenses the transpose: hold a
//! block of anchors' slate scores as one `[b × k]` array and advance the dim loop by a
//! rank-1 update, `S ← S + x ⊗ c_d`. The flop count is unchanged; what changes is that
//! the inner loop becomes contiguous, branch-free array work.
//!
//! Dims stay **sequential**. `S` carries `Σ_{k≠d} z_k·β_k`, with `k < d` already
//! updated this sweep, so the scan is a valid Gibbs order and cannot be reordered.
//!
//! # Independent moves, batched evaluation
//!
//! Each anchor gets its own 1-D elliptical-slice transition with its own angle,
//! bracket and acceptance — see [`elliptical_slice_batch`]. Only the *evaluation* is
//! shared. A joint move over the column would need every anchor's log-likelihood to
//! clear one threshold, and its acceptance region shrinks exponentially in the column
//! width, so at tens of thousands of anchors it would never accept.
//!
//! # Tiling
//!
//! An untiled `S` over 30k anchors and a 1024-wide slate is 124 MB, re-read `h` times
//! a sweep — bandwidth-bound for no reason. [`tile_size`] instead picks `b` so `b·k·4`
//! stays near [`TILE_BYTES`], and the whole dim loop runs on one tile before moving on.
//! `S` is then computed once per tile and never leaves cache, and rayon parallelizes
//! over tiles: dense inner loops *and* coarse-grained parallelism, where the per-anchor
//! layout has the parallelism without the density and an untiled column has the density
//! without the locality.
//!
//! Tiling is for **locality, not mixing**. `b` independent moves are the exact
//! conditional; there is no cross-anchor dependence a wider tile could capture. The one
//! thing that would create such dependence is a negative slate drawn on the anchors'
//! own end — every anchor would then appear in every other anchor's slate — which is
//! why the trainer's negative scheme cannot be transplanted onto this side.
//!
//! Results must not depend on `b`. Every anchor's randomness comes from [`gene_rng`]
//! keyed by its GLOBAL index, never from a shared stream, so a tile boundary cannot
//! shift a draw.

use super::hyper::{gene_rng, sigmoid};
use super::lnpdf::{AnchorMoment, FrozenSide, NodeTerm};
use super::score::{AnchorScore, ColumnCtx, SlateSlab};
use crate::cell_projection::SCORE_CLAMP;
use mcmc_util::engine::elliptical_slice_batch;
use rand::rngs::SmallRng;
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal};

/// Target working-set size for one tile's score block, so the `h` passes over it are
/// cache hits rather than DRAM traffic. Not a user knob: it describes the machine, not
/// the model.
pub(super) const TILE_BYTES: usize = 2 << 20;

/// Anchors per tile, given the slate width and how many likelihood terms each anchor
/// carries — gem's spliced and unspliced tracks are two, and each needs its own block.
///
/// Clamped below so a very wide slate still amortizes per-tile setup, and above so a
/// narrow one does not build a tile too large to stay resident.
#[must_use]
pub(super) fn tile_size(k: usize, n_terms: usize) -> usize {
    let per_anchor = 4 * k.max(1) * n_terms.max(1);
    (TILE_BYTES / per_anchor.max(1)).clamp(32, 4096)
}

/// What one tile's pass produces.
pub(super) struct TileDraw {
    /// `[b × h]` slab values. This is `β`, not `z ⊙ β`: the caller's reduction needs
    /// the latent value to form the slab variance.
    pub beta: Vec<f32>,
    /// `[b × h]` inclusion indicators after this sweep.
    pub z: Vec<bool>,
    /// Bracket fallbacks summed over dims — the batched kernel's analogue of a
    /// rejected move. Reported, never swallowed.
    pub fallbacks: usize,
    /// Likelihood-evaluation rounds summed over dims, i.e. the tile's actual cost.
    pub rounds: usize,
    /// Slice transitions ATTEMPTED — the honest denominator for [`Self::fallbacks`].
    ///
    /// Counts only the coordinates that actually ran a slice move, i.e. those included on
    /// entry. A coordinate that is off has a flat likelihood in `β`, so it takes a prior
    /// draw with no bracket and cannot fall back; counting it would inflate the
    /// denominator by `1/(1−π₀)` and make a badly stalled run read as healthy — at the
    /// measured `π₀ ≈ 0.885` that is a factor of nearly nine.
    pub transitions: usize,
    /// `[b × n_terms]` profiled intercepts at the end of the sweep, anchor-major.
    ///
    /// One per TERM, not per anchor: gem's spliced and unspliced rows are separate
    /// observations with independent biases, and sharing one would force the static
    /// level onto a rank-`h` bilinear. Read [`TermTile::profiled_bias`] for the form.
    pub b_profiled: Vec<f32>,
}

/// Everything a tile pass needs that is constant across tiles.
pub(super) struct ColumnArgs<'a> {
    pub side: &'a FrozenSide<'a>,
    /// `[h]` slab standard deviations for this sweep.
    pub sd: &'a [f32],
    /// `[h]` `ln((1−π₀)/π₀)`, the inclusion prior's log-odds.
    pub log_prior_odds: &'a [f64],
    /// `[n_anchors × h]` mask of coordinates the model allows to be included, in
    /// GLOBAL anchor indexing — gem's nesting `z_δ ⊆ z_β` arrives here.
    pub veto: Option<&'a [bool]>,
    /// When false, every coordinate stays included and no `z` is drawn.
    pub selection: bool,
    /// Slice transitions per `(anchor, dim)` per sweep. The column pass already gives
    /// each anchor `h` transitions a sweep, so `1` here is not the same `1` a
    /// whole-anchor sampler would mean.
    pub transitions: usize,
    pub seed: u64,
    pub sweep: usize,
    pub h: usize,
}

/// One term's per-tile state. A term is one likelihood factor of an anchor; gem's
/// velocity gate sums a spliced and an unspliced term, each with its own slate,
/// biases and frozen offset.
struct TermTile<'a> {
    /// `[b × k]` slate scores with dim `d`'s SAMPLED contribution peeled out. A frozen
    /// offset stays folded in — it is not being sampled.
    s: Vec<f32>,
    /// `[b]` data term, peeled, including the `Σ_pos n·b_o` the moment collapse omits.
    data: Vec<f64>,
    /// `[b]` `Σ_{k≠d} (v + off)²`, the radius guard's other-coordinate part.
    sumsq: Vec<f32>,
    m: Vec<f32>,
    total: Vec<f64>,
    safe_radius: Vec<f32>,
    /// `[b × h]` frozen offset, or `None` in the plain case.
    offset: Option<Vec<f32>>,
    /// `[b]` observed edges, BORROWED from the caller's nodes — see `ColumnCtx::pos`.
    pos: Vec<&'a [(u32, f32)]>,
    slab: &'a SlateSlab,
    partition: &'a [u32],
    partition_scale: f64,
    k: usize,
}

impl<'a> TermTile<'a> {
    /// Build term `t`'s block for this tile from the anchors' current effective
    /// loading. One `O(b·k·h)` pass, once per tile per sweep; every dim afterwards is
    /// `O(b·k)`.
    fn seed(
        nodes: &[Vec<NodeTerm<'a>>],
        t: usize,
        slab: &'a SlateSlab,
        side: &FrozenSide<'_>,
        v: &[f32],
        h: usize,
    ) -> Self {
        let b = nodes.len();
        let k = slab.k();
        let first = &nodes[0][t];
        let partition = first.partition;
        let partition_scale = first.partition_scale;

        let has_offset = nodes.iter().any(|terms| terms[t].offset.is_some());
        let offset = has_offset.then(|| {
            let mut o = vec![0.0f32; b * h];
            for (i, terms) in nodes.iter().enumerate() {
                if let Some(src) = terms[t].offset {
                    o[i * h..(i + 1) * h].copy_from_slice(src);
                }
            }
            o
        });
        let off_at = |i: usize, d: usize| offset.as_ref().map_or(0.0f32, |o| o[i * h + d]);

        let pos: Vec<&'a [(u32, f32)]> = nodes.iter().map(|terms| terms[t].pos).collect();

        // Moment statistics. Rebuilt per sweep because the frozen side moves per sweep,
        // which is also why the caller cannot hand these in.
        let mut m = vec![0.0f32; b * h];
        let mut total = vec![0.0f64; b];
        let mut bias_dot = vec![0.0f64; b];
        let mut safe_radius = vec![0.0f32; b];
        for (i, p) in pos.iter().enumerate() {
            let mom = AnchorMoment::new(p, side);
            m[i * h..(i + 1) * h].copy_from_slice(&mom.m);
            total[i] = mom.total;
            bias_dot[i] = mom.bias_dot;
            safe_radius[i] = mom.safe_radius;
        }

        // s[i,j] = b_j + Σ_d (v_id + off_id)·col[d*k + j]
        let mut s = vec![0.0f32; b * k];
        for i in 0..b {
            let row = &mut s[i * k..(i + 1) * k];
            row.copy_from_slice(slab.biases());
            for d in 0..h {
                let w = v[i * h + d] + off_at(i, d);
                if w == 0.0 {
                    continue;
                }
                for (sj, cj) in row.iter_mut().zip(slab.dim(d)) {
                    *sj += w * cj;
                }
            }
        }

        // data[i] = Σ_pos n·b_o + Σ_d (v_id + off_id)·m_id ; sumsq[i] = Σ_d (·)²
        // `bias_dot` has no other reader, so it IS the data term's starting value.
        let mut data = bias_dot;
        let mut sumsq = vec![0.0f32; b];
        for i in 0..b {
            for d in 0..h {
                let w = v[i * h + d] + off_at(i, d);
                data[i] += f64::from(w) * f64::from(m[i * h + d]);
                sumsq[i] += w * w;
            }
        }

        Self {
            s,
            data,
            sumsq,
            m,
            total,
            safe_radius,
            offset,
            pos,
            slab,
            partition,
            partition_scale,
            k,
        }
    }

    /// Remove dim `d`'s SAMPLED contribution. The frozen offset stays, so the peeled
    /// state still describes "this coordinate contributes only what is not being
    /// sampled" — which is exactly what an evaluation at `x` then adds `x` to.
    fn peel(&mut self, v: &[f32], d: usize, h: usize) {
        let k = self.k;
        let col = self.slab.dim(d);
        for i in 0..self.total.len() {
            let w = v[i * h + d];
            let off = self.offset.as_ref().map_or(0.0f32, |o| o[i * h + d]);
            // `sumsq` tracks the OFFSET-INCLUDED coordinate, so remove the full square.
            self.sumsq[i] -= (w + off) * (w + off);
            if w == 0.0 {
                continue;
            }
            self.data[i] -= f64::from(w) * f64::from(self.m[i * h + d]);
            for (sj, cj) in self.s[i * k..(i + 1) * k].iter_mut().zip(col) {
                *sj -= w * cj;
            }
        }
    }

    /// Fold dim `d`'s new effective value back in, so the next dim conditions on the
    /// state this one just produced.
    fn restore(&mut self, v: &[f32], d: usize, h: usize) {
        let k = self.k;
        let col = self.slab.dim(d);
        for i in 0..self.total.len() {
            let w = v[i * h + d];
            let off = self.offset.as_ref().map_or(0.0f32, |o| o[i * h + d]);
            self.sumsq[i] += (w + off) * (w + off);
            if w == 0.0 {
                continue;
            }
            self.data[i] += f64::from(w) * f64::from(self.m[i * h + d]);
            for (sj, cj) in self.s[i * k..(i + 1) * k].iter_mut().zip(col) {
                *sj += w * cj;
            }
        }
    }

    /// Anchor `i`'s profiled intercept, `b* = ln(T_i) − ln(scale · Σ_slate exp(s_j))`.
    ///
    /// The closed-form maximizer of the Poisson likelihood in `b`, which is what makes
    /// the sampled likelihood free of an intercept at all. Cheap here and only here:
    /// after the dim loop `s` already holds the final scores, so this is `O(k)` rather
    /// than the `O(k·h)` a fresh evaluation would cost. Scores are deliberately NOT
    /// clamped, matching the estimand the profile was derived from.
    fn profiled_bias(&self, i: usize) -> f32 {
        if self.total[i] <= 0.0 {
            // No counts ⇒ no rate to match. Park it at the floor rather than emit a
            // `ln(0)`, so a silent anchor cannot contribute to another side's scores.
            return -(SCORE_CLAMP as f32);
        }
        let s = &self.s[i * self.k..(i + 1) * self.k];
        let hi = f64::from(s.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        if !hi.is_finite() {
            return 0.0;
        }
        let acc: f64 = s.iter().map(|&v| (f64::from(v) - hi).exp()).sum();
        let log_part = hi + (self.partition_scale * acc).max(f64::MIN_POSITIVE).ln();
        (self.total[i].ln() - log_part).clamp(-SCORE_CLAMP, SCORE_CLAMP) as f32
    }

    /// Borrow this term's state as a [`ColumnCtx`]. `v` is shared across terms — it is
    /// the anchor's loading, not a per-term quantity — so it is passed in.
    fn ctx<'c>(&'c self, v: &'c [f32], side: &'c FrozenSide<'c>, h: usize) -> ColumnCtx<'c> {
        ColumnCtx {
            s: &self.s,
            v,
            data: &self.data,
            sumsq: &self.sumsq,
            m: &self.m,
            total: &self.total,
            safe_radius: &self.safe_radius,
            offset: self.offset.as_deref(),
            pos: &self.pos,
            partition: self.partition,
            partition_scale: self.partition_scale,
            side,
            k: self.k,
            h,
        }
    }
}

/// Sample one tile: every dim, every anchor in `lo .. lo + nodes.len()`.
///
/// `slabs` are the per-term transposed slates, shared read-only across tiles.
/// `beta_in` / `z_in` are the tile's slice of the chain's current state. `lo` is
/// needed to index [`ColumnArgs::veto`] and to key each anchor's RNG, both global.
pub(super) fn sample_tile<S: AnchorScore>(
    score: &S,
    args: &ColumnArgs<'_>,
    nodes: &[Vec<NodeTerm<'_>>],
    slabs: &[SlateSlab],
    lo: usize,
    beta_in: &[f32],
    z_in: &[bool],
) -> TileDraw {
    let h = args.h;
    let b = nodes.len();
    let mut beta = beta_in.to_vec();
    let mut z = z_in.to_vec();
    let mut fallbacks = 0usize;
    let mut rounds = 0usize;
    let mut transitions = 0usize;
    if b == 0 {
        return TileDraw {
            beta,
            z,
            fallbacks,
            rounds,
            transitions,
            b_profiled: Vec::new(),
        };
    }
    let n_terms = nodes[0].len();

    // Per-anchor streams, keyed globally so a tile boundary cannot shift a draw.
    let mut rngs: Vec<SmallRng> = (0..b)
        .map(|i| gene_rng(args.seed, lo + i, args.sweep))
        .collect();

    // `v` is the EFFECTIVE loading `z ⊙ β`, carried alongside `beta` because the
    // likelihood sees the gated value while the slab variance needs the latent one.
    let mut v: Vec<f32> = (0..b * h)
        .map(|idx| if z[idx] { beta[idx] } else { 0.0 })
        .collect();

    let mut terms: Vec<TermTile> = (0..n_terms)
        .map(|t| TermTile::seed(nodes, t, &slabs[t], args.side, &v, h))
        .collect();

    // Scratch reused across dims, so the dim loop allocates nothing.
    let mut ll_off = vec![0.0f32; b];
    let mut ll_on = vec![0.0f32; b];
    let mut nu = vec![0.0f32; b];
    let mut scratch = vec![0.0f32; b];
    let mut inner = vec![0.0f32; b];
    let zeros = vec![0.0f32; b];
    let all: Vec<u32> = (0..b as u32).collect();
    // Hoisted out of the dim loop. Each of these was an allocation per (tile, dim) —
    // roughly `h` per tile per sweep each — and `sub` was the expensive one: a fresh
    // `Vec<SmallRng>` of up to `b` entries, 16 KB at `b = 512`, which also evicts the
    // tile's score block from the cache it was tiled to stay in.
    let mut on: Vec<u32> = Vec::with_capacity(b);
    let mut off: Vec<u32> = Vec::with_capacity(b);
    let mut xs: Vec<f32> = Vec::with_capacity(b);
    let mut out_buf: Vec<f32> = Vec::with_capacity(b);
    let mut cur: Vec<f32> = Vec::with_capacity(b);
    let mut start: Vec<f32> = Vec::with_capacity(b);
    let mut draw: Vec<f32> = Vec::with_capacity(b);
    let mut idx_buf: Vec<u32> = Vec::with_capacity(b);
    let mut sub: Vec<SmallRng> = (0..b)
        .map(|i| gene_rng(args.seed, lo + i, args.sweep))
        .collect();

    for d in 0..h {
        // ---- peel dim `d` out of every term ----
        for term in &mut terms {
            term.peel(&v, d, h);
        }
        for i in 0..b {
            v[i * h + d] = 0.0;
        }

        // ---- ll with dim `d` contributing nothing ----
        // Needed for the `z` draw, and it is where an excluded coordinate starts, so
        // it is not extra work.
        eval(
            score,
            args,
            &terms,
            &v,
            d,
            &zeros[..b],
            &all,
            &mut ll_off,
            &mut scratch,
        );

        // ---- β | z ----
        let sd_d = args.sd[d];
        for i in 0..b {
            let g: f64 = StandardNormal.sample(&mut rngs[i]);
            nu[i] = g as f32 * sd_d;
        }

        // Split by entry state so both arms stay batched. A coordinate that is ON gets
        // fit against the likelihood; one that is OFF has a flat likelihood in `β`, so
        // its exact conditional IS the prior. Drawing it from the prior rather than
        // keeping the stale fit is what makes the `z` step below a valid Gibbs move,
        // and why no Occam correction is needed: a fitted β always beats 0, so
        // comparing against one would bias every coordinate on.
        on.clear();
        off.clear();
        for i in 0..b {
            if z[i * h + d] {
                on.push(i as u32);
            } else {
                off.push(i as u32);
            }
        }

        if !off.is_empty() {
            xs.clear();
            xs.extend(off.iter().map(|&i| nu[i as usize]));
            out_buf.clear();
            out_buf.resize(off.len(), 0.0);
            eval(
                score,
                args,
                &terms,
                &v,
                d,
                &xs,
                &off,
                &mut out_buf,
                &mut scratch,
            );
            for (slot, &i) in off.iter().enumerate() {
                let i = i as usize;
                beta[i * h + d] = xs[slot];
                ll_on[i] = out_buf[slot];
            }
        }

        if !on.is_empty() {
            cur.clear();
            cur.extend(on.iter().map(|&i| beta[i as usize * h + d]));
            start.clear();
            start.resize(on.len(), 0.0);
            eval(
                score,
                args,
                &terms,
                &v,
                d,
                &cur,
                &on,
                &mut start,
                &mut scratch,
            );

            // The kernel needs one stream per ITEM, and a scattered mutable subset of
            // `rngs` is not expressible — so derive a fresh stream per (anchor, dim).
            // Keyed on the GLOBAL anchor, so it stays tile-invariant, and salted by dim
            // so no two dims share a stream.
            // Re-keyed in place rather than re-collected: the reseeding is required for
            // reproducibility, the reallocation is not.
            sub.truncate(on.len());
            while sub.len() < on.len() {
                sub.push(gene_rng(args.seed, lo, args.sweep));
            }
            for (slot, &i) in on.iter().enumerate() {
                sub[slot] = gene_rng(
                    args.seed ^ ((d as u64 + 1) << 32),
                    lo + i as usize,
                    args.sweep,
                );
            }

            for t in 0..args.transitions.max(1) {
                // Each transition needs its own prior draw; the first reuses the `nu`
                // already drawn above so `transitions = 1` consumes exactly one.
                draw.clear();
                if t == 0 {
                    draw.extend(on.iter().map(|&i| nu[i as usize]));
                } else {
                    for &i in &on {
                        let g: f64 = StandardNormal.sample(&mut rngs[i as usize]);
                        draw.push(g as f32 * sd_d);
                    }
                }
                let step = elliptical_slice_batch(
                    &cur,
                    &draw,
                    &start,
                    &mut sub[..on.len()],
                    &mut |x, active, out| {
                        // `active` indexes into `on`; map back to tile-local anchors.
                        idx_buf.clear();
                        idx_buf.extend(active.iter().map(|&a| on[a as usize]));
                        eval(score, args, &terms, &v, d, x, &idx_buf, out, &mut inner);
                    },
                );
                fallbacks += step.fallbacks;
                rounds += step.rounds;
                transitions += on.len();
                cur.copy_from_slice(&step.value);
                start.copy_from_slice(&step.lnpdf);
            }
            for (slot, &i) in on.iter().enumerate() {
                let i = i as usize;
                beta[i * h + d] = cur[slot];
                ll_on[i] = start[slot];
            }
        }

        // ---- z | β: exact Gibbs, `p(z|β,y) ∝ p(y|z·β)·p(z)` ----
        for i in 0..b {
            let idx = i * h + d;
            z[idx] = if args.selection {
                match args.veto {
                    // Vetoed coordinates are excluded BY THE MODEL, not merely
                    // disfavoured by the data, so they are pinned rather than drawn.
                    Some(mask) if !mask[(lo + i) * h + d] => false,
                    _ => {
                        let logit =
                            args.log_prior_odds[d] + f64::from(ll_on[i]) - f64::from(ll_off[i]);
                        rngs[i].random::<f64>() < sigmoid(logit)
                    }
                }
            } else {
                true
            };
            v[idx] = if z[idx] { beta[idx] } else { 0.0 };
        }

        // ---- fold the new effective values back in ----
        for term in &mut terms {
            term.restore(&v, d, h);
        }
    }

    // The intercepts, from the state the sweep ended in. `terms[t].s` already holds
    // the final scores, so this is one extra pass over the slate rather than a fresh
    // `O(k·h)` evaluation per anchor.
    let mut b_profiled = vec![0.0f32; b * n_terms];
    for (t, term) in terms.iter().enumerate() {
        for i in 0..b {
            b_profiled[i * n_terms + t] = term.profiled_bias(i);
        }
    }

    TileDraw {
        beta,
        z,
        fallbacks,
        rounds,
        transitions,
        b_profiled,
    }
}

/// `out[slot] = Σ_terms ll` for anchor `active[slot]` with dim `d` set to `x[slot]`.
///
/// Summing over terms is what makes gem's two splice tracks one likelihood. Legitimate
/// because the tracks are separate rows with independent biases, so each intercept
/// profiles out on its own.
///
/// Takes its inputs positionally rather than through a context struct: `terms` is borrowed
/// mutably by the peel/restore either side of every call, so a struct holding it across
/// the dim loop would not borrow-check, and one built per call would be pure ceremony.
#[allow(clippy::too_many_arguments)]
fn eval<S: AnchorScore>(
    score: &S,
    args: &ColumnArgs<'_>,
    terms: &[TermTile<'_>],
    v: &[f32],
    d: usize,
    x: &[f32],
    active: &[u32],
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let n = active.len();
    out[..n].fill(0.0);
    for term in terms {
        let ctx = term.ctx(v, args.side, args.h);
        let part = &mut scratch[..n];
        score.ll_column(&ctx, d, term.slab.dim(d), &x[..n], active, part);
        for (o, p) in out[..n].iter_mut().zip(part.iter()) {
            *o += *p;
        }
    }
}

#[cfg(test)]
#[path = "column_tests.rs"]
mod column_tests;
