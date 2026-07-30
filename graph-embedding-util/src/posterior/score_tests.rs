//! The column score must be interchangeable with the scalar oracle.
//!
//! [`multinomial_ll`] is the audited path and stays in the tree for exactly this
//! reason: every claim about the batched form is checked against it POINTWISE, not
//! through an aggregate recovery number. Without that the dense path would be
//! unfalsifiable — a plausible-looking `pip` table is not evidence.
//!
//! The context here is built by NAIVE loops straight from the field definitions,
//! deliberately not through whatever helper the sampler uses. A shared builder
//! would let one bug satisfy both sides.

use super::*;

const H: usize = 6;
const K: usize = 32;
const B: usize = 5;

/// A frozen side with small, non-degenerate rows — nowhere near [`SCORE_CLAMP`], so
/// the fast path is what gets exercised unless a test deliberately leaves it.
fn side_buffers() -> (Vec<f32>, Vec<f32>) {
    let mut e = vec![0.0f32; K * H];
    for o in 0..K {
        for d in 0..H {
            e[o * H + d] = ((o * (d + 3) + 2 * d) % 11) as f32 * 0.05 - 0.25;
        }
    }
    let b: Vec<f32> = (0..K).map(|o| (o % 7) as f32 * 0.03 - 0.1).collect();
    (e, b)
}

/// Per-anchor edge lists. Anchor 3 is deliberately EMPTY so the flat-likelihood
/// branch is covered; over half the anchors on a real annotation have no counts.
fn edges() -> Vec<Vec<(u32, f32)>> {
    (0..B)
        .map(|i| {
            if i == 3 {
                return Vec::new();
            }
            (0..K as u32)
                .filter(|o| (*o as usize + i).is_multiple_of(4))
                .map(|o| (o, 1.0 + ((o as usize + i) % 5) as f32))
                .collect()
        })
        .collect()
}

/// `[B × H]` loadings. `z` is folded in by the caller, so this is already `z ⊙ β`.
fn loadings(scale: f32) -> Vec<f32> {
    (0..B * H)
        .map(|i| scale * (((i * 7) % 13) as f32 * 0.02 - 0.12))
        .collect()
}

/// Everything `ColumnCtx` needs, computed from the definitions by hand.
struct Naive {
    s: Vec<f32>,
    v: Vec<f32>,
    data: Vec<f64>,
    sumsq: Vec<f32>,
    m: Vec<f32>,
    total: Vec<f64>,
    safe_radius: Vec<f32>,
}

/// Build the peeled state for dim `d` directly from its definition.
///
/// `beta` is the effective loading `z ⊙ β`; `offset` is frozen and therefore NOT
/// peeled, so its dim-`d` component stays folded into every derived quantity.
fn naive(d: usize, beta: &[f32], offset: Option<&[f32]>, side: &FrozenSide) -> Naive {
    let off = |i: usize, k: usize| offset.map_or(0.0f32, |o| o[i * H + k]);

    // `v` is the sampled loading with dim `d` zeroed — the sampler is about to
    // replace that coordinate, so it must not be in the peeled state.
    let mut v = beta.to_vec();
    for i in 0..B {
        v[i * H + d] = 0.0;
    }

    // Moments and their radii, one per anchor.
    let pos = edges();
    let mut m = vec![0.0f32; B * H];
    let mut total = vec![0.0f64; B];
    let mut bias_dot = vec![0.0f64; B];
    let mut safe_radius = vec![0.0f32; B];
    for (i, p) in pos.iter().enumerate() {
        let mut max_row = 0.0f32;
        let mut max_b = 0.0f32;
        for &(o, n) in p {
            let row = &side.e[o as usize * H..(o as usize + 1) * H];
            let mut nrm2 = 0.0f32;
            for (k, val) in row.iter().enumerate() {
                m[i * H + k] += n * val;
                nrm2 += val * val;
            }
            max_row = max_row.max(nrm2.sqrt());
            max_b = max_b.max(side.b[o as usize].abs());
            total[i] += f64::from(n);
            bias_dot[i] += f64::from(n) * f64::from(side.b[o as usize]);
        }
        let headroom = SCORE_CLAMP as f32 - max_b;
        safe_radius[i] = if max_row > 0.0 && headroom > 0.0 {
            headroom / max_row
        } else {
            0.0
        };
    }

    // s[i,j] = Σ_k (v_ik + off_ik)·e_j[k] + b_j, with v_id = 0.
    let mut s = vec![0.0f32; B * K];
    for i in 0..B {
        for j in 0..K {
            let mut acc = side.b[j];
            for k in 0..H {
                acc += (v[i * H + k] + off(i, k)) * side.e[j * H + k];
            }
            s[i * K + j] = acc;
        }
    }

    // data[i] = Σ_k (v_ik + off_ik)·m_i[k] + Σ_pos n·b_o, the second term being what
    // the `⟨v, m⟩` collapse leaves out; sumsq[i] = Σ_{k≠d} (v_ik + off_ik)².
    let mut data = bias_dot;
    let mut sumsq = vec![0.0f32; B];
    for i in 0..B {
        for k in 0..H {
            let w = v[i * H + k] + off(i, k);
            data[i] += f64::from(w) * f64::from(m[i * H + k]);
            if k != d {
                sumsq[i] += w * w;
            }
        }
    }

    Naive {
        s,
        v,
        data,
        sumsq,
        m,
        total,
        safe_radius,
    }
}

/// The scalar oracle at the same point: rebuild the full loading and call
/// [`multinomial_ll`], which is what the sampler used to do one coordinate at a time.
///
/// Deliberately takes every input positionally rather than bundling them: this is the
/// reference the batched path is checked against, so it should share as little structure
/// with the thing under test as possible.
#[allow(clippy::too_many_arguments)]
fn oracle(
    d: usize,
    i: usize,
    x: f32,
    beta: &[f32],
    offset: Option<&[f32]>,
    side: &FrozenSide,
    partition: &[u32],
    scale: f64,
) -> f32 {
    let pos = edges();
    let mut full: Vec<f32> = beta[i * H..(i + 1) * H].to_vec();
    full[d] = x;
    let node = NodeTerm {
        offset: offset.map(|o| &o[i * H..(i + 1) * H]),
        ..NodeTerm::new(&pos[i], partition, scale)
    };
    multinomial_ll(&full, &node, side)
}

/// Drive `ll_column` over every anchor and compare against the oracle, returning the
/// worst relative gap seen.
fn worst_gap(beta: &[f32], offset: Option<&[f32]>, xs: &[f32]) -> f64 {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edges();
    let pos_ref: Vec<&[(u32, f32)]> = pos.iter().map(Vec::as_slice).collect();
    let scale = 1.0f64;
    let active: Vec<u32> = (0..B as u32).collect();

    let mut worst = 0.0f64;
    for d in 0..H {
        let n = naive(d, beta, offset, &side);
        let ctx = ColumnCtx {
            s: &n.s,
            v: &n.v,
            data: &n.data,
            sumsq: &n.sumsq,
            m: &n.m,
            total: &n.total,
            safe_radius: &n.safe_radius,
            offset,
            pos: &pos_ref,
            partition: &partition,
            partition_scale: scale,
            side: &side,
            k: K,
            h: H,
        };
        let c_d: Vec<f32> = (0..K).map(|j| e[j * H + d]).collect();

        for &x in xs {
            let xv = vec![x; B];
            let mut out = vec![0.0f32; B];
            ProfiledPoisson.ll_column(&ctx, d, &c_d, &xv, &active, &mut out);
            for (i, &got) in out.iter().enumerate() {
                let want = oracle(d, i, x, beta, offset, &side, &partition, scale);
                let gap = f64::from(got - want).abs() / f64::from(want).abs().max(1.0);
                worst = worst.max(gap);
            }
        }
    }
    worst
}

/// THE test. Batched column evaluation must agree with the scalar oracle at every
/// anchor, every dim and every candidate value the bracket could propose.
#[test]
fn the_column_score_matches_the_scalar_oracle() {
    let beta = loadings(1.0);
    let xs = [0.0f32, 0.05, -0.05, 0.3, -0.3, 0.9];
    let worst = worst_gap(&beta, None, &xs);
    assert!(
        worst < 1e-3,
        "column vs scalar disagree by {worst:.3e} — the batched path is not the same \
         estimand as `multinomial_ll`"
    );
}

/// A frozen offset must be folded in identically on both paths. gem samples `δ_g`
/// with `β_g` carried here, so a column form that dropped it would silently sample
/// the wrong conditional — and every shape check would still pass.
#[test]
fn the_column_score_honours_a_frozen_offset() {
    let beta = loadings(1.0);
    let offset: Vec<f32> = (0..B * H)
        .map(|i| ((i * 5) % 9) as f32 * 0.03 - 0.12)
        .collect();
    let xs = [0.0f32, 0.2, -0.4];
    let worst = worst_gap(&beta, Some(&offset), &xs);
    assert!(worst < 1e-3, "offset handling diverged by {worst:.3e}");

    // Sanity: the offset actually moves the answer, so the check above is not
    // comparing two copies of "offset ignored".
    let without = worst_gap(&beta, None, &xs);
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let a = oracle(2, 0, 0.2, &beta, Some(&offset), &side, &partition, 1.0);
    let c = oracle(2, 0, 0.2, &beta, None, &side, &partition, 1.0);
    assert!(
        (a - c).abs() > 1e-3,
        "fixture offset is inert ({a} vs {c}), so the agreement above proves nothing"
    );
    assert!(without < 1e-3, "no-offset arm regressed: {without:.3e}");
}

/// REGRESSION SHAPE: outside the safe radius the collapsed data term is not the
/// clamped one, so the score must defer to the walk. There the two forms are the
/// SAME CALL, so they must agree exactly, not approximately — a gap here means the
/// guard did not fire and the fast path was used where it is invalid.
#[test]
fn past_the_safe_radius_it_defers_to_the_walk_exactly() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edges();
    let pos_ref: Vec<&[(u32, f32)]> = pos.iter().map(Vec::as_slice).collect();
    let active: Vec<u32> = (0..B as u32).collect();

    // Blow the loading far past any plausible radius.
    let beta = loadings(4000.0);
    let d = 1usize;
    let n = naive(d, &beta, None, &side);
    assert!(
        n.safe_radius.iter().any(|&r| r > 0.0),
        "fixture must have a usable safe region for the guard to be meaningful"
    );

    let ctx = ColumnCtx {
        s: &n.s,
        v: &n.v,
        data: &n.data,
        sumsq: &n.sumsq,
        m: &n.m,
        total: &n.total,
        safe_radius: &n.safe_radius,
        offset: None,
        pos: &pos_ref,
        partition: &partition,
        partition_scale: 1.0,
        side: &side,
        k: K,
        h: H,
    };
    let c_d: Vec<f32> = (0..K).map(|j| e[j * H + d]).collect();
    let x = 5000.0f32;
    let xv = vec![x; B];
    let mut out = vec![0.0f32; B];
    ProfiledPoisson.ll_column(&ctx, d, &c_d, &xv, &active, &mut out);

    for (i, &got) in out.iter().enumerate() {
        let want = oracle(d, i, x, &beta, None, &side, &partition, 1.0);
        assert_eq!(
            got, want,
            "anchor {i} outside the radius must be the exact walk, not a collapse"
        );
    }
}

/// An anchor with no counts has a flat likelihood, so every candidate must score the
/// same. Reported as `0` rather than a walk over an empty edge list, which is what
/// makes the `z` draw fall back to the prior for such anchors.
#[test]
fn an_empty_anchor_is_flat() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edges();
    let pos_ref: Vec<&[(u32, f32)]> = pos.iter().map(Vec::as_slice).collect();
    assert!(
        pos[3].is_empty(),
        "anchor 3 is the empty one by construction"
    );

    let beta = loadings(1.0);
    let d = 4usize;
    let n = naive(d, &beta, None, &side);
    let ctx = ColumnCtx {
        s: &n.s,
        v: &n.v,
        data: &n.data,
        sumsq: &n.sumsq,
        m: &n.m,
        total: &n.total,
        safe_radius: &n.safe_radius,
        offset: None,
        pos: &pos_ref,
        partition: &partition,
        partition_scale: 1.0,
        side: &side,
        k: K,
        h: H,
    };
    let c_d: Vec<f32> = (0..K).map(|j| e[j * H + d]).collect();

    let mut seen = Vec::new();
    for x in [0.0f32, 1.0, -7.5] {
        let mut out = vec![0.0f32; 1];
        ProfiledPoisson.ll_column(&ctx, d, &c_d, &[x], &[3u32], &mut out);
        seen.push(out[0]);
    }
    assert!(
        seen.iter().all(|v| *v == 0.0),
        "an empty anchor must be flat, got {seen:?}"
    );
}

/// The batched ESS retires anchors as they accept, so later rounds pass a SUBSET of
/// `active`. Evaluating a subset must give bit-identical values to the full batch —
/// otherwise results would depend on how the active set happened to shrink, which is
/// a reproducibility bug with no visible symptom.
#[test]
fn subsetting_the_active_set_changes_nothing() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edges();
    let pos_ref: Vec<&[(u32, f32)]> = pos.iter().map(Vec::as_slice).collect();
    let beta = loadings(1.0);
    let d = 2usize;
    let n = naive(d, &beta, None, &side);
    let ctx = ColumnCtx {
        s: &n.s,
        v: &n.v,
        data: &n.data,
        sumsq: &n.sumsq,
        m: &n.m,
        total: &n.total,
        safe_radius: &n.safe_radius,
        offset: None,
        pos: &pos_ref,
        partition: &partition,
        partition_scale: 1.0,
        side: &side,
        k: K,
        h: H,
    };
    let c_d: Vec<f32> = (0..K).map(|j| e[j * H + d]).collect();

    let xs: Vec<f32> = (0..B).map(|i| 0.1 * i as f32 - 0.2).collect();
    let all: Vec<u32> = (0..B as u32).collect();
    let mut full_out = vec![0.0f32; B];
    ProfiledPoisson.ll_column(&ctx, d, &c_d, &xs, &all, &mut full_out);

    // A scattered subset, in a different order than the full pass visited them.
    let subset = [4u32, 1, 2];
    let sub_x: Vec<f32> = subset.iter().map(|&i| xs[i as usize]).collect();
    let mut sub_out = vec![0.0f32; subset.len()];
    ProfiledPoisson.ll_column(&ctx, d, &c_d, &sub_x, &subset, &mut sub_out);

    for (slot, &i) in sub_out.iter().zip(&subset) {
        assert_eq!(
            *slot, full_out[i as usize],
            "anchor {i} changed value when evaluated as part of a subset"
        );
    }
}

/// The capability predicates are load-bearing — they decide whether the caller may
/// carry scores by a rank-1 update and collapse the data term. Assert them, so a
/// future edit that breaks the algebra has to notice it is also changing a promise.
#[test]
fn profiled_poisson_declares_both_fast_paths() {
    assert!(
        ProfiledPoisson.affine_in_anchor(),
        "s_j is affine in v, so the rank-1 dim update is licensed"
    );
    assert!(
        ProfiledPoisson.data_term_is_linear(),
        "the data term is linear in the scores, so the moment collapse is licensed"
    );
    assert_eq!(ProfiledPoisson.label(), "profiled-poisson");
}

///////////////////////////////////////////////////////////
// Precision gate for a dense (candle/CUDA) backend       //
///////////////////////////////////////////////////////////

/// The same estimand accumulated entirely in `f32` — what a candle/CUDA backend would be
/// forced into, since candle's element types are `f32` and its reductions accumulate there.
///
/// Written as a second [`AnchorScore`] rather than a flag on the first, which is the point
/// of the seam: a whole alternative numeric path is a ~20-line impl, and the two can be
/// diffed on the same context with no production code touched.
struct F32Poisson;

impl AnchorScore for F32Poisson {
    fn label(&self) -> &'static str {
        "profiled-poisson-f32"
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
        let clamp = SCORE_CLAMP as f32;
        for (slot, (&i, &xi)) in out.iter_mut().zip(active.iter().zip(x)) {
            let i = i as usize;
            let total = ctx.total[i] as f32;
            if total == 0.0 {
                *slot = 0.0;
                continue;
            }
            // No radius fallback: the point is to measure the reduction, and a fixture
            // that left the safe radius would be measuring the walk instead.
            let data = ctx.data[i] as f32 + xi * ctx.moment_at(i, d);
            let mut acc = 0.0f32;
            for (sj, cj) in ctx.scores(i).iter().zip(c_d) {
                let v = (sj + xi * cj).clamp(-clamp, clamp);
                acc += (v - clamp).exp();
            }
            *slot = data - total * (clamp + acc.max(f32::MIN_POSITIVE).ln());
        }
    }
}

/// A wide-slate, high-count context — the two things that amplify a summation error.
/// `k` at the production default of 1024, and counts big enough that `T_a` reaches the
/// thousands, because `T_a` multiplies the log-normalizer's error directly.
#[allow(clippy::type_complexity)]
fn big_fixture(
    hh: usize,
    kk: usize,
    bb: usize,
    count_scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<Vec<(u32, f32)>>, Vec<f32>) {
    let mut e = vec![0.0f32; kk * hh];
    for o in 0..kk {
        for d in 0..hh {
            e[o * hh + d] = ((o * (d + 5) + 3 * d) % 17) as f32 * 0.03 - 0.24;
        }
    }
    let b: Vec<f32> = (0..kk).map(|o| (o % 13) as f32 * 0.02 - 0.12).collect();
    let partition: Vec<u32> = (0..kk as u32).collect();
    let pos: Vec<Vec<(u32, f32)>> = (0..bb)
        .map(|i| {
            (0..kk as u32)
                .filter(|o| (*o as usize + i).is_multiple_of(3))
                .map(|o| (o, count_scale * (1.0 + ((o as usize + i) % 9) as f32)))
                .collect()
        })
        .collect();
    let beta: Vec<f32> = (0..bb * hh)
        .map(|i| ((i * 11) % 19) as f32 * 0.015 - 0.13)
        .collect();
    (e, b, partition, pos, beta)
}

/// THE GATE for a dense backend. `f32` accumulation must not disturb what the sampler
/// actually consumes.
///
/// Two quantities, and the second is the one that matters. The absolute `ll` error scales
/// with `T_a`, so on a high-count anchor it can look alarming — but no consumer reads an
/// absolute `ll`. Both consumers read DIFFERENCES: the slice threshold compares one `ll`
/// against another, and the `z` draw uses `ll_on − ll_off`. Those two evaluations sum the
/// same terms in the same order and differ only in one coordinate, so their errors are
/// highly correlated and largely cancel. This measures both so the cancellation is
/// demonstrated rather than assumed.
///
/// # Result, measured
///
/// ```text
/// counts×1   T_a max   1368   |Δll| 0.0029   |Δlogit| 0.0039   ⇒ ΔPIP 9.8e-4
/// counts×50  T_a max  68400   |Δll| 0.1562   |Δlogit| 0.2500   ⇒ ΔPIP 6.3e-2
/// ```
///
/// **The error is LINEAR IN `T_a`** — 54× for a 50× count increase — because
/// `ll = data − T_a·ln(acc)` multiplies the reduction's rounding by the anchor's total. A
/// pseudobulk's total genuinely is tens of thousands, so at that scale `f32` accumulation
/// moves an inclusion probability by ~6 percentage points. A naive dense port would change
/// the answer, not just the speed.
///
/// That does not kill the idea, it constrains it. Three routes, each measurable with this
/// same harness before any kernel is written: `f64` reductions on device (a non-starter on
/// the consumer cards here, where FP64 runs at 1/64 rate); pairwise or Kahan summation,
/// which trades `O(k·ε)` for `O(log k · ε)` and is what a GPU tree reduction does anyway;
/// or returning the LOGIT from the device rather than two `ll`s, so the `T_a` scaling
/// happens once in `f64` on the host. Note the errors did NOT cancel across the two
/// evaluations here (`Δ(ln acc_on − ln acc_off) ≈ 3.7e-6`, about 30 ulp), so the third
/// route needs the difference formed on-device to help.
///
/// Ignored by default: this is a decision-support measurement, not an invariant of the
/// shipped code, which accumulates in `f64` and is unaffected. Run it deliberately:
///
/// ```text
/// cargo test -p graph-embedding-util --release f32_accumulation -- --ignored --nocapture
/// ```
#[test]
#[ignore = "precision measurement for a dense backend; run with --ignored --nocapture"]
fn f32_accumulation_preserves_the_logit_a_dense_backend_would_need() {
    const HH: usize = 32;
    const KK: usize = 1024;
    const BB: usize = 64;

    for &count_scale in &[1.0f32, 50.0] {
        let (e, b, partition, pos, beta) = big_fixture(HH, KK, BB, count_scale);
        let side = FrozenSide {
            e: &e,
            b: &b,
            h: HH,
        };
        let pos_ref: Vec<&[(u32, f32)]> = pos.iter().map(Vec::as_slice).collect();
        let active: Vec<u32> = (0..BB as u32).collect();

        // Build the peeled state for dim 0 from the definitions, as elsewhere in this file.
        let d = 0usize;
        let mut v = beta.clone();
        for i in 0..BB {
            v[i * HH + d] = 0.0;
        }
        let (mut m, mut total, mut data, mut sumsq, radius) = (
            vec![0.0f32; BB * HH],
            vec![0.0f64; BB],
            vec![0.0f64; BB],
            vec![0.0f32; BB],
            vec![f32::INFINITY; BB],
        );
        for (i, p) in pos.iter().enumerate() {
            for &(o, n) in p {
                let row = &e[o as usize * HH..(o as usize + 1) * HH];
                for (k, val) in row.iter().enumerate() {
                    m[i * HH + k] += n * val;
                }
                total[i] += f64::from(n);
                data[i] += f64::from(n) * f64::from(b[o as usize]);
            }
        }
        let mut s = vec![0.0f32; BB * KK];
        for i in 0..BB {
            for j in 0..KK {
                let mut acc = b[j];
                for k in 0..HH {
                    acc += v[i * HH + k] * e[j * HH + k];
                }
                s[i * KK + j] = acc;
            }
            for k in 0..HH {
                let w = v[i * HH + k];
                data[i] += f64::from(w) * f64::from(m[i * HH + k]);
                if k != d {
                    sumsq[i] += w * w;
                }
            }
        }
        let ctx = ColumnCtx {
            s: &s,
            v: &v,
            data: &data,
            sumsq: &sumsq,
            m: &m,
            total: &total,
            safe_radius: &radius,
            offset: None,
            pos: &pos_ref,
            partition: &partition,
            partition_scale: 1.0,
            side: &side,
            k: KK,
            h: HH,
        };
        let c_d: Vec<f32> = (0..KK).map(|j| e[j * HH + d]).collect();

        // `ll_off` at x = 0 and `ll_on` at a plausible slab draw, in both precisions.
        let zeros = vec![0.0f32; BB];
        let xs: Vec<f32> = (0..BB).map(|i| 0.05 + (i % 7) as f32 * 0.02).collect();
        let mut off64 = vec![0.0f32; BB];
        let mut on64 = vec![0.0f32; BB];
        let mut off32 = vec![0.0f32; BB];
        let mut on32 = vec![0.0f32; BB];
        ProfiledPoisson.ll_column(&ctx, d, &c_d, &zeros, &active, &mut off64);
        ProfiledPoisson.ll_column(&ctx, d, &c_d, &xs, &active, &mut on64);
        F32Poisson.ll_column(&ctx, d, &c_d, &zeros, &active, &mut off32);
        F32Poisson.ll_column(&ctx, d, &c_d, &xs, &active, &mut on32);

        let mut worst_ll = 0.0f64;
        let mut worst_logit = 0.0f64;
        let mut typical_logit = 0.0f64;
        for i in 0..BB {
            worst_ll = worst_ll.max(f64::from(on64[i] - on32[i]).abs());
            let l64 = f64::from(on64[i]) - f64::from(off64[i]);
            let l32 = f64::from(on32[i]) - f64::from(off32[i]);
            worst_logit = worst_logit.max((l64 - l32).abs());
            typical_logit += l64.abs() / BB as f64;
        }
        let t_max = total.iter().copied().fold(0.0f64, f64::max);
        // What the error COSTS, in the unit the output is reported in. The logit is
        // consumed through `sigmoid`, whose derivative is at most 1/4, so a logit
        // perturbation of `δ` moves the inclusion probability by at most `δ/4`. That —
        // not the logit's relative error — is the number to judge, because a PIP is what
        // reaches a reader and it is reported to about three digits.
        let pip_error = worst_logit / 4.0;
        println!(
            "f32 gate  counts×{count_scale:<4}  T_a max {t_max:>9.0}  \
             worst |Δll| {worst_ll:>10.4}  worst |Δlogit| {worst_logit:>9.5}  \
             mean |logit| {typical_logit:>9.4}  ⇒ worst ΔPIP {pip_error:.2e}"
        );

        // Deliberately NOT asserting a bound on `pip_error`. The high-count arm exceeds
        // any bound worth setting — that IS the finding — so an assertion here would just
        // be a permanently red test restating the doc comment. What is asserted is that
        // the measurement actually ran: a silent zero would look like a clean pass.
        assert!(
            worst_ll.is_finite() && worst_logit.is_finite() && typical_logit > 0.0,
            "the fixture produced no usable signal, so these numbers mean nothing"
        );
    }
}
