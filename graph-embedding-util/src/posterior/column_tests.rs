//! Structural checks on the column pass.
//!
//! The claim that the batched evaluation IS the scalar estimand lives in
//! `score_tests.rs`, asserted pointwise against `multinomial_ll`. What is checked here
//! is everything the tiling and the peel/restore bookkeeping could break without
//! changing any shape: drift in the carried state, and dependence on a tile boundary.

use super::*;
use crate::posterior::score::ProfiledPoisson;

const H: usize = 5;
const K: usize = 24;
const B: usize = 12;

fn side_buffers() -> (Vec<f32>, Vec<f32>) {
    let mut e = vec![0.0f32; K * H];
    for o in 0..K {
        for d in 0..H {
            e[o * H + d] = ((o * (d + 2) + 3 * d) % 13) as f32 * 0.04 - 0.24;
        }
    }
    let b: Vec<f32> = (0..K).map(|o| (o % 5) as f32 * 0.04 - 0.08).collect();
    (e, b)
}

/// Anchor 7 is deliberately EMPTY — over half the anchors on a real annotation have no
/// counts, and their conditional is the prior.
fn edge_lists() -> Vec<Vec<(u32, f32)>> {
    (0..B)
        .map(|i| {
            if i == 7 {
                return Vec::new();
            }
            (0..K as u32)
                .filter(|o| (*o as usize + i).is_multiple_of(3))
                .map(|o| (o, 1.0 + ((o as usize * 2 + i) % 4) as f32))
                .collect()
        })
        .collect()
}

/// Build `nodes` for a single-term model over `lo..hi` of the fixture.
fn nodes_for<'a>(
    pos: &'a [Vec<(u32, f32)>],
    partition: &'a [u32],
    lo: usize,
    hi: usize,
) -> Vec<Vec<NodeTerm<'a>>> {
    (lo..hi)
        .map(|i| vec![NodeTerm::new(&pos[i], partition, 1.0)])
        .collect()
}

fn args<'a>(side: &'a FrozenSide<'a>, selection: bool, veto: Option<&'a [bool]>) -> ColumnArgs<'a> {
    ColumnArgs {
        side,
        sd: &[0.6f32; H],
        log_prior_odds: &[-1.0f64; H],
        veto,
        selection,
        transitions: 1,
        seed: 4242,
        sweep: 0,
        h: H,
    }
}

fn initial_state() -> (Vec<f32>, Vec<bool>) {
    let beta: Vec<f32> = (0..B * H)
        .map(|i| ((i * 5) % 11) as f32 * 0.03 - 0.15)
        .collect();
    let z: Vec<bool> = (0..B * H).map(|i| i % 3 != 0).collect();
    (beta, z)
}

/// The tile width is a cache-locality choice, so it must respect the byte budget at
/// every slate width the sampler can actually pick — including an exact normalizer over
/// the whole feature axis, where an untiled block would be gigabytes.
#[test]
fn tile_size_keeps_the_block_resident() {
    for &k in &[16usize, 128, 1024, 1460, 30392] {
        for &terms in &[1usize, 2] {
            let b = tile_size(k, terms);
            assert!((32..=4096).contains(&b), "k={k} terms={terms} gave b={b}");
            let bytes = 4 * b * k * terms;
            // The clamp can force a small `k` past the target; a large one must not be.
            if b > 32 {
                assert!(
                    bytes <= TILE_BYTES * 2,
                    "k={k} terms={terms}: {bytes} bytes for b={b} blows the budget"
                );
            }
        }
    }
    // The case the exact-normalizer option turns on: 30392-wide slate must still tile
    // to something resident, not to a 3.7 GB block.
    let b = tile_size(30392, 1);
    assert!(
        4 * b * 30392 < 8 << 20,
        "an exact feature-axis normalizer tiles to {} bytes",
        4 * b * 30392
    );
}

/// THE tiling test. Splitting the anchor axis is a performance decision, so it must not
/// move a single draw. Every anchor's randomness is keyed on its GLOBAL index precisely
/// so this holds; if it ever fails, some stream is keyed off tile position and
/// reproducibility has silently become a function of cache geometry.
#[test]
fn results_do_not_depend_on_the_tile_boundary() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slabs = vec![SlateSlab::new(&partition, &side)];
    let a = args(&side, true, None);
    let (beta0, z0) = initial_state();

    // One tile over everything.
    let whole = {
        let nodes = nodes_for(&pos, &partition, 0, B);
        sample_tile(&ProfiledPoisson, &a, &nodes, &slabs, 0, &beta0, &z0)
    };

    // The same anchors in three uneven tiles — uneven on purpose, so an off-by-one in
    // the global keying cannot cancel.
    let mut beta_split = vec![0.0f32; B * H];
    let mut z_split = vec![false; B * H];
    for &(lo, hi) in &[(0usize, 5usize), (5, 6), (6, B)] {
        let nodes = nodes_for(&pos, &partition, lo, hi);
        let part = sample_tile(
            &ProfiledPoisson,
            &a,
            &nodes,
            &slabs,
            lo,
            &beta0[lo * H..hi * H],
            &z0[lo * H..hi * H],
        );
        beta_split[lo * H..hi * H].copy_from_slice(&part.beta);
        z_split[lo * H..hi * H].copy_from_slice(&part.z);
    }

    assert_eq!(
        whole.beta, beta_split,
        "loadings depend on the tile boundary"
    );
    assert_eq!(whole.z, z_split, "inclusions depend on the tile boundary");
}

/// Peel then restore with the same value must return the carried state to where it
/// started. This is the drift check: `s` is advanced by rank-1 updates for `h` dims a
/// sweep, and a mismatched peel would corrupt every later dim with no visible symptom.
#[test]
fn peel_and_restore_round_trip() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slab = SlateSlab::new(&partition, &side);
    let nodes = nodes_for(&pos, &partition, 0, B);

    let (beta0, z0) = initial_state();
    let v: Vec<f32> = (0..B * H)
        .map(|i| if z0[i] { beta0[i] } else { 0.0 })
        .collect();

    let mut term = TermTile::seed(&nodes, 0, &slab, &side, &v, H, true);
    let s0 = term.s.clone();
    let data0 = term.data.clone();
    let sumsq0 = term.sumsq.clone();

    for d in 0..H {
        term.peel(&v, d, H);
        term.restore(&v, d, H);
    }

    for (i, (got, want)) in term.s.iter().zip(&s0).enumerate() {
        assert!((got - want).abs() < 1e-4, "s[{i}] drifted: {got} vs {want}");
    }
    for (i, (got, want)) in term.data.iter().zip(&data0).enumerate() {
        assert!(
            (got - want).abs() / want.abs().max(1.0) < 1e-6,
            "data[{i}] drifted: {got} vs {want}"
        );
    }
    for (i, (got, want)) in term.sumsq.iter().zip(&sumsq0).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "sumsq[{i}] drifted: {got} vs {want}"
        );
    }
}

/// A peel must leave the state describing "this coordinate contributes nothing
/// sampled", which is checkable against an independent reseed at `v[d] = 0`. Guards the
/// specific hazard that a frozen offset is peeled along with the sampled part — gem
/// carries `β_g` there, and losing it would sample the wrong conditional silently.
#[test]
fn a_peel_matches_reseeding_with_that_coordinate_zeroed() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slab = SlateSlab::new(&partition, &side);
    let offset: Vec<f32> = (0..B * H)
        .map(|i| ((i * 3) % 7) as f32 * 0.05 - 0.15)
        .collect();
    let nodes: Vec<Vec<NodeTerm>> = (0..B)
        .map(|i| {
            vec![NodeTerm {
                offset: Some(&offset[i * H..(i + 1) * H]),
                ..NodeTerm::new(&pos[i], &partition, 1.0)
            }]
        })
        .collect();

    let (beta0, z0) = initial_state();
    let v: Vec<f32> = (0..B * H)
        .map(|i| if z0[i] { beta0[i] } else { 0.0 })
        .collect();

    let d = 2usize;
    let mut peeled = TermTile::seed(&nodes, 0, &slab, &side, &v, H, true);
    peeled.peel(&v, d, H);

    // Independent construction: seed from a loading whose dim `d` is already zero. The
    // offset is untouched, so its dim-`d` share must still be present in both.
    let mut v_zero = v.clone();
    for i in 0..B {
        v_zero[i * H + d] = 0.0;
    }
    let fresh = TermTile::seed(&nodes, 0, &slab, &side, &v_zero, H, true);

    for (i, (got, want)) in peeled.s.iter().zip(&fresh.s).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "peeled s[{i}] {got} != reseeded {want} — the offset was probably peeled too"
        );
    }
    for i in 0..B {
        assert!(
            (peeled.data[i] - fresh.data[i]).abs() / fresh.data[i].abs().max(1.0) < 1e-6,
            "peeled data[{i}] {} != reseeded {}",
            peeled.data[i],
            fresh.data[i]
        );
    }
    // `sumsq` differs by construction: `fresh` includes dim `d`'s offset-only square,
    // `peeled` excludes the coordinate entirely. Assert that exact relationship rather
    // than equality, so the difference is pinned down instead of tolerated.
    for i in 0..B {
        let off_d = offset[i * H + d];
        let want = fresh.sumsq[i] - off_d * off_d;
        assert!(
            (peeled.sumsq[i] - want).abs() < 1e-5,
            "peeled sumsq[{i}] {} should be reseeded {} minus off_d² {}",
            peeled.sumsq[i],
            fresh.sumsq[i],
            off_d * off_d
        );
    }
}

/// With selection off every coordinate stays included, and nothing is written back as
/// an exact zero — a pseudobulk is a location, not a presence, so its block runs this
/// way.
#[test]
fn selection_off_keeps_every_coordinate_included() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slabs = vec![SlateSlab::new(&partition, &side)];
    let nodes = nodes_for(&pos, &partition, 0, B);
    let (beta0, z0) = initial_state();

    let draw = sample_tile(
        &ProfiledPoisson,
        &args(&side, false, None),
        &nodes,
        &slabs,
        0,
        &beta0,
        &z0,
    );
    assert!(draw.z.iter().all(|&v| v), "every coordinate must stay on");
    assert!(
        draw.beta.iter().all(|v| v.is_finite()),
        "no non-finite loading"
    );
}

/// A vetoed coordinate is excluded by the model, so it must come back off regardless of
/// what the data says — this is how gem's nesting `z_δ ⊆ z_β` is enforced.
#[test]
fn a_vetoed_coordinate_is_pinned_off() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slabs = vec![SlateSlab::new(&partition, &side)];
    let nodes = nodes_for(&pos, &partition, 0, B);
    let (beta0, z0) = initial_state();

    // Veto dim 1 everywhere, allow everything else.
    let mut allowed = vec![true; B * H];
    for i in 0..B {
        allowed[i * H + 1] = false;
    }
    let draw = sample_tile(
        &ProfiledPoisson,
        &args(&side, true, Some(&allowed)),
        &nodes,
        &slabs,
        0,
        &beta0,
        &z0,
    );
    for i in 0..B {
        assert!(
            !draw.z[i * H + 1],
            "anchor {i} dim 1 was vetoed but came back on"
        );
    }
    // And the veto must not be a blanket off-switch.
    assert!(
        (0..B).any(|i| (0..H).any(|d| d != 1 && draw.z[i * H + d])),
        "the veto turned off dims it was not asked to"
    );
}

/// The same seed must reproduce the run — there is no shared mutable state, so a rayon
/// reschedule must not change the answer, or no A/B against this sampler is meaningful.
#[test]
fn the_same_seed_reproduces_the_tile() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slabs = vec![SlateSlab::new(&partition, &side)];
    let nodes = nodes_for(&pos, &partition, 0, B);
    let (beta0, z0) = initial_state();
    let a = args(&side, true, None);

    let one = sample_tile(&ProfiledPoisson, &a, &nodes, &slabs, 0, &beta0, &z0);
    let two = sample_tile(&ProfiledPoisson, &a, &nodes, &slabs, 0, &beta0, &z0);
    assert_eq!(one.beta, two.beta);
    assert_eq!(one.z, two.z);
    assert_eq!(one.fallbacks, two.fallbacks);
}

/// The profiled intercept must satisfy its defining identity, `exp(b*)·A(θ) = T`, i.e.
/// `b* = ln T − ln(scale · Σ_slate exp(s))`.
///
/// This is the mechanism the alternating sampler leans on to keep the two sides'
/// intercepts live: each block profiles its own anchor's intercept exactly, so the other
/// side can score against a value that describes the state it is conditioning on rather
/// than one snapshotted before sampling began. Checked against an independent
/// computation from the naive definition, because `profiled_bias` reads the carried
/// scores and a bug there would be invisible in every shape.
#[test]
fn the_profiled_intercept_reproduces_the_observed_total() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slab = SlateSlab::new(&partition, &side);
    let nodes = nodes_for(&pos, &partition, 0, B);

    let (beta0, z0) = initial_state();
    let v: Vec<f32> = (0..B * H)
        .map(|i| if z0[i] { beta0[i] } else { 0.0 })
        .collect();
    let term = TermTile::seed(&nodes, 0, &slab, &side, &v, H, true);

    for i in 0..B {
        let got = f64::from(term.profiled_bias(i));
        if pos[i].is_empty() {
            // No counts ⇒ no rate to match, so the floor rather than a `ln(0)`.
            assert!(
                got < -20.0,
                "empty anchor {i} should be parked at the floor, got {got}"
            );
            continue;
        }
        // Independent construction: score every slate entry from scratch.
        let total: f64 = pos[i].iter().map(|&(_, n)| f64::from(n)).sum();
        let part: f64 = partition
            .iter()
            .map(|&o| {
                let e_o = &e[o as usize * H..(o as usize + 1) * H];
                let dot: f64 = (0..H)
                    .map(|d| f64::from(v[i * H + d]) * f64::from(e_o[d]))
                    .sum();
                (dot + f64::from(b[o as usize])).exp()
            })
            .sum();
        let want = total.ln() - part.ln();
        assert!(
            (got - want).abs() < 1e-3,
            "anchor {i}: profiled intercept {got} should be ln({total}) − ln({part}) = {want}"
        );

        // And state the identity the way it is used: the implied rate integrates to the
        // observed total.
        let implied = got.exp() * part;
        assert!(
            (implied - total).abs() / total < 1e-3,
            "anchor {i}: exp(b*)·A = {implied} should equal the observed total {total}"
        );
    }
}

/// The pass must report its own cost rather than hiding it. `rounds` is the number of
/// batched likelihood calls, so it bounds the work and makes a stalled sampler visible;
/// an active-set loop is exactly where such a count goes missing.
#[test]
fn the_tile_reports_its_cost() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slabs = vec![SlateSlab::new(&partition, &side)];
    let nodes = nodes_for(&pos, &partition, 0, B);
    let (beta0, z0) = initial_state();

    let draw = sample_tile(
        &ProfiledPoisson,
        &args(&side, true, None),
        &nodes,
        &slabs,
        0,
        &beta0,
        &z0,
    );
    assert!(
        draw.rounds > 0,
        "a sweep that evaluated nothing did not sample"
    );
    // One batched call per shrinkage round per dim; a healthy bracket takes a handful.
    assert!(
        draw.rounds < H * 64,
        "{} rounds over {H} dims means brackets are stalling",
        draw.rounds
    );
}

////////////////////////////////////////////////////////////
// The oracle claim: ANY log-likelihood, not just bilinear //
////////////////////////////////////////////////////////////

/// A likelihood with nothing to do with the bilinear-Poisson form: a Gaussian pulling each
/// anchor's loading toward a planted target.
///
/// It reads NOTHING precomputed — no `s`, no `data`, no `sumsq`, no `m`, no slate at all —
/// only `ctx.v` and the candidate `x`. Declares both capabilities false, because neither
/// holds: there are no per-slate scores to keep by a rank-1 update and no data term to
/// collapse.
///
/// This is what "the sampler is a likelihood oracle" has to mean if it means anything:
/// elliptical slice sampling constrains the PRIOR, and reads the likelihood only through
/// `ll(θ) > threshold`. Nothing in the transition kernel may depend on the score's algebra.
struct PlantedGaussian<'t> {
    /// `[b × h]` target the likelihood pulls each anchor toward.
    target: &'t [f32],
    /// Likelihood SD. Smaller ⇒ the data dominates the prior.
    sd: f32,
}

impl AnchorScore for PlantedGaussian<'_> {
    fn label(&self) -> &'static str {
        "planted-gaussian"
    }
    // Deliberately both false: this score is the Tier-3 case.
    fn affine_in_anchor(&self) -> bool {
        false
    }
    fn data_term_is_linear(&self) -> bool {
        false
    }

    fn ll_column(
        &self,
        ctx: &ColumnCtx<'_>,
        d: usize,
        _c_d: &[f32],
        x: &[f32],
        active: &[u32],
        out: &mut [f32],
    ) {
        let prec = 1.0f64 / f64::from(self.sd * self.sd);
        for (slot, (&i, &xi)) in out.iter_mut().zip(active.iter().zip(x)) {
            let i = i as usize;
            // Rebuild the full loading from `v` — the only field this score touches — and
            // score it directly. `v` has dim `d` zeroed, so `x` goes there.
            let mut acc = 0.0f64;
            for k in 0..ctx.h {
                let vk = if k == d { xi } else { ctx.loading(i)[k] };
                let r = f64::from(vk - self.target[i * ctx.h + k]);
                acc += r * r;
            }
            *slot = (-0.5 * prec * acc) as f32;
        }
    }
}

/// A non-bilinear likelihood must be sampled correctly, and correctness here is checkable
/// against a closed form: with prior `N(0, σ₀²)` per coordinate and likelihood
/// `N(target, σ²)`, the posterior is Gaussian with
///
/// ```text
///   mean = target · (1/σ²) / (1/σ₀² + 1/σ²),   var = 1 / (1/σ₀² + 1/σ²)
/// ```
///
/// So this asserts the *shrunk* target rather than the target itself — a sampler that
/// ignored the prior would hit `target` exactly and fail, and one that ignored the
/// likelihood would sit at 0 and fail. Both failure modes are excluded.
///
/// `selection` is off so `z ≡ 1` and `v == β`: this is about the ESS move, not the gate.
#[test]
fn a_non_bilinear_likelihood_is_sampled_correctly() {
    const SWEEPS: usize = 1200;
    const BURN: usize = 200;
    const SD0: f32 = 0.6; // prior SD, matching `args()`
    const SD: f32 = 0.35; // likelihood SD

    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let partition: Vec<u32> = (0..K as u32).collect();
    let pos = edge_lists();
    let slabs = vec![SlateSlab::new(&partition, &side)];
    let nodes = nodes_for(&pos, &partition, 0, B);

    let target: Vec<f32> = (0..B * H)
        .map(|i| ((i * 7) % 11) as f32 * 0.12 - 0.6)
        .collect();
    let score = PlantedGaussian {
        target: &target,
        sd: SD,
    };

    let (mut beta, _) = initial_state();
    let mut z = vec![true; B * H];
    let mut acc = vec![0.0f64; B * H];
    let mut kept = 0usize;

    for sweep in 0..SWEEPS {
        let a = ColumnArgs {
            side: &side,
            sd: &[SD0; H],
            log_prior_odds: &[0.0f64; H],
            veto: None,
            selection: false,
            transitions: 1,
            seed: 909,
            sweep,
            h: H,
        };
        let draw = sample_tile(&score, &a, &nodes, &slabs, 0, &beta, &z);
        beta = draw.beta;
        z = draw.z;
        if sweep >= BURN {
            for (s, &v) in acc.iter_mut().zip(&beta) {
                *s += f64::from(v);
            }
            kept += 1;
        }
    }

    let prec0 = 1.0 / f64::from(SD0 * SD0);
    let prec = 1.0 / f64::from(SD * SD);
    let shrink = prec / (prec0 + prec);
    let post_sd = (1.0 / (prec0 + prec)).sqrt();
    let mut worst = 0.0f64;
    let mut mean_dev = 0.0f64;
    let mut mean_target = 0.0f64;
    for i in 0..B * H {
        let got = acc[i] / kept as f64;
        let want = f64::from(target[i]) * shrink;
        worst = worst.max((got - want).abs());
        mean_dev += (got - want).abs() / (B * H) as f64;
        mean_target += f64::from(target[i]).abs() / (B * H) as f64;
    }
    println!(
        "blackbox oracle: {kept} kept sweeps, shrinkage {shrink:.4} (posterior SD {post_sd:.3}), \
         mean |dev| {mean_dev:.4}, worst |dev| {worst:.4}"
    );

    // The bound is DERIVED, not picked. Assert on the MEAN deviation, whose Monte-Carlo
    // variance is `B·H` times smaller than the worst coordinate's, and check it against the
    // two ways this could be wrong rather than against a round number:
    //
    //   ignoring the prior      ⇒ shrinkage 1, mean deviation ≈ mean|target|·(1−shrink)
    //   ignoring the likelihood ⇒ shrinkage 0, mean deviation ≈ mean|target|·shrink
    //
    // so the tolerance must sit well below both, and it does — by roughly 2× and 6×.
    let if_no_prior = mean_target * (1.0 - shrink);
    let if_no_likelihood = mean_target * shrink;
    let tol = 0.25 * if_no_prior.min(if_no_likelihood);
    assert!(
        mean_dev < tol,
        "a non-bilinear likelihood was not sampled to its analytic posterior — mean \
         deviation {mean_dev:.4} against a tolerance of {tol:.4}; ignoring the prior would \
         give {if_no_prior:.4} and ignoring the likelihood {if_no_likelihood:.4}"
    );
    // And no single coordinate may be wildly off, at a bound set by the posterior's own
    // spread rather than by a constant.
    assert!(
        worst < 0.6 * post_sd,
        "coordinate-wise deviation {worst:.4} exceeds 0.6 posterior SD ({post_sd:.3})"
    );
}
