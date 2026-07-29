//! The collapsed data term must be interchangeable with the edge walk.
//!
//! The `⟨v, m⟩` collapse omits `Σ_pos n·b_o`, and [`AnchorMoment`] carries that term
//! in `bias_dot` so the two paths agree in **absolute value**, not merely up to a
//! constant.
//!
//! An earlier version of this file asserted only the weaker property, reasoning that
//! both consumers are differences — a slice threshold derived from another
//! likelihood, and the `z` draw's `ll_on − ll_off` — so a constant cancels. That is
//! true only if both sides of a comparison take the SAME form, and the radius guard
//! chooses per evaluation: a `z` draw whose `ll_off` at `x = 0` is inside the radius
//! while its `ll_on` is outside compares a collapse against a walk, and the constant
//! lands in the logit at full size.

use super::*;

const H: usize = 5;
const N_OTHER: usize = 40;

fn side_buffers() -> (Vec<f32>, Vec<f32>) {
    let mut e = vec![0.0f32; N_OTHER * H];
    for o in 0..N_OTHER {
        for d in 0..H {
            // Small, non-degenerate, and nowhere near SCORE_CLAMP.
            e[o * H + d] = ((o * (d + 3) + 2 * d) % 11) as f32 * 0.05 - 0.25;
        }
    }
    let b: Vec<f32> = (0..N_OTHER).map(|o| (o % 7) as f32 * 0.03 - 0.2).collect();
    (e, b)
}

fn probe(k: f32) -> Vec<f32> {
    (0..H).map(|d| k * (d as f32 * 0.17 - 0.4)).collect()
}

#[test]
fn the_moment_path_equals_the_walk() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let pos: Vec<(u32, f32)> = (0..N_OTHER as u32)
        .filter(|o| o % 3 == 0)
        .map(|o| (o, 1.0 + (o % 5) as f32))
        .collect();
    let partition: Vec<u32> = (0..N_OTHER as u32).collect();

    let walk = NodeTerm::new(&pos, &partition, 1.0);
    let mom = AnchorMoment::new(&pos, &side);
    let collapsed = walk.with_moment(&mom);

    // `bias_dot` is the term the collapse omits, so it must be exactly Σ n·b_o.
    let expected: f64 = pos
        .iter()
        .map(|&(o, n)| f64::from(n) * f64::from(b[o as usize]))
        .sum();
    assert!(
        (mom.bias_dot - expected).abs() < 1e-6,
        "bias_dot {} should be Σ n·b_o = {expected}",
        mom.bias_dot
    );

    // With it carried, the two forms agree outright at every probe — so which form
    // the radius guard picked is not observable to any consumer.
    for &k in &[0.0f32, 0.5, 1.0, -1.3, 2.7] {
        let x = probe(k);
        let w = f64::from(multinomial_ll(&x, &walk, &side));
        let c = f64::from(multinomial_ll(&x, &collapsed, &side));
        assert!(
            (w - c).abs() / w.abs().max(1.0) < 1e-3,
            "probe {k}: walk {w} vs collapse {c} — the forms must agree in absolute \
             value, not up to a constant"
        );
    }
}

/// REGRESSION: the guard chooses per evaluation, so a single `z` logit can straddle
/// it. With the omitted `Σ n·b_o` restored that is harmless; without it the logit is
/// wrong by that term, which is not small — it grows with the anchor's edge count.
///
/// Constructed to straddle deliberately: `x = 0` sits inside the radius and the
/// probe sits outside, which is exactly the `ll_off` / `ll_on` pair the gate draws.
#[test]
fn a_logit_straddling_the_safe_radius_is_not_corrupted() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let pos: Vec<(u32, f32)> = (0..N_OTHER as u32).map(|o| (o, 1.0 + (o % 4) as f32)).collect();
    let partition: Vec<u32> = (0..N_OTHER as u32).collect();

    let walk = NodeTerm::new(&pos, &partition, 1.0);
    let mom = AnchorMoment::new(&pos, &side);
    let collapsed = walk.with_moment(&mom);
    assert!(mom.safe_radius > 0.0, "fixture needs a usable safe region");
    assert!(
        mom.bias_dot.abs() > 1.0,
        "fixture's Σ n·b_o is {:.3}, too small for this test to detect the bug",
        mom.bias_dot
    );

    let off = vec![0f32; H];
    let inside = off.clone(); // ‖0‖ = 0, comfortably inside
    let outside: Vec<f32> = probe(1.0)
        .iter()
        .map(|v| v * 100.0 * mom.safe_radius.max(1.0))
        .collect();

    // The collapsed node must produce the same logit as the all-walk node, even
    // though `inside` takes the fast path and `outside` takes the slow one.
    let logit_mixed = f64::from(multinomial_ll(&outside, &collapsed, &side))
        - f64::from(multinomial_ll(&inside, &collapsed, &side));
    let logit_walk = f64::from(multinomial_ll(&outside, &walk, &side))
        - f64::from(multinomial_ll(&inside, &walk, &side));
    assert!(
        (logit_mixed - logit_walk).abs() / logit_walk.abs().max(1.0) < 1e-3,
        "a logit straddling the radius is off by {:.3} ({logit_mixed} vs {logit_walk}); \
         Σ n·b_o is {:.3}",
        logit_mixed - logit_walk,
        mom.bias_dot
    );
}


/// A frozen offset has to be folded in on both paths — gem's velocity track
/// sends `β_g` through here, and a moment path that ignored it would silently
/// sample the wrong conditional.
#[test]
fn the_offset_is_honoured_by_the_moment_path() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let pos: Vec<(u32, f32)> = vec![(2, 4.0), (11, 1.5), (23, 6.0)];
    let partition: Vec<u32> = (0..N_OTHER as u32).collect();
    let off = probe(0.6);

    let mom = AnchorMoment::new(&pos, &side);
    let walk = NodeTerm {
        offset: Some(&off),
        ..NodeTerm::new(&pos, &partition, 1.0)
    };
    let collapsed = walk.with_moment(&mom);

    let (x1, x2) = (probe(1.1), probe(-0.4));
    let d_walk = f64::from(multinomial_ll(&x1, &walk, &side))
        - f64::from(multinomial_ll(&x2, &walk, &side));
    let d_mom = f64::from(multinomial_ll(&x1, &collapsed, &side))
        - f64::from(multinomial_ll(&x2, &collapsed, &side));
    assert!(
        (d_walk - d_mom).abs() < 1e-3,
        "offset handling diverged: {d_walk} vs {d_mom}"
    );

    // Sanity: the offset actually moves the answer, so the check above is not
    // comparing two copies of "offset ignored".
    let no_off = NodeTerm::new(&pos, &partition, 1.0);
    assert!(
        (f64::from(multinomial_ll(&x1, &walk, &side))
            - f64::from(multinomial_ll(&x1, &no_off, &side)))
        .abs()
            > 1e-3,
        "the offset should change the likelihood"
    );
}


/// REGRESSION (review finding): SCORE_CLAMP is 30, a modelling bound, and `clamp`
/// is nonlinear in `e_a` — so once scores saturate the collapsed data term stops
/// differing from the walk by a constant. The earlier tests all probed the small-
/// score regime and so could never see this. Outside the safe radius the fast path
/// must not be used, and the two forms must still agree exactly.
#[test]
fn the_moment_path_defers_to_the_walk_once_scores_saturate() {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let pos: Vec<(u32, f32)> = (0..N_OTHER as u32).map(|o| (o, 1.0 + o as f32)).collect();
    let partition: Vec<u32> = (0..N_OTHER as u32).collect();

    let walk = NodeTerm::new(&pos, &partition, 1.0);
    let mom = AnchorMoment::new(&pos, &side);
    let collapsed = walk.with_moment(&mom);
    assert!(mom.safe_radius > 0.0, "fixture should have a usable safe region");

    // Well outside the radius, every score saturates and the two forms MUST agree
    // exactly — the guard has to have sent this through the clamped walk.
    let huge: Vec<f32> = probe(1.0)
        .iter()
        .map(|v| v * 400.0 * mom.safe_radius.max(1.0))
        .collect();
    let a = multinomial_ll(&huge, &walk, &side);
    let c = multinomial_ll(&huge, &collapsed, &side);
    assert_eq!(
        a, c,
        "outside the safe radius the collapsed form must fall back to the exact walk"
    );

    // And the unbounded-likelihood symptom is gone: scaling further must not keep
    // buying likelihood without limit.
    let huger: Vec<f32> = huge.iter().map(|v| v * 10.0).collect();
    let l1 = f64::from(multinomial_ll(&huge, &collapsed, &side));
    let l2 = f64::from(multinomial_ll(&huger, &collapsed, &side));
    assert!(
        (l2 - l1).abs() < 1e-3,
        "likelihood still grows with ||e_a|| past saturation: {l1} -> {l2}"
    );
}

