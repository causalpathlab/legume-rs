//! The collapsed data term must be interchangeable with the edge walk.
//!
//! [`AnchorMoment`] drops a `Σ_pos n·b_o` constant, so the two paths do **not**
//! agree in absolute value — and they are not supposed to. What has to hold is
//! that every comparison a sampler actually makes is unchanged, i.e. the two
//! differ by a constant that does not depend on `e_a`. Both consumers are
//! differences: elliptical slice sampling compares a likelihood against a
//! threshold derived from another likelihood, and the `z` draw uses
//! `ll_on − ll_off`.

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
fn the_moment_path_shifts_the_likelihood_by_a_constant() {
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

    // The gap must be the same at every probe — that is what "constant in e_a"
    // means, and it is the whole claim.
    let gaps: Vec<f64> = [0.0f32, 0.5, 1.0, -1.3, 2.7]
        .iter()
        .map(|&k| {
            let x = probe(k);
            f64::from(multinomial_ll(&x, &walk, &side))
                - f64::from(multinomial_ll(&x, &collapsed, &side))
        })
        .collect();

    let first = gaps[0];
    for (i, g) in gaps.iter().enumerate() {
        assert!(
            (g - first).abs() < 1e-3,
            "gap must not depend on e_a: probe {i} gave {g}, probe 0 gave {first}"
        );
    }

    // And it really is `Σ_pos n·b_o`, not some other constant.
    let expected: f64 = pos
        .iter()
        .map(|&(o, n)| f64::from(n) * f64::from(b[o as usize]))
        .sum();
    assert!(
        (first - expected).abs() < 1e-3,
        "gap {first} should be Σ n·b_o = {expected}"
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

