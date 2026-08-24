//! Contract of `testable_weighted_cov`: the statistic, or `None` when the
//! pair carries no testable hypothesis. See its doc for why an exactly-zero
//! covariance is an absent measurement rather than a measured null; these
//! tests pin each way that zero arises, and the boundary that keeps small
//! real signals in.

use crate::lr_activity::fit::{testable_weighted_cov, weighted_cov};

const W: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// A constant side forces the covariance to zero whatever the other side
/// does, so the statistic carries no information about the pair.
#[test]
fn a_constant_side_is_untestable() {
    let varying = [1.0f32, 2.0, 3.0, 4.0];
    let constant = [7.0f32, 7.0, 7.0, 7.0];

    assert_eq!(testable_weighted_cov(&constant, &varying, &W), None);
    assert_eq!(testable_weighted_cov(&varying, &constant, &W), None);
    assert_eq!(testable_weighted_cov(&constant, &constant, &W), None);
    // A genuinely varying pair passes, or the checks above prove nothing.
    assert!(testable_weighted_cov(&varying, &varying, &W).unwrap() > 1e-6);
}

/// Variation confined to zero-weight samples is no variation: if every
/// weighted sample holds one value, the covariance is zero regardless of
/// what the zero-weight samples do.
#[test]
fn variation_only_in_zero_weight_samples_is_untestable() {
    let w = [1.0f32, 1.0, 0.0, 0.0];
    let l = [5.0f32, 5.0, 99.0, -99.0];
    let r = [1.0f32, 2.0, 3.0, 4.0];

    assert_eq!(weighted_cov(&l, &r, &w), 0.0, "the statistic is zero");
    assert_eq!(testable_weighted_cov(&l, &r, &w), None);
}

/// No usable weight at all means no statistic either; that must read as
/// untestable, not as a measured zero.
#[test]
fn a_pair_with_no_weight_is_untestable() {
    let w = [0.0f32; 4];
    let l = [1.0f32, 2.0, 3.0, 4.0];
    let r = [4.0f32, 3.0, 2.0, 1.0];

    assert!(
        !weighted_cov(&l, &r, &w).is_finite(),
        "no weight, no statistic"
    );
    assert_eq!(testable_weighted_cov(&l, &r, &w), None);
}

/// Disjoint support: both sides vary, but never in the same weighted sample,
/// so every term of the covariance sum carries a zero factor. Common in
/// sparse strata where two genes are detected in non-overlapping samples.
#[test]
fn disjoint_support_is_untestable_despite_variance_on_both_sides() {
    let l = [2.0f32, 0.0, 1.0, 1.0];
    let r = [5.0f32, 5.0, 8.0, 2.0];

    assert_eq!(weighted_cov(&l, &r, &W), 0.0, "no sample contributes");
    assert_eq!(testable_weighted_cov(&l, &r, &W), None);
}

/// Exact cancellation: samples do co-deviate, yet the contributions sum to
/// bit-exact zero. This is what the surviving zero statistics in real runs
/// turned out to be — near-floor genes whose few contributions pull in
/// exactly opposite directions.
#[test]
fn exact_cancellation_is_untestable() {
    let l = [1.0f32, 1.0, -1.0, -1.0];
    let r = [1.0f32, -1.0, 1.0, -1.0];

    assert_eq!(weighted_cov(&l, &r, &W), 0.0, "the terms cancel exactly");
    assert_eq!(testable_weighted_cov(&l, &r, &W), None);

    // A pair that genuinely co-varies is untouched.
    let r2 = [1.0f32, 1.0, -1.0, -1.0];
    assert!(testable_weighted_cov(&l, &r2, &W).unwrap() > 0.5);
}

/// Near-constant is NOT untestable: the rule keys on exact absence, so a
/// small real signal is still a hypothesis. Widening this into a magnitude
/// threshold would silently drop weak true effects.
#[test]
fn a_small_but_real_signal_stays_testable() {
    let l = [1.0f32, 1.0, 1.0, 1.000_01];
    let r = [1.0f32, 2.0, 3.0, 4.0];

    let t = testable_weighted_cov(&l, &r, &W);
    assert!(
        t.is_some(),
        "a tiny real variation must remain a hypothesis"
    );
    assert!(t.unwrap().abs() < 1e-4, "and it is genuinely tiny");
}
