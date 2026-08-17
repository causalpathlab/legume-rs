//! Tests for the shared sample-level inference helpers.
//!
//! The BH cases below are the union of the two independent suites that existed
//! while this function was duplicated — one written against the q-value contract
//! (monotone, `q >= p`, the m/1 inflation on the smallest p) and one against the
//! textbook worked example. Both are kept: they check different properties, and
//! between them they pin the behaviour the two former copies happened to share.

use super::*;
use approx::assert_relative_eq;
use rand::rngs::SmallRng;
use rand::SeedableRng;

//////////////////////////////
// Benjamini-Hochberg (FDR) //
//////////////////////////////

#[test]
fn benjamini_hochberg_qvalues() {
    let p = [0.001f32, 0.01, 0.5, 0.9];
    let q = benjamini_hochberg(&p);
    // q monotone w.r.t. sorted p and >= p.
    for i in 0..p.len() {
        assert!(q[i] >= p[i] - 1e-6, "q {} < p {}", q[i], p[i]);
        assert!(q[i] <= 1.0);
    }
    // Smallest p gets the largest inflation factor (m/1).
    assert!((q[0] - 0.004).abs() < 1e-6, "q0 = {}", q[0]);
}

#[test]
fn bh_preserves_input_order() {
    let p = vec![0.01, 0.50, 0.03, 0.20];
    let q = benjamini_hochberg(&p);
    assert_eq!(q.len(), p.len());
}

#[test]
fn bh_classic_fixture() {
    // Classic BH example: p = [0.01, 0.04, 0.03, 0.005]
    // sorted ascending: [0.005, 0.01, 0.03, 0.04] with m=4
    // scaled: [0.02, 0.02, 0.04, 0.04]
    let p = vec![0.01, 0.04, 0.03, 0.005];
    let q = benjamini_hochberg(&p);
    assert_relative_eq!(q[3], 0.02, epsilon = 1e-5);
    assert_relative_eq!(q[0], 0.02, epsilon = 1e-5);
    assert_relative_eq!(q[2], 0.04, epsilon = 1e-5);
    assert_relative_eq!(q[1], 0.04, epsilon = 1e-5);
}

#[test]
fn bh_empty_returns_empty() {
    let q = benjamini_hochberg(&[]);
    assert!(q.is_empty());
}

#[test]
fn bh_clamps_to_unit_interval() {
    let p = vec![0.001, 0.5, 0.9];
    let q = benjamini_hochberg(&p);
    for &qi in &q {
        assert!((0.0..=1.0).contains(&qi));
    }
}

#[test]
fn bh_monotone_in_p() {
    // Smaller p should get smaller-or-equal q (in sorted rank).
    let p = vec![0.9, 0.01, 0.5, 0.02, 0.3];
    let q = benjamini_hochberg(&p);
    let mut pairs: Vec<(f32, f32)> = p.iter().copied().zip(q.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for w in pairs.windows(2) {
        assert!(w[0].1 <= w[1].1 + 1e-6, "{:?} violates monotone", w);
    }
}

#[test]
fn bh_tolerates_nan_without_panicking() {
    // The pre-merge copies disagreed here: one sorted with `unwrap()` and
    // aborted the process on a NaN p-value. Callers derive p-values from fits
    // that can degenerate, so the surviving contract is "return something
    // usable", pinned here so a future edit cannot quietly restore the panic.
    let q = benjamini_hochberg(&[0.01, f32::NAN, 0.5]);
    assert_eq!(q.len(), 3);
    assert!((0.0..=1.0).contains(&q[0]));
}

/////////////////////////////////
// Bootstrap + permutation p's //
/////////////////////////////////

#[test]
fn bootstrap_ci_brackets_a_clear_positive_mean() {
    // A tight positive sample: the mean is ~5 and the CI should sit well above 0.
    let x: Vec<f32> = (0..40).map(|i| 5.0 + 0.01 * (i as f32 - 20.0)).collect();
    let mut rng = SmallRng::seed_from_u64(1);
    let (se, lo, hi) = bootstrap_mean_ci(&x, 500, 0.05, &mut rng);
    assert!(lo > 0.0 && hi > lo, "CI [{lo}, {hi}] should clear 0");
    assert!((0.0..0.1).contains(&se), "SE should be small, got {se}");
    assert!(mean(&x) > lo && mean(&x) < hi, "CI brackets the mean");
}

#[test]
fn sign_flip_rejects_strong_signal_and_not_zero_mean() {
    let mut rng = SmallRng::seed_from_u64(2);
    // All positive → the observed |mean| is never exceeded by a sign-flip → tiny p.
    let pos = vec![1.0f32; 30];
    assert!(sign_flip_pvalue(&pos, 500, &mut rng) < 0.01);
    // Symmetric ±1 → mean 0 → every flip ties or exceeds → p ≈ 1.
    let sym: Vec<f32> = (0..30)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    assert!(sign_flip_pvalue(&sym, 500, &mut rng) > 0.5);
}

#[test]
fn bootstrap_and_sign_flip_handle_empty_input() {
    let mut rng = SmallRng::seed_from_u64(3);
    let (se, lo, hi) = bootstrap_mean_ci(&[], 100, 0.05, &mut rng);
    assert!(se.is_nan() && lo.is_nan() && hi.is_nan());
    assert!(sign_flip_pvalue(&[], 100, &mut rng).is_nan());
}

#[test]
fn mean_of_empty_is_nan() {
    assert!(mean(&[]).is_nan());
    assert_relative_eq!(mean(&[1.0, 2.0, 3.0]), 2.0, epsilon = 1e-6);
}
