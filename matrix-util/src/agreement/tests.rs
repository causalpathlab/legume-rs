//! The correlations must behave on the shapes a held-out profile actually has:
//! mostly zeros (so ties dominate the ranks), counts spanning orders of
//! magnitude, and the degenerate cases a real run will hit.

use super::*;

#[test]
fn a_perfect_prediction_correlates_at_one() {
    let obs = [0.0f32, 0.0, 3.0, 1.0, 40.0, 0.0, 7.0];
    let pred = obs;
    assert!((spearman(&obs, &pred) - 1.0).abs() < 1e-5);
    assert!((pearson_log1p(&obs, &pred) - 1.0).abs() < 1e-5);
}

/// A monotone but non-linear prediction is a perfect Spearman and a degraded
/// Pearson — the reason both are reported.
///
/// The distortion has to be non-linear *after* `log1p`, which rules out the
/// obvious `pred = obs²`: `log1p(v²) ≈ 2·log(v)`, so that stays at r = 0.9997
/// and would have made this test pass for no reason. A compressed low end with
/// one large value is genuinely non-linear on the log scale.
#[test]
fn spearman_ignores_a_monotone_distortion_that_pearson_sees() {
    let obs = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let pred = [0.0f32, 0.1, 0.2, 0.3, 0.4, 5000.0];
    assert!(
        (spearman(&obs, &pred) - 1.0).abs() < 1e-5,
        "ranks are identical"
    );
    let r = pearson_log1p(&obs, &pred);
    assert!(r < 0.7, "log-scale distortion should show: r = {r}");
}

#[test]
fn a_reversed_prediction_correlates_at_minus_one() {
    let obs = [1.0f32, 2.0, 3.0, 4.0];
    let pred = [4.0f32, 3.0, 2.0, 1.0];
    assert!((spearman(&obs, &pred) + 1.0).abs() < 1e-5);
}

/// Held-out profiles are mostly zero, so tie handling is load-bearing: every
/// zero must share one averaged rank, not an arbitrary order-dependent one.
#[test]
fn tied_zeros_share_an_averaged_rank() {
    let v = [0.0f32, 0.0, 0.0, 5.0];
    let r = average_ranks(&v);
    assert_eq!(r[0], r[1]);
    assert_eq!(r[1], r[2]);
    assert!(
        (r[0] - 2.0).abs() < 1e-9,
        "three tied firsts average to 2, got {}",
        r[0]
    );
    assert!((r[3] - 4.0).abs() < 1e-9);
    // Order of the tied run must not change the answer.
    let shuffled = [5.0f32, 0.0, 0.0, 0.0];
    let rs = average_ranks(&shuffled);
    assert!((rs[0] - 4.0).abs() < 1e-9);
    assert!((rs[1] - 2.0).abs() < 1e-9);
}

/// An all-zero observed vector has no variance, so there is no correlation to
/// report. NaN says that; 0.0 would claim "no agreement", which is different
/// and would drag a mean down.
#[test]
fn a_constant_vector_gives_nan_not_zero() {
    let obs = [0.0f32; 5];
    let pred = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    assert!(spearman(&obs, &pred).is_nan());
    assert!(pearson_log1p(&obs, &pred).is_nan());
}

#[test]
fn a_single_gene_cannot_be_correlated() {
    assert!(spearman(&[1.0], &[1.0]).is_nan());
    assert!(pearson_log1p(&[1.0], &[1.0]).is_nan());
}

#[test]
fn pearson_log1p_is_not_scale_invariant() {
    // The property that forces callers to hand over predicted COUNTS rather than
    // a rate. A prediction proportional to truth is perfect, but log1p(c·p) is
    // not an affine function of log1p(p) — the zeros anchor the low end — so a
    // rate on some other scale scores below 1, and by an amount that depends on
    // the scale. Two models whose rates carry different arbitrary scales would
    // therefore be ranked by their scales.
    let obs = [0.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 5.0, 40.0];
    let total: f32 = obs.iter().sum();
    let comp: Vec<f32> = obs.iter().map(|o| o / total).collect();

    let on_counts: Vec<f32> = comp.iter().map(|c| c * total).collect();
    assert!(
        (pearson_log1p(&obs, &on_counts) - 1.0).abs() < 1e-4,
        "a proportional prediction on the count scale must score 1"
    );

    let as_rate = pearson_log1p(&obs, &comp);
    assert!(
        as_rate < 0.95,
        "the same prediction as a composition must score visibly lower, got {as_rate}"
    );

    // Spearman is rank-based and therefore immune — which is why only the
    // Pearson column needed the fix.
    assert!((spearman(&obs, &comp) - 1.0).abs() < 1e-6);
}
