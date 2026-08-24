//! A pair with no variance on one side is not a testable hypothesis.
//!
//! `weighted_cov` is identically zero when either side is constant across the
//! stratum's samples, whatever the other side does. Such a pair cannot reject
//! at any threshold, so counting it as a test is wrong three times over: it
//! pays multiplicity in the Westfall-Young family, it drags the MAD that sets
//! every other pair's `z_re` scale, and it ships a row that reads as a
//! measured null rather than as an absent measurement.
//!
//! The pre-existing guard keyed on `t_obs = 0 AND null_sd = 0`. That holds
//! only when both sides are computed from the same quantities; the null is
//! built from fresh posterior draws, which fluctuate even where the observed
//! posterior mean is flat, so `null_sd > 0` and the guard never fired. These
//! tests pin the property directly on the observed side, where it is decidable.

use crate::lr_activity::fit::{is_degenerate_pair, pair_is_untestable, weighted_cov};

const W: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// The motivating identity: a constant side forces the covariance to zero, so
/// the statistic carries no information about the other side at all.
#[test]
fn a_constant_side_forces_the_statistic_to_zero() {
    let varying = [1.0f32, 2.0, 3.0, 4.0];
    let constant = [7.0f32, 7.0, 7.0, 7.0];

    assert_eq!(weighted_cov(&constant, &varying, &W), 0.0);
    assert_eq!(weighted_cov(&varying, &constant, &W), 0.0);
    // Both constant is the same story.
    assert_eq!(weighted_cov(&constant, &constant, &W), 0.0);
    // A genuinely varying pair is not zero, or the test above proves nothing.
    assert!(weighted_cov(&varying, &varying, &W).abs() > 1e-6);
}

/// The predicate must catch exactly that case, on either side.
#[test]
fn degeneracy_is_detected_on_either_side() {
    let varying = [1.0f32, 2.0, 3.0, 4.0];
    let constant = [7.0f32, 7.0, 7.0, 7.0];

    assert!(is_degenerate_pair(&constant, &varying, &W), "left constant");
    assert!(
        is_degenerate_pair(&varying, &constant, &W),
        "right constant"
    );
    assert!(
        is_degenerate_pair(&constant, &constant, &W),
        "both constant"
    );
    assert!(
        !is_degenerate_pair(&varying, &varying, &W),
        "a varying pair must stay testable"
    );
}

/// Zero-weight samples do not count as variation: if every sample carrying
/// weight holds the same value, the weighted covariance is zero regardless of
/// what the zero-weight samples do, so the predicate must agree.
#[test]
fn variation_only_in_zero_weight_samples_is_still_degenerate() {
    let w = [1.0f32, 1.0, 0.0, 0.0];
    let l = [5.0f32, 5.0, 99.0, -99.0];
    let r = [1.0f32, 2.0, 3.0, 4.0];

    assert_eq!(weighted_cov(&l, &r, &w), 0.0, "the statistic is zero");
    assert!(
        is_degenerate_pair(&l, &r, &w),
        "so the pair must be called degenerate"
    );
}

/// A pair with no usable weight at all has no statistic either; it must be
/// excluded rather than treated as a measured zero.
#[test]
fn a_pair_with_no_weight_is_degenerate() {
    let w = [0.0f32; 4];
    let l = [1.0f32, 2.0, 3.0, 4.0];
    let r = [4.0f32, 3.0, 2.0, 1.0];

    assert!(
        !weighted_cov(&l, &r, &w).is_finite(),
        "no weight, no statistic"
    );
    assert!(is_degenerate_pair(&l, &r, &w));
}

/// Near-constant is NOT degenerate: the predicate keys on exact absence of
/// weighted variance, so a real but small signal is still tested. Widening
/// this into a magnitude threshold would silently drop weak true effects.
#[test]
fn a_small_but_real_signal_is_not_degenerate() {
    let l = [1.0f32, 1.0, 1.0, 1.000_01];
    let r = [1.0f32, 2.0, 3.0, 4.0];

    assert!(
        !is_degenerate_pair(&l, &r, &W),
        "a tiny real variation must remain a hypothesis"
    );
}

/// Disjoint support: L varies only in samples where R sits at its mean, and
/// vice versa. Both sides have real weighted variance, yet EVERY term of the
/// covariance sum carries a zero factor, so the statistic is structurally
/// zero — no sample can contribute evidence about this pair. Common in sparse
/// strata, where two genes are detected in non-overlapping samples.
#[test]
fn disjoint_support_is_degenerate_despite_variance_on_both_sides() {
    // L moves only in samples 0,1; R only in samples 2,3.
    let l = [2.0f32, 0.0, 1.0, 1.0];
    let r = [5.0f32, 5.0, 8.0, 2.0];
    let w = [1.0f32, 1.0, 1.0, 1.0];

    // Both marginal variances are non-zero, so the variance check alone passes.
    assert_eq!(weighted_cov(&l, &r, &w), 0.0, "no sample contributes");
    assert!(
        is_degenerate_pair(&l, &r, &w),
        "structurally zero covariance is an absent measurement, not a null result"
    );
}

/// Exact cancellation: both sides vary, samples do co-deviate, yet the
/// contributions sum to bit-exact zero. This is what the remaining zero
/// statistics in real runs turn out to be — near-floor genes whose few
/// contributing samples pull in exactly opposite directions. The structural
/// checks all pass, so the statistic itself has to be inspected.
#[test]
fn exact_cancellation_is_untestable() {
    let w = [1.0f32, 1.0, 1.0, 1.0];
    let l = [1.0f32, 1.0, -1.0, -1.0];
    let r = [1.0f32, -1.0, 1.0, -1.0];

    // Every structural check passes: real variance on both sides, and every
    // sample deviates on both.
    assert!(
        !is_degenerate_pair(&l, &r, &w),
        "structurally this looks testable"
    );
    assert_eq!(
        weighted_cov(&l, &r, &w),
        0.0,
        "yet the terms cancel exactly"
    );
    assert!(
        pair_is_untestable(&l, &r, &w),
        "a bit-exact zero statistic is an absent measurement"
    );

    // A pair that genuinely co-varies is untouched.
    let r2 = [1.0f32, 1.0, -1.0, -1.0];
    assert!(weighted_cov(&l, &r2, &w) > 0.5);
    assert!(!pair_is_untestable(&l, &r2, &w));
}

/// A stratum's samples are POSITIONS into its own list; the rate matrices are
/// indexed by GLOBAL sample id. The two coincide only when a stratum holds
/// every sample in order, which is the single-batch case — so a per-batch
/// stratum, holding a filtered subset, is where conflating them breaks.
#[test]
fn a_permuted_position_maps_back_to_a_global_sample_id() {
    use crate::lr_activity::fit::permuted_global_id;

    // A second batch's samples: a subset, and deliberately none of them small
    // enough to be mistaken for a valid position into a 4-long list.
    let samples_in_stratum = [900usize, 901, 902, 903];
    // Identity permutation: position k must map to that stratum's k-th sample.
    let identity = [0usize, 1, 2, 3];
    for k in 0..4 {
        assert_eq!(
            permuted_global_id(&samples_in_stratum, &identity, k),
            samples_in_stratum[k]
        );
    }

    // A real permutation reorders WITHIN the subset and never leaves it.
    let sigma = [2usize, 0, 3, 1];
    let got: Vec<usize> = (0..4)
        .map(|k| permuted_global_id(&samples_in_stratum, &sigma, k))
        .collect();
    assert_eq!(got, vec![902, 900, 903, 901]);
    for g in &got {
        assert!(
            samples_in_stratum.contains(g),
            "a permutation must stay inside the stratum"
        );
    }
}
