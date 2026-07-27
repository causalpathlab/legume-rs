use super::*;

////////////////////////////////////////
// Fisher exact (one-sided, WT > MUT) //
////////////////////////////////////////

#[test]
fn fisher_strong_wt_enrichment_is_significant() {
    // All WT reads converted, no MUT conversion → strong WT enrichment.
    let p = fisher_exact_greater(20, 0, 0, 20);
    assert!(p < 1e-4, "expected tiny p, got {p}");
}

#[test]
fn fisher_equal_rates_not_significant() {
    // Identical conversion fraction in both arms (a genomic variant) → p ≈ 1/2+.
    let p = fisher_exact_greater(10, 10, 10, 10);
    assert!(p > 0.3, "equal rates should not be significant, got {p}");
}

#[test]
fn fisher_wt_below_mut_is_one_ish() {
    // WT *lower* than MUT → upper tail is large.
    let p = fisher_exact_greater(2, 8, 8, 2);
    assert!(p > 0.9, "WT<MUT should give large p, got {p}");
}

#[test]
fn fisher_degenerate_margins_return_one() {
    assert_eq!(fisher_exact_greater(0, 0, 0, 0), 1.0);
    assert_eq!(fisher_exact_greater(0, 10, 0, 10), 1.0); // no converted reads anywhere
}

/////////////////////////////////////////////
// Beta-binomial LRT (one-sided, WT > MUT) //
/////////////////////////////////////////////

/// The contrast is ONE exact test now, so this pins the property the old
/// two-branch version could not have: the p-value never gets larger when the
/// evidence gets stronger. The dispatch broke exactly this -- one extra
/// converted control read moved p 7.6e6-fold, and doubling coverage at a fixed
/// effect made a site less significant.
#[test]
fn contrast_is_monotone_in_the_evidence() {
    // Sweep the control's converted count across the OLD branch boundary at 5.
    let ps: Vec<f32> = (0..10)
        .map(|a_m| contrast_pvalue(40, 360, a_m, 400 - a_m))
        .collect();
    for w in ps.windows(2) {
        assert!(
            w[1] >= w[0],
            "more control signal must not make a site MORE significant: {ps:?}"
        );
    }
    // ...and more data at a fixed effect must not weaken the call.
    let small = contrast_pvalue(20, 180, 2, 198);
    let big = contrast_pvalue(40, 360, 4, 396);
    assert!(
        big <= small,
        "doubling coverage at the same rates weakened the call: {small} -> {big}"
    );
}

//////////////////////////////////////////
// Bootstrap mean CI + sign-flip p-value //
//////////////////////////////////////////

use rand::rngs::SmallRng;
use rand::SeedableRng;

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
