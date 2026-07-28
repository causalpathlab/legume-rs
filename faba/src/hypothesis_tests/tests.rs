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

///////////////////////////////
// Contrast p-value monotone //
///////////////////////////////

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

/////////////////
// Odds ratio  //
/////////////////

/// A genomic C/T variant converts equally in both arms. The two cross-products
/// are then the SAME f64, so the logs cancel to a true `0.0` rather than landing
/// near it -- which is what lets the site guard use a positive floor and reject
/// variants at any depth without a tolerance.
#[test]
fn log_odds_is_exactly_zero_on_a_genomic_variant() {
    for (a_w, u_w, a_m, u_m) in [
        (80u64, 20u64, 80u64, 20u64),
        (1, 99, 1, 99),
        (24, 396, 24, 396),
    ] {
        assert_eq!(log_odds_ratio(a_w, u_w, a_m, u_m), 0.0);
    }
}

/// `a_m = 0` at most DART sites (57% on chr19+MYC), so this is the common case, not an edge
/// case: a control that never converts puts the WT odds infinitely above it.
#[test]
fn log_odds_is_infinite_when_the_control_never_converts() {
    assert_eq!(log_odds_ratio(4, 96, 0, 5), f64::INFINITY);
    assert_eq!(log_odds_ratio(20, 0, 0, 50), f64::INFINITY);
    assert_eq!(log_odds_ratio(0, 96, 5, 0), f64::NEG_INFINITY);
}

/// Mirrors `fisher_degenerate_margins_return_one`: a table with no information
/// reads as `OR = 1`, not as an effect in either direction.
#[test]
fn log_odds_is_zero_when_the_table_says_nothing() {
    assert_eq!(log_odds_ratio(0, 0, 0, 0), 0.0);
    assert_eq!(log_odds_ratio(20, 0, 5, 0), 0.0);
}

/// The guard compares with `<`, so a `NaN` would silently KEEP a site rather
/// than reject it -- the exact failure the retired rate rule needed `.max(1)`
/// denominators to dodge. Every branch here must be explicit instead.
#[test]
fn log_odds_never_returns_nan() {
    for a_w in 0..=3u64 {
        for u_w in 0..=3u64 {
            for a_m in 0..=3u64 {
                for u_m in 0..=3u64 {
                    let l = log_odds_ratio(a_w, u_w, a_m, u_m);
                    assert!(!l.is_nan(), "NaN at ({a_w},{u_w},{a_m},{u_m})");
                }
            }
        }
    }
}

/// The whole point of the change: the guard's statistic and the test's statistic
/// now move together. As the control converts more, the odds ratio falls and the
/// p-value rises, monotonically. The retired delta guard was only correlated
/// with this (rho = 0.172 on real data).
#[test]
fn log_odds_and_fisher_order_sites_the_same_way() {
    let ls: Vec<f64> = (0..10)
        .map(|a_m| log_odds_ratio(40, 360, a_m, 400 - a_m))
        .collect();
    let ps: Vec<f32> = (0..10)
        .map(|a_m| contrast_pvalue(40, 360, a_m, 400 - a_m))
        .collect();
    for i in 1..ls.len() {
        assert!(ls[i] <= ls[i - 1], "log odds must not rise: {ls:?}");
        assert!(ps[i] >= ps[i - 1], "p must not fall: {ps:?}");
    }
}

/// The SE is dominated by the SMALLEST cell, not by either library's depth. Here
/// the MUT arm is the DEEPER one (4999 vs 5000 is a wash, but 2 converted vs 30
/// is not), and it still supplies almost all the variance -- which is why
/// subsampling the control to match the WT arm's sample size would discard the
/// term that was already free.
#[test]
fn woolf_se_is_dominated_by_the_smallest_cell() {
    let (a_w, u_w, a_m, u_m) = (40.5f64, 704.5, 1.5, 1200.5);
    let var_wt = 1.0 / a_w + 1.0 / u_w;
    let var_mut = 1.0 / a_m + 1.0 / u_m;
    assert!(
        var_mut / (var_wt + var_mut) > 0.9,
        "the deeper arm supplies {:.0}% of the variance",
        100.0 * var_mut / (var_wt + var_mut)
    );
}

#[test]
fn woolf_se_shrinks_when_every_cell_grows() {
    let (_, se_thin) = log_odds_ratio_woolf(30, 4970, 2, 4998);
    let (_, se_deep) = log_odds_ratio_woolf(300, 49700, 20, 49980);
    assert!(
        se_thin > 2.5 * se_deep,
        "10x the counts at the same rates must sharpen the estimate: {se_thin} vs {se_deep}"
    );
}

/// The caveat that would otherwise arrive as a bug report. With `a_m = 0` the
/// corrected control cell is 0.5, so `1/0.5 = 2` floors the variance and the SE
/// cannot fall below ~1.41 however deep the run goes. It still says "this is a
/// lower bound, uncertain ~16-fold", but it does NOT rank those sites -- and at
/// most DART sites (57% on chr19+MYC) that is the situation.
#[test]
fn woolf_se_is_floored_by_the_pseudo_count_at_an_empty_control() {
    let (_, se_shallow) = log_odds_ratio_woolf(1, 4, 0, 50);
    let (_, se_deep) = log_odds_ratio_woolf(100, 400, 0, 5000);
    assert!(se_deep > 1.41, "floored near sqrt(2): {se_deep}");
    assert!(
        se_shallow - se_deep < 0.3,
        "100x the depth barely moves it: {se_shallow} vs {se_deep}"
    );
}

/// Why the continuity correction must never reach a decision path. This 2×2 is
/// a 0.6% WT site whose control converts 0 of 3 reads. Raw, the odds ratio is
/// `+inf`; corrected, it is −3.15, i.e. a claim that the CONTROL converts ~23x
/// more, off three reads. A guard built on the corrected value would reject it
/// as "no effect" when the truth is "no evidence" -- which is the p-value's
/// verdict, and the reason the two functions are kept apart.
#[test]
fn haldane_disagrees_in_sign_with_the_raw_guard_at_an_empty_control() {
    let (corrected, se) = log_odds_ratio_woolf(30, 4970, 0, 3);
    assert_eq!(log_odds_ratio(30, 4970, 0, 3), f64::INFINITY);
    assert!(corrected < -3.0 && corrected > -3.3, "got {corrected}");
    assert!(se > 1.5, "and it knows it is uncertain: {se}");
    assert!(contrast_pvalue(30, 4970, 0, 3) > 0.9);
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
