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

#[test]
fn lrt_wt_enriched_low_fraction_is_significant() {
    // The regime faba was missing: ~50% WT vs ~5% MUT at good depth.
    let p = betabinom_lrt_greater(50, 100, 5, 100, 0.02);
    assert!(p < 0.01, "WT-enriched site should be significant, got {p}");
}

#[test]
fn lrt_equal_rates_not_significant() {
    // p̂_W == p̂_M → WT is not strictly above MUT → one-sided p = 1 (conservative).
    let p = betabinom_lrt_greater(50, 100, 50, 100, 0.02);
    assert_eq!(p, 1.0, "equal rates carry no WT>MUT evidence, got {p}");
}

#[test]
fn lrt_marginal_excess_is_near_half() {
    // A hair above equal → D ≈ 0 → one-sided p ≈ 0.5 (the boundary of the
    // half-chi-bar), i.e. not significant.
    let p = betabinom_lrt_greater(51, 100, 50, 100, 0.02);
    assert!(
        p > 0.3,
        "marginal WT excess should be non-significant, got {p}"
    );
}

#[test]
fn lrt_wt_below_mut_returns_one() {
    assert_eq!(betabinom_lrt_greater(5, 100, 50, 100, 0.02), 1.0);
}

#[test]
fn lrt_overdispersion_rescues_high_coverage_variant() {
    // A hom/het variant: near-equal, very high coverage, tiny WT excess.
    // With real overdispersion the LRT must NOT call it significant, even
    // though sheer n would make a naive binomial test tiny.
    let p = betabinom_lrt_greater(1030, 2000, 1000, 2000, 0.1);
    assert!(
        p > 0.05,
        "overdispersion should spare a high-cov variant, got {p}"
    );
}

///////////////////////
// contrast dispatch //
///////////////////////

#[test]
fn contrast_small_counts_match_fisher() {
    let a_w = 6;
    let u_w = 1;
    let a_m = 1;
    let u_m = 8;
    assert_eq!(
        contrast_pvalue(a_w, u_w, a_m, u_m, 0.02),
        fisher_exact_greater(a_w, u_w, a_m, u_m)
    );
}

#[test]
fn contrast_large_counts_use_lrt() {
    // All cells ≥ 5 and total ≥ 100 → LRT branch.
    let (a_w, u_w, a_m, u_m) = (60u64, 140u64, 6u64, 194u64);
    assert_eq!(
        contrast_pvalue(a_w, u_w, a_m, u_m, 0.02),
        betabinom_lrt_greater(a_w, a_w + u_w, a_m, a_m + u_m, 0.02)
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
