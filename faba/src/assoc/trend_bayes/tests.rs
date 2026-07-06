use super::*;
use crate::assoc::test_util::two_branch_panel as synth;

fn cfg() -> BayesTrendConfig {
    BayesTrendConfig {
        n_knots: 5,
        min_total_coverage: 30,
        min_cells: 10,
        prior_sd: 3.0,
        n_samples: 600,
        warmup: 200,
        seed: 1,
    }
}

#[test]
fn rising_branch_has_positive_effect_low_lfsr() {
    let (lin, sites) = synth();
    let res = run_trends_bayes(&sites, &lin, &cfg());
    let a0 = res
        .iter()
        .find(|r| r.site == 0 && r.branch == 0)
        .expect("site 0 branch 0");
    assert!(a0.effect > 0.0, "rising ⇒ +effect, got {}", a0.effect);
    assert!(a0.lfsr < 0.05, "rising ⇒ small lfsr, got {}", a0.lfsr);
    assert!(a0.effect_lo > 0.0, "90% CI excludes 0, lo={}", a0.effect_lo);
}

#[test]
fn flat_branches_have_high_lfsr() {
    let (lin, sites) = synth();
    let res = run_trends_bayes(&sites, &lin, &cfg());
    // Site A branch 1 (flat) and all of site B (flat) should be uncertain in sign.
    let a1 = res
        .iter()
        .find(|r| r.site == 0 && r.branch == 1)
        .expect("site 0 branch 1");
    assert!(
        a1.lfsr > 0.15,
        "flat branch lfsr should be high, got {}",
        a1.lfsr
    );
    for r in res.iter().filter(|r| r.site == 1) {
        assert!(
            r.effect_lo < 0.0 && r.effect_hi > 0.0,
            "flat site CI should straddle 0: [{}, {}]",
            r.effect_lo,
            r.effect_hi
        );
    }
}

#[test]
fn reproducible_across_runs() {
    let (lin, sites) = synth();
    let a = run_trends_bayes(&sites, &lin, &cfg());
    let b = run_trends_bayes(&sites, &lin, &cfg());
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.site, rb.site);
        assert_eq!(ra.branch, rb.branch);
        assert!((ra.effect - rb.effect).abs() < 1e-6, "effect reproducible");
        assert!((ra.lfsr - rb.lfsr).abs() < 1e-6, "lfsr reproducible");
    }
}
