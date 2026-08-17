//! Unit tests for the single-sample editing statistics exposed via the lib
//! surface (`faba::hypothesis_tests`): the beta-binomial p-value.
//!
//! The FDR adjustment these once sat beside is not an editing statistic and is
//! now shared workspace-wide; its tests live with it in
//! `matrix_util::hypothesis`.

use faba::hypothesis_tests::betabinom_pvalue_greater;
use statrs::distribution::{Binomial, DiscreteCDF};

#[test]
fn betabinom_basic_ordering() {
    // Strong editing is far more significant than sparse noise.
    let p_edit = betabinom_pvalue_greater(50, 100, 0.01, 0.1);
    let p_noise = betabinom_pvalue_greater(1, 100, 0.01, 0.1);
    assert!(
        p_edit < p_noise,
        "edit {p_edit} should be < noise {p_noise}"
    );
    assert!(
        p_edit < 1e-3,
        "strong editing should be significant: {p_edit}"
    );
    // Zero alt reads => no evidence.
    assert_eq!(betabinom_pvalue_greater(0, 100, 0.01, 0.1), 1.0);
}

#[test]
fn betabinom_binomial_limit() {
    // rho -> 0 must match the plain binomial upper tail.
    let n = 80u64;
    let k = 6u64;
    let eps = 0.01;
    let bb = betabinom_pvalue_greater(k, n, eps, 0.0);
    let binom = Binomial::new(eps, n).unwrap();
    let expected = binom.sf(k - 1) as f32;
    assert!(
        (bb - expected).abs() < 1e-4,
        "bb {bb} vs binomial {expected}"
    );
}

#[test]
fn betabinom_no_underflow_at_tail_crossover() {
    // The old `1 - lower_tail` branch collapsed very significant sites to
    // exactly 0.0 around k ≈ n/2. The direct upper-tail sum must stay
    // strictly positive and monotonic (more editing ⇒ smaller p).
    let mut prev = 1.0f32;
    for k in [48u64, 49, 50, 51, 52] {
        let p = betabinom_pvalue_greater(k, 100, 0.01, 0.1);
        assert!(p > 0.0, "p must not underflow to 0 at k={k}");
        assert!(p <= prev, "p must be monotonic non-increasing in k (k={k})");
        prev = p;
    }
}

#[test]
fn betabinom_overdispersion_is_conservative() {
    // Overdispersion (rho>0) yields a heavier tail => larger p than binomial.
    let p_binom = betabinom_pvalue_greater(8, 100, 0.01, 0.0);
    let p_bb = betabinom_pvalue_greater(8, 100, 0.01, 0.2);
    assert!(
        p_bb > p_binom,
        "overdispersed {p_bb} should exceed {p_binom}"
    );
}
