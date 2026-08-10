//! Pins WHICH posterior the weighted-cohort NB-Fisher trend is fitted on.
//!
//! `fisher_weights_for_weighted_cohort` must read `mu_observed`, not the
//! batch-adjusted posterior. This is exactly the kind of line a well-meaning
//! cleanup reverts — `preferred_posterior_mean` is the idiom everywhere else,
//! and swapping it in compiles, passes every shape check, and quietly drops
//! the agreement with the exact re-collapse from ρ 0.98 to 0.47 (δ-removal
//! strips between-batch spread, which is variance the observation-process
//! trend is supposed to see). It happened once in this feature's history;
//! this test is why it cannot happen silently again.

use data_beans_alg::collapse_data::CollapsedOut;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::{Inference, TwoStatParam};
use nalgebra::DMatrix;
use senna::pb_reference::fisher_weights_for_weighted_cohort;

const D: usize = 30;
const S: usize = 8;

/// A Gamma posterior whose mean is (numerically) `mean`, via a large
/// pseudo-count: `posterior_mean = (a0 + m·BIG) / (b0 + BIG) ≈ m`.
fn gamma_with_mean(mean: &DMatrix<f32>) -> GammaMatrix {
    const BIG: f32 = 1e4;
    let mut g = GammaMatrix::new((mean.nrows(), mean.ncols()), 1.0, 1.0);
    g.update_stat(
        &(mean * BIG),
        &DMatrix::from_element(mean.nrows(), mean.ncols(), BIG),
    );
    g.calibrate();
    g
}

/// Observed and adjusted posteriors that RANK genes differently: the observed
/// mean ramps up the gene axis, the adjusted one ramps down. Any statistic
/// keyed on abundance therefore orders genes oppositely between the two, so
/// the test cannot pass by accident of symmetric inputs.
fn collapsed_with_divergent_posteriors() -> CollapsedOut {
    // Rates large and spread wide enough that, after ×100 count-rescaling,
    // per-gene variance clears the Poisson floor — otherwise every fit point
    // drops, both trends degenerate to "no weighting", and the guard-the-guard
    // assertion below fires (it did, on a first version of this fixture).
    let observed = DMatrix::from_fn(D, S, |g, s| {
        2.0 * (1.0 + g as f32) * (1.0 + 0.5 * (s % 3) as f32)
    });
    let adjusted = DMatrix::from_fn(D, S, |g, s| {
        2.0 * (D - g) as f32 * (1.0 + 0.5 * (s % 3) as f32)
    });
    CollapsedOut {
        mu_observed: gamma_with_mean(&observed),
        mu_adjusted: Some(gamma_with_mean(&adjusted)),
        mu_residual: None,
        gamma: None,
        delta: None,
    }
}

#[test]
fn the_trend_is_fitted_on_the_observed_posterior_not_the_adjusted_one() {
    let collapsed = collapsed_with_divergent_posteriors();
    // One column per pseudobulk (the degenerate membership), 100 cells each.
    let cell_to_pb: Vec<usize> = (0..S).collect();
    let weights = vec![100.0f32; S];

    let got = fisher_weights_for_weighted_cohort(&collapsed, &cell_to_pb, Some(&weights), None)
        .expect("well-formed inputs");

    let expect_from = |mu: &DMatrix<f32>| {
        data_beans_alg::gene_weighting::fisher_weights_from_pseudobulk(mu, &[100.0f32; S], None)
            .expect("estimator")
    };
    let from_observed = expect_from(collapsed.mu_observed.posterior_mean());
    let from_adjusted = expect_from(
        collapsed
            .mu_adjusted
            .as_ref()
            .expect("built with Some")
            .posterior_mean(),
    );

    assert_eq!(
        got, from_observed,
        "the weighted-cohort trend must come from mu_observed"
    );
    // Guard the guard: if both posteriors ever produced the same weights,
    // the assertion above would stop testing the choice at all.
    assert_ne!(
        from_observed, from_adjusted,
        "test inputs no longer distinguish the two posteriors"
    );
    // And the difference is not a tie in disguise: the two rankings disagree.
    let rank = |w: &[f32]| {
        let mut idx: Vec<usize> = (0..w.len()).collect();
        idx.sort_by(|&a, &b| w[a].total_cmp(&w[b]));
        idx
    };
    assert_ne!(rank(&from_observed), rank(&from_adjusted));
}
