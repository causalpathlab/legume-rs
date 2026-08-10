//! The NB-Fisher trend can be fitted on pseudobulks instead of on cells —
//! *if* the rates are rescaled back to counts first. The why lives on
//! [`fisher_weights_from_pseudobulk`] itself; these tests pin that the
//! rescaling actually is the difference between a working trend and a
//! silently degenerate one.

use data_beans_alg::feature_coarsening::FeatureCoarsening;
use data_beans_alg::gene_weighting::fisher_weights_from_pseudobulk;
use nalgebra::DMatrix;

/// `[D, S]` of per-cell rates: gene `g` sits near `base[g]` with a spread of
/// `spread[g]`, so Poisson-like and over-dispersed genes are distinguishable.
/// Deterministic rather than sampled — the trend fit only reads per-gene mean
/// and variance, so an unseeded draw would buy nothing but flakiness.
fn rates(base: &[f32], spread: &[f32], s: usize) -> DMatrix<f32> {
    DMatrix::from_fn(base.len(), s, |g, j| {
        let swing = if j % 2 == 0 { 1.0 } else { -1.0 };
        (base[g] + swing * spread[g] * (1.0 + (j % 3) as f32)).max(0.0)
    })
}

const SIZES: [f32; 8] = [40.0, 55.0, 30.0, 60.0, 45.0, 35.0, 50.0, 25.0];

/// `w_g = 1 / (1 + π_g·s̄·φ(μ_g))`, so what it attenuates is **abundance**
/// against the fitted trend — not a gene's own excess variance. Two genes at
/// the same mean get the same `φ(μ)` by construction, however differently they
/// scatter; the trend is a single global regression, not a per-gene estimate.
#[test]
fn weights_are_bounded_and_fall_with_abundance() {
    let m = rates(&[0.05, 0.5, 5.0], &[0.01, 0.1, 1.0], SIZES.len());
    let w = fisher_weights_from_pseudobulk(&m, &SIZES, None).expect("matched lengths");

    assert_eq!(w.len(), 3);
    for &x in &w {
        assert!(x > 0.0 && x <= 1.0, "weight out of (0, 1]: {x}");
    }
    assert!(
        w[0] > w[1] && w[1] > w[2],
        "a more abundant gene should weigh less, got {w:?}"
    );
}

/// Rates and counts are not interchangeable inputs. Passing unit sizes leaves
/// the matrix on the rate scale, where the floor subtraction is wrong — this
/// pins that the two disagree, so a caller cannot quietly drop `size_s`.
#[test]
fn the_count_rescaling_is_what_makes_the_trend_meaningful() {
    let m = rates(&[0.5, 0.5, 0.5], &[0.01, 0.2, 0.45], SIZES.len());
    let ones = [1.0f32; SIZES.len()];

    let counts = fisher_weights_from_pseudobulk(&m, &SIZES, None).expect("matched lengths");
    let as_rates = fisher_weights_from_pseudobulk(&m, &ones, None).expect("rates");

    // On the rate scale every mean is below the Poisson floor the estimator
    // assumes, so the trend degenerates and nothing is attenuated at all.
    assert!(
        as_rates.iter().all(|&x| x == 1.0),
        "expected the rate-scale trend to collapse to no weighting, got {as_rates:?}"
    );
    assert!(
        counts.iter().any(|&x| x < 0.99),
        "expected the count-scale trend to attenuate something, got {counts:?}"
    );
}

/// Coarsening sums features into meta-features, so the trend is fitted at the
/// resolution the decoder actually runs at and returns one weight per group.
#[test]
fn coarsening_gives_one_weight_per_meta_feature() {
    let m = rates(&[0.4, 0.4, 0.6, 0.6], &[0.01, 0.01, 0.5, 0.5], SIZES.len());
    let fc = FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3]],
        num_coarse: 2,
    };

    let w = fisher_weights_from_pseudobulk(&m, &SIZES, Some(&fc)).expect("matched lengths");
    assert_eq!(w.len(), 2);
    for &x in &w {
        assert!(x > 0.0 && x <= 1.0, "weight out of (0, 1]: {x}");
    }
    assert!(
        w[0] > w[1],
        "the less abundant meta-feature should outweigh the more abundant one, got {w:?}"
    );
}

/// A degenerate input must not produce NaN — these weights multiply a
/// log-likelihood, so one NaN silently poisons the whole gradient.
#[test]
fn a_flat_or_empty_matrix_still_yields_finite_weights() {
    for m in [
        DMatrix::<f32>::from_element(4, SIZES.len(), 3.0),
        DMatrix::<f32>::zeros(4, SIZES.len()),
    ] {
        for x in fisher_weights_from_pseudobulk(&m, &SIZES, None).expect("degenerate") {
            assert!(x.is_finite() && x > 0.0 && x <= 1.0, "degenerate gave {x}");
        }
    }
}

/// A `size_s` of the wrong length rescales the wrong columns and returns
/// perfectly plausible weights — no shape mismatch, no NaN. Refused instead.
#[test]
fn a_wrong_length_size_s_is_refused() {
    let m = DMatrix::<f32>::from_element(4, SIZES.len(), 3.0);
    let err = fisher_weights_from_pseudobulk(&m, &[2.0], None).expect_err("length mismatch");
    assert!(err.to_string().contains("1 cell counts for 8"), "{err}");
}
