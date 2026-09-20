use crate::dmatrix_gamma::GammaMatrix;
use crate::traits::{Inference, TwoStatParam};
use nalgebra::DMatrix;

/// `sd[ln X]` for `X ~ Gamma(a, b)` is `sqrt(trigamma(a))`, independent of `b`.
///
/// The values below are exact: `trigamma(1) = pi^2/6`, `trigamma(2) = pi^2/6 - 1`,
/// `trigamma(0.5) = pi^2/2`.
#[test]
fn log_sd_is_sqrt_trigamma_of_the_shape() {
    let cases = [
        (1.0_f32, (std::f32::consts::PI.powi(2) / 6.0).sqrt()),
        (2.0, (std::f32::consts::PI.powi(2) / 6.0 - 1.0).sqrt()),
        (0.5, (std::f32::consts::PI.powi(2) / 2.0).sqrt()),
    ];
    for (a, want) in cases {
        let mut p = GammaMatrix::new((1, 1), 0.0, 0.0);
        // `add_stat` accumulates onto (a0, b0); start from zero so a_stat == a.
        p.update_stat(
            &DMatrix::from_element(1, 1, a),
            &DMatrix::from_element(1, 1, 3.0),
        );
        p.calibrate();
        let got = p.posterior_log_sd()[(0, 0)];
        assert!(
            (got - want).abs() < 1e-4,
            "a={a}: log_sd {got} != sqrt(trigamma(a)) {want}"
        );
    }
}

/// The shape-1 case is what the old `1/sqrt(a-1)` could not express: it returned
/// 0, i.e. perfect certainty, for a feature whose posterior is still the prior.
#[test]
fn an_unobserved_feature_has_the_largest_log_sd_not_zero() {
    let mut p = GammaMatrix::new((2, 1), 1.0, 1.0);
    // Row 0 sees nothing; row 1 sees plenty.
    let a_obs = DMatrix::from_row_slice(2, 1, &[0.0, 500.0]);
    let b_obs = DMatrix::from_row_slice(2, 1, &[0.0, 500.0]);
    p.update_stat(&a_obs, &b_obs);
    p.calibrate();
    let sd = p.posterior_log_sd();
    assert!(
        sd[(0, 0)] > 1.0,
        "unobserved row should be uncertain, got {}",
        sd[(0, 0)]
    );
    assert!(
        sd[(1, 0)] < 0.1,
        "well-observed row should be precise, got {}",
        sd[(1, 0)]
    );
    assert!(
        sd[(0, 0)] > sd[(1, 0)] * 10.0,
        "uncertainty should be ordered by evidence"
    );
}

// ---------------------------------------------------------------------------
// Per-row prior
// ---------------------------------------------------------------------------

use nalgebra::DVector;

fn rows(v: &[f32]) -> DVector<f32> {
    DVector::from_column_slice(v)
}

/// With a row prior and no data, the posterior mean of row `d` is `a0[d] / b0[d]`
/// in every column: `reset_stat` filled each row with its own hyper-parameters.
#[test]
fn row_prior_reset_fills_each_row_with_its_own_hyperparameter() {
    let a0 = rows(&[1.0, 2.0, 3.0]);
    let b0 = rows(&[4.0, 5.0, 6.0]);
    let mut p = GammaMatrix::with_row_prior((3, 2), &a0, &b0);
    p.update_stat(&DMatrix::zeros(3, 2), &DMatrix::zeros(3, 2));
    p.calibrate();
    let m = p.posterior_mean();
    for d in 0..3 {
        for c in 0..2 {
            let want = a0[d] / b0[d];
            assert!(
                (m[(d, c)] - want).abs() < 1e-6,
                "row {d} col {c}: mean {} != {want}",
                m[(d, c)]
            );
        }
    }
}

/// A row prior whose rows are all equal is the scalar prior, bit for bit, on all
/// four posterior planes.
#[test]
fn row_prior_with_equal_rows_matches_scalar_path_exactly() {
    let a = DMatrix::from_row_slice(3, 2, &[1.0, 5.0, 0.0, 2.0, 7.5, 3.0]);
    let b = DMatrix::from_row_slice(3, 2, &[2.0, 4.0, 1.0, 1.5, 6.0, 2.5]);
    let mut scalar = GammaMatrix::new((3, 2), 0.5, 2.0);
    let mut row = GammaMatrix::with_row_prior((3, 2), &rows(&[0.5; 3]), &rows(&[2.0; 3]));
    scalar.update_stat(&a, &b);
    row.update_stat(&a, &b);
    scalar.calibrate();
    row.calibrate();
    assert_eq!(scalar.posterior_mean(), row.posterior_mean());
    assert_eq!(scalar.posterior_sd(), row.posterior_sd());
    assert_eq!(scalar.posterior_log_mean(), row.posterior_log_mean());
    assert_eq!(scalar.posterior_log_sd(), row.posterior_log_sd());
}

/// The row prior is exactly the hand-folded prior: adding `a0[d]`/`b0[d]` to the
/// statistics of a zero-prior matrix gives the same posterior.
#[test]
fn row_prior_update_stat_equals_hand_folded_prior() {
    let a = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 10.0, 4.0]);
    let b = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 8.0, 3.0]);
    let a0 = rows(&[0.7, 9.0]);
    let b0 = rows(&[0.7, 9.0]);
    let mut folded = GammaMatrix::new((2, 2), 0.0, 0.0);
    let mut a_f = a.clone();
    let mut b_f = b.clone();
    for d in 0..2 {
        a_f.row_mut(d).add_scalar_mut(a0[d]);
        b_f.row_mut(d).add_scalar_mut(b0[d]);
    }
    folded.update_stat(&a_f, &b_f);
    folded.calibrate();
    let mut row = GammaMatrix::with_row_prior((2, 2), &a0, &b0);
    row.update_stat(&a, &b);
    row.calibrate();
    for (x, y) in folded
        .posterior_mean()
        .iter()
        .zip(row.posterior_mean().iter())
    {
        assert!((x - y).abs() < 1e-6, "mean {x} != {y}");
    }
    for (x, y) in folded
        .posterior_log_mean()
        .iter()
        .zip(row.posterior_log_mean().iter())
    {
        assert!((x - y).abs() < 1e-6, "log mean {x} != {y}");
    }
}

/// `evidence_mean` strips the row's own prior, and `has_data_support` compares
/// against the row's own `a0`.
#[test]
fn evidence_mean_subtracts_the_row_prior() {
    let a = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 10.0, 4.0]);
    let b = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 8.0, 3.0]);
    let mut p = GammaMatrix::with_row_prior((2, 2), &rows(&[1.0, 2.0]), &rows(&[3.0, 4.0]));
    p.update_stat(&a, &b);
    for d in 0..2 {
        for c in 0..2 {
            let want = if a[(d, c)] > 0.0 {
                a[(d, c)] / b[(d, c)]
            } else {
                0.0
            };
            assert!((p.evidence_mean(d, c) - want).abs() < 1e-6);
            assert_eq!(p.has_data_support(d, c), a[(d, c)] > 0.0);
        }
    }
}

/// Row-stacking gene blocks concatenates their row priors, so `evidence_mean` on
/// the stacked matrix equals the per-block value at the shifted row.
#[test]
fn vconcat_concatenates_row_priors() {
    let mut top = GammaMatrix::with_row_prior((2, 2), &rows(&[1.0, 2.0]), &rows(&[3.0, 4.0]));
    let mut bottom = GammaMatrix::with_row_prior((1, 2), &rows(&[5.0]), &rows(&[6.0]));
    top.update_stat(
        &DMatrix::from_element(2, 2, 10.0),
        &DMatrix::from_element(2, 2, 2.0),
    );
    bottom.update_stat(
        &DMatrix::from_element(1, 2, 7.0),
        &DMatrix::from_element(1, 2, 1.0),
    );
    let want_bottom = bottom.evidence_mean(0, 1);
    let want_top = top.evidence_mean(1, 0);
    let stacked = GammaMatrix::vconcat(vec![top, bottom], true);
    let (a0, b0) = stacked.row_prior().expect("row prior carried through");
    assert_eq!(a0, &rows(&[1.0, 2.0, 5.0]));
    assert_eq!(b0, &rows(&[3.0, 4.0, 6.0]));
    assert!((stacked.evidence_mean(2, 1) - want_bottom).abs() < 1e-6);
    assert!((stacked.evidence_mean(1, 0) - want_top).abs() < 1e-6);
}

#[test]
#[should_panic]
fn vconcat_scalar_blocks_with_different_hyperparameters_panics() {
    let a = GammaMatrix::new((1, 1), 1.0, 1.0);
    let b = GammaMatrix::new((1, 1), 2.0, 1.0);
    let _ = GammaMatrix::vconcat(vec![a, b], true);
}

#[test]
#[should_panic]
fn vconcat_mixed_scalar_and_row_prior_panics() {
    let a = GammaMatrix::new((1, 1), 1.0, 1.0);
    let b = GammaMatrix::with_row_prior((1, 1), &rows(&[1.0]), &rows(&[1.0]));
    let _ = GammaMatrix::vconcat(vec![a, b], true);
}

/// `set_row_prior` does not rewrite statistics already accumulated; it applies
/// at the next `update_stat` (which resets).
#[test]
fn set_row_prior_takes_effect_at_next_reset_only() {
    let a = DMatrix::from_element(2, 1, 4.0);
    let b = DMatrix::from_element(2, 1, 2.0);
    let mut p = GammaMatrix::with_row_prior((2, 1), &rows(&[1.0, 1.0]), &rows(&[1.0, 1.0]));
    p.update_stat(&a, &b);
    p.calibrate();
    let before = p.posterior_mean().clone();
    p.set_row_prior(&rows(&[10.0, 10.0]), &rows(&[10.0, 10.0]));
    p.calibrate();
    assert_eq!(p.posterior_mean(), &before);
    p.update_stat(&a, &b);
    p.calibrate();
    let want = (10.0 + 4.0) / (10.0 + 2.0);
    for m in p.posterior_mean().iter() {
        assert!((m - want).abs() < 1e-6, "mean {m} != {want}");
    }
}

#[test]
#[should_panic]
fn set_row_prior_wrong_length_panics() {
    let mut p = GammaMatrix::new((2, 1), 1.0, 1.0);
    p.set_row_prior(&rows(&[1.0]), &rows(&[1.0]));
}
