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
