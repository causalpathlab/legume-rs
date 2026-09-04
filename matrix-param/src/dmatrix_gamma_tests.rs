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

//////////////////////////////////////
// Seeded posterior draws (jitter) //
//////////////////////////////////////

/// A `[rows × cols]` Gamma parameter with every entry at shape `a`, rate `b`.
fn uniform_param(rows: usize, cols: usize, a: f32, b: f32) -> GammaMatrix {
    let mut p = GammaMatrix::new((rows, cols), 0.0, 0.0);
    p.update_stat(
        &DMatrix::from_element(rows, cols, a),
        &DMatrix::from_element(rows, cols, b),
    );
    p.calibrate();
    p
}

fn max_rel_diff(x: &DMatrix<f32>, y: &DMatrix<f32>) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6))
        .fold(0.0, f32::max)
}

#[test]
fn seeded_sample_is_reproducible_and_seed_sensitive() {
    let p = uniform_param(40, 30, 5.0, 2.0);
    let a = p.posterior_sample_seeded(11).unwrap();
    let b = p.posterior_sample_seeded(11).unwrap();
    let c = p.posterior_sample_seeded(12).unwrap();
    assert_eq!(a, b, "same seed must give the same draw, bit for bit");
    assert!(
        max_rel_diff(&a, &c) > 1e-3,
        "different seeds must give different draws"
    );
}

/// The draw is seeded per fixed-width chunk of the element stream, so how rayon
/// splits the work cannot leak into the numbers. A thread-local `rand::rng()`
/// would fail this.
#[test]
fn seeded_sample_is_independent_of_the_thread_count() {
    let p = uniform_param(64, 80, 3.0, 1.5);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| p.posterior_sample_seeded(5).unwrap());
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| p.posterior_sample_seeded(5).unwrap());
    assert_eq!(one, four);
}

/// `mean = a/b`, so `(mean, a)` determines `b`; dropping the rate plane costs
/// nothing but the rounding of one division, and a released parameter must draw
/// the same sample as a retained one at the same seed.
#[test]
fn a_released_rate_reconstructs_the_same_draw() {
    let full = uniform_param(30, 20, 7.0, 3.0);
    let mut shape_only = full.clone();
    shape_only.release_rate_stat();
    assert!(shape_only.has_shape_stat(), "the shape plane must survive");
    let a = full.posterior_sample_seeded(9).unwrap();
    let b = shape_only.posterior_sample_seeded(9).unwrap();
    assert!(
        max_rel_diff(&a, &b) < 1e-5,
        "reconstructed rate must reproduce the draw: max rel diff {}",
        max_rel_diff(&a, &b)
    );
}

/// At large shape the Gamma is tight, so the sample mean over many entries pins
/// the posterior mean to a fraction of a percent.
#[test]
fn seeded_sample_mean_tracks_the_posterior_mean() {
    let (a, b) = (400.0f32, 4.0f32);
    let p = uniform_param(200, 50, a, b);
    let s = p.posterior_sample_seeded(1).unwrap();
    let mean = s.iter().sum::<f32>() / (s.len() as f32);
    let want = a / b;
    assert!(
        (mean - want).abs() < 0.5,
        "sample mean {mean} vs posterior mean {want}"
    );
}

/// After `release_stats` there is nothing to draw from; that must be an error,
/// not a silent prior draw or a panic on an empty vector.
#[test]
fn a_fully_released_parameter_refuses_to_sample() {
    let mut p = uniform_param(4, 4, 2.0, 1.0);
    p.release_stats();
    assert!(!p.has_shape_stat());
    assert!(p.posterior_sample_seeded(0).is_err());
}
