//! Tests for the per-epoch Poisson draw behind `MaskedTrainOpts::poisson_thin`.

use super::poisson_draw;
use super::Mat;

/// Zero rates must draw zero — a gene absent from a pseudobulk cannot acquire a
/// count — and every draw must be a non-negative integer.
#[test]
fn zero_rates_stay_zero_and_draws_are_counts() {
    let mut rates = Mat::zeros(50, 40);
    for i in 0..50 {
        for j in 0..40 {
            rates[(i, j)] = if (i + j) % 3 == 0 { 0.0 } else { 2.5 };
        }
    }
    let x = poisson_draw(&rates, 7);
    assert_eq!(x.shape(), rates.shape());
    for i in 0..50 {
        for j in 0..40 {
            let v = x[(i, j)];
            if rates[(i, j)] == 0.0 {
                assert_eq!(v, 0.0, "({i},{j}) drew {v} from a zero rate");
            }
            assert!(
                v >= 0.0 && v.fract() == 0.0,
                "({i},{j}) = {v} is not a count"
            );
        }
    }
}

/// The draw is unbiased: over enough entries its mean recovers the rate.
#[test]
fn draws_are_unbiased_around_the_rate() {
    for &rate in &[0.05f32, 1.0, 7.5] {
        let rates = Mat::from_element(400, 100, rate);
        let x = poisson_draw(&rates, 7);
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        // 40k draws; the standard error is sqrt(rate / 40k).
        let se = (rate / 40_000.0).sqrt();
        assert!(
            (mean - rate).abs() < 5.0 * se + 1e-3,
            "rate {rate}: mean {mean} is off by more than 5 SE ({se})"
        );
    }
}

/// Non-finite rates are treated as absent rather than panicking inside
/// `Poisson::new`.
#[test]
fn non_finite_rates_draw_zero() {
    let mut rates = Mat::from_element(4, 4, 1.0);
    rates[(0, 0)] = f32::NAN;
    rates[(1, 1)] = f32::INFINITY;
    let x = poisson_draw(&rates, 7);
    assert_eq!(x[(0, 0)], 0.0);
    assert_eq!(x[(1, 1)], 0.0);
}

/// The draw must be a function of its seed alone — not of the thread count,
/// and not of run-to-run OS entropy. This is the property that lets a
/// `--poisson-thin` result be replicated or bisected at all.
#[test]
fn the_draw_is_reproducible_from_its_seed() {
    let mut rates = Mat::zeros(200, 60);
    for i in 0..200 {
        for j in 0..60 {
            rates[(i, j)] = 0.5 + (i % 7) as f32;
        }
    }
    let a = poisson_draw(&rates, 11);
    let b = poisson_draw(&rates, 11);
    assert_eq!(a, b, "same seed must give the same draw");
    let c = poisson_draw(&rates, 12);
    assert_ne!(a, c, "a different seed must give a different draw");
}
