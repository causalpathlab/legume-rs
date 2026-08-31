//! Tests for the latent-sharpness summary.

use super::{latent_sharpness, Mat};

#[test]
fn a_flat_latent_has_k_effective_topics() {
    let k = 8;
    let theta = Mat::from_element(10, k, 1.0 / k as f32);
    let (eff, mx) = latent_sharpness(&theta);
    assert!((eff - k as f32).abs() < 1e-3, "effective topics {eff}");
    assert!((mx - 1.0 / k as f32).abs() < 1e-6, "max weight {mx}");
}

#[test]
fn a_one_hot_latent_has_one_effective_topic() {
    let k = 8;
    let mut theta = Mat::zeros(10, k);
    for i in 0..10 {
        theta[(i, i % k)] = 1.0;
    }
    let (eff, mx) = latent_sharpness(&theta);
    assert!((eff - 1.0).abs() < 1e-6, "effective topics {eff}");
    assert!((mx - 1.0).abs() < 1e-6, "max weight {mx}");
}

/// Rows are renormalized, so unnormalized weights and proportions agree.
#[test]
fn rows_are_renormalized_before_the_entropy() {
    let theta = Mat::from_row_slice(2, 3, &[0.2, 0.3, 0.5, 0.2, 0.3, 0.5]);
    let scaled = &theta * 7.0;
    assert_eq!(latent_sharpness(&theta), latent_sharpness(&scaled));
}

#[test]
fn an_empty_latent_is_nan_not_zero() {
    let (eff, mx) = latent_sharpness(&Mat::zeros(0, 4));
    assert!(eff.is_nan() && mx.is_nan());
}

/// A diverged latent must report NaN, not `+inf`. `f32::max` drops a NaN operand,
/// so the guarded row sum has to check finiteness explicitly.
#[test]
fn a_non_finite_latent_is_nan_not_infinity() {
    let mut theta = Mat::from_element(3, 4, 0.25);
    theta[(1, 2)] = f32::NAN;
    let (eff, mx) = latent_sharpness(&theta);
    assert!(eff.is_nan(), "effective topics was {eff}");
    assert!(mx.is_nan(), "mean max theta was {mx}");
}

#[test]
fn zero_rows_survive_l2_normalization_untouched() {
    let mut m = Mat::from_row_slice(2, 2, &[3.0, 4.0, 0.0, 0.0]);
    super::l2_normalize_rows_inplace(&mut m);
    assert!((m.row(0).norm() - 1.0).abs() < 1e-6);
    assert!(m.row(1).iter().all(|&x| x == 0.0));
}
