use super::*;
use crate::traits::{RandomizedAlgs, SampleOps};

/// A planted rank-one spike over unit noise, well above the noise edge:
/// the randomised SVD must recover its singular value and direction.
fn spiked(n: usize, p: usize, strength: f32, seed: u64) -> (DMatrix<f32>, DVector<f32>) {
    let noise = DMatrix::<f32>::rnorm_seeded(n, p, seed);
    let mut u = DVector::<f32>::from_fn(n, |i, _| if i % 2 == 0 { 1.0 } else { -1.0 });
    u /= u.norm();
    let mut v = DVector::<f32>::from_fn(p, |j, _| if j < p / 3 { 1.0 } else { 0.0 });
    v /= v.norm();
    (noise + strength * (&u * v.transpose()), v)
}

#[test]
fn rsvd_recovers_a_planted_spike() {
    let (x, v) = spiked(120, 200, 40.0, 1);
    let full = x.clone().svd(false, true);
    let exact = full.singular_values[0];
    let exact_v = full.v_t.as_ref().unwrap().row(0).transpose();
    let (_, s, vv) = x.rsvd(2).unwrap();
    let s1 = s[0];
    assert!(
        (s1 - exact).abs() <= 0.02 * exact,
        "leading singular value {s1} vs exact {exact}"
    );
    let cos_exact = vv.column(0).dot(&exact_v).abs();
    assert!(
        cos_exact >= 0.99,
        "cosine with the exact leading vector {cos_exact}"
    );
    // and the exact vector itself is the plant up to noise tilt
    let cos_plant = exact_v.dot(&v).abs();
    assert!(cos_plant >= 0.85, "exact vs planted cosine {cos_plant}");
}

#[test]
fn rsvd_leading_value_of_noise_stays_at_the_edge() {
    let x = DMatrix::<f32>::rnorm_seeded(120, 200, 2);
    let exact = x.clone().svd(false, false).singular_values[0];
    let (_, s, _) = x.rsvd(2).unwrap();
    assert!(
        s[0] <= exact * 1.001,
        "randomised value cannot exceed the exact one"
    );
    assert!(
        s[0] >= 0.9 * exact,
        "randomised value {} far below exact {exact}",
        s[0]
    );
}
