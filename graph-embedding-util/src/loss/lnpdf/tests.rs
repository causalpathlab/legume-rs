//! The profile likelihood's defining properties, checked against closed forms
//! rather than against a second implementation.

use super::*;

fn dev_side(e: &[f32], b: &[f32], h: usize) -> (Vec<f32>, Vec<f32>, usize) {
    (e.to_vec(), b.to_vec(), h)
}

/// `ℓ_p = Σ_pos n·s_o − T·ln Σ_partition exp(s_o)`, computed by hand at `h = 1`
/// where every score is a single product.
#[test]
fn matches_the_closed_form_at_h_one() {
    let (e, b, h) = dev_side(&[1.0, 2.0, -1.0], &[0.5, -0.5, 0.0], 1);
    let side = FrozenSide { e: &e, b: &b, h };
    let pos = [(0u32, 3.0f32), (2, 1.0)];
    let partition = [0u32, 1, 2];
    let node = NodeTerm::new(&pos, &partition, 1.0);

    let a = 0.7f64;
    let s: Vec<f64> = (0..3)
        .map(|o| a * f64::from(e[o]) + f64::from(b[o]))
        .collect();
    let data = 3.0 * s[0] + 1.0 * s[2];
    let total = 4.0f64;
    let lse = s.iter().map(|x| x.exp()).sum::<f64>().ln();
    let want = (data - total * lse) as f32;

    let got = multinomial_ll(&[a as f32], &node, &side);
    assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
}

/// The likelihood is linear in the counts: both the data term and `T` scale, so
/// scaling every observed count by `c` scales the whole thing by `c`.
#[test]
fn scales_linearly_in_the_counts() {
    let (e, b, h) = dev_side(&[1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[0.1, -0.2, 0.3], 2);
    let side = FrozenSide { e: &e, b: &b, h };
    let partition = [0u32, 1, 2];
    let x = [0.3f32, -0.4];

    let pos1 = [(0u32, 2.0f32), (2, 5.0)];
    let pos5 = [(0u32, 10.0f32), (2, 25.0)];
    let one = multinomial_ll(&x, &NodeTerm::new(&pos1, &partition, 1.0), &side);
    let five = multinomial_ll(&x, &NodeTerm::new(&pos5, &partition, 1.0), &side);
    assert!(
        (five - 5.0 * one).abs() < 1e-3 * five.abs().max(1.0),
        "5x counts: {five} vs 5 * {one}"
    );
}

/// Shifting every frozen bias by one constant moves the data term by `T·k` and
/// the log-normalizer by `k`, which cancel exactly. This is the property that
/// makes the score depth-invariant — the reason the profile form is used at all.
#[test]
fn a_constant_bias_shift_cancels() {
    let e = vec![1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5];
    let b0 = vec![0.1f32, -0.2, 0.3];
    let k = 1.75f32;
    let b1: Vec<f32> = b0.iter().map(|v| v + k).collect();
    let pos = [(0u32, 2.0f32), (2, 5.0)];
    let partition = [0u32, 1, 2];
    let node = NodeTerm::new(&pos, &partition, 1.0);
    let x = [0.3f32, -0.4];

    let lo = multinomial_ll(
        &x,
        &node,
        &FrozenSide {
            e: &e,
            b: &b0,
            h: 2,
        },
    );
    let hi = multinomial_ll(
        &x,
        &node,
        &FrozenSide {
            e: &e,
            b: &b1,
            h: 2,
        },
    );
    assert!(
        (lo - hi).abs() < 1e-3,
        "shift changed the score: {lo} vs {hi}"
    );
}

/// An anchor with no observed counts has a flat likelihood — reported as an
/// exact 0 rather than a `NaN` from `0 * ln 0`.
#[test]
fn no_counts_is_flat_not_nan() {
    let (e, b, h) = dev_side(&[1.0, 2.0], &[0.0, 0.0], 1);
    let side = FrozenSide { e: &e, b: &b, h };
    let partition = [0u32, 1];
    let got = multinomial_ll(&[0.5], &NodeTerm::new(&[], &partition, 1.0), &side);
    assert_eq!(got, 0.0);
}

/// Scores are clamped before `exp`, so a loading large enough to overflow `f32`
/// exp still returns a finite number.
#[test]
fn a_saturating_loading_stays_finite() {
    let (e, b, h) = dev_side(&[1.0, 2.0], &[0.0, 0.0], 1);
    let side = FrozenSide { e: &e, b: &b, h };
    let partition = [0u32, 1];
    let pos = [(0u32, 1.0f32)];
    let got = multinomial_ll(&[1e6], &NodeTerm::new(&pos, &partition, 1.0), &side);
    assert!(got.is_finite(), "saturating loading gave {got}");
}
