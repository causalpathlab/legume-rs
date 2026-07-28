//! Each test plants a frozen side whose geometry is known by construction, so a
//! wrong number is unambiguous rather than merely surprising.

use super::*;

/// Build a `FrozenSide` view over row-major `e` with `h` columns.
fn side<'a>(e: &'a [f32], b: &'a [f32], h: usize) -> FrozenSide<'a> {
    FrozenSide { e, b, h }
}

/// `h` orthogonal directions, each used by an equal number of rows, centered at
/// the origin: full effective rank, no common mode, no collinearity.
#[test]
fn an_orthogonal_side_is_full_rank_and_uncorrelated() {
    const H: usize = 4;
    let n = 4 * H;
    let mut e = vec![0.0f32; n * H];
    for i in 0..n {
        // +1 and -1 on one axis each, so every column has zero mean.
        e[i * H + (i % H)] = if (i / H).is_multiple_of(2) { 1.0 } else { -1.0 };
    }
    let b = vec![0.0f32; n];
    let d = frozen_side_diag(&side(&e, &b, H));

    assert!(
        (d.eff_rank_raw - H as f32).abs() < 1e-3,
        "orthogonal side must use all {H} dims, got {}",
        d.eff_rank_raw
    );
    assert!(
        (d.eff_rank_centered - H as f32).abs() < 1e-3,
        "and the same after centering, got {}",
        d.eff_rank_centered
    );
    assert!(
        d.common_mode_cos < 1e-3,
        "zero-mean side has no common direction, got {}",
        d.common_mode_cos
    );
    assert!(
        d.max_abs_corr < 1e-3 && (d.max_vif - 1.0).abs() < 1e-3,
        "orthogonal dims: corr {} vif {}",
        d.max_abs_corr,
        d.max_vif
    );
}

/// Every row a multiple of ONE direction: a genuine rank-1 side, which centering
/// does NOT rescue. This is the case that must read differently from a common
/// mode (below), since the two have different causes and different fixes.
#[test]
fn a_rank_one_side_stays_rank_one_after_centering() {
    const H: usize = 4;
    let dir = [0.5f32, -0.5, 0.5, -0.5];
    let n = 20;
    let mut e = vec![0.0f32; n * H];
    for i in 0..n {
        // Scales straddling zero, so the low rank is not itself a mean offset.
        let s = (i as f32) - (n as f32) / 2.0;
        for k in 0..H {
            e[i * H + k] = s * dir[k];
        }
    }
    let b = vec![0.0f32; n];
    let d = frozen_side_diag(&side(&e, &b, H));

    assert!(
        d.eff_rank_raw < 1.05,
        "one direction ⇒ effective rank 1, got {}",
        d.eff_rank_raw
    );
    assert!(
        d.eff_rank_centered < 1.05,
        "centering must NOT rescue a genuine rank-1 side, got {}",
        d.eff_rank_centered
    );
    // Perfectly collinear dims: correlation saturates and VIF blows up. Both are
    // the honest report, and `max_vif` must not come back NaN.
    assert!(
        d.max_abs_corr > 0.99,
        "collinear dims, got {}",
        d.max_abs_corr
    );
    assert!(
        d.max_vif > 5.0,
        "collinear dims must exceed the trust threshold, got {}",
        d.max_vif
    );
}

/// A full-rank cloud shifted far off the origin: the RAW Gram looks near-rank-1
/// because the offset dominates, but centering recovers the true rank. This is
/// the discriminator the struct exists for — on real fits `common_mode_cos`
/// reaches 0.96 with the centered rank several times the raw one.
#[test]
fn a_common_mode_depresses_the_raw_rank_but_not_the_centered_one() {
    const H: usize = 4;
    let n = 4 * H;
    let offset = 25.0f32; // ≫ the unit-scale spread below
    let mut e = vec![0.0f32; n * H];
    for i in 0..n {
        for k in 0..H {
            e[i * H + k] = offset;
        }
        e[i * H + (i % H)] += if (i / H).is_multiple_of(2) { 1.0 } else { -1.0 };
    }
    let b = vec![0.0f32; n];
    let d = frozen_side_diag(&side(&e, &b, H));

    assert!(
        d.common_mode_cos > 0.99,
        "every row points along the shared offset, got {}",
        d.common_mode_cos
    );
    assert!(
        d.eff_rank_raw < 1.5,
        "the offset should dominate the uncentered Gram, got {}",
        d.eff_rank_raw
    );
    assert!(
        d.eff_rank_centered > 3.5,
        "centering must recover the real rank ({H}), got {}",
        d.eff_rank_centered
    );
    assert!(
        d.eff_rank_centered > 2.0 * d.eff_rank_raw,
        "raw {} vs centered {} must read as a MEAN OFFSET, not a collapse",
        d.eff_rank_raw,
        d.eff_rank_centered
    );
}

/// Degenerate inputs report zeros, never NaN — these values are serialized into
/// `{out}.posterior_hyper.json`, and a NaN there reads as "not measured".
#[test]
fn degenerate_sides_report_zero_not_nan() {
    let empty = frozen_side_diag(&side(&[], &[], 4));
    assert_eq!(empty.eff_rank_raw, 0.0);
    assert_eq!(empty.common_mode_cos, 0.0);

    // Every row identical and every column constant: no variance anywhere.
    const H: usize = 3;
    let e = vec![2.0f32; 6 * H];
    let b = vec![0.0f32; 6];
    let d = frozen_side_diag(&side(&e, &b, H));
    assert!(
        d.eff_rank_raw.is_finite() && d.eff_rank_centered.is_finite(),
        "constant side must not produce NaN ranks: {d:?}"
    );
    assert_eq!(
        d.eff_rank_centered, 0.0,
        "a constant side has no centered variance at all"
    );
    assert!(
        d.max_abs_corr.is_finite(),
        "constant dims correlate with nothing: {d:?}"
    );
    assert!(
        (d.common_mode_cos - 1.0).abs() < 1e-5,
        "identical rows all point the same way, got {}",
        d.common_mode_cos
    );
}
