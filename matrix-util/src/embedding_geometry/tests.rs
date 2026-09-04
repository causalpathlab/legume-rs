//! Each test plants a table whose geometry is known by construction, so a wrong
//! number is unambiguous rather than merely surprising.

use super::*;
use nalgebra::DMatrix;

/// `h` orthogonal directions, each used by an equal number of rows, centered at
/// the origin: full effective rank, no common mode, no collinearity.
fn balanced_axes(h: usize, per_axis: usize) -> DMatrix<f32> {
    let n = per_axis * h;
    let mut e = vec![0.0f32; n * h];
    for i in 0..n {
        // +1 and -1 on one axis each, so every column has zero mean.
        e[i * h + (i % h)] = if (i / h).is_multiple_of(2) { 1.0 } else { -1.0 };
    }
    DMatrix::from_row_slice(n, h, &e)
}

#[test]
fn an_orthogonal_table_is_full_rank_and_uncorrelated() {
    const H: usize = 4;
    let d = embedding_geometry(&balanced_axes(H, 4));

    assert!(
        (d.eff_rank_raw - H as f32).abs() < 1e-3,
        "orthogonal table must use all {H} dims, got {}",
        d.eff_rank_raw
    );
    assert!(
        (d.eff_rank_centered - H as f32).abs() < 1e-3,
        "and the same after centering, got {}",
        d.eff_rank_centered
    );
    assert!(
        d.common_mode_cos < 1e-3,
        "zero-mean table has no common direction, got {}",
        d.common_mode_cos
    );
    assert!(
        d.max_abs_corr < 1e-3 && (d.max_vif - 1.0).abs() < 1e-3,
        "orthogonal dims: corr {} vif {}",
        d.max_abs_corr,
        d.max_vif
    );
}

/// Every row a multiple of ONE direction: a genuine rank-1 table, which
/// centering does NOT rescue. This is the case that must read differently from
/// a common mode (below), since the two have different causes and different
/// fixes.
#[test]
fn a_rank_one_table_stays_rank_one_after_centering() {
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
    let d = embedding_geometry(&DMatrix::from_row_slice(n, H, &e));

    assert!(
        d.eff_rank_raw < 1.05,
        "one direction ⇒ effective rank 1, got {}",
        d.eff_rank_raw
    );
    assert!(
        d.eff_rank_centered < 1.05,
        "centering must NOT rescue a genuine rank-1 table, got {}",
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
/// the discriminator the struct exists for.
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
    let d = embedding_geometry(&DMatrix::from_row_slice(n, H, &e));

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
    // The pairwise cosine agrees: rows sharing a large offset all point the
    // same way.
    assert!(
        d.mean_pairwise_cos > 0.99,
        "shared offset ⇒ every pair nearly parallel, got {}",
        d.mean_pairwise_cos
    );
}

/// Degenerate inputs report zeros, never NaN — a NaN in a report reads as
/// "not measured".
#[test]
fn degenerate_tables_report_zero_not_nan() {
    let empty = embedding_geometry(&DMatrix::<f32>::zeros(0, 4));
    assert_eq!(empty.eff_rank_raw, 0.0);
    assert_eq!(empty.common_mode_cos, 0.0);
    assert_eq!(empty.mean_pairwise_cos, 0.0);

    // Every row identical and every column constant: no variance anywhere.
    const H: usize = 3;
    let d = embedding_geometry(&DMatrix::from_element(6, H, 2.0f32));
    assert!(
        d.eff_rank_raw.is_finite() && d.eff_rank_centered.is_finite(),
        "constant table must not produce NaN ranks: {d:?}"
    );
    assert_eq!(
        d.eff_rank_centered, 0.0,
        "a constant table has no centered variance at all"
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
    assert!(
        (d.mean_pairwise_cos - 1.0).abs() < 1e-5,
        "identical rows are pairwise parallel, got {}",
        d.mean_pairwise_cos
    );
}

////////////////////////////////////
// Mean pairwise cosine, pinned  //
////////////////////////////////////

/// A balanced ± cloud has a SIGNED mean pairwise cosine of exactly `−1/(n−1)`,
/// not 0: the unit rows sum to zero, so `‖Σê‖² = 0` and the closed form gives
/// `(0 − n)/(n(n−1))`. Pinning the exact value, rather than "≈ 0", is what
/// catches an off-by-one in the pair count.
#[test]
fn balanced_axes_score_minus_one_over_n_minus_one() {
    const H: usize = 4;
    let e = balanced_axes(H, 4);
    let n = e.nrows() as f32;
    let d = embedding_geometry(&e);
    assert!(
        (d.mean_pairwise_cos + 1.0 / (n - 1.0)).abs() < 1e-5,
        "balanced ± rows: expected {}, got {}",
        -1.0 / (n - 1.0),
        d.mean_pairwise_cos
    );
}

/// The `O(n·h)` closed form must equal the explicit `O(n²·h)` double loop on a
/// table with no special structure.
#[test]
fn closed_form_pairwise_cosine_matches_brute_force() {
    const H: usize = 3;
    let n = 7;
    // Deterministic, non-degenerate, no symmetry to hide a bug behind.
    let e = DMatrix::<f32>::from_fn(n, H, |i, j| {
        ((i * 7 + j * 3) % 11) as f32 - 5.0 + 0.1 * (i as f32)
    });
    let d = embedding_geometry(&e);

    let unit: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let r: Vec<f64> = (0..H).map(|j| f64::from(e[(i, j)])).collect();
            let nrm = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            r.iter().map(|x| x / nrm).collect()
        })
        .collect();
    let mut acc = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                acc += (0..H).map(|k| unit[i][k] * unit[j][k]).sum::<f64>();
            }
        }
    }
    let brute = (acc / (n * (n - 1)) as f64) as f32;
    assert!(
        (d.mean_pairwise_cos - brute).abs() < 1e-5,
        "closed form {} vs brute force {}",
        d.mean_pairwise_cos,
        brute
    );
}

/// A zero row has no direction. It must be left out of the pair count, not
/// counted as a cosine of zero — otherwise the mean is silently diluted.
#[test]
fn zero_norm_rows_are_excluded_from_the_pairwise_mean() {
    const H: usize = 3;
    let live = DMatrix::<f32>::from_row_slice(3, H, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let with_zeros = DMatrix::<f32>::from_row_slice(
        5,
        H,
        &[
            1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0,
        ],
    );
    let a = embedding_geometry(&live).mean_pairwise_cos;
    let b = embedding_geometry(&with_zeros).mean_pairwise_cos;
    assert!(
        (a - b).abs() < 1e-6,
        "zero rows must not change the pairwise mean: {a} vs {b}"
    );
}

///////////////////////////////////
// Participation ratio, pinned  //
///////////////////////////////////

#[test]
fn participation_ratio_is_h_for_identity_and_one_for_rank_one() {
    let eye = DMatrix::<f64>::identity(5, 5);
    assert!((participation_ratio(&eye) - 5.0).abs() < 1e-6);

    // Rank-one PSD: `v vᵀ` has one nonzero eigenvalue.
    let v = nalgebra::DVector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
    let r1 = &v * v.transpose();
    assert!((participation_ratio(&r1) - 1.0).abs() < 1e-6);

    // Zero matrix has no spectrum to report: 0, not NaN.
    assert_eq!(participation_ratio(&DMatrix::<f64>::zeros(3, 3)), 0.0);
}
