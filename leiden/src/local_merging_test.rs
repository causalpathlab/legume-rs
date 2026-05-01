//! Edge-case tests for the cumulative-weight sampler used inside
//! `LocalMerging::run`. The sampler picks the smallest index `i` where
//! `cum[i] >= r`, given a non-decreasing cumulative array. The current
//! implementation expresses this as
//! `cum.partition_point(|&x| x < r)`; this file exercises that against a
//! reference Java-style binary search to lock in equivalence and document
//! the boundary behaviour.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Reference: the original Java-style binary search that the
/// `partition_point` call replaced. Kept here as the oracle.
///
/// **Caller invariant** (matches the algorithm's call site): `r` must be
/// strictly less than `cum[cum.len() - 1]`. The legacy loop reads past
/// the end of `cum` if `r` exceeds the total — `partition_point` is
/// strictly more robust in this regard.
fn legacy_search(cum: &[f64], r: f64) -> usize {
    let mut min_idx: isize = -1;
    let mut max_idx: isize = (cum.len() + 1) as isize;
    while min_idx < max_idx - 1 {
        let mid_idx = isize::midpoint(min_idx, max_idx);
        if cum[mid_idx as usize] >= r {
            max_idx = mid_idx;
        } else {
            min_idx = mid_idx;
        }
    }
    max_idx as usize
}

/// Current implementation: what `LocalMerging::run` actually uses.
fn current_search(cum: &[f64], r: f64) -> usize {
    cum.partition_point(|&x| x < r)
}

#[test]
fn r_below_all_picks_first() {
    let cum = vec![0.5, 1.0, 1.5];
    assert_eq!(current_search(&cum, 0.0), 0);
    assert_eq!(current_search(&cum, -1e9), 0);
}

#[test]
fn r_exactly_at_boundary_picks_that_index() {
    // partition_point(|&x| x < 1.0) on [0.5, 1.0, 1.5] → first idx where
    // x >= 1.0, which is 1. The legacy search agrees.
    let cum = vec![0.5, 1.0, 1.5];
    assert_eq!(current_search(&cum, 1.0), 1);
    assert_eq!(legacy_search(&cum, 1.0), 1);
}

#[test]
fn r_just_below_boundary_picks_that_index() {
    let cum = vec![0.5, 1.0, 1.5];
    assert_eq!(current_search(&cum, 0.999_999), 1);
}

#[test]
fn r_near_total_picks_last() {
    let cum = vec![0.5, 1.0, 1.5];
    assert_eq!(current_search(&cum, 1.499_999), 2);
    assert_eq!(current_search(&cum, 1.5), 2);
}

#[test]
fn single_element() {
    let cum = vec![1.0];
    assert_eq!(current_search(&cum, 0.0), 0);
    assert_eq!(current_search(&cum, 0.5), 0);
    assert_eq!(current_search(&cum, 1.0), 0);
    assert_eq!(legacy_search(&cum, 0.5), 0);
}

#[test]
fn flat_regions_pick_first_of_run() {
    // Constant cum (zero-mass clusters in a row) — partition_point
    // returns the first index where cum[i] >= r, which is the start of
    // the flat run. This matches the algorithm's intended semantics:
    // ties on a tied cum value all map to the same neighbor cluster,
    // so the run order doesn't perturb the sample.
    let cum = vec![0.5, 0.5, 0.5, 1.0];
    assert_eq!(current_search(&cum, 0.5), 0);
    assert_eq!(current_search(&cum, 0.5 + f64::EPSILON), 3);
}

#[test]
fn matches_legacy_on_handpicked_arrays() {
    // (cum, r-values) — every r is < cum[N-1] to honour the legacy
    // search's caller invariant.
    let cases: &[(&[f64], &[f64])] = &[
        (&[0.1, 0.3, 0.6, 1.0], &[0.0, 0.05, 0.5, 0.999]),
        (&[1.0], &[0.0, 0.5, 0.999_999]),
        (&[0.0, 0.0, 1.0], &[0.0, 0.5, 0.999]),
        (&[0.5, 0.5, 0.5, 0.5, 1.0], &[0.0, 0.5, 0.999]),
    ];
    for (cum, rs) in cases {
        for &r in *rs {
            assert_eq!(
                current_search(cum, r),
                legacy_search(cum, r),
                "diverged on cum={cum:?}, r={r}"
            );
        }
    }
}

#[test]
fn partition_point_handles_r_above_total_gracefully() {
    // `partition_point` returns `cum.len()` when no element satisfies
    // `cum[i] >= r`. The legacy loop would have read past the end. The
    // production call site keeps `r < total` so this case never occurs,
    // but the more robust handling is a small win.
    let cum = vec![0.5, 1.0, 1.5];
    assert_eq!(current_search(&cum, 2.0), 3);
    assert_eq!(current_search(&cum, f64::INFINITY), 3);
}

#[test]
fn defensive_clamp_keeps_index_in_bounds() {
    // Mirrors the production call site: `idx.min(len - 1)` after
    // partition_point. Even pathological `r >= total` values must not
    // produce an out-of-range index.
    fn clamped(cum: &[f64], r: f64) -> usize {
        cum.partition_point(|&x| x < r).min(cum.len() - 1)
    }
    let cum = vec![0.5, 1.0, 1.5];
    assert_eq!(clamped(&cum, 2.0), 2);
    assert_eq!(clamped(&cum, f64::INFINITY), 2);
}

#[test]
fn nan_r_is_handled_without_panicking() {
    // NaN comparisons are always false, so `partition_point(|&x| x < NaN)`
    // returns 0 (predicate false at every position). Combined with the
    // clamp, a stray NaN-r picks neighbour 0 (the original cluster `j`)
    // instead of panicking. Algorithm degrades gracefully rather than
    // crashing if any upstream computation goes off the rails.
    let cum = [0.5_f64, 1.0, 1.5];
    let r = f64::NAN;
    let idx = cum.partition_point(|&x| x < r).min(cum.len() - 1);
    assert_eq!(idx, 0);
}

#[test]
fn matches_legacy_on_random_monotone_arrays() {
    let mut rng = SmallRng::seed_from_u64(0xC0FF_EE42);
    for _ in 0..1000 {
        let n = rng.random_range(1..=32_usize);
        let mut cum = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        for _ in 0..n {
            acc += rng.random_range(0.0..1.0);
            cum.push(acc);
        }
        let total = *cum.last().unwrap();
        // Match the algorithm's call site: r = total * uniform.
        let r = total * rng.random_range(0.0..1.0);
        assert_eq!(
            current_search(&cum, r),
            legacy_search(&cum, r),
            "diverged on cum={cum:?}, r={r}"
        );
    }
}

#[test]
fn algorithm_call_site_invariant_holds() {
    // The caller in LocalMerging::run guarantees `r < total = cum[n-1]`
    // because `r = total * rng.random::<f64>()` and `random()` is in
    // [0, 1). This test verifies the search never returns an
    // out-of-range index under that invariant.
    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..1000 {
        let n = rng.random_range(1..=16_usize);
        let mut cum = Vec::with_capacity(n);
        let mut acc = 1e-12; // avoid zero total
        for _ in 0..n {
            acc += rng.random_range(1e-6..1.0);
            cum.push(acc);
        }
        let total = *cum.last().unwrap();
        let r = total * rng.random_range(0.0..1.0);
        let idx = current_search(&cum, r);
        assert!(idx < n, "idx {idx} out of range for n={n}");
    }
}
