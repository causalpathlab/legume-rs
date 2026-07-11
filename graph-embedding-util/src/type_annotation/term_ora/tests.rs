//! Unit tests for the firm term-ORA internals. Kept in a sibling file; `use super::*`
//! still reaches the private items since this is a child module of `term_ora`.

use super::*;

/// `[g × 2]` row-major embedding from per-gene rows.
fn emb(rows: &[[f32; 2]]) -> Vec<f32> {
    rows.iter().flat_map(|r| r.iter().copied()).collect()
}

/// Unweighted marker list (IDF off).
fn markers(ids: &[u32]) -> Vec<(u32, f32)> {
    ids.iter().map(|&g| (g, 1.0)).collect()
}

#[track_caller]
fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{actual:?} != {expected:?}");
    }
}

#[test]
fn centroid_is_the_weighted_mean_when_every_marker_is_live() {
    // Two live markers at (2,0) and (0,2) → mean (1,1).
    let e = emb(&[[2.0, 0.0], [0.0, 2.0]]);
    let (c, n_live) = term_centroids(&e, &[markers(&[0, 1])], 2);
    assert_close(&c, &[1.0, 1.0]);
    assert_eq!(n_live, vec![2]);
}

#[test]
fn dead_markers_do_not_shrink_the_centroid() {
    // One live marker at (3,4) plus three all-zero rows. Counting the dead rows in
    // `wsum` would divide by 4 and leave a centroid of norm 1.25 — a short vector that
    // `assign_nearest` treats as a near-origin magnet. It must stay at the live marker.
    let e = emb(&[[3.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    let (c, n_live) = term_centroids(&e, &[markers(&[0, 1, 2, 3])], 2);
    assert_close(&c, &[3.0, 4.0]);
    assert_eq!(n_live, vec![1]);
}

#[test]
fn dead_marker_fraction_does_not_bias_one_type_against_another() {
    // Two types with the SAME live prototype, differing only in how many dead markers
    // they drag along. Their centroids must come out identical — otherwise
    // nearest-centroid assignment is decided by panel decay, not by biology.
    let e = emb(&[
        [1.0, 0.0], // type A live
        [1.0, 0.0], // type B live
        [0.0, 0.0], // dead
        [0.0, 0.0], // dead
        [0.0, 0.0], // dead
    ]);
    let (c, n_live) = term_centroids(&e, &[markers(&[0]), markers(&[1, 2, 3, 4])], 2);
    assert_close(&c, &[1.0, 0.0, 1.0, 0.0]);
    assert_eq!(n_live, vec![1, 1]);
}

#[test]
fn all_dead_markers_leave_the_centroid_at_the_origin() {
    // The limiting case must still land on zero, so the `assign_nearest` guard keeps
    // excluding it rather than letting it become a magnet.
    let e = emb(&[[0.0, 0.0], [0.0, 0.0]]);
    let (c, n_live) = term_centroids(&e, &[markers(&[0, 1])], 2);
    assert_close(&c, &[0.0, 0.0]);
    assert_eq!(n_live, vec![0]);
}

#[test]
fn idf_weights_are_honoured_among_the_live_markers() {
    // Weighted mean over live rows only: (3·(2,0) + 1·(0,2)) / 4 = (1.5, 0.5). The dead
    // row's weight of 9 must not reach the denominator.
    let e = emb(&[[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]);
    let (c, n_live) = term_centroids(&e, &[vec![(0, 3.0), (1, 1.0), (2, 9.0)]], 2);
    assert_close(&c, &[1.5, 0.5]);
    assert_eq!(n_live, vec![2]);
}

#[test]
fn out_of_range_marker_index_is_skipped() {
    let e = emb(&[[2.0, 2.0]]);
    let (c, n_live) = term_centroids(&e, &[markers(&[0, 7])], 2);
    assert_close(&c, &[2.0, 2.0]);
    assert_eq!(n_live, vec![1]);
}
