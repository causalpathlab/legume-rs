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

//////////////////////////////////////
// the cluster × term ORA core //
//////////////////////////////////////

fn cfg_with(n_perm: usize) -> TermOraConfig {
    TermOraConfig {
        n_perm,
        seed: 7,
        ..Default::default()
    }
}

/// Drive the ORA core the way the reported run does.
fn ora_of(
    assign: &[usize],
    community: &[usize],
    n_comm: usize,
    c: usize,
    n_perm: usize,
) -> OraResult {
    let lnfact = ln_factorials(assign.len());
    cluster_term_ora(
        assign,
        community,
        n_comm,
        c,
        &lnfact,
        Want::Report,
        &cfg_with(n_perm),
    )
}

/// A partition of 4-cell clusters, hand-checked against the hypergeometric. Cluster 0 is 4/4
/// type-0 out of 8 cells with 4 type-0 in total, so `P(X≥4) = 1/C(8,4) = 1/70`.
#[test]
fn cluster_ora_matches_the_hypergeometric_by_hand() {
    let assign = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let community = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let ora = ora_of(&assign, &community, 2, 2, 0);
    let expected = -(1.0f32 / 70.0).ln();
    assert!(
        (ora.stat[0] - expected).abs() < 1e-4,
        "cluster 0 term 0: {} vs {expected}",
        ora.stat[0]
    );
}

/// The margins are counted over the cells, so an `unassigned` cell is out of the population
/// entirely — it must not swell `N` and dilute everyone else's enrichment.
#[test]
fn unassigned_cells_leave_the_population() {
    let community = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let clean = vec![0, 0, 0, 0, 1, 1, 1, 1];
    // Same cells, but two of cluster 1's are pruned. Cluster 0 is still 4-of-4 out of the 6
    // cells that remain, so its p must be the 6-cell one (1/C(6,4) = 1/15), not the 8-cell one.
    let pruned = vec![0, 0, 0, 0, 1, 1, UNASSIGNED, UNASSIGNED];
    let a = ora_of(&clean, &community, 2, 2, 0);
    let b = ora_of(&pruned, &community, 2, 2, 0);
    assert!((a.stat[0] - -(1.0f32 / 70.0).ln()).abs() < 1e-4);
    assert!(
        (b.stat[0] - -(1.0f32 / 15.0).ln()).abs() < 1e-4,
        "pruned cells must leave the hypergeometric population: {}",
        b.stat[0]
    );
}

/// A partition with no structure in it must produce no calls — the permutation null has to be
/// able to say "nothing here", or it is not a test.
#[test]
fn no_signal_yields_no_calls() {
    let n = 60;
    // Four contiguous clusters of 15; the labels alternate, so every cluster is 50/50 and no
    // term is over-represented anywhere. (`i % 4` for the cluster would *not* work — it lines up
    // with `i % 2` and hands cluster 0 every type-0 cell.)
    let assign: Vec<usize> = (0..n).map(|i| i % 2).collect();
    let community: Vec<usize> = (0..n).map(|i| i / 15).collect();
    let ora = ora_of(&assign, &community, 4, 2, 200);
    let calls = cluster_calls(&ora, 4, 2, 0.1);
    assert!(
        calls.iter().all(|&t| t == UNASSIGNED),
        "called {calls:?} on labels with no cluster structure"
    );
}

/// The null pool is `n_perm × n_comm`, so a partition that is already fine buys its own
/// precision and needs fewer draws. It never exceeds what the caller asked for, so at the
/// resolutions worth using this is a no-op.
#[test]
fn capped_n_perm_bounds_the_pool_without_touching_sane_partitions() {
    assert_eq!(capped_n_perm(500, 35), 500, "few clusters ⇒ the ask stands");
    assert_eq!(capped_n_perm(500, 137), 500);
    // A runaway partition (--resolution 8 has produced 1713 communities on 15k cells).
    assert!(capped_n_perm(500, 1713) < 500);
    assert!(capped_n_perm(500, 1713) * 1713 >= MAX_NULL_POOL);
    assert_eq!(capped_n_perm(500, 10_000_000), 1, "never below one draw");
    assert_eq!(capped_n_perm(0, 137), 0, "opting out stays opted out");
}
