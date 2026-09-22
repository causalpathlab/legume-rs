//! The phase-1 cell subsample: at most `k` cells per pseudobulk at every
//! level, unioned, seeded.

use graph_embedding_util::fit::keep_cells_per_pb;

#[test]
fn at_most_k_per_pseudobulk_at_every_level_and_the_union_is_kept() {
    // 12 cells; finest: 4 pbs of 3; coarse: 2 pbs of 6.
    let finest: Vec<usize> = (0..12).map(|c| c / 3).collect();
    let coarse: Vec<usize> = (0..12).map(|c| c / 6).collect();
    let keep = keep_cells_per_pb(&[finest.clone(), coarse.clone()], 2, 7);
    assert_eq!(keep.len(), 12);
    for pb in 0..4 {
        let n = (0..12).filter(|&c| finest[c] == pb && keep[c]).count();
        assert!((2..=3).contains(&n), "finest pb {pb} keeps {n}");
    }
    // Every coarse pb keeps at least its own 2 draws.
    for pb in 0..2 {
        assert!((0..12).filter(|&c| coarse[c] == pb && keep[c]).count() >= 2);
    }
    assert_eq!(keep, keep_cells_per_pb(&[finest, coarse], 2, 7), "seeded");
}

#[test]
fn k_at_least_the_bucket_size_keeps_everything() {
    let finest: Vec<usize> = (0..6).map(|c| c / 3).collect();
    assert!(keep_cells_per_pb(&[finest], 3, 1).iter().all(|&k| k));
}
