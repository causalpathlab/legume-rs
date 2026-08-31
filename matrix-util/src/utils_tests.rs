//! The byte-budget blocker exists because the work heuristic bounds columns ×
//! features, which says nothing about the nnz a block actually materialises —
//! and nnz is what OOMs.

use super::byte_budget_intervals;

#[test]
fn blocks_respect_the_byte_budget() {
    // 10 columns of 100 nnz at 24 B = 2400 B/col; a 5000 B budget fits two.
    let nnz = vec![100u64; 10];
    let blocks = byte_budget_intervals(&nnz, 5000, 24);
    for &(lb, ub) in &blocks {
        let bytes: u64 = nnz[lb..ub].iter().sum::<u64>() * 24;
        assert!(bytes <= 5000, "block {lb}..{ub} holds {bytes} B");
    }
    // coverage: contiguous, disjoint, complete
    assert_eq!(blocks.first().unwrap().0, 0);
    assert_eq!(blocks.last().unwrap().1, 10);
    for w in blocks.windows(2) {
        assert_eq!(w[0].1, w[1].0, "gap or overlap between blocks");
    }
}

#[test]
fn one_dense_column_still_gets_a_block() {
    // A single column heavier than the whole budget cannot be split here; it
    // must become its own block rather than an error or an infinite loop.
    let nnz = vec![5u64, 1_000_000, 5];
    let blocks = byte_budget_intervals(&nnz, 1000, 24);
    assert!(
        blocks.contains(&(1, 2)),
        "the heavy column stands alone: {blocks:?}"
    );
    assert_eq!(blocks.first().unwrap().0, 0);
    assert_eq!(blocks.last().unwrap().1, 3);
}

#[test]
fn millions_of_empty_columns_collapse_into_one_block() {
    // Empty columns cost nothing; the work heuristic would cut thousands of
    // pointless blocks here, the byte budget cuts one.
    let nnz = vec![0u64; 1_000_000];
    let blocks = byte_budget_intervals(&nnz, 1 << 20, 24);
    assert_eq!(blocks, vec![(0, 1_000_000)]);
}

#[test]
fn a_zero_budget_degrades_to_one_column_per_nonempty_block() {
    let nnz = vec![3u64, 3, 3];
    let blocks = byte_budget_intervals(&nnz, 0, 24);
    assert_eq!(
        blocks.len(),
        3,
        "budget floor is one nnz, so each column stands alone"
    );
    assert_eq!(blocks.last().unwrap().1, 3);
}
