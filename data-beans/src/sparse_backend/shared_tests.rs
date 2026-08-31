//! The coalescer's job is to keep the RETRIEVE COUNT proportional to the data
//! span, not to the number of requested ranges. A subsample keeps every other
//! column, so no two ranges abut — and abutting-only merging degenerated into
//! one retrieve per column, several hundred thousand cache round-trips where a
//! few dozen block reads carry the same bytes.

use super::*;
use std::cell::Cell;

/// A fake backing array 0..n as both values and indices, with a counter on the
/// retrieve closure — the quantity under test.
fn run(tagged: &[(u64, u64, u64)], n: u64) -> (Vec<(u64, u64, f32)>, usize) {
    let calls = Cell::new(0usize);
    let out = coalesce_and_emit(
        tagged,
        n as usize,
        |tag, inner, val| (inner, tag, val),
        |s, e| {
            calls.set(calls.get() + 1);
            let data: Vec<f32> = (s..e).map(|v| v as f32).collect();
            let indices: Vec<u64> = (s..e).collect();
            Ok((data, indices))
        },
    )
    .expect("emit");
    (out, calls.get())
}

#[test]
fn small_gaps_fuse_into_one_retrieve() {
    // 1000 ranges of 60 entries with 60-entry gaps: the alternating-columns
    // shape a subsample produces. Fused, this is a handful of bounded reads;
    // unfused it is 1000.
    let tagged: Vec<(u64, u64, u64)> = (0..1000).map(|k| (k, k * 120, k * 120 + 60)).collect();
    let (out, calls) = run(&tagged, 200_000);
    assert_eq!(out.len(), 60_000, "every requested entry is emitted");
    assert!(
        calls <= 8,
        "1000 near-adjacent ranges must fuse into a few reads, got {calls}"
    );
    // and nothing from the gaps leaks in
    assert!(out.iter().all(|&(inner, _, _)| inner % 120 < 60));
}

#[test]
fn a_huge_gap_still_splits_the_read() {
    // Two ranges a hundred million entries apart must NOT be fused into one
    // retrieve spanning the void.
    let tagged = vec![(0u64, 0u64, 10u64), (1, 100_000_000, 100_000_010)];
    let (out, calls) = run(&tagged, 200_000_000);
    assert_eq!(out.len(), 20);
    assert_eq!(calls, 2, "fusing across the void would read 100M entries");
}

#[test]
fn merged_spans_stay_bounded() {
    // Many small gaps in a long run: fusion must cap the merged span rather
    // than growing one retrieve without limit.
    let tagged: Vec<(u64, u64, u64)> = (0..100_000).map(|k| (k, k * 120, k * 120 + 60)).collect();
    let (out, calls) = run(&tagged, 100_000 * 120);
    assert_eq!(out.len(), 6_000_000);
    assert!(calls >= 2, "a 12M-entry span must not become one retrieve");
    assert!(
        calls <= 64,
        "but it must stay a bounded handful, got {calls}"
    );
}
