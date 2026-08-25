//! The chunk-sizing arithmetic, without a device: the doubling rule,
//! the halved-budget reservation for backward, and the clamps. The
//! live CUDA query is exercised by real training runs, not unit tests.

use candle_util::device::chunk_from_measurements;

/// 8 GiB free, 1 MiB per item, cap 2048: half of 60% of 8 GiB is
/// ~2.4 GiB, which fits 2048 items with room, so the cap binds.
#[test]
fn ample_memory_runs_to_the_cap() {
    let free = 8usize << 30;
    let n = chunk_from_measurements(free, 1 << 20, 2048, 16, 0.6);
    assert_eq!(n, 2048);
}

/// 2 GiB free, 4 MiB per item: budget = 0.6 GiB -> 153 items; the
/// doubling ladder from 16 stops at 128 (256 * 4 MiB = 1 GiB > budget).
#[test]
fn tight_memory_stops_the_doubling_early() {
    let free = 2usize << 30;
    let n = chunk_from_measurements(free, 4 << 20, 2048, 16, 0.6);
    assert_eq!(n, 128);
}

/// The floor holds even when memory says "smaller": a chunk of zero
/// trains nothing, so the caller's floor is a promise.
#[test]
fn the_floor_is_never_undercut() {
    let n = chunk_from_measurements(1 << 20, 1 << 20, 2048, 16, 0.6);
    assert_eq!(n, 16);
}

/// Halving the budget for backward is load-bearing: without it, a
/// forward-fitting chunk can OOM in backward. 1 GiB free, 1 MiB/item,
/// 60%: full budget would allow 512, the halved one stops at 256.
#[test]
fn backward_reservation_halves_the_budget() {
    let free = 1usize << 30;
    let n = chunk_from_measurements(free, 1 << 20, 2048, 16, 0.6);
    assert_eq!(n, 256, "half of 0.6 GiB at 1 MiB/item -> 307 -> ladder 256");
}

/// Degenerate inputs stay sane: zero per-item cost (measurement noise)
/// must not divide by zero or explode past the cap.
#[test]
fn degenerate_measurements_clamp() {
    assert_eq!(chunk_from_measurements(1 << 30, 0, 512, 8, 0.6), 512);
    assert_eq!(chunk_from_measurements(0, 1 << 20, 512, 8, 0.6), 8);
    assert_eq!(chunk_from_measurements(1 << 30, 1 << 20, 8, 64, 0.6), 8);
}
