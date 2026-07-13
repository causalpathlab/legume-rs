use super::*;

/// Elementwise sum of two equal-length vectors.
fn add(mut a: Vec<u64>, b: Vec<u64>) -> Vec<u64> {
    for (x, y) in a.iter_mut().zip(&b) {
        *x += y;
    }
    a
}

/// The fold must agree with the serial sum: rayon combines subtrees in whatever order it happened
/// to split the range, and the callers assume that cannot change the answer.
#[test]
fn the_fold_matches_the_serial_sum() {
    let n = 512;
    let go = AtomicBool::new(false);
    let (done, acc) = reduce_in(
        &go,
        n,
        "test",
        |i| Ok(vec![i as u64, 1, (i * i) as u64]),
        add,
    )
    .expect("no error")
    .expect("something completed");

    let want = (0..n).fold(vec![0u64; 3], |a, i| {
        add(a, vec![i as u64, 1, (i * i) as u64])
    });
    assert_eq!(done, n, "every replicate ran");
    assert_eq!(acc, want, "the folded total must equal the serial total");
}

/// **What an order-sensitive `combine` would expose.** The sum above is commutative, so no split
/// order could distinguish it; this one appends, so it *can* see the order rayon picked. Asserting
/// on the multiset pins the contract the callers actually rely on: every replicate contributes
/// exactly once, however the subtrees merged.
#[test]
fn every_replicate_contributes_exactly_once() {
    let n = 300;
    let go = AtomicBool::new(false);
    let (_, mut acc) = reduce_in(
        &go,
        n,
        "test",
        |i| Ok(vec![i]),
        |mut a: Vec<usize>, b: Vec<usize>| {
            a.extend(b);
            a
        },
    )
    .expect("no error")
    .expect("something completed");

    acc.sort_unstable();
    assert_eq!(
        acc,
        (0..n).collect::<Vec<_>>(),
        "no replicate lost or doubled"
    );
}

/// **An already-latched flag must yield `None`, not an error.** The flag never resets, so a stage
/// that *begins* after an interrupt completes zero replicates. Erroring there would propagate and
/// destroy the output of every stage that already finished — the exact loss the graceful stop
/// exists to prevent, delivered because the user used it. `None` hands the decision to the caller.
#[test]
fn an_already_stopped_run_yields_none_not_an_error() {
    let go = AtomicBool::new(true); // as if a Ctrl+C had landed in an earlier stage
    let out = reduce_in(&go, 100, "test", |i| Ok(i as u64), |a, b| a + b).expect("must not error");
    assert!(out.is_none(), "zero replicates is a None, never an Err");
}

/// An interrupt partway through keeps what finished, and `done` counts *exactly* those. It is the
/// denominator of every support-null p-value (`(k + 1) / (done + 1)`), so an off-by-one here
/// silently deflates every p-value in the run.
#[test]
fn done_counts_only_the_replicates_that_completed() {
    let n = 200;
    let go = AtomicBool::new(false);
    let (done, sum) = reduce_in(
        &go,
        n,
        "test",
        |_| {
            // Each replicate contributes 1 and trips the flag on its way out. Rayon has several in
            // flight, so `done` is not predictable — but it must equal what was folded in.
            go.store(true, Ordering::Relaxed);
            Ok(1u64)
        },
        |a, b| a + b,
    )
    .expect("must not error")
    .expect("the first replicates completed");

    assert!(done < n, "the flag should have cut the run short");
    assert_eq!(
        sum, done as u64,
        "`done` must count exactly the replicates folded in — no more, no fewer"
    );
}

/// An error in one replicate aborts the fold rather than being folded into a plausible partial sum.
#[test]
fn an_error_propagates() {
    let go = AtomicBool::new(false);
    let r = reduce_in(
        &go,
        64,
        "test",
        |i| {
            if i == 37 {
                anyhow::bail!("boom")
            } else {
                Ok(1u64)
            }
        },
        |a, b| a + b,
    );
    assert!(r.is_err(), "a failing replicate must not yield a total");
}
