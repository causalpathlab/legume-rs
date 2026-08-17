use super::*;

#[test]
fn bins_span_and_clamp() {
    // Values spanning [0, 10] into 5 bins → ids 0..4, monotone, endpoints clamped.
    let pt = [0.0, 2.4, 5.0, 7.6, 10.0];
    let b = bin_pseudotime(&pt, 5);
    assert_eq!(b.len(), 5);
    assert_eq!(b[0], 0, "min → first bin");
    assert_eq!(*b.last().unwrap(), 4, "max → last bin (clamped, not 5)");
    assert!(
        b.windows(2).all(|w| w[0] <= w[1]),
        "bins must be monotone: {b:?}"
    );
}

#[test]
fn constant_pseudotime_is_single_bin() {
    // Degenerate (all equal) → every cell in bin 0, no divide-by-zero.
    let b = bin_pseudotime(&[3.0, 3.0, 3.0], 4);
    assert!(b.iter().all(|&x| x == 0));
}
