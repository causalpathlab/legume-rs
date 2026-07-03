use super::*;

#[test]
fn col_of_endpoints_and_midpoint() {
    let e = BinEdges::new(100, 200, 10);
    // first site -> first column, last site -> last column
    assert_eq!(e.col_of(100), 0);
    assert_eq!(e.col_of(200), 9);
    assert_eq!(e.col_of(150), 5);
    // out-of-range low clamps to col 0
    assert_eq!(e.col_of(50), 0);
    // degenerate single-position extent collapses to column 0
    assert_eq!(BinEdges::new(100, 100, 10).col_of(100), 0);
}

#[test]
fn bin_conserves_mass_without_log() {
    let e = BinEdges::new(0, 99, 10);
    let pts = [(0, 1.0), (5, 2.0), (50, 3.0), (99, 4.0)];
    let bins = e.bin(&pts, false);
    let total: f64 = bins.iter().sum();
    assert!((total - 10.0).abs() < 1e-9, "got {total}");
    // positions outside the extent are dropped
    let bins2 = e.bin(&[(0, 1.0), (1000, 9.0)], false);
    assert!((bins2.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn x_px_spans_plot_band() {
    let e = BinEdges::new(100, 200, 10);
    assert!((e.x_px(100, 10.0, 80.0) - 10.0).abs() < 1e-4);
    assert!((e.x_px(200, 10.0, 80.0) - 90.0).abs() < 1e-4);
    assert!((e.x_px(150, 10.0, 80.0) - 50.0).abs() < 1e-4);
}

#[test]
fn robust_max_ignores_zeros() {
    assert_eq!(robust_max([0.0, 0.0].into_iter()), 0.0);
    let m = robust_max([1.0, 2.0, 3.0, 4.0].into_iter());
    assert!(m >= 3.0);
}
