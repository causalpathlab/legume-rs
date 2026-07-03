use super::*;

#[test]
fn block_bp_overlap_splits_across_bins() {
    // span = max-min = 100, 10 bins -> 10bp bins ([0,10),[10,20),...).
    let edges = BinEdges::new(0, 100, 10);
    let mut bins = vec![0.0; 10];
    // block [5, 25): 5bp in bin0, 10bp in bin1, 5bp in bin2.
    accumulate_block(&mut bins, &edges, 5, 25);
    assert!((bins[0] - 5.0).abs() < 1e-9, "{:?}", bins);
    assert!((bins[1] - 10.0).abs() < 1e-9, "{:?}", bins);
    assert!((bins[2] - 5.0).abs() < 1e-9, "{:?}", bins);
    // total bp == clipped block length, regardless of bin boundaries
    assert!((bins.iter().sum::<f64>() - 20.0).abs() < 1e-9);
}

#[test]
fn block_clipped_to_extent() {
    let edges = BinEdges::new(0, 99, 10);
    let mut bins = vec![0.0; 10];
    // block extends past both ends; only [0,100) of it counts -> 100bp.
    accumulate_block(&mut bins, &edges, -50, 500);
    assert!((bins.iter().sum::<f64>() - 100.0).abs() < 1e-9);
    // fully outside -> nothing
    let mut bins2 = vec![0.0; 10];
    accumulate_block(&mut bins2, &edges, 500, 600);
    assert_eq!(bins2.iter().sum::<f64>(), 0.0);
}
