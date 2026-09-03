use super::*;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn log_norm_is_log1p_of_the_library_scaled_count() {
    // ln(1 + 1e4 * 1 / 100) = ln(101)
    assert!(approx(f64::from(log_norm(1.0, 100.0)), 101f64.ln(), 1e-5));
    assert!(approx(f64::from(log_norm(0.0, 100.0)), 0.0, 1e-9));
}

#[test]
fn histogram_bins_values_like_numpy_including_the_closed_last_bin() {
    let mut h = Histogram::new(0.0, 10.0, 10).expect("range");
    h.add(0.0); // first bin
    h.add(1.0); // an interior edge belongs to the bin on its right
    h.add(9.99); // last bin
    h.add(10.0); // the upper bound is INCLUDED in the last bin (numpy)
    let counts = h.counts();
    assert_eq!(counts.len(), 10);
    assert_eq!(counts[0], 1);
    assert_eq!(counts[1], 1);
    assert_eq!(counts[9], 2);
    assert_eq!(counts.iter().sum::<u64>(), 4);
    let c = h.centroids();
    assert!(approx(c[0], 0.5, 1e-12));
    assert!(approx(c[9], 9.5, 1e-12));
    assert!(Histogram::new(1.0, 1.0, 10).is_err());
}

/// Five well-separated clusters of histogram mass. Two of them span two
/// histogram bins with unequal counts, so the centre must be the WEIGHTED mean
/// of the bin centroids, not the plain mean.
fn five_cluster_histogram() -> Histogram {
    let mut h = Histogram::new(0.0, 10.0, 100).expect("range");
    let mut put = |bin: usize, n: usize| {
        let v = (bin as f64 + 0.5) * 0.1; // the bin's centroid
        for _ in 0..n {
            h.add(v);
        }
    };
    put(10, 3);
    put(11, 1); // A: (3·1.05 + 1·1.15)/4 = 1.075
    put(30, 10); // B: 3.05
    put(50, 5);
    put(52, 5); // C: (5.05 + 5.25)/2 = 5.15
    put(70, 10); // D: 7.05
    put(90, 5); // E: 9.05
    h
}

#[test]
fn discretize_reproduces_hand_computed_edges_on_five_separated_value_clusters() {
    let disc = Discretization::fit(&five_cluster_histogram(), 5).expect("fit");
    assert_eq!(disc.n_levels(), 5);
    let want_centers = [1.075, 3.05, 5.15, 7.05, 9.05];
    for (c, w) in disc.centers.iter().zip(want_centers) {
        assert!(approx(*c, w, 1e-9), "centre {c} vs {w}");
    }
    // padding = (hi − lo) / (100 · 10) = 0.01; interior edges are midpoints.
    let want_edges = [-0.01, 2.0625, 4.1, 6.1, 8.05, 10.01];
    assert_eq!(disc.bin_edges.len(), want_edges.len());
    for (e, w) in disc.bin_edges.iter().zip(want_edges) {
        assert!(approx(*e, w, 1e-9), "edge {e} vs {w}");
    }
    assert_eq!(disc.hist_counts.iter().sum::<u64>(), 39);
    assert!(approx(disc.hist_range.0, 0.0, 0.0) && approx(disc.hist_range.1, 10.0, 0.0));
}

#[test]
fn bin_levels_are_monotone_in_value_and_span_one_to_n_bins() {
    let disc = Discretization::fit(&five_cluster_histogram(), 5).expect("fit");
    let probes: Vec<f32> = (0..=100).map(|i| i as f32 * 0.1).collect();
    let levels: Vec<u8> = probes.iter().map(|&v| disc.level(v)).collect();
    assert!(levels.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(*levels.first().unwrap(), 1);
    assert_eq!(*levels.last().unwrap(), 5);
    // np.digitize: a value ON an interior edge goes to the bin on its right.
    assert_eq!(disc.level(2.0625), 2);
    assert_eq!(disc.level(2.06), 1);
    assert_eq!(disc.level(1.075), 1);
    assert_eq!(disc.level(9.05), 5);
}

#[test]
fn weighted_kmeans_1d_returns_the_sorted_optimal_partition_centres() {
    let x = [0.0, 0.2, 0.9, 1.0, 1.1, 3.0, 3.4, 10.0];
    let w = [1.0, 2.0, 1.0, 5.0, 1.0, 1.0, 1.0, 3.0];
    let k = 3;
    // Brute force: optimal 1-D k-means partitions are contiguous in sorted
    // order, so enumerate every split into k contiguous non-empty groups.
    let n = x.len();
    let mut best = (f64::INFINITY, Vec::new());
    for a in 1..n - 1 {
        for b in a + 1..n {
            let groups = [&x[..a], &x[a..b], &x[b..]];
            let wg = [&w[..a], &w[a..b], &w[b..]];
            let mut sse = 0.0;
            let mut centres = Vec::new();
            for (g, wgt) in groups.iter().zip(wg) {
                let ws: f64 = wgt.iter().sum();
                let m = g.iter().zip(wgt.iter()).map(|(v, w)| v * w).sum::<f64>() / ws;
                sse += g
                    .iter()
                    .zip(wgt.iter())
                    .map(|(v, w)| w * (v - m).powi(2))
                    .sum::<f64>();
                centres.push(m);
            }
            if sse < best.0 {
                best = (sse, centres);
            }
        }
    }
    let got = weighted_kmeans_1d(&x, &w, k);
    assert_eq!(got.len(), k);
    for (g, b) in got.iter().zip(&best.1) {
        assert!(approx(*g, *b, 1e-9), "centre {g} vs brute force {b}");
    }
}

#[test]
fn fewer_nonempty_histogram_bins_than_requested_shrinks_the_number_of_levels() {
    let mut h = Histogram::new(0.0, 10.0, 100).expect("range");
    for _ in 0..7 {
        h.add(1.05);
    }
    for _ in 0..3 {
        h.add(8.05);
    }
    let disc = Discretization::fit(&h, 5).expect("fit");
    assert_eq!(disc.n_levels(), 2);
    assert_eq!(disc.bin_edges.len(), 3);
    assert_eq!(disc.level(0.5), 1);
    assert_eq!(disc.level(9.0), 2);
}
