//! The inverted-file search is exact when every cell is probed, recovers most
//! of the exact neighbours when a few are, and answers the same on any thread
//! count.

use super::*;
use crate::knn::all_pairs::knn_rows_l2;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A mixture of `n_centres` Gaussian-ish blobs in `d` dims: the shape a
/// clustered latent has, where most of a row's neighbours share its blob.
fn mixture_rows(n: usize, d: usize, n_centres: usize, seed: u64) -> DMatrix<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..n_centres)
        .map(|_| (0..d).map(|_| rng.random_range(-3.0f32..3.0)).collect())
        .collect();
    DMatrix::from_fn(n, d, |i, j| {
        let c = &centres[i % n_centres];
        c[j] + rng.random_range(-0.5f32..0.5)
    })
}

fn args(k: usize, n_lists: usize, n_probe: usize) -> IvfArgs {
    IvfArgs {
        k,
        n_lists,
        n_probe,
        seed: 7,
    }
}

/// Mean over rows of the share of exact neighbours the approximate search found.
fn recall(approx: &[Vec<usize>], exact: &[Vec<usize>]) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for (a, e) in approx.iter().zip(exact) {
        total += e.len();
        hit += e.iter().filter(|j| a.contains(j)).count();
    }
    hit as f64 / total.max(1) as f64
}

#[test]
fn probing_every_cell_is_exact() {
    let x = mixture_rows(600, 8, 6, 1);
    let (nb, ds) = knn_rows_ivf(&x, &args(7, 12, 12));
    let (tnb, tds) = knn_rows_l2(&x, 7);
    assert_eq!(nb, tnb);
    for (a, b) in ds.iter().zip(&tds) {
        for (u, v) in a.iter().zip(b) {
            assert!((u - v).abs() < 1e-4, "{u} vs {v}");
        }
    }
}

#[test]
fn a_few_probes_recover_most_neighbours_and_more_probes_never_fewer() {
    let x = mixture_rows(20_000, 16, 25, 2);
    let (exact, _) = knn_rows_l2(&x, 30);
    let mut last = 0.0;
    for n_probe in [4usize, 8, 16] {
        let (nb, _) = knn_rows_ivf(&x, &args(30, 0, n_probe));
        let r = recall(&nb, &exact);
        assert!(
            r >= last - 1e-9,
            "recall fell from {last} to {r} at n_probe={n_probe}"
        );
        if n_probe == DEFAULT_N_PROBE {
            assert!(r >= 0.9, "recall {r} at the default n_probe");
        }
        last = r;
    }
}

#[test]
fn results_are_nearest_first_without_self_and_ties_break_by_index() {
    // Three distinct points, four copies each: every copy's nearest three
    // are the other copies at distance zero, in index order.
    let base = mixture_rows(3, 5, 3, 3);
    let x = DMatrix::from_fn(12, 5, |i, c| base[(i % 3, c)]);
    let (nb, ds) = knn_rows_ivf(&x, &args(3, 2, 2));
    for i in 0..12 {
        assert!(!nb[i].contains(&i), "row {i} returned itself");
        let expect: Vec<usize> = (0..12).filter(|&j| j != i && j % 3 == i % 3).collect();
        assert_eq!(nb[i], expect, "row {i}");
        assert!(ds[i].iter().all(|&d| d == 0.0));
    }
    let x = mixture_rows(300, 6, 4, 4);
    let (_, ds) = knn_rows_ivf(&x, &args(9, 0, 3));
    for row in &ds {
        for w in row.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }
}

#[test]
fn thread_count_does_not_change_the_result() {
    let x = mixture_rows(9_000, 10, 12, 5);
    let a = args(15, 0, 6);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| knn_rows_ivf(&x, &a));
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()
        .unwrap()
        .install(|| knn_rows_ivf(&x, &a));
    assert_eq!(one.0, many.0);
    assert_eq!(one.1, many.1);
}

#[test]
fn small_inputs_clamp_k_lists_and_probes() {
    let x = mixture_rows(50, 4, 2, 6);
    // More lists and probes than rows, and k beyond the other rows.
    let (nb, ds) = knn_rows_ivf(&x, &args(80, 200, 500));
    let (tnb, tds) = knn_rows_l2(&x, 49);
    assert_eq!(nb, tnb);
    assert_eq!(ds.len(), tds.len());
    let (nb, _) = knn_rows_ivf(&DMatrix::<f32>::zeros(1, 3), &args(4, 0, 1));
    assert_eq!(nb, vec![Vec::<usize>::new()]);
    let (nb, _) = knn_rows_ivf(&DMatrix::<f32>::zeros(0, 3), &args(4, 0, 1));
    assert!(nb.is_empty());
}
