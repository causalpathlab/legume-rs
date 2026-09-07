use super::*;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn random_rows(n: usize, s: usize, seed: u64) -> DMatrix<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    DMatrix::from_fn(n, s, |_, _| rng.random_range(-1.0f32..1.0))
}

/// The shared oracle over every row: nearest first, ties by index, self
/// excluded.
fn brute(x: &DMatrix<f32>, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let points: Vec<Vec<f32>> = (0..x.nrows())
        .map(|i| x.row(i).iter().copied().collect())
        .collect();
    (0..x.nrows())
        .map(|q| crate::knn::tests::brute_others(&points, q, k))
        .unzip()
}

////////////////////////////////////////
// Exactness across block boundaries //
////////////////////////////////////////

#[test]
fn matches_brute_force_across_block_boundaries() {
    let x = random_rows(97, 6, 1);
    let k = 5;
    let (nb, ds) = knn_rows_l2_blocked(&x, k, 8);
    let (tnb, tds) = brute(&x, k);
    assert_eq!(nb.len(), 97);
    for i in 0..97 {
        assert_eq!(nb[i], tnb[i], "row {i} neighbours");
        for (a, b) in ds[i].iter().zip(&tds[i]) {
            assert!((a - b).abs() < 1e-4, "row {i}: {a} vs {b}");
        }
        for w in ds[i].windows(2) {
            assert!(w[0] <= w[1], "row {i} not nearest-first");
        }
    }
}

#[test]
fn default_block_matches_small_block() {
    let x = random_rows(150, 7, 2);
    assert_eq!(knn_rows_l2(&x, 4), knn_rows_l2_blocked(&x, 4, 5));
}

////////////////////////////
// Ties and degeneracies //
////////////////////////////

#[test]
fn ties_break_by_index_and_are_deterministic() {
    // Four copies of each of three distinct points: every copy's nearest
    // three are the other copies, at distance zero, in index order.
    let base = random_rows(3, 5, 3);
    let x = DMatrix::from_fn(12, 5, |i, c| base[(i % 3, c)]);
    let (nb, ds) = knn_rows_l2_blocked(&x, 3, 4);
    for i in 0..12 {
        let expect: Vec<usize> = (0..12).filter(|&j| j != i && j % 3 == i % 3).collect();
        assert_eq!(nb[i], expect, "row {i}");
        assert!(ds[i].iter().all(|&d| d == 0.0), "row {i}: {:?}", ds[i]);
    }
    assert_eq!(knn_rows_l2_blocked(&x, 3, 4), knn_rows_l2_blocked(&x, 3, 4));
}

#[test]
fn k_is_clamped_to_the_other_rows() {
    let x = random_rows(4, 3, 4);
    let (nb, ds) = knn_rows_l2(&x, 10);
    assert!(nb.iter().all(|v| v.len() == 3));
    assert!(ds.iter().all(|v| v.len() == 3));
    let one = random_rows(1, 3, 5);
    let (nb, ds) = knn_rows_l2(&one, 3);
    assert_eq!(nb, vec![Vec::<usize>::new()]);
    assert_eq!(ds, vec![Vec::<f32>::new()]);
    let none = DMatrix::<f32>::zeros(0, 3);
    assert!(knn_rows_l2(&none, 3).0.is_empty());
}

#[test]
fn non_finite_rows_sort_last() {
    let mut x = random_rows(20, 4, 6);
    x[(7, 2)] = f32::NAN;
    let (nb, ds) = knn_rows_l2_blocked(&x, 5, 6);
    for i in 0..20 {
        if i == 7 {
            assert!(ds[i].iter().all(|d| !d.is_finite()), "NaN row's distances");
        } else {
            assert!(!nb[i].contains(&7), "row {i} lists the NaN row");
            assert!(ds[i].iter().all(|d| d.is_finite()));
        }
    }
}

/////////////////////////////////////////////
// Precision: a large shared offset must   //
// not corrupt the neighbourhoods.         //
/////////////////////////////////////////////

#[test]
fn large_shared_offset_keeps_exact_neighbours() {
    // Gram-matrix distances cancel catastrophically when the norms dwarf the
    // separations; the returned neighbours and distances must still be the
    // direct-difference truth.
    let small = random_rows(64, 8, 7);
    let x = small.map(|v| 1.0e4 + 1.0e-2 * v);
    let (nb, ds) = knn_rows_l2_blocked(&x, 6, 16);
    let (tnb, tds) = brute(&x, 6);
    for i in 0..64 {
        assert_eq!(nb[i], tnb[i], "row {i} neighbours");
        for (a, b) in ds[i].iter().zip(&tds[i]) {
            assert!(((a - b) / b).abs() < 1e-3, "row {i}: {a} vs {b}");
        }
    }
}

////////////////////////////
// Thread-count invariance //
////////////////////////////

#[test]
fn independent_of_thread_count() {
    let x = random_rows(300, 6, 8);
    let reference = knn_rows_l2_blocked(&x, 8, 32);
    for threads in [1usize, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let got = pool.install(|| knn_rows_l2_blocked(&x, 8, 32));
        assert_eq!(got, reference, "{threads} threads");
    }
}
