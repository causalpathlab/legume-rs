use super::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Deterministic random n×d data as column views fed to `ColumnDict`.
fn random_points(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect()
}

fn dict_from(points: &[Vec<f32>]) -> ColumnDict<usize> {
    let views: Vec<_> = points.iter().map(|p| p.as_slice().to_vec()).collect();
    let names: Vec<usize> = (0..points.len()).collect();
    <ColumnDict<usize> as ColumnDictOps<usize, Vec<f32>>>::from_column_views(views, names)
}

/// Forces the approximate HNSW backend even at small `n` (fast to build in
/// debug) by passing an exact-threshold of 0.
fn dict_from_approx(points: &[Vec<f32>]) -> ColumnDict<usize> {
    let views: Vec<_> = points.iter().map(|p| p.as_slice().to_vec()).collect();
    let names: Vec<usize> = (0..points.len()).collect();
    super::backend::build_column_dict(views, names, 0)
}

/// Independent ground-truth top-k by exact L2 (excluding self).
fn brute_others(points: &[Vec<f32>], q: usize, k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != q)
        .map(|(i, p)| (i, l2_simd(p, &points[q])))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[test]
fn l2_simd_matches_scalar() {
    // Cover a length that is not a multiple of 8 to exercise the scalar tail.
    let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.3).collect();
    let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.1).sin()).collect();
    let scalar: f32 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    assert!((l2_simd(&a, &b) - scalar).abs() < 1e-4);
}

#[test]
fn approx_recall_vs_exact() {
    // Force the HNSW path at small n (fast in debug) and check recall vs exact.
    let points = random_points(2_000, 32, 1);
    let dict = dict_from_approx(&points);
    let k = 10;

    let mut hits = 0usize;
    let mut total = 0usize;
    for q in (0..points.len()).step_by(23) {
        // search_others returns exactly k *others* (self excluded).
        let (got, _) = dict.search_others(&q, k).unwrap();
        let truth = brute_others(&points, q, k);
        let truth: std::collections::HashSet<usize> = truth.into_iter().collect();
        hits += got.iter().filter(|i| truth.contains(i)).count();
        total += k;
    }
    let recall = hits as f64 / total as f64;
    assert!(recall >= 0.95, "approx recall too low: {recall}");
}

#[test]
fn exact_path_is_perfect() {
    // n below EXACT_THRESHOLD uses the brute-force backend: recall must be 1.0.
    let points = random_points(500, 16, 2);
    let dict = dict_from(&points);
    let k = 8;
    for q in 0..points.len() {
        // search_others returns exactly k *others* (self excluded).
        let (got, dists) = dict.search_others(&q, k).unwrap();
        let truth = brute_others(&points, q, k);
        assert_eq!(
            got, truth,
            "exact backend disagreed with brute force at {q}"
        );
        // distances sorted ascending and self excluded
        assert!(dists.windows(2).all(|w| w[0] <= w[1]));
        assert!(!got.contains(&q));
    }
}

#[test]
fn build_is_deterministic() {
    // The whole point of the swap: identical HNSW index across builds (fixed seed).
    let points = random_points(2_000, 24, 3);
    let a = dict_from_approx(&points);
    let b = dict_from_approx(&points);
    for q in (0..points.len()).step_by(23) {
        let (ia, da) = a.search_others(&q, 12).unwrap();
        let (ib, db) = b.search_others(&q, 12).unwrap();
        assert_eq!(ia, ib, "neighbour indices differ between builds at {q}");
        assert_eq!(da, db, "distances differ between builds at {q}");
    }
}

#[test]
fn cross_dict_match() {
    let a_pts = random_points(300, 12, 4);
    let b_pts = random_points(400, 12, 5);
    let a = dict_from(&a_pts);
    let b = dict_from(&b_pts);

    let q = 7usize;
    let (names, dists) = a.match_by_query_name_against(&q, 5, &b).unwrap();
    // ground truth: nearest in b_pts to a_pts[q]
    let mut truth: Vec<(usize, f32)> = b_pts
        .iter()
        .enumerate()
        .map(|(i, p)| (i, l2_simd(p, &a_pts[q])))
        .collect();
    truth.sort_by(|x, y| x.1.total_cmp(&y.1));
    let truth: Vec<usize> = truth.into_iter().take(5).map(|(i, _)| i).collect();
    assert_eq!(names, truth);
    assert_eq!(dists.len(), 5);
}

#[test]
fn query_by_slice_matches_exact() {
    // The public query API takes an arbitrary &[f32] (no VecPoint construction).
    let points = random_points(400, 16, 6);
    let dict = dict_from(&points);
    let query: Vec<f32> = (0..16).map(|d| d as f32 * 0.05 - 0.4).collect();
    let k = 6;

    let (got, dists) = dict.search_by_query_data(&query, k).unwrap();

    let mut truth: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (i, l2_simd(p, &query)))
        .collect();
    truth.sort_by(|a, b| a.1.total_cmp(&b.1));
    let truth: Vec<usize> = truth.into_iter().take(k).map(|(i, _)| i).collect();

    assert_eq!(got, truth);
    assert!(dists.windows(2).all(|w| w[0] <= w[1]));
}
