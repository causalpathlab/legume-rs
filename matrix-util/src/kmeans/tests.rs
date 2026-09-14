//! The k-means contract: recovers a planted partition on the sphere and in the
//! plane, agrees with a naive serial Lloyd from the same initialisation, and
//! returns bit-identical results regardless of how rayon splits the work.

use super::*;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// `n_per` unit-norm rows around each of `k` planted unit directions in `d`
/// dims (small angular jitter), rows in planted order. Returns the rows and
/// the planted labels.
fn planted_sphere(
    k: usize,
    n_per: usize,
    d: usize,
    jitter: f32,
    seed: u64,
) -> (DMatrix<f32>, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut dirs = DMatrix::<f32>::zeros(k, d);
    for c in 0..k {
        // Axis-aligned directions with a little spread, so no two coincide.
        dirs[(c, c % d)] = 1.0;
        dirs[(c, (c + 1) % d)] = 0.3;
    }
    let mut z = DMatrix::<f32>::zeros(k * n_per, d);
    let mut labels = Vec::with_capacity(k * n_per);
    for c in 0..k {
        for i in 0..n_per {
            let r = c * n_per + i;
            let mut norm = 0f32;
            for j in 0..d {
                let v = dirs[(c, j)] + jitter * (rng.random::<f32>() - 0.5);
                z[(r, j)] = v;
                norm += v * v;
            }
            let norm = norm.sqrt();
            for j in 0..d {
                z[(r, j)] /= norm;
            }
            labels.push(c);
        }
    }
    (z, labels)
}

/// `n_per` rows per planted blob in `d` dims, blobs far apart, uniform jitter.
fn planted_blobs(
    k: usize,
    n_per: usize,
    d: usize,
    jitter: f32,
    seed: u64,
) -> (DMatrix<f32>, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut z = DMatrix::<f32>::zeros(k * n_per, d);
    let mut labels = Vec::with_capacity(k * n_per);
    for c in 0..k {
        for i in 0..n_per {
            let r = c * n_per + i;
            for j in 0..d {
                let centre = if j == c % d {
                    10.0 * (c / d + 1) as f32
                } else {
                    0.0
                };
                z[(r, j)] = centre + jitter * (rng.random::<f32>() - 0.5);
            }
            labels.push(c);
        }
    }
    (z, labels)
}

/// Two labelings induce the same partition.
fn same_partition(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a_to_b = std::collections::HashMap::new();
    let mut b_to_a = std::collections::HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        if *a_to_b.entry(x).or_insert(y) != y || *b_to_a.entry(y).or_insert(x) != x {
            return false;
        }
    }
    true
}

fn opts(k: usize, seed: u64, metric: KmeansMetric) -> KmeansRowsOpts {
    KmeansRowsOpts {
        k,
        max_iter: 100,
        seed,
        metric,
        min_changed_frac: 0.0,
        init_sample: 0,
    }
}

/// Naive serial f64 Lloyd from a given initial centroid table: the oracle.
fn naive_lloyd(
    z: &DMatrix<f32>,
    init: &[f32],
    k: usize,
    max_iter: usize,
) -> (Vec<f32>, Vec<usize>) {
    let (n, d) = (z.nrows(), z.ncols());
    let mut centres: Vec<f64> = init.iter().map(|&v| f64::from(v)).collect();
    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let (mut best, mut best_d) = (0usize, f64::INFINITY);
            for c in 0..k {
                let mut s = 0f64;
                for j in 0..d {
                    let v = f64::from(z[(i, j)]) - centres[c * d + j];
                    s += v * v;
                }
                if s < best_d {
                    best_d = s;
                    best = c;
                }
            }
            if labels[i] != best {
                changed = true;
                labels[i] = best;
            }
        }
        let mut sums = vec![0f64; k * d];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..d {
                sums[c * d + j] += f64::from(z[(i, j)]);
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    centres[c * d + j] = sums[c * d + j] / counts[c] as f64;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (centres.iter().map(|&v| v as f32).collect(), labels)
}

#[test]
fn spherical_recovers_planted_directions() {
    let (z, planted) = planted_sphere(4, 200, 8, 0.2, 1);
    let fit = kmeans_rows_seeded(&z, &opts(4, 3, KmeansMetric::Cosine));
    assert!(
        same_partition(&fit.labels, &planted),
        "partition differs from the planted one"
    );
    for c in 0..4 {
        let norm: f32 = (0..8)
            .map(|j| fit.centroids[(c, j)].powi(2))
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "centroid {c} has norm {norm}");
    }
}

#[test]
fn euclidean_agrees_with_a_naive_oracle() {
    let (z, _) = planted_blobs(5, 100, 3, 1.0, 2);
    let o = opts(5, 11, KmeansMetric::Euclidean);
    let rows = z.transpose();
    let init = kmeans_pp_init(rows.as_slice(), z.nrows(), z.ncols(), 5, o.seed, 0);
    let (oracle_c, oracle_l) = naive_lloyd(&z, &init, 5, 100);
    let fit = kmeans_rows_seeded(&z, &o);
    assert_eq!(fit.labels, oracle_l);
    for c in 0..5 {
        for j in 0..3 {
            assert!(
                (fit.centroids[(c, j)] - oracle_c[c * 3 + j]).abs() < 1e-5,
                "centroid ({c},{j}) differs from the oracle"
            );
        }
    }
}

#[test]
fn centroids_are_the_means_of_their_members() {
    for metric in [KmeansMetric::Euclidean, KmeansMetric::Cosine] {
        let (mut z, _) = planted_blobs(3, 70, 5, 4.0, 5);
        if matches!(metric, KmeansMetric::Cosine) {
            for mut row in z.row_iter_mut() {
                let norm = row.norm();
                row /= norm;
            }
        }
        let fit = kmeans_rows_seeded(&z, &opts(3, 9, metric));
        for c in 0..3 {
            let members: Vec<usize> = (0..z.nrows()).filter(|&i| fit.labels[i] == c).collect();
            assert!(!members.is_empty(), "cluster {c} is empty");
            let mut mean = [0f64; 5];
            for &i in &members {
                for (j, m) in mean.iter_mut().enumerate() {
                    *m += f64::from(z[(i, j)]);
                }
            }
            for m in mean.iter_mut() {
                *m /= members.len() as f64;
            }
            if matches!(metric, KmeansMetric::Cosine) {
                let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
                for m in mean.iter_mut() {
                    *m /= norm;
                }
            }
            for (j, &m) in mean.iter().enumerate() {
                assert!(
                    (f64::from(fit.centroids[(c, j)]) - m).abs() < 1e-5,
                    "centroid ({c},{j}) is not its members' mean"
                );
            }
        }
    }
}

#[test]
fn thread_count_does_not_change_the_result() {
    let n = 5 * CHUNK_ROWS + 17;
    let d = 6;
    let mut rng = SmallRng::seed_from_u64(77);
    let z = DMatrix::<f32>::from_fn(n, d, |_, _| rng.random::<f32>());
    for metric in [KmeansMetric::Euclidean, KmeansMetric::Cosine] {
        let o = opts(7, 4, metric);
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| kmeans_rows_seeded(&z, &o));
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .build()
            .unwrap()
            .install(|| kmeans_rows_seeded(&z, &o));
        assert_eq!(one.labels, many.labels);
        assert_eq!(one.centroids.as_slice(), many.centroids.as_slice());
        assert_eq!(one.n_iter, many.n_iter);
    }
}

#[test]
fn same_seed_same_result_and_separable_data_is_seed_invariant() {
    let (z, planted) = planted_blobs(4, 50, 4, 0.5, 8);
    let a = kmeans_rows_seeded(&z, &opts(4, 1, KmeansMetric::Euclidean));
    let b = kmeans_rows_seeded(&z, &opts(4, 1, KmeansMetric::Euclidean));
    assert_eq!(a.labels, b.labels);
    assert_eq!(a.centroids.as_slice(), b.centroids.as_slice());
    let c = kmeans_rows_seeded(&z, &opts(4, 2, KmeansMetric::Euclidean));
    assert!(same_partition(&a.labels, &planted));
    assert!(same_partition(&c.labels, &planted));
}

#[test]
fn reseed_takes_the_worst_served_row_then_the_next_worst() {
    let d = 2;
    let rows: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 3.0, 3.0];
    let mut next = vec![9.0f32; 4 * d];
    let counts = vec![1usize, 0, 2, 0];
    // Row 2 is worst served, row 3 next; rows 0 and 1 tie at the bottom.
    let mut nearest = vec![0.5f32, 0.5, 4.0, 2.0];
    reseed_empty_clusters(&mut next, &counts, &mut nearest, &rows, d);
    assert_eq!(&next[d..2 * d], &[5.0, 5.0]);
    assert_eq!(&next[3 * d..4 * d], &[3.0, 3.0]);
    assert_eq!(nearest[2], 0.0);
    assert_eq!(nearest[3], 0.0);
    // Non-empty clusters were left alone.
    assert_eq!(&next[0..d], &[9.0, 9.0]);
    // Ties resolve to the lowest index.
    let mut next = vec![9.0f32; 2 * d];
    let mut nearest = vec![1.0f32, 1.0, 1.0, 1.0];
    reseed_empty_clusters(&mut next, &[1, 0], &mut nearest, &rows, d);
    assert_eq!(&next[d..2 * d], &[0.0, 0.0]);
}

#[test]
fn k_greater_than_n_does_not_panic() {
    let z = DMatrix::<f32>::from_row_slice(3, 2, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    let fit = kmeans_rows_seeded(&z, &opts(5, 1, KmeansMetric::Euclidean));
    assert_eq!(fit.centroids.nrows(), 5);
    assert!(fit.centroids.iter().all(|v| v.is_finite()));
    assert!(fit.labels.iter().all(|&l| l < 5));
}

#[test]
fn cosine_mode_normalises_rows_and_parks_zero_rows() {
    let (unit, _) = planted_sphere(3, 60, 5, 0.2, 4);
    let n = unit.nrows();
    let mut scaled = DMatrix::<f32>::zeros(n + 3, 5);
    let mut rng = SmallRng::seed_from_u64(6);
    for i in 0..n {
        let s = 0.1 + 5.0 * rng.random::<f32>();
        for j in 0..5 {
            scaled[(i, j)] = unit[(i, j)] * s;
        }
    }
    let on_unit = kmeans_rows_seeded(&unit, &opts(3, 2, KmeansMetric::Cosine));
    let on_scaled = kmeans_rows_seeded(&scaled, &opts(3, 2, KmeansMetric::Cosine));
    assert!(same_partition(&on_unit.labels, &on_scaled.labels[..n]));
    for c in 0..3 {
        let norm: f32 = (0..5)
            .map(|j| on_scaled.centroids[(c, j)].powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(norm > 0.5, "centroid {c} collapsed to {norm}");
    }
}

#[test]
fn changed_fraction_tolerance_stops_no_later() {
    let mut rng = SmallRng::seed_from_u64(3);
    let z = DMatrix::<f32>::from_fn(3000, 4, |_, _| rng.random::<f32>());
    let mut strict = opts(6, 5, KmeansMetric::Euclidean);
    strict.min_changed_frac = 0.0;
    let mut loose = strict.clone();
    loose.min_changed_frac = 0.05;
    let a = kmeans_rows_seeded(&z, &strict);
    let b = kmeans_rows_seeded(&z, &loose);
    assert!(b.n_iter <= a.n_iter, "{} > {}", b.n_iter, a.n_iter);
}

#[test]
fn init_from_a_subsample_is_reproducible_and_still_recovers_blobs() {
    let (z, planted) = planted_blobs(4, 300, 3, 0.5, 12);
    let mut o = opts(4, 5, KmeansMetric::Euclidean);
    o.init_sample = 40;
    let a = kmeans_rows_seeded(&z, &o);
    let b = kmeans_rows_seeded(&z, &o);
    assert_eq!(a.labels, b.labels);
    assert!(same_partition(&a.labels, &planted));
}

#[test]
fn degenerate_inputs_mirror_the_single_centroid_return() {
    let z = DMatrix::<f32>::from_row_slice(3, 2, &[0.0, 2.0, 2.0, 4.0, 4.0, 6.0]);
    let fit = kmeans_rows_seeded(&z, &opts(1, 1, KmeansMetric::Euclidean));
    assert_eq!(fit.centroids.nrows(), 1);
    assert_eq!(fit.centroids.as_slice(), &[2.0, 4.0]);
    assert!(fit.labels.iter().all(|&l| l == 0));
    let empty = DMatrix::<f32>::zeros(0, 2);
    let fit = kmeans_rows_seeded(&empty, &opts(3, 1, KmeansMetric::Euclidean));
    assert_eq!(fit.centroids.nrows(), 3);
    assert!(fit.labels.is_empty());
    let flat = DMatrix::<f32>::zeros(4, 0);
    let fit = kmeans_rows_seeded(&flat, &opts(2, 1, KmeansMetric::Cosine));
    assert_eq!(fit.labels, vec![0; 4]);
}
