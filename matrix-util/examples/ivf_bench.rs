//! Timing and recall of the IVF search at scale on a clustered synthetic
//! latent: `ivf_bench [n] [d] [k] [n_probe]`.
use matrix_util::knn::ivf::{knn_rows_ivf, IvfArgs};
use matrix_util::knn::metric::l2_simd;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).map_or(2_300_000, |s| s.parse().unwrap());
    let d: usize = a.get(2).map_or(16, |s| s.parse().unwrap());
    let k: usize = a.get(3).map_or(30, |s| s.parse().unwrap());
    let n_probe: usize = a.get(4).map_or(8, |s| s.parse().unwrap());
    let mut rng = StdRng::seed_from_u64(1);
    let n_centres = 40;
    let centres: Vec<Vec<f32>> = (0..n_centres)
        .map(|_| (0..d).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    let mut x = DMatrix::from_fn(n, d, |i, j| {
        centres[i % n_centres][j] + rng.random_range(-0.3f32..0.3)
    });
    for mut row in x.row_iter_mut() {
        let nrm = row.norm();
        row /= nrm;
    }
    let t = Instant::now();
    let (nb, _) = knn_rows_ivf(
        &x,
        &IvfArgs {
            k,
            n_lists: 0,
            n_probe,
            seed: 42,
        },
    );
    eprintln!(
        "ivf: n={n} d={d} k={k} n_probe={n_probe}: {:.1} s",
        t.elapsed().as_secs_f64()
    );
    // Recall on sampled queries against a brute-force scan of every row.
    let xt = x.transpose();
    let rows = xt.as_slice();
    let queries: Vec<usize> = (0..2000).map(|_| rng.random_range(0..n)).collect();
    let t = Instant::now();
    let recall: f64 = queries
        .par_iter()
        .map(|&q| {
            let qv = &rows[q * d..(q + 1) * d];
            let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
            for j in 0..n {
                if j == q {
                    continue;
                }
                let dd = l2_simd(qv, &rows[j * d..(j + 1) * d]);
                if best.len() < k || dd < best[k - 1].0 {
                    let pos = best.partition_point(|x| x.0 < dd || (x.0 == dd && x.1 < j));
                    best.insert(pos, (dd, j));
                    best.truncate(k);
                }
            }
            best.iter().filter(|(_, j)| nb[q].contains(j)).count() as f64 / k as f64
        })
        .sum::<f64>()
        / queries.len() as f64;
    eprintln!(
        "recall@{k} on 2000 queries: {recall:.3} (brute force {:.1} s)",
        t.elapsed().as_secs_f64()
    );
}
