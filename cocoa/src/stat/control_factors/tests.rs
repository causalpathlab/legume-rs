//! Control-gene factor extraction.

use super::*;
use crate::stat::{CocoaStat, CocoaStatArgs};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn abs_corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let sa: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
    let sb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
    (cov / (sa * sb)).abs()
}

#[test]
fn leading_factor_recovers_planted_confounder() {
    let (n, f) = (40, 200);
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    let normal = Normal::new(0f32, 1f32).unwrap();
    let v: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let w: Vec<f32> = (0..f).map(|_| normal.sample(&mut rng)).collect();
    let x = Mat::from_fn(n, f, |i, j| v[i] * w[j] + 0.5 * normal.sample(&mut rng));
    let (factors, s) = leading_factors(&x, None);
    assert_eq!(factors.ncols(), 1, "singular values {:?}", &s[..4]);
    let r = abs_corr(factors.column(0).as_slice(), &v);
    assert!(r > 0.95, "|corr| = {r}");
}

#[test]
fn log_rate_features_skip_sparse_topics_and_genes() {
    // two topics; topic 1 has too few cells, gene 1 is never expressed
    let n = 6;
    let t = ControlTotals {
        y: vec![
            Mat::from_fn(2, n, |g, i| if g == 0 { 10.0 + i as f32 } else { 0.0 }),
            Mat::from_element(2, n, 50.0),
        ],
        lib: vec![DVec::from_element(n, 1000.0), DVec::from_element(n, 1000.0)],
        cells: vec![DVec::from_element(n, 20.0), DVec::from_element(n, 1.0)],
    };
    let f = log_rate_features(&t);
    assert_eq!(f.ncols(), 1);
    let col = f.column(0);
    assert!(col.iter().sum::<f32>().abs() < 1e-4);
}

#[test]
fn totals_come_from_the_stage_one_statistics() {
    // 3 genes, 2 individuals, 2 pseudobulks, 2 topics
    let mut stat = CocoaStat::new(
        CocoaStatArgs {
            n_genes: 3,
            n_topics: 2,
            n_indv: 2,
            n_samples: 2,
        },
        None,
        None,
    );
    for k in 0..2 {
        stat.indv_y1_stat_mut(k)
            .copy_from(&Mat::from_fn(3, 2, |g, i| (1 + g + 3 * i + 10 * k) as f32));
        stat.indv_size_stat_mut(k)
            .copy_from(&Mat::from_fn(2, 2, |i, p| (1 + i + 2 * p + k) as f32));
    }
    let rows = [0usize, 2];
    let t = totals_from_stat(&stat, &rows);
    for k in 0..2 {
        let y1 = stat.indv_y1_stat(k);
        for (c, &g) in rows.iter().enumerate() {
            for i in 0..2 {
                assert_eq!(t.y[k][(c, i)], y1[(g, i)]);
            }
        }
        assert_eq!(t.lib[k], y1.row_sum_tr());
        assert_eq!(t.cells[k], stat.indv_size_stat(k).column_sum());
    }
}
