//! Label-free stage-1 baseline on simulated pseudobulk tables.
//!
//! ```text
//!   y(d,i,p) ~ Poisson( n(i,p) * mu(d,p) * omega(d,i) )
//! ```
//!
//! Individuals are spread over pseudobulks unevenly; some pseudobulks hold a
//! single individual. The fit must recover the offsets m(d,i) = sum_p mu n
//! up to one scale per gene, with no exposure labels.

use super::*;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Poisson};

struct Table {
    stat: CocoaStat,
    /// planted m(d,i) = sum_p mu(d,p) n(i,p)
    m_true: Mat,
}

fn simulate(n_genes: usize, n_indv: usize, n_pb: usize, seed: u64) -> Table {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0f32, 1f32).unwrap();
    // cells of individual i in pseudobulk p; the last pseudobulk holds only
    // individual 0
    let n_ip = Mat::from_fn(n_indv, n_pb, |i, p| {
        if p == n_pb - 1 {
            if i == 0 {
                30.0
            } else {
                0.0
            }
        } else if rng.random::<f32>() < 0.7 {
            5.0 + 40.0 * rng.random::<f32>()
        } else {
            0.0
        }
    });
    let mu = Mat::from_fn(n_genes, n_pb, |_, _| (0.8 * normal.sample(&mut rng)).exp());
    let omega = Mat::from_fn(n_genes, n_indv, |_, _| {
        (0.5 * normal.sample(&mut rng)).exp()
    });
    let mut y_dp = Mat::zeros(n_genes, n_pb);
    let mut y_di = Mat::zeros(n_genes, n_indv);
    for d in 0..n_genes {
        for i in 0..n_indv {
            for p in 0..n_pb {
                let rate = n_ip[(i, p)] * mu[(d, p)] * omega[(d, i)];
                if rate > 0.0 {
                    let y = Poisson::new(rate).unwrap().sample(&mut rng);
                    y_dp[(d, p)] += y;
                    y_di[(d, i)] += y;
                }
            }
        }
    }
    let mut stat = CocoaStat::new(
        CocoaStatArgs {
            n_genes,
            n_topics: 1,
            n_indv,
            n_samples: n_pb,
        },
        Some(200),
        Some((1e-2, 1e-2)),
    );
    stat.y1_stat_mut(0).copy_from(&y_dp);
    stat.indv_y1_stat_mut(0).copy_from(&y_di);
    stat.indv_size_stat_mut(0).copy_from(&n_ip);
    let m_true = &mu * n_ip.transpose();
    Table { stat, m_true }
}

/// Pearson correlation of log values over positive entries.
fn log_corr(a: &[f32], b: &[f32]) -> f32 {
    let (x, y): (Vec<f32>, Vec<f32>) = a
        .iter()
        .zip(b)
        .filter(|(p, q)| **p > 0.0 && **q > 0.0)
        .map(|(p, q)| (p.ln(), q.ln()))
        .unzip();
    let n = x.len() as f32;
    let (mx, my) = (x.iter().sum::<f32>() / n, y.iter().sum::<f32>() / n);
    let c: f32 = x.iter().zip(&y).map(|(p, q)| (p - mx) * (q - my)).sum();
    let sx: f32 = x.iter().map(|p| (p - mx).powi(2)).sum::<f32>().sqrt();
    let sy: f32 = y.iter().map(|q| (q - my).powi(2)).sum::<f32>().sqrt();
    c / (sx * sy)
}

#[test]
fn recovers_the_planted_offsets_up_to_a_gene_scale() {
    // low-count genes are noisier, so check the spread over genes
    let t = simulate(30, 12, 8, 1);
    let fit = t.stat.estimate_baseline();
    let m = &fit[0].indv_offset;
    let mut r: Vec<f32> = (0..30)
        .map(|d| {
            let got: Vec<f32> = m.row(d).iter().cloned().collect();
            let want: Vec<f32> = t.m_true.row(d).iter().cloned().collect();
            log_corr(&got, &want)
        })
        .collect();
    r.sort_by(f32::total_cmp);
    assert!(r[15] > 0.98, "median log corr(m, planted m) = {}", r[15]);
    assert!(r[0] > 0.9, "worst log corr(m, planted m) = {}", r[0]);
}

#[test]
fn every_offset_is_finite_and_positive() {
    let t = simulate(20, 10, 6, 3);
    let fit = t.stat.estimate_baseline();
    assert!(fit[0].indv_offset.iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(fit[0].log_mean.iter().all(|v| v.is_finite()));
}
