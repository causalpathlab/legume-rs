//! 1D kmeans + Gaussian-BIC for choosing the number of CN states and
//! seeding emission means / variances from the marginal signal distribution.
//!
//! Hard-assignment kmeans is faster than full GMM-EM and sufficient for
//! seeding the HMM (the HMM's own EM refines emission params). BIC is computed
//! under a Gaussian mixture assumption using each cluster's empirical mean
//! and variance.

use matrix_util::clustering::{Kmeans, KmeansArgs};
use nalgebra::DMatrix;

/// Kmeans-based 1D component fit at a fixed K.
///
/// Returns `(means, variances, weights, log_likelihood)` where
/// `log_likelihood` is the **soft** Gaussian-mixture log-likelihood evaluated
/// at the kmeans centers and per-cluster variances — this gives a tighter
/// BIC than the hard-assignment likelihood and avoids favouring extreme K.
pub fn cluster_stats_kmeans(values: &[f32], k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let n = values.len();
    if n == 0 {
        return (vec![], vec![], vec![], f32::NEG_INFINITY);
    }
    if k <= 1 {
        let mean = values.iter().sum::<f32>() / (n as f32);
        let var = (values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (n as f32)).max(1e-6);
        let mut ll = 0f32;
        for &v in values {
            ll += -0.5 * ((v - mean).powi(2) / var + (2.0 * std::f32::consts::PI * var).ln());
        }
        return (vec![mean], vec![var], vec![1.0], ll);
    }

    let mat = DMatrix::<f32>::from_column_slice(n, 1, values);
    let labels = mat.kmeans_rows(KmeansArgs {
        num_clusters: k,
        max_iter: 200,
    });

    let mut sums = vec![0f32; k];
    let mut counts = vec![0usize; k];
    for (i, &l) in labels.iter().enumerate() {
        sums[l] += values[i];
        counts[l] += 1;
    }
    let mut means = vec![0f32; k];
    for j in 0..k {
        means[j] = if counts[j] > 0 {
            sums[j] / counts[j] as f32
        } else {
            0.0
        };
    }
    let mut sq = vec![0f32; k];
    for (i, &l) in labels.iter().enumerate() {
        sq[l] += (values[i] - means[l]).powi(2);
    }
    let mut variances = vec![1e-4f32; k];
    for j in 0..k {
        if counts[j] > 0 {
            variances[j] = (sq[j] / counts[j] as f32).max(1e-4);
        }
    }
    let weights: Vec<f32> = counts.iter().map(|&c| c as f32 / n as f32).collect();

    let two_pi = 2.0 * std::f32::consts::PI;
    let mut ll = 0f32;
    for &v in values {
        let mut log_p = vec![0f32; k];
        for j in 0..k {
            let var = variances[j].max(1e-6);
            log_p[j] = weights[j].max(1e-30).ln()
                - 0.5 * ((v - means[j]).powi(2) / var + (two_pi * var).ln());
        }
        let m = log_p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse: f32 = m + log_p.iter().map(|&l| (l - m).exp()).sum::<f32>().ln();
        ll += lse;
    }
    (means, variances, weights, ll)
}

type KmeansFit = (f32, usize, Vec<f32>, Vec<f32>, Vec<f32>);

/// BIC-select K over `k_range` using kmeans-based component stats. Returns
/// `(selected_k, means, variances, weights)` for the BIC-best K.
pub fn select_kmeans_k(values: &[f32], k_range: &[usize]) -> (usize, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = values.len() as f32;
    let mut best: Option<KmeansFit> = None;
    for &k in k_range {
        let (means, variances, weights, ll) = cluster_stats_kmeans(values, k);
        let p = (3 * k - 1) as f32; // means + variances + weights (sum-to-1)
        let bic = -2.0 * ll + p * n.max(1.0).ln();
        log::debug!("kmeans k={}: ll={:.2}, bic={:.2}", k, ll, bic);
        match &best {
            None => best = Some((bic, k, means, variances, weights)),
            Some((b_bic, _, _, _, _)) if bic < *b_bic => {
                best = Some((bic, k, means, variances, weights))
            }
            _ => {}
        }
    }
    let (_, k, means, variances, weights) = best.expect("k_range non-empty");
    (k, means, variances, weights)
}

/// Sort GMM components by mean and return them in ascending order, plus the
/// index of the component closest to 0 (= "neutral").
pub fn sort_components(
    means: Vec<f32>,
    variances: Vec<f32>,
    weights: Vec<f32>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let mut idx: Vec<usize> = (0..means.len()).collect();
    idx.sort_by(|&a, &b| {
        means[a]
            .partial_cmp(&means[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let m: Vec<f32> = idx.iter().map(|&i| means[i]).collect();
    let v: Vec<f32> = idx.iter().map(|&i| variances[i]).collect();
    let w: Vec<f32> = idx.iter().map(|&i| weights[i]).collect();
    let neutral = m
        .iter()
        .enumerate()
        .min_by(|a, b| {
            a.1.abs()
                .partial_cmp(&b.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(m.len() / 2);
    (m, v, w, neutral)
}
