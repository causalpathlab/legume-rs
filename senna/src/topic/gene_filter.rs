//! Data-driven gene filtering for topic models.
//!
//! K-means on per-gene CV separates housekeeping genes (low CV),
//! informative genes (moderate CV), and hyper-variable outliers
//! (Ig/TCR, extreme CV). The majority cluster is kept.

use crate::embed_common::*;

/// K-means gene filter on log(CV). Fits K=2 and K=3, picks via BIC,
/// and keeps the majority cluster (informative genes). Filters out
/// both low-CV housekeeping (uniform across PBs) and extreme-CV
/// outliers (Ig/TCR expressed in 1-2 PBs).
pub(crate) fn kmeans_cv_filter(cv: &[f32]) -> Vec<bool> {
    use matrix_util::clustering::{Kmeans, KmeansArgs};

    let n = cv.len();
    if n < 10 {
        return vec![true; n];
    }

    // Work in log(CV + eps) space as a 1D matrix [N, 1].
    let eps = 1e-6f32;
    let log_cv: Vec<f32> = cv.iter().map(|&v| (v + eps).ln()).collect();
    let mat = Mat::from_column_slice(n, 1, &log_cv);

    // Try K=2 and K=3
    let labels_2 = mat.kmeans_rows(KmeansArgs::with_clusters(2));
    let labels_3 = mat.kmeans_rows(KmeansArgs::with_clusters(3));
    let bic_2 = kmeans_bic(&log_cv, &labels_2, 2);
    let bic_3 = kmeans_bic(&log_cv, &labels_3, 3);

    let (labels, k_best) = if bic_2 < bic_3 {
        (labels_2, 2)
    } else {
        (labels_3, 3)
    };

    // Find the largest cluster = informative genes
    let mut counts = vec![0usize; k_best];
    for &l in &labels {
        if l < k_best {
            counts[l] += 1;
        }
    }
    let majority = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap();

    // Log cluster stats
    let mut means = vec![0.0f32; k_best];
    for (i, &l) in labels.iter().enumerate() {
        if l < k_best {
            means[l] += log_cv[i];
        }
    }
    for k in 0..k_best {
        if counts[k] > 0 {
            means[k] /= counts[k] as f32;
        }
    }
    log::info!(
        "gene filter: K={}, BIC(2)={:.0} BIC(3)={:.0}, clusters: {:?} (log_cv means: {:?}), keeping cluster {}",
        k_best, bic_2, bic_3, counts,
        means.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>(),
        majority
    );

    labels.iter().map(|&l| l == majority).collect()
}

/// BIC for 1D k-means: n·log(RSS/n) + k·log(n).
fn kmeans_bic(data: &[f32], labels: &[usize], k: usize) -> f64 {
    let n = data.len();
    let mut centroids = vec![0.0f64; k];
    let mut counts = vec![0usize; k];
    for (i, &l) in labels.iter().enumerate() {
        if l < k {
            centroids[l] += data[i] as f64;
            counts[l] += 1;
        }
    }
    for j in 0..k {
        if counts[j] > 0 {
            centroids[j] /= counts[j] as f64;
        }
    }
    let rss: f64 = data
        .iter()
        .zip(labels)
        .map(|(&x, &l)| {
            let c = if l < k { centroids[l] } else { 0.0 };
            let d = x as f64 - c;
            d * d
        })
        .sum();
    n as f64 * (rss / n as f64 + 1e-30).ln() + k as f64 * (n as f64).ln()
}
