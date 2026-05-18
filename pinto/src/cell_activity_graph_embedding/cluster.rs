//! K-means++ clustering for `pinto cage` cell embeddings.
//!
//! cage trains cells and genes in a shared dot-product embedding space,
//! so we cluster cells on the unit sphere (row-L2-normalize → Euclidean
//! distance on the sphere ≡ chord distance, monotone with cosine). The
//! k-means++ initialization + Lloyd's algorithm come from
//! [`matrix_util::clustering::Kmeans`] (which wraps the `clustering`
//! crate's k-means++).
//!
//! Soft propensity is derived from the hard partition via temperature-
//! sharpened cosine to per-cluster L2-normalized centroids:
//!
//! ```text
//!   propensity[n, k] ∝ exp(τ · ⟨X̂[n], Ĉ[k]⟩)        (row-softmax)
//! ```
//!
//! Centroids `Ĉ ∈ ℝ^{K×D}` are reused on the gene side via
//! [`propensity_against_centroids`] to produce a feature dictionary
//! `[G × K]` — each gene's soft affinity to each cell cluster.
//!
//! Per-edge labels (`pinto lr-activity` consumes these) come from
//! [`edge_community_from_propensity`]: argmax over the Hadamard product
//! of the two endpoints' propensities.

use crate::util::common::*;
use matrix_util::clustering::{Kmeans, KmeansArgs};

#[derive(Debug, Clone)]
pub struct KmeansClusteringArgs {
    pub n_clusters: usize,
    /// Softmax concentration τ for the soft propensity. Higher = sharper.
    pub propensity_temp: f32,
    /// Lloyd's algorithm iteration cap.
    pub max_iter: usize,
}

pub struct KmeansClusteringResult {
    /// Hard cluster label per cell (argmax of propensity row).
    pub labels: Vec<usize>,
    /// Number of clusters in the chosen fit.
    pub n_clusters: usize,
    /// `[N × n_clusters]` row-stochastic soft assignment.
    pub propensity: Mat,
    /// `[n_clusters × D]` L2-normalized cluster centroids on the unit
    /// sphere. Reuse via [`propensity_against_centroids`] to score
    /// other point sets (e.g. genes) against the same clusters.
    pub centroids: Mat,
}

pub fn run_kmeans_clustering(
    e_cell: &Mat,
    args: &KmeansClusteringArgs,
) -> anyhow::Result<KmeansClusteringResult> {
    let n = e_cell.nrows();
    let d = e_cell.ncols();
    anyhow::ensure!(n >= 2, "need ≥2 cells for clustering");
    anyhow::ensure!(args.n_clusters >= 1, "--n-clusters must be ≥1");

    let x_norm = l2_normalize_rows(e_cell);

    info!(
        "k-means++ on the unit sphere: N={n}, D={d}, K={}, τ={}",
        args.n_clusters, args.propensity_temp
    );
    let labels = x_norm.kmeans_rows(KmeansArgs {
        num_clusters: args.n_clusters,
        max_iter: args.max_iter,
    });

    // Per-cluster mean → L2-normalize. Empty clusters fall back to the
    // first row (rare; only if k-means++ degenerates).
    let k = args.n_clusters;
    let mut centroids = Mat::zeros(k, d);
    let mut counts = vec![0u32; k];
    for (i, &c) in labels.iter().enumerate() {
        counts[c] += 1;
        let mut row = centroids.row_mut(c);
        row += x_norm.row(i);
    }
    let mut n_empty = 0usize;
    for (c, &count) in counts.iter().enumerate() {
        if count == 0 {
            n_empty += 1;
            centroids.row_mut(c).copy_from(&x_norm.row(0));
            continue;
        }
        let mut row = centroids.row_mut(c);
        row /= count as f32;
        let denom = row.norm().max(1e-8);
        row /= denom;
    }
    if n_empty > 0 {
        warn!(
            "k-means: {} of {} clusters empty (centroids fell back to row 0). \
             Lower --n-clusters or expect the empty-row indicator to be ignored downstream.",
            n_empty, k
        );
    }

    let propensity =
        propensity_against_centroids_prenormed(&x_norm, &centroids, args.propensity_temp);

    info!(
        "k-means cluster counts: {:?}",
        counts.iter().map(|&c| c as usize).collect::<Vec<_>>()
    );

    Ok(KmeansClusteringResult {
        labels,
        n_clusters: k,
        propensity,
        centroids,
    })
}

/// Per-edge community: argmax of the Hadamard product of endpoint
/// cluster propensities. Used by `pinto lr-activity`.
pub fn edge_community_from_propensity(edges: &[(u32, u32)], propensity: &Mat) -> Vec<usize> {
    let k = propensity.ncols();
    edges
        .iter()
        .map(|&(u, v)| {
            let u = u as usize;
            let v = v as usize;
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..k {
                let s = propensity[(u, c)] * propensity[(v, c)];
                if s > best_score {
                    best_score = s;
                    best = c;
                }
            }
            best
        })
        .collect()
}

/// Soft affinity of arbitrary embedding rows (e.g. genes) to the cell
/// clusters represented by L2-normalized `centroids` `[K × D]`. Each
/// input row gets L2-normalized, scored against every centroid, then
/// row-softmaxed at temperature `temp`. Returns `[rows × K]`.
pub fn propensity_against_centroids(rows: &Mat, centroids: &Mat, temp: f32) -> Mat {
    let normed = l2_normalize_rows(rows);
    propensity_against_centroids_prenormed(&normed, centroids, temp)
}

fn propensity_against_centroids_prenormed(rows_norm: &Mat, centroids: &Mat, temp: f32) -> Mat {
    let mut scores = rows_norm * centroids.transpose();
    if temp != 1.0 {
        scores *= temp;
    }
    softmax_rows_inplace(&mut scores);
    scores
}

fn l2_normalize_rows(m: &Mat) -> Mat {
    let mut out = m.clone();
    for mut row in out.row_iter_mut() {
        let denom = row.norm().max(1e-8);
        row /= denom;
    }
    out
}

fn softmax_rows_inplace(m: &mut Mat) {
    for mut row in m.row_iter_mut() {
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > max {
                max = v;
            }
        }
        if !max.is_finite() {
            for v in row.iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    fn make_blobs() -> (Mat, Vec<usize>) {
        // 3 well-separated blobs on a 4-dim unit sphere, 30 cells each.
        let mut rng = SmallRng::seed_from_u64(7);
        let centers = [
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let n_per = 30;
        let mut rows: Vec<f32> = Vec::with_capacity(centers.len() * n_per * 4);
        let mut truth: Vec<usize> = Vec::with_capacity(centers.len() * n_per);
        for (k, c) in centers.iter().enumerate() {
            for _ in 0..n_per {
                let mut v = *c;
                for x in v.iter_mut() {
                    let noise: f32 = (rng.random::<u32>() as f32 / u32::MAX as f32) * 0.2 - 0.1;
                    *x += noise;
                }
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in v.iter_mut() {
                    *x /= norm;
                }
                rows.extend_from_slice(&v);
                truth.push(k);
            }
        }
        let n = truth.len();
        let d = 4;
        let m = Mat::from_row_slice(n, d, &rows);
        (m, truth)
    }

    #[test]
    fn kmeans_recovers_blobs() {
        let (x, truth) = make_blobs();
        let res = run_kmeans_clustering(
            &x,
            &KmeansClusteringArgs {
                n_clusters: 3,
                propensity_temp: 20.0,
                max_iter: 100,
            },
        )
        .unwrap();
        assert_eq!(res.n_clusters, 3);
        // Confusion is permutation-invariant: each true blob should be
        // dominated by a single predicted label.
        let mut conf = [[0usize; 3]; 3];
        for (i, &t) in truth.iter().enumerate() {
            conf[t][res.labels[i]] += 1;
        }
        let n_per = 30;
        for (t, row) in conf.iter().enumerate() {
            let max = *row.iter().max().unwrap();
            assert!(
                max >= (n_per * 8 / 10),
                "truth cluster {t} confusion {row:?}"
            );
        }
    }
}
