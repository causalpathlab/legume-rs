//! Cosine-similarity agglomerative merging over the gene × community
//! dictionary.
//!
//! In `pinto lc`, each fitted link community has a posterior gene-expression
//! profile (the "dictionary atom") produced by `compute_gene_community_stat`.
//! Many fine link communities share a near-identical profile — they encode
//! the same cell type or program sitting at different spatial locations or
//! neighbourhoods. To recover a consistent cell-type-level annotation we
//! collapse such columns by hierarchical agglomerative merging on cosine
//! similarity of per-gene-centred log-rates.
//!
//! The output `Vec<BhcMerge>` reuses the merge-tree node type from
//! `data_beans_alg::bhc` purely as a binary-merge-tree carrier. The
//! cut function is the same union-find walk BHC uses; under the
//! cosine alias, `log_bf` carries cosine similarity and `cutoff` is a
//! user-chosen similarity threshold rather than a Bayes-factor break
//! point.
//!
//! Linkage: average linkage (UPGMA). When two clusters merge, the
//! similarity between the new node and any other cluster `o` is the
//! `n_samples`-weighted mean of `(left, o)` and `(right, o)` similarities,
//! which is equivalent to the average pairwise similarity between leaves.

use crate::util::common::Mat;
pub use data_beans_alg::bhc::{bhc_cut as cosine_cut, BhcMerge};

/// Build an agglomerative average-linkage merge tree over the K columns of
/// `post_log_mean` (gene × community posterior log-mean) using cosine
/// similarity of per-gene-centred community vectors.
///
/// Returns `K - 1` merges in increasing-id order. Each merge records the
/// cosine similarity at which the two children were joined in `log_bf`.
/// The resulting tree can be cut with `data_beans_alg::bhc::bhc_cut`
/// using a cosine-similarity threshold (e.g. 0.9 = collapse columns whose
/// merge happened at cosine ≥ 0.9).
///
/// Returns an empty vector for `K < 2`.
pub fn cosine_merge(post_log_mean: &Mat) -> Vec<BhcMerge> {
    let n_genes = post_log_mean.nrows();
    let k = post_log_mean.ncols();
    if k < 2 {
        return Vec::new();
    }

    // 1. Per-gene centring (subtract row mean across communities).
    //    Cosine on the centred matrix == Pearson on log-rates.
    let mut z = post_log_mean.clone();
    for g in 0..n_genes {
        let mu: f32 = z.row(g).iter().sum::<f32>() / k as f32;
        for j in 0..k {
            z[(g, j)] -= mu;
        }
    }

    // 2. L2-normalize columns. Pairwise cosine = Z^T Z (symmetric).
    for j in 0..k {
        let nrm = z.column(j).iter().map(|v| v * v).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for g in 0..n_genes {
                z[(g, j)] /= nrm;
            }
        }
    }

    // 3. Cosine similarity matrix S = Zᵀ Z (K × K, f32 via nalgebra),
    //    promoted to f64 and padded to (2K-1) × (2K-1) for in-place
    //    UPGMA updates as new internal nodes are written into the
    //    extended rows/cols.
    let total_nodes = 2 * k - 1;
    let sim_kk = z.transpose() * &z;
    let mut sim = vec![vec![0.0f64; total_nodes]; total_nodes];
    for i in 0..k {
        for j in 0..k {
            sim[i][j] = sim_kk[(i, j)] as f64;
        }
    }

    // 4. Agglomeration state. `alive[c]` indicates whether cluster `c` is
    //    still a root (clusters are addressed by their public id, which
    //    starts at 0..k for leaves and grows to 2k-2 at the final merge).
    //    `n_leaves[c]` = number of original leaves under `c` — the
    //    UPGMA weight for similarity averaging.
    let mut alive = vec![false; total_nodes];
    let mut n_leaves = vec![0i32; total_nodes];
    for c in 0..k {
        alive[c] = true;
        n_leaves[c] = 1;
    }

    let mut merges: Vec<BhcMerge> = Vec::with_capacity(k - 1);

    for step in 0..(k - 1) {
        // Find the most-similar live pair.
        let mut best_i = usize::MAX;
        let mut best_j = usize::MAX;
        let mut best_s = f64::NEG_INFINITY;
        for i in 0..(k + step) {
            if !alive[i] {
                continue;
            }
            for j in (i + 1)..(k + step) {
                if !alive[j] {
                    continue;
                }
                if sim[i][j] > best_s {
                    best_s = sim[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        debug_assert!(best_i != usize::MAX);

        let new_id = (k + step) as i32;
        let (left, right) = if (best_i as i32) < (best_j as i32) {
            (best_i as i32, best_j as i32)
        } else {
            (best_j as i32, best_i as i32)
        };
        let n_new = n_leaves[best_i] + n_leaves[best_j];

        merges.push(BhcMerge {
            id: new_id,
            left,
            right,
            log_bf: best_s,
            n_samples: n_new,
        });

        // UPGMA update: similarity from the new cluster to every surviving
        // other cluster `o` is the leaf-count-weighted mean of `(best_i, o)`
        // and `(best_j, o)`.
        let new_idx = new_id as usize;
        let w_i = n_leaves[best_i] as f64;
        let w_j = n_leaves[best_j] as f64;
        let w_sum = w_i + w_j;
        for o in 0..(k + step) {
            if !alive[o] || o == best_i || o == best_j {
                continue;
            }
            let s = (w_i * sim[best_i][o] + w_j * sim[best_j][o]) / w_sum;
            sim[new_idx][o] = s;
            sim[o][new_idx] = s;
        }
        sim[new_idx][new_idx] = 1.0;

        alive[best_i] = false;
        alive[best_j] = false;
        alive[new_idx] = true;
        n_leaves[new_idx] = n_new;
    }

    merges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::common::Mat;

    fn col_from(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    /// Build an `(n_genes × k)` matrix from K column vectors.
    fn mat_from_columns(cols: &[Vec<f32>]) -> Mat {
        let n_genes = cols[0].len();
        let k = cols.len();
        Mat::from_fn(n_genes, k, |g, j| cols[j][g])
    }

    #[test]
    fn merges_two_identical_columns_first() {
        // Three columns: 0 and 1 identical, 2 anti-correlated.
        let cols = vec![
            col_from(&[1.0, 0.0, -1.0, 0.5]),
            col_from(&[1.0, 0.0, -1.0, 0.5]),
            col_from(&[-1.0, 0.0, 1.0, -0.5]),
        ];
        let m = mat_from_columns(&cols);
        let merges = cosine_merge(&m);
        assert_eq!(merges.len(), 2);
        // First merge should be (0, 1) with cosine very close to 1.
        let first = &merges[0];
        assert!(
            (first.left == 0 && first.right == 1) || (first.left == 1 && first.right == 0),
            "first merge should join cols 0 and 1, got ({}, {})",
            first.left,
            first.right
        );
        assert!(
            (first.log_bf - 1.0).abs() < 1e-6,
            "identical-column merge should have cosine ≈ 1, got {}",
            first.log_bf
        );
        assert_eq!(first.n_samples, 2);
        assert_eq!(first.id, 3);
    }

    #[test]
    fn tree_shape_invariants() {
        // Random-ish 6 columns over 8 genes. Just check structural invariants.
        let cols: Vec<Vec<f32>> = (0..6)
            .map(|j| {
                (0..8)
                    .map(|g| ((g + j) as f32 * 0.7).sin())
                    .collect::<Vec<_>>()
            })
            .collect();
        let m = mat_from_columns(&cols);
        let merges = cosine_merge(&m);
        let k = 6usize;
        assert_eq!(merges.len(), k - 1);
        // Ids are exactly k, k+1, ..., 2k-2 in order.
        for (step, m) in merges.iter().enumerate() {
            assert_eq!(m.id as usize, k + step);
            assert!(m.left < m.id);
            assert!(m.right < m.id);
            assert!(m.left != m.right);
        }
        // Root node n_samples == K.
        assert_eq!(merges.last().unwrap().n_samples as usize, k);
        // Cosine similarity is monotonically non-increasing as we go up
        // an average-linkage UPGMA tree (the so-called Lance-Williams
        // monotonicity for UPGMA on a similarity).
        for w in merges.windows(2) {
            assert!(
                w[0].log_bf + 1e-9 >= w[1].log_bf,
                "UPGMA similarity should be non-increasing: {} -> {}",
                w[0].log_bf,
                w[1].log_bf
            );
        }
    }

    #[test]
    fn cut_at_threshold_recovers_groups() {
        // Two groups of similar columns: {0, 1, 2} and {3, 4}; group means
        // anti-correlated. Each within-group column has small jitter so
        // cosine within a group is close to 1; across groups close to -1.
        let g0a = vec![1.0, 0.0, -1.0, 0.5, 0.2];
        let g0b = vec![1.01, 0.02, -0.99, 0.51, 0.18];
        let g0c = vec![0.99, -0.01, -1.02, 0.49, 0.22];
        let g1a = vec![-1.0, 0.0, 1.0, -0.5, -0.2];
        let g1b = vec![-1.02, 0.01, 0.98, -0.49, -0.22];
        let m = mat_from_columns(&[g0a, g0b, g0c, g1a, g1b]);
        let merges = cosine_merge(&m);
        let k = 5usize;
        // Cut at cosine ≥ 0.5: only within-group merges happen.
        let labels = cosine_cut(&merges, k, 0.5);
        assert_eq!(labels.len(), k);
        let n_super: usize = labels.iter().copied().filter(|&v| v >= 0).count();
        let max_label = labels.iter().copied().max().unwrap();
        assert_eq!(n_super, k);
        assert_eq!(max_label, 1, "expected exactly 2 super-clusters");
        // Cols 0, 1, 2 share a label; cols 3, 4 share a label; the two are different.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn empty_for_k_lt_two() {
        let m = Mat::from_element(4, 1, 0.5);
        assert!(cosine_merge(&m).is_empty());
    }
}
