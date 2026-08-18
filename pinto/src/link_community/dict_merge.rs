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
/// `keep_genes` is a per-row mask of genes to score on, indexed like
/// `post_log_mean`'s rows; `None` uses every row. A shorter mask drops the rows
/// it does not cover rather than panicking.
/// Passing the detected-gene mask is strongly recommended — see the comment on
/// step 0 for what leaving undetected genes in does to the similarity.
///
/// Returns an empty vector for `K < 2`, or when the mask keeps no genes.
pub fn cosine_merge(post_log_mean: &Mat, keep_genes: Option<&[bool]>) -> Vec<BhcMerge> {
    let k = post_log_mean.ncols();
    if k < 2 {
        return Vec::new();
    }

    // 0. Restrict to informative genes.
    //
    // UNDETECTED GENES DOMINATE THIS SIMILARITY IF LEFT IN, and not because they
    // are flat — because they are the LOUDEST rows. A gene with no counts gets a
    // Poisson-Gamma posterior driven entirely by each community's exposure, and
    // `log` of a near-zero rate swings hard between communities, while a
    // well-measured gene's log-rate is stable. Measured, the
    // centred row sum-of-squares was ~13x LARGER for zero-count genes than for
    // genes seen in >= 20 spots (medians 235 vs 18.5), and
    // `spearman(row SS, nnz) = -0.986`. Cosine is dominated by the
    // largest-magnitude rows, so 19k noise genes outvoted 17.7k real ones: 43% of
    // community pairs scored >= 0.9 and the default cut collapsed 50 communities
    // to 4, with 97% of cells in one — a segmentation at chance (1.03x a
    // neighbour-agreement null, against 11.1x before the merge).
    //
    // Filtering by DETECTION is the only criterion that works here. Two plausible
    // alternatives are both backwards, and were measured to be so:
    //   * NB-Fisher `gene_weights` (already computed upstream) give undetected
    //     genes weight exactly 1.0 and well-detected ones a median of 0.878 —
    //     they are built to suppress high-dispersion housekeeping genes, which is
    //     the opposite question.
    //   * Filtering on low dictionary variance keeps precisely the noise, per the
    //     inverted correlation above.
    let rows: Vec<usize> = (0..post_log_mean.nrows())
        .filter(|&g| keep_genes.is_none_or(|mask| mask.get(g).copied().unwrap_or(false)))
        .collect();
    let n_genes = rows.len();
    if n_genes == 0 {
        return Vec::new();
    }

    // 1. Per-gene centring (subtract row mean across communities), so a gene's
    //    shared level across communities drops out and only its contrast
    //    survives. NOTE this is NOT a Pearson correlation between communities —
    //    that would need the COLUMNS centred too. Column centring was measured to
    //    add nothing once the gene filter above is applied.
    let mut z = Mat::zeros(n_genes, k);
    for (gi, &g) in rows.iter().enumerate() {
        let mu: f32 = post_log_mean.row(g).iter().sum::<f32>() / k as f32;
        for j in 0..k {
            z[(gi, j)] = post_log_mean[(g, j)] - mu;
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
    // `tr_mul` is `Zᵀ · Z` without materializing `Zᵀ` (same idiom as
    // `selection.rs`'s frozen-side product).
    let sim_kk = z.tr_mul(&z);
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
