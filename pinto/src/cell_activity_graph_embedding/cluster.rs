//! Leiden clustering + soft propensity for `pinto cage`.
//!
//! Cage trains cells and genes in a shared dot-product embedding space,
//! so we cluster cells on the unit sphere (row-L2-normalize → Euclidean
//! kNN ≡ angular). After Leiden gives hard labels, the propensity of
//! cell `n` for cluster `k` is `softmax(τ · ⟨X̂[n], Ĉ[k]⟩)` where `Ĉ[k]`
//! is the L2-normalized mean of in-cluster L2-normalized embeddings.
//!
//! The same centroids `Ĉ ∈ ℝ^{K×D}` are reused on the gene side to
//! produce a feature dictionary `[G × K]` — each gene's soft affinity
//! to each cell cluster, computed in identical fashion via
//! [`propensity_against_centroids`].

use crate::util::common::*;
use matrix_util::knn_graph::{
    compact_labels, modularity_to_cpm_resolution, run_leiden, tune_leiden_resolution, KnnGraph,
    KnnGraphArgs,
};

#[derive(Debug, Clone)]
pub struct LeidenPropensityArgs {
    pub knn: usize,
    pub resolution: f64,
    pub target_clusters: Option<usize>,
    pub min_cluster_size: usize,
    pub propensity_temp: f32,
    pub seed: u64,
}

pub struct LeidenPropensityResult {
    /// Cluster id per cell. `usize::MAX` = filtered (small cluster).
    pub labels: Vec<usize>,
    /// Number of retained clusters (after `min_cluster_size` filtering).
    pub n_clusters: usize,
    /// `[N × n_clusters]` row-stochastic soft assignment.
    pub propensity: Mat,
    /// `[n_clusters × D]` L2-normalized cluster centroids on the unit
    /// sphere. Reuse via [`propensity_against_centroids`] to score
    /// other point sets (e.g. genes) against the same clusters.
    pub centroids: Mat,
}

pub fn run_leiden_and_propensity(
    e_cell: &Mat,
    args: &LeidenPropensityArgs,
) -> anyhow::Result<LeidenPropensityResult> {
    let n = e_cell.nrows();
    let d = e_cell.ncols();
    anyhow::ensure!(n >= 2, "need ≥2 cells for leiden clustering");
    anyhow::ensure!(args.knn >= 1, "--leiden-knn must be ≥1 when enabled");

    info!(
        "cage leiden: {} cells × {} dims, knn={}, resolution={}",
        n, d, args.knn, args.resolution
    );

    // 1. L2-normalize rows — cosine kNN on the cage dot-product space.
    let x_norm = l2_normalize_rows(e_cell);

    // 2. kNN graph + leiden network.
    let graph = KnnGraph::from_rows(
        &x_norm,
        KnnGraphArgs {
            knn: args.knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;
    let n_components = matrix_util::graph::num_connected_components(&graph);
    info!(
        "cage leiden: knn graph {} nodes, {} edges, {} components",
        graph.num_nodes(),
        graph.num_edges(),
        n_components
    );
    let (network, total_w) = graph.to_leiden_network();
    let r = modularity_to_cpm_resolution(args.resolution, total_w);
    let seed = Some(args.seed as usize);

    // 3. Run leiden (with optional target-k tuning) → compact 0..K-1.
    let mut labels = match args.target_clusters {
        Some(k) => tune_leiden_resolution(&network, n, k, r, seed),
        None => run_leiden(&network, n, r, seed),
    };
    compact_labels(&mut labels);
    let mut n_clusters = labels.iter().copied().max().unwrap_or(0) + 1;

    // 4. Filter tiny clusters → usize::MAX, renumber remaining 0..K'-1.
    if args.min_cluster_size > 0 {
        let mut sizes = vec![0usize; n_clusters];
        for &lab in &labels {
            sizes[lab] += 1;
        }
        let mut remap = vec![usize::MAX; n_clusters];
        let mut next = 0usize;
        for (old, &sz) in sizes.iter().enumerate() {
            if sz >= args.min_cluster_size {
                remap[old] = next;
                next += 1;
            }
        }
        let dropped = n_clusters - next;
        if dropped > 0 {
            info!(
                "cage leiden: dropped {} clusters below min-size {}",
                dropped, args.min_cluster_size
            );
        }
        for lab in labels.iter_mut() {
            *lab = remap[*lab];
        }
        n_clusters = next;
    }
    anyhow::ensure!(
        n_clusters > 0,
        "no clusters survived min-cluster-size filter"
    );
    info!("cage leiden: {} retained clusters", n_clusters);

    // 5. Centroids: mean of L2-normalized rows per cluster, re-L2-normalize.
    let mut centroids = Mat::zeros(n_clusters, d);
    let mut counts = vec![0usize; n_clusters];
    for i in 0..n {
        let lab = labels[i];
        if lab == usize::MAX {
            continue;
        }
        counts[lab] += 1;
        let xi = x_norm.row(i);
        let mut ci = centroids.row_mut(lab);
        ci += xi;
    }
    for k in 0..n_clusters {
        if counts[k] > 0 {
            let mut row = centroids.row_mut(k);
            row /= counts[k] as f32;
            let denom = row.norm().max(1e-8);
            row /= denom;
        }
    }

    // 6. Propensity = row-wise softmax(τ · X̂ · Ĉᵀ) — every cell gets
    //    a row, including filtered-out ones (still a valid soft mass
    //    over retained clusters).
    let propensity =
        propensity_against_centroids_prenormed(&x_norm, &centroids, args.propensity_temp);

    Ok(LeidenPropensityResult {
        labels,
        n_clusters,
        propensity,
        centroids,
    })
}

/// Soft affinity of arbitrary embedding rows (e.g. genes) to the cell
/// clusters represented by L2-normalized `centroids` `[K × D]`. Each
/// input row gets L2-normalized, scored against every centroid, then
/// row-softmaxed at temperature `temp`. Returns `[rows × K]`.
/// Per-edge community label from per-cell soft propensity.
///
/// `community(e=(u,v)) = argmax_k propensity[u,k] · propensity[v,k]`
/// — Hadamard product → argmax. Picks the cluster where both endpoints
/// agree most strongly. Produces the `.link_community.parquet` schema
/// `pinto lr-activity` expects, bridging the per-cell → per-edge gap
/// that cage / cage-mcmc otherwise leave to the user.
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
