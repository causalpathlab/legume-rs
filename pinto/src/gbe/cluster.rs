//! Light wrapper over `leiden` + `matrix-util` for post-fit
//! clustering of a `pinto gbe` cell-embedding latent. Kept inside
//! `pinto` (no senna dependency) so we only pull in what we need.

use crate::util::common::*;
use matrix_util::clustering::KmeansArgs;
use matrix_util::knn_graph::{self, KnnGraph, KnnGraphArgs};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterMethod {
    Leiden,
    Kmeans,
}

pub struct LeidenParams {
    pub knn: usize,
    pub resolution: f64,
    /// When set, auto-tunes resolution to approximate this many clusters.
    pub target_clusters: Option<usize>,
    pub seed: Option<u64>,
}

/// Leiden modularity clustering on `latent` (rows = cells). Z-scores
/// columns, builds a kNN graph in latent space, converts to a Leiden
/// network, optionally auto-tunes resolution to a target cluster count,
/// runs Leiden, and compacts labels. Returns one cluster id per row.
pub fn leiden_clusters(latent: &Mat, params: &LeidenParams) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(latent.nrows() >= 2, "Need at least 2 cells for Leiden");
    let n = latent.nrows();
    info!(
        "Leiden: {n} cells × {} features, knn={}, resolution={:.4}, target_k={:?}",
        latent.ncols(),
        params.knn,
        params.resolution,
        params.target_clusters,
    );

    let mut z = latent.clone();
    z.scale_columns_inplace();

    let graph = KnnGraph::from_rows(
        &z,
        KnnGraphArgs {
            knn: params.knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;
    info!(
        "kNN graph: {} nodes, {} edges",
        graph.num_nodes(),
        graph.num_edges()
    );

    let (network, total_w) = graph.to_leiden_network();
    let scaled = knn_graph::modularity_to_cpm_resolution(params.resolution, total_w);
    let seed_val = params.seed.map(|s| s as usize);

    let mut labels = if let Some(k) = params.target_clusters {
        knn_graph::tune_leiden_resolution(&network, n, k, scaled, seed_val)
    } else {
        knn_graph::run_leiden(&network, n, scaled, seed_val)
    };
    knn_graph::compact_labels(&mut labels);

    let n_clusters = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut sizes = vec![0usize; n_clusters];
    for &l in &labels {
        sizes[l] += 1;
    }
    sizes.sort_unstable();
    info!(
        "Leiden: {n_clusters} clusters; sizes min={} median={} max={}",
        sizes.first().copied().unwrap_or(0),
        sizes.get(sizes.len() / 2).copied().unwrap_or(0),
        sizes.last().copied().unwrap_or(0),
    );
    Ok(labels)
}

/// K-means clustering on `latent` rows. Returns one cluster id per row.
pub fn kmeans_clusters(latent: &Mat, k: usize, max_iter: usize) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(k > 0, "k-means requires k > 0");
    anyhow::ensure!(
        k <= latent.nrows(),
        "k ({k}) exceeds number of cells ({})",
        latent.nrows()
    );
    info!(
        "K-means: {} cells × {} features, k={k}, max_iter={max_iter}",
        latent.nrows(),
        latent.ncols()
    );
    let labels = latent.kmeans_rows(KmeansArgs {
        num_clusters: k,
        max_iter,
    });
    Ok(labels)
}

/// Write per-cell cluster assignments as a parquet — `cell` (string) +
/// `cluster` (f32, NaN for unassigned). Mirrors the format senna
/// clustering writes, so downstream senna tools (annotate, plot) can
/// read it without a translation step.
pub fn write_clusters_parquet(
    labels: &[usize],
    cell_names: &[Box<str>],
    output_path: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        labels.len() == cell_names.len(),
        "labels ({}) and cell_names ({}) length mismatch",
        labels.len(),
        cell_names.len(),
    );
    let mut data = Mat::zeros(cell_names.len(), 1);
    for (i, &c) in labels.iter().enumerate() {
        data[(i, 0)] = if c == usize::MAX { f32::NAN } else { c as f32 };
    }
    let col_names: Vec<Box<str>> = vec!["cluster".into()];
    data.to_parquet_with_names(
        output_path,
        (Some(cell_names), Some("cell")),
        Some(&col_names),
    )?;
    Ok(())
}
