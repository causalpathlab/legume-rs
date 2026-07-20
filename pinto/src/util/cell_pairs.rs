//! Spatial layer over [`CellPairs`].
//!
//! The general "cell-cell graph + the counts behind it" structure lives in
//! [`data_beans_alg::cell_pairs`] so senna / faba can share it. Everything
//! here is what pinto adds on top: per-cell coordinates, the pair table that
//! carries them, and the two ways pinto has of getting a graph in the first
//! place (tissue positions, or a layout synthesized from expression).

use crate::util::common::*;
use crate::util::knn_graph::{KnnGraph, KnnGraphArgs};
use dashmap::DashMap;
use data_beans_alg::cell_pairs::CellPairs;
use matrix_util::parquet::Column;
use matrix_util::traits::RandomizedAlgs;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

pub struct SrtCellPairs<'a> {
    /// The coordinate-free core: counts + the graph whose edges are the pairs.
    pub inner: CellPairs<'a>,
    /// Per-cell positions, `n_cells × n_dims`. In expression mode these are a
    /// synthesized 2D layout rather than tissue coordinates.
    pub coordinates: &'a Mat,
}

pub struct SrtCellPairsArgs {
    pub knn: usize,
    pub block_size: Option<usize>,
    pub reciprocal: bool,
}

impl<'a> SrtCellPairs<'a> {
    /// Wrap a pre-built KNN graph with data and coordinates. The graph is
    /// borrowed, not consumed — callers keep it for the graph algorithms
    /// (coarsening, component decomposition) that need its adjacency.
    pub fn with_graph(
        data: &'a SparseIoVec,
        coordinates: &'a Mat,
        graph: &'a KnnGraph,
    ) -> SrtCellPairs<'a> {
        SrtCellPairs {
            inner: CellPairs::from_graph(data, graph),
            coordinates,
        }
    }

    pub fn num_coordinates(&self) -> usize {
        self.coordinates.ncols()
    }

    /// Write all the coordinate pairs into `.parquet` file
    /// * `file_path`: destination file name (try to include a recognizable extension in the end, e.g., `.parquet`)
    /// * `coordinate_names`: column names for the left (`left_{}`) and right (`right_{}`) where each `{}` will be replaced with the corresponding column name
    pub fn to_parquet(
        &self,
        file_path: &str,
        coordinate_names: Option<Vec<Box<str>>>,
    ) -> anyhow::Result<()> {
        let coordinate_names = coordinate_names.unwrap_or(
            (0..self.num_coordinates())
                .map(|x| x.to_string().into_boxed_str())
                .collect(),
        );

        if coordinate_names.len() != self.num_coordinates() {
            return Err(anyhow::anyhow!("invalid coordinate names"));
        }

        let coords = self.pair_coordinate_columns(&coordinate_names);
        let columns: Vec<(Box<str>, Column<'_>)> = coords
            .iter()
            .map(|(name, values)| (name.clone(), Column::F32(values)))
            .collect();

        self.inner.to_parquet(file_path, &columns)
    }

    /// Per-pair endpoint coordinates, named and ordered the way the pair
    /// table wants them: every `left_{dim}`, then every `right_{dim}`.
    fn pair_coordinate_columns(&self, names: &[Box<str>]) -> Vec<(Box<str>, Vec<f32>)> {
        let pairs = self.inner.pairs();
        let mut out = Vec::with_capacity(names.len() * 2);
        for (prefix, take_left) in [("left", true), ("right", false)] {
            for (name, coord) in names.iter().zip(self.coordinates.column_iter()) {
                let values = pairs
                    .iter()
                    .map(|&(l, r)| coord[if take_left { l } else { r }])
                    .collect();
                out.push((format!("{prefix}_{name}").into_boxed_str(), values));
            }
        }
        out
    }
}

/// Build a KNN graph from a row-major point matrix (`n_points × n_dims`).
pub fn build_spatial_graph(coordinates: &Mat, args: SrtCellPairsArgs) -> anyhow::Result<KnnGraph> {
    KnnGraph::from_rows(
        coordinates,
        KnnGraphArgs {
            knn: args.knn,
            block_size: args.block_size.unwrap_or(1000),
            reciprocal: args.reciprocal,
        },
    )
}

/// Build a KNN graph from expression embeddings (random projection).
///
/// When spatial coordinates are not available, we build the cell-cell graph
/// from random-projected gene expression. Returns the graph and a 2D
/// force-directed layout for visualization in output files.
pub fn build_expression_graph(
    cell_proj: &Mat,
    args: SrtCellPairsArgs,
) -> anyhow::Result<(KnnGraph, Mat)> {
    // cell_proj is proj_dim × N from RandProjOps; transpose to N × proj_dim
    // so each row is a cell embedding for KNN
    let embedding_nk = cell_proj.transpose();
    let graph = build_spatial_graph(&embedding_nk, args)?;

    // Compute 2D layout: PCA init → force-directed refinement using graph
    info!("Computing 2D layout (PCA + force-directed)...");
    let coords_2d = force_directed_layout(&embedding_nk, &graph)?;

    Ok((graph, coords_2d))
}

/// Compute a 2D PCA initialization from an N × D embedding matrix.
fn pca_2d(embedding: &Mat) -> anyhow::Result<Mat> {
    let n = embedding.nrows();
    let d = embedding.ncols();

    if d <= 2 {
        return Ok(embedding.clone());
    }

    // Column-centre
    let col_means: Vec<f32> = (0..d)
        .map(|j| embedding.column(j).sum() / n as f32)
        .collect();

    let mut centred = embedding.clone();
    for (j, &m) in col_means.iter().enumerate() {
        centred.column_mut(j).add_scalar_mut(-m);
    }

    // Top-2 SVD → N × 2
    let (u, s, _) = centred.rsvd(2)?;
    let mut coords = Mat::zeros(n, 2);
    for k in 0..2 {
        for i in 0..n {
            coords[(i, k)] = u[(i, k)] * s[k];
        }
    }

    Ok(coords)
}

/// Force-directed 2D layout with negative sampling.
///
/// PCA-initialized, then refined with:
/// - Attractive forces along KNN graph edges (pull neighbours closer)
/// - Repulsive forces against random negative samples (push non-neighbours apart)
///
/// This is essentially the UMAP/LargeVis optimization step applied to the
/// existing KNN graph, producing a visually informative 2D embedding.
fn force_directed_layout(embedding: &Mat, graph: &KnnGraph) -> anyhow::Result<Mat> {
    use rand::rngs::SmallRng;
    use rand::RngExt;
    use rand::SeedableRng;

    let n = graph.n_nodes;
    let n_edges = graph.edges.len();

    // Initialize from PCA
    let mut coords = pca_2d(embedding)?;

    // Scale initial coordinates to unit variance per dimension
    for d in 0..2 {
        let mean = coords.column(d).sum() / n as f32;
        let var = coords
            .column(d)
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / n as f32;
        let std = var.sqrt().max(1e-8);
        for i in 0..n {
            coords[(i, d)] = (coords[(i, d)] - mean) / std;
        }
    }

    // Layout parameters
    let n_epochs = 200;
    let neg_samples_per_edge = 5usize;
    let initial_lr: f32 = 1.0;
    let min_dist: f32 = 0.01;
    let a: f32 = 1.0; // attractive curve shape
    let b: f32 = 1.0; // repulsive curve shape

    let mut rng = SmallRng::seed_from_u64(42);

    for epoch in 0..n_epochs {
        let lr = initial_lr * (1.0 - epoch as f32 / n_epochs as f32);
        let lr = lr.max(initial_lr * 0.01);

        // Attractive forces: pull edge endpoints together
        for e in 0..n_edges {
            let (i, j) = graph.edges[e];

            let dx = coords[(i, 0)] - coords[(j, 0)];
            let dy = coords[(i, 1)] - coords[(j, 1)];
            let dist_sq = dx * dx + dy * dy + min_dist;
            let dist = dist_sq.sqrt();

            // Attractive gradient: 2ab * d^(2b-2) / (1 + a * d^(2b))
            let grad = -2.0 * a * b * dist.powf(2.0 * b - 2.0) / (1.0 + a * dist.powf(2.0 * b));
            let fx = grad * dx * lr;
            let fy = grad * dy * lr;

            coords[(i, 0)] += fx;
            coords[(i, 1)] += fy;
            coords[(j, 0)] -= fx;
            coords[(j, 1)] -= fy;
        }

        // Repulsive forces: push random non-neighbours apart
        for e in 0..n_edges {
            let (i, _) = graph.edges[e];
            for _ in 0..neg_samples_per_edge {
                let k = rng.random_range(0..n);
                if k == i {
                    continue;
                }

                let dx = coords[(i, 0)] - coords[(k, 0)];
                let dy = coords[(i, 1)] - coords[(k, 1)];
                let dist_sq = dx * dx + dy * dy + min_dist;
                let dist = dist_sq.sqrt();

                // Repulsive gradient: 2b / (d * (1 + a * d^(2b)))
                let grad = 2.0 * b / (dist * (1.0 + a * dist.powf(2.0 * b)) + 1e-6);
                let fx = (grad * dx / dist).clamp(-4.0, 4.0) * lr;
                let fy = (grad * dy / dist).clamp(-4.0, 4.0) * lr;

                coords[(i, 0)] += fx;
                coords[(i, 1)] += fy;
            }
        }
    }

    Ok(coords)
}

/// Find connected components of a KNN graph.
///
/// Returns `(labels, n_components)` where `labels[i]` is the component index
/// of node `i`. Uses Union-Find for edge processing, then DashMap for parallel
/// label compaction.
pub fn connected_components(graph: &KnnGraph) -> (Vec<usize>, usize) {
    let n = graph.n_nodes;

    // Union-Find with path halving and union by rank
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    };

    for &(i, j) in &graph.edges {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            let (big, small) = if rank[ri] >= rank[rj] {
                (ri, rj)
            } else {
                (rj, ri)
            };
            parent[small] = big;
            if rank[big] == rank[small] {
                rank[big] += 1;
            }
        }
    }

    // Resolve all roots (serial, since find mutates)
    let roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();

    // Parallel label compaction with DashMap
    let rep_to_label = DashMap::new();
    let next = AtomicUsize::new(0);
    let labels: Vec<usize> = roots
        .par_iter()
        .map(|&r| {
            *rep_to_label
                .entry(r)
                .or_insert_with(|| next.fetch_add(1, AtomicOrdering::Relaxed))
        })
        .collect();

    (labels, next.load(AtomicOrdering::Relaxed))
}
