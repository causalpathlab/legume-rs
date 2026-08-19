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

/// `edge_kind` for a physically adjacent pair.
pub const EDGE_KIND_SPATIAL: i32 = 0;
/// `edge_kind` for a pair that is expression-similar but not adjacent.
pub const EDGE_KIND_EXPRESSION: i32 = 1;

/// How a pair reads in the per-edge outputs.
///
/// [`EdgeSource::Both`] is deliberately spatial: such a pair is physically
/// adjacent as well as expression-similar, and a consumer filtering on
/// spatial is asking about adjacency.
pub fn edge_kind_code(source: matrix_util::knn_graph::EdgeSource) -> i32 {
    use matrix_util::knn_graph::EdgeSource::*;
    match source {
        Primary | Both => EDGE_KIND_SPATIAL,
        Secondary => EDGE_KIND_EXPRESSION,
    }
}

pub struct SrtCellPairs<'a> {
    /// The coordinate-free core: counts + the graph whose edges are the pairs.
    pub inner: CellPairs<'a>,
    /// Per-cell positions, `n_cells × n_dims`. In expression mode these are a
    /// synthesized 2D layout rather than tissue coordinates.
    pub coordinates: &'a Mat,
    /// Per-pair `edge_kind`, parallel to `inner.pairs()`. `None` when the
    /// graph was not augmented, in which case the column is not written at
    /// all and an unaugmented run stays byte-identical.
    pub edge_kind: Option<Vec<i32>>,
}

pub struct SrtCellPairsArgs {
    pub knn: usize,
    pub block_size: Option<usize>,
    pub reciprocal: bool,
}

impl<'a> SrtCellPairs<'a> {
    /// Wrap a pre-built KNN graph with data and coordinates, recording which
    /// graph each pair came from. The graph is borrowed, not consumed, so
    /// callers keep it for the graph algorithms (coarsening, component
    /// decomposition) that need its adjacency. Pass `None` for `edge_source`
    /// when the graph was not augmented.
    pub fn with_graph_and_source(
        data: &'a SparseIoVec,
        coordinates: &'a Mat,
        graph: &'a KnnGraph,
        edge_source: Option<&[matrix_util::knn_graph::EdgeSource]>,
    ) -> SrtCellPairs<'a> {
        SrtCellPairs {
            inner: CellPairs::from_graph(data, graph),
            coordinates,
            edge_kind: edge_source.map(|src| src.iter().copied().map(edge_kind_code).collect()),
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
        let mut columns: Vec<(Box<str>, Column<'_>)> = coords
            .iter()
            .map(|(name, values)| (name.clone(), Column::F32(values)))
            .collect();
        // Unprefixed on purpose: the plot layer discovers coordinates by
        // scanning for a `left_` prefix, so `left_edge_kind` would be read
        // back as a coordinate.
        if let Some(kind) = self.edge_kind.as_ref() {
            columns.push(("edge_kind".into(), Column::I32(kind)));
        }

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
/// `cell_proj` is `proj_dim x n_cells`, which is the layout
/// [`KnnGraph::from_columns`] already wants, so no transpose is needed.
pub fn build_expression_knn(cell_proj: &Mat, args: SrtCellPairsArgs) -> anyhow::Result<KnnGraph> {
    KnnGraph::from_columns(
        cell_proj,
        KnnGraphArgs {
            knn: args.knn,
            block_size: args.block_size.unwrap_or(1000),
            reciprocal: args.reciprocal,
        },
    )
}

/// Expression KNN restricted to within each spatial component.
///
/// Expression similarity ignores geometry, so a global search pairs cells
/// across whatever the spatial graph left disconnected: separate sections, or
/// the cores of a tissue microarray. Those are usually separate samples, and
/// the rest of this pipeline treats a spatial component as a batch, so joining
/// them silently contradicts that.
///
/// Each component is searched on its own, so every cell gets its neighbours
/// from its own sample. A component smaller than `knn + 1` is skipped: it
/// cannot supply that many neighbours, and taking whatever it has would give
/// its cells a denser neighbourhood than everyone else's.
pub fn build_expression_knn_within(
    cell_proj: &Mat,
    component_of_cell: &[usize],
    n_components: usize,
    args: SrtCellPairsArgs,
) -> anyhow::Result<KnnGraph> {
    let n_cells = cell_proj.ncols();
    anyhow::ensure!(
        component_of_cell.len() == n_cells,
        "one component label per cell: got {} for {} cells",
        component_of_cell.len(),
        n_cells
    );

    let mut cells_in: Vec<Vec<usize>> = vec![Vec::new(); n_components];
    for (cell, &c) in component_of_cell.iter().enumerate() {
        if c < n_components {
            cells_in[c].push(cell);
        }
    }

    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut distances: Vec<f32> = Vec::new();
    let mut skipped = 0usize;
    for members in cells_in.iter() {
        if members.len() <= args.knn {
            skipped += members.len();
            continue;
        }
        // A component's own sub-projection, columns in `members` order.
        let sub = Mat::from_fn(cell_proj.nrows(), members.len(), |r, c| {
            cell_proj[(r, members[c])]
        });
        let g = build_expression_knn(&sub, SrtCellPairsArgs { ..args })?;
        for (&(i, j), &d) in g.edges.iter().zip(g.distances.iter()) {
            let (a, b) = (members[i], members[j]);
            edges.push((a.min(b), a.max(b)));
            distances.push(d);
        }
    }
    if skipped > 0 {
        info!(
            "{} cells sit in a spatial component too small for {} expression neighbours; \
             they get none",
            skipped, args.knn
        );
    }

    // Sorted before it leaves, so the result does not depend on the order the
    // components happened to be labelled in. That labelling comes from a
    // parallel union-find and varies between runs; without this the ranks the
    // union assigns to tied distances would move with it, and those ranks are
    // written to a file.
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_unstable_by_key(|&i| edges[i]);
    let edges: Vec<(usize, usize)> = order.iter().map(|&i| edges[i]).collect();
    let distances: Vec<f32> = order.iter().map(|&i| distances[i]).collect();

    // The one implementation of this invariant lives with the type: the COO to
    // CSC conversion SUMS entries sharing a coordinate, so an edge pushed twice
    // silently doubles its weight.
    let adjacency = matrix_util::knn_graph::symmetric_adjacency(n_cells, &edges, &distances);
    Ok(KnnGraph {
        adjacency,
        edges,
        distances,
        n_nodes: n_cells,
    })
}

/// The expression graph plus a 2D layout to stand in for real coordinates.
///
/// For the graph alone use [`build_expression_knn`]: the layout costs a
/// force-directed pass over the whole edge list and is pure waste when the
/// caller already has coordinates.
pub fn build_expression_graph(
    cell_proj: &Mat,
    args: SrtCellPairsArgs,
) -> anyhow::Result<(KnnGraph, Mat)> {
    let graph = build_expression_knn(cell_proj, args)?;

    // The layout wants one row per cell, unlike the KNN above.
    let embedding_nk = cell_proj.transpose();
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
