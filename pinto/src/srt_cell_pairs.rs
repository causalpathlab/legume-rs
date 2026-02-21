use crate::srt_common::*;
use crate::srt_knn_graph::{KnnGraph, KnnGraphArgs};
use dashmap::DashMap;
use matrix_util::parquet::*;
use matrix_util::traits::RandomizedAlgs;
use matrix_util::utils::generate_minibatch_intervals;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use parquet::basic::Type as ParquetType;

pub struct SrtCellPairs<'a> {
    pub data: &'a SparseIoVec,
    pub coordinates: &'a Mat,
    pub graph: KnnGraph,
    pub pairs: Vec<Pair>,
}

pub struct SrtCellPairsArgs {
    pub knn: usize,
    pub block_size: usize,
    pub reciprocal: bool,
}

impl<'a> SrtCellPairs<'a> {
    /// number of pairs
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
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

        let mut column_names = vec![];
        let mut column_types = vec![];

        // 1. left data column names
        column_names.push("left_cell".to_string().into_boxed_str());
        column_types.push(ParquetType::BYTE_ARRAY);

        // 2. right data column names
        column_names.push("right_cell".to_string().into_boxed_str());
        column_types.push(ParquetType::BYTE_ARRAY);

        // 3. left coordinate names
        for x in coordinate_names.iter() {
            column_names.push(format!("left_{}", x).into_boxed_str());
            column_types.push(ParquetType::FLOAT);
        }

        // 4. right coordinate names
        for x in coordinate_names.iter() {
            column_names.push(format!("right_{}", x).into_boxed_str());
            column_types.push(ParquetType::FLOAT);
        }

        // 5. distance
        column_names.push("distance".to_string().into_boxed_str());
        column_types.push(ParquetType::FLOAT);

        //////////////////////////////////////
        // write them down column by column //
        //////////////////////////////////////

        let shape = (self.num_pairs(), column_names.len());

        let writer = ParquetWriter::new(
            file_path,
            shape,
            (None, Some(&column_names)),
            Some(&column_types),
            Some("cell_pair"),
        )?;
        let row_names = writer.row_names_vec();

        let mut writer = writer.get_writer()?;
        let mut row_group_writer = writer.next_row_group()?;

        // 0. row names
        parquet_add_bytearray(&mut row_group_writer, row_names)?;

        let cell_names = self.data.column_names()?;
        // 1. left column names
        parquet_add_string_column(
            &mut row_group_writer,
            &self
                .pairs
                .iter()
                .map(|pp| cell_names[pp.left].clone())
                .collect::<Vec<_>>(),
        )?;

        // 2. right column names
        parquet_add_string_column(
            &mut row_group_writer,
            &self
                .pairs
                .iter()
                .map(|pp| cell_names[pp.right].clone())
                .collect::<Vec<_>>(),
        )?;

        let (left_coord, right_coord) = self.all_pairs_positions()?;

        // 3. left coordinates
        for coord in left_coord.iter() {
            parquet_add_numeric_column(&mut row_group_writer, coord)?;
        }

        // 4. right coordinates
        for coord in right_coord.iter() {
            parquet_add_numeric_column(&mut row_group_writer, coord)?;
        }

        // 5. distances
        parquet_add_numeric_column(&mut row_group_writer, &self.graph.distances)?;

        row_group_writer.close()?;
        writer.close()?;
        Ok(())
    }

    ///
    /// Take all pairs' positions
    ///
    /// returns `(left_coordinates, right_coordinates)`
    ///
    #[allow(clippy::type_complexity)]
    pub fn all_pairs_positions(&self) -> anyhow::Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let left: Vec<Vec<f32>> = self
            .coordinates
            .column_iter()
            .map(|coord_vec| self.pairs.iter().map(|pp| coord_vec[pp.left]).collect())
            .collect();
        let right: Vec<Vec<f32>> = self
            .coordinates
            .column_iter()
            .map(|coord_vec| self.pairs.iter().map(|pp| coord_vec[pp.right]).collect())
            .collect();

        Ok((left, right))
    }

    /// visit cell pairs by regular-sized block
    ///
    /// A visitor function takes
    /// - `(lb,ub)` `(usize,usize)`
    /// - data itself
    /// - `shared_input`
    /// - `shared_out` (`Arc(Mutex())`)
    pub fn visit_pairs_by_block<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
        block_size: usize,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(
                (usize, usize),
                &SrtCellPairs,
                &SharedIn,
                Arc<Mutex<&mut SharedOut>>,
            ) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send,
    {
        let all_pairs = &self.pairs;
        let ntot = all_pairs.len();
        let jobs = generate_minibatch_intervals(ntot, block_size);
        let arc_shared_out = Arc::new(Mutex::new(shared_out));

        let pb = new_progress_bar(
            jobs.len() as u64,
            "Processing {bar:40} {pos}/{len} blocks ({eta})",
        );
        jobs.par_iter()
            .progress_with(pb)
            .map(|&(lb, ub)| -> anyhow::Result<()> {
                visitor((lb, ub), self, shared_in, arc_shared_out.clone())
            })
            .collect::<anyhow::Result<()>>()
    }

    pub fn num_coordinates(&self) -> usize {
        self.coordinates.ncols()
    }

    /// Wrap a pre-built KNN graph with data and coordinates.
    pub fn with_graph(
        data: &'a SparseIoVec,
        coordinates: &'a Mat,
        graph: KnnGraph,
    ) -> SrtCellPairs<'a> {
        let pairs = graph
            .edges
            .iter()
            .map(|&(i, j)| Pair { left: i, right: j })
            .collect::<Vec<_>>();
        SrtCellPairs {
            data,
            coordinates,
            graph,
            pairs,
        }
    }
}

/// Build a spatial KNN graph from coordinate matrix.
pub fn build_spatial_graph(coordinates: &Mat, args: SrtCellPairsArgs) -> anyhow::Result<KnnGraph> {
    let points = coordinates.transpose();
    KnnGraph::from_columns(
        &points,
        KnnGraphArgs {
            knn: args.knn,
            block_size: args.block_size,
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
    use rand::Rng;
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    fn make_test_graph(n_nodes: usize, edges: Vec<(usize, usize)>) -> KnnGraph {
        let distances = vec![1.0; edges.len()];
        let mut coo = CooMatrix::new(n_nodes, n_nodes);
        for &(i, j) in &edges {
            coo.push(i, j, 1.0f32);
            coo.push(j, i, 1.0f32);
        }
        let adjacency = CscMatrix::from(&coo);
        KnnGraph {
            adjacency,
            edges,
            distances,
            n_nodes,
        }
    }

    #[test]
    fn test_connected_components_single() {
        let graph = make_test_graph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        let (labels, n_components) = connected_components(&graph);
        assert_eq!(n_components, 1);
        assert!(labels.iter().all(|&l| l == labels[0]));
    }

    #[test]
    fn test_connected_components_two_cliques() {
        let graph = make_test_graph(6, vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]);
        let (labels, n_components) = connected_components(&graph);
        assert_eq!(n_components, 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_connected_components_isolates() {
        let graph = make_test_graph(4, vec![]);
        let (labels, n_components) = connected_components(&graph);
        assert_eq!(n_components, 4);
        let unique: HashSet<usize> = labels.iter().cloned().collect();
        assert_eq!(unique.len(), 4);
    }
}
