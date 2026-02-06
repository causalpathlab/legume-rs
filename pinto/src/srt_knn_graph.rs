use crate::srt_common::*;

use dashmap::DashMap;
use nalgebra_sparse::{CooMatrix, CscMatrix};

#[allow(dead_code)]
pub struct KnnGraph {
    /// Symmetric CSC adjacency matrix (n_nodes x n_nodes)
    pub adjacency: CscMatrix<f32>,
    /// Sorted edge list (i < j), deduplicated
    pub edges: Vec<(usize, usize)>,
    /// Edge distances/weights, parallel to `edges`
    pub distances: Vec<f32>,
    /// Number of nodes
    pub n_nodes: usize,
}

pub struct KnnGraphArgs {
    pub knn: usize,
    pub block_size: usize,
}

#[allow(dead_code)]
impl KnnGraph {
    /// Build a KNN graph from column vectors.
    ///
    /// * `points` - transposed coordinate matrix (d x n), where each column is a point
    /// * `args` - KNN graph construction parameters
    pub fn from_columns(points: &Mat, args: KnnGraphArgs) -> anyhow::Result<KnnGraph> {
        let nn = points.ncols();
        let points_vec = points.column_iter().collect::<Vec<_>>();
        let names = (0..nn).collect::<Vec<_>>();

        let dict = ColumnDict::from_dvector_views(points_vec, names);
        let nquery = (args.knn + 1).min(nn).max(2);

        let jobs = create_jobs(nn, Some(args.block_size));
        let njobs = jobs.len() as u64;

        /////////////////////////////////////////////////////////////////
        // step 1: searching nearest neighbours in spatial coordinates //
        /////////////////////////////////////////////////////////////////

        let triplets: DashMap<(usize, usize), f32> = DashMap::new();

        jobs.into_par_iter()
            .progress_count(njobs)
            .try_for_each(|(lb, ub)| -> anyhow::Result<()> {
                for i in lb..ub {
                    let (_indices, _distances) = dict.search_others(&i, nquery)?;
                    for (j, d_ij) in _indices.into_iter().zip(_distances) {
                        triplets.insert((i, j), d_ij);
                    }
                }
                Ok(())
            })?;

        info!("{} triplets by spatial kNN matching", triplets.len());

        if triplets.is_empty() {
            return Err(anyhow::anyhow!("empty triplets"));
        }

        //////////////////////////////////////////////////
        // step 2: reciprocal filtering (i <-> j edges) //
        //////////////////////////////////////////////////

        let mut edges: Vec<((usize, usize), f32)> = triplets
            .par_iter()
            .filter_map(|entry| {
                let &(i, j) = entry.key();
                if i < j && triplets.contains_key(&(j, i)) {
                    Some(((i, j), *entry.value()))
                } else {
                    None
                }
            })
            .collect();

        edges.par_sort_by_key(|&(ij, _)| ij);
        edges.dedup();

        info!("{} triplets after reciprocal matching", edges.len());

        ///////////////////////////////////////////////
        // step 3: construct sparse network backbone //
        ///////////////////////////////////////////////

        let mut coo = CooMatrix::new(nn, nn);
        for &((i, j), v) in edges.iter() {
            coo.push(i, j, v);
            coo.push(j, i, v);
        }

        let adjacency = CscMatrix::from(&coo);

        let (edge_pairs, distances): (Vec<_>, Vec<_>) = edges.into_iter().unzip();

        Ok(KnnGraph {
            adjacency,
            edges: edge_pairs,
            distances,
            n_nodes: nn,
        })
    }

    /// Get neighbors of a node from the CSC adjacency matrix
    pub fn neighbors(&self, node: usize) -> &[usize] {
        let offsets = self.adjacency.col_offsets();
        let start = offsets[node];
        let end = offsets[node + 1];
        &self.adjacency.row_indices()[start..end]
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }
}
