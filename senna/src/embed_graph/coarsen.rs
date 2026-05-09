//! Multi-seed sketch coarsening for cells. K `FeatureCoarsening`s
//! built from random-projection sketches over the triplet stream.

use crate::embed_common::*;
use crate::embed_graph::data::Triplet;
use data_beans_alg::feature_coarsening::compute_feature_coarsening;
use nalgebra::DMatrix;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

pub struct AxisCoarsenings {
    pub coarsenings: Vec<FeatureCoarsening>,
}

impl AxisCoarsenings {
    pub fn avg_n_coarse(&self) -> f32 {
        if self.coarsenings.is_empty() {
            return 0.0;
        }
        let sum: usize = self.coarsenings.iter().map(|c| c.num_coarse).sum();
        sum as f32 / self.coarsenings.len() as f32
    }
}

pub struct CellCoarseningArgs<'a> {
    pub triplets: &'a [Triplet],
    pub n_cells: usize,
    pub n_features: usize,
    pub target_blocks: usize,
    pub sketch_dim: usize,
    pub n_seeds: usize,
    pub base_seed: u64,
}

// par_iter over seeds is safe to add: `compute_feature_coarsening`
// is not internally rayon-parallel, so this doesn't nest.
pub fn build_cell_coarsenings(args: CellCoarseningArgs) -> anyhow::Result<AxisCoarsenings> {
    let coarsenings: anyhow::Result<Vec<FeatureCoarsening>> = (0..args.n_seeds)
        .into_par_iter()
        .map(|k| {
            let seed = args.base_seed.wrapping_add(k as u64);
            let sketch = sketch_cells_via_random_projection(
                args.triplets,
                args.n_cells,
                args.n_features,
                args.sketch_dim,
                seed,
            );
            let fc = compute_feature_coarsening(&sketch, args.target_blocks)?;
            info!(
                "Cell coarsening seed={}: {} → {} blocks",
                k, args.n_cells, fc.num_coarse
            );
            Ok(fc)
        })
        .collect();

    Ok(AxisCoarsenings {
        coarsenings: coarsenings?,
    })
}

fn sketch_cells_via_random_projection(
    triplets: &[Triplet],
    n_cells: usize,
    n_features: usize,
    sketch_dim: usize,
    seed: u64,
) -> DMatrix<f32> {
    let r = gaussian_matrix(n_features, sketch_dim, seed);
    let mut sketch = DMatrix::<f32>::zeros(n_cells, sketch_dim);
    for t in triplets {
        let c = t.cell as usize;
        let f = t.feature as usize;
        let row_r = r.row(f);
        let mut row = sketch.row_mut(c);
        for s in 0..sketch_dim {
            row[s] += t.count * row_r[s];
        }
    }
    sketch
}

fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> DMatrix<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    DMatrix::from_fn(rows, cols, |_, _| {
        let v: f32 = rng.sample(StandardNormal);
        v
    })
}
