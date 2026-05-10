//! Cell-axis coarsening for the gbe training loop. With the two-stage
//! design (stage-1 pseudobulks, stage-2 per-cell), both stages embed
//! the underlying entities directly — every minibatch picks a "super-
//! cell" that maps to exactly one entity. So the coarsening here is
//! always the identity partition; the type stays around because
//! `nce_loss` / `pool_cells` are written generically over coarse →
//! fine maps and an identity partition is a 1-line zero-cost path
//! through the same code.

use data_beans_alg::feature_coarsening::FeatureCoarsening;

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

/// One identity coarsening over `n` items. Each item is its own
/// super-cell; `pool_cells` becomes a no-op (mean of one row = that
/// row) and `sample_edge_batch` resolves `coarse_cell == fine_cell`.
pub fn identity_axis(n: usize) -> AxisCoarsenings {
    let fine_to_coarse: Vec<usize> = (0..n).collect();
    let coarse_to_fine: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    AxisCoarsenings {
        coarsenings: vec![FeatureCoarsening {
            fine_to_coarse,
            coarse_to_fine,
            num_coarse: n,
        }],
    }
}
