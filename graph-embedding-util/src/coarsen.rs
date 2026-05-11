//! Cell-axis coarsening for the gbe training loop. Every axis (per-cell
//! and each pb level) embeds its entities directly, so the coarsening
//! here is always identity. Kept only so `nce_loss` / `pool_cells` stay
//! generic over coarse→fine maps — identity is a zero-cost fast path
//! (see `is_identity` short-circuit in `loss.rs`).

use data_beans_alg::feature_coarsening::FeatureCoarsening;

pub struct AxisCoarsenings {
    pub coarsenings: Vec<FeatureCoarsening>,
    /// Set when every coarsening in `coarsenings` is the identity
    /// partition (each fine row is its own super-cell). Lets the
    /// training loop bypass `pool_cells`'s gather/scatter and use a
    /// single `index_select` per side. Set by [`identity_axis`].
    pub is_identity: bool,
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
        is_identity: true,
    }
}
