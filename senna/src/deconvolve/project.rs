//! Project bulk samples into the shared embedding latent and initialize
//! cell-type fractions.
//!
//! The bulk-into-embedding projection is the cross-platform bridge: each bulk
//! sample is embedded by the same frozen-ρ Poisson solver
//! ([`graph_embedding_util::cell_projection::project_cells`]) that placed the
//! reference cells, so bulk, cells, and anchors share one geometry. The
//! projected position also seeds the Gibbs sampler via a simplex fit against
//! the anchor rows.

use crate::embed_common::{DVec, Mat};
use graph_embedding_util::cell_projection::project_cells;
use matrix_util::archetypal::simplex_lsq;

/// Project each bulk sample (a column of `bulk`, genes × samples) into the ρ
/// latent under the Poisson rate `μ = exp(ρ·z + a_g + b_s)`. Returns the `S×H`
/// sample embeddings; the per-sample depth intercept is fitted internally and
/// discarded (it only absorbs library size).
#[must_use]
pub fn project_bulk(rho: &Mat, gene_offset: &DVec, bulk: &Mat, ridge: f64) -> Mat {
    let (d, h) = (rho.nrows(), rho.ncols());
    let s = bulk.ncols();

    // project_cells wants ρ row-major [D×H] + per-sample sparse (gene, count)
    // lists. `Mat` is column-major, so ρᵀ's buffer is exactly the row-major ρ.
    let rho_rm = rho.transpose();
    let frozen_b: Vec<f32> = gene_offset.iter().copied().collect();
    let per_cell: Vec<Vec<(u32, f32)>> = (0..s)
        .map(|si| {
            (0..d)
                .filter_map(|g| {
                    let y = bulk[(g, si)];
                    (y > 0.0).then_some((g as u32, y))
                })
                .collect()
        })
        .collect();

    let (e, _b_cell) = project_cells(rho_rm.as_slice(), &frozen_b, &per_cell, h, ridge, None);
    Mat::from_row_slice(s, h, &e)
}

/// Warm-start fractions `w` `[S×C]`: each sample's projected position is fit as
/// a simplex combination of the anchor rows (`anchor_mean` `[C×H]`). A small
/// floor keeps every type strictly positive for the Gamma sampler.
#[must_use]
pub fn init_fractions(anchor_mean: &Mat, z: &Mat, fw_iters: usize) -> Mat {
    let (s, c, h) = (z.nrows(), anchor_mean.nrows(), z.ncols());
    let mut w = Mat::zeros(s, c);
    for si in 0..s {
        let x = DVec::from_iterator(h, z.row(si).iter().copied());
        let a = simplex_lsq(anchor_mean, &x, fw_iters);
        for ct in 0..c {
            w[(si, ct)] = a[ct].max(1e-6);
        }
    }
    w
}
