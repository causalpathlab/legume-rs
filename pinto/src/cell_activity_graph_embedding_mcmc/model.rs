//! Mutable model state + numerically-stable softplus_floored, on
//! plain `nalgebra::DMatrix` / `DVector`. No Candle, no autograd.

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Linear-floor coefficient mirroring
/// `cell_activity_graph_embedding::gene_gating::GAMMA_EPS`. Keeps γ moving
/// after the softplus saturates on the positive side.
pub const GAMMA_EPS: f32 = 1e-2;

#[derive(Clone)]
pub struct McmcState {
    pub e_cell: DMatrix<f32>, // [N, D]
    pub e_gene: DMatrix<f32>, // [G, D]
    pub gamma: DMatrix<f32>,  // [L, D] pre-softplus
    pub b_cell: DVector<f32>, // [N]
}

/// Draw an `[rows, cols]` matrix with i.i.d. `N(0, sigma²)` entries.
pub fn randn_matrix<R: Rng>(rng: &mut R, rows: usize, cols: usize, sigma: f32) -> DMatrix<f32> {
    let n = rows * cols;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let v: f64 = StandardNormal.sample(rng);
        data.push(v as f32 * sigma);
    }
    DMatrix::from_vec(rows, cols, data)
}

/// Draw a length-`n` vector with i.i.d. `N(0, sigma²)` entries.
pub fn randn_vector<R: Rng>(rng: &mut R, n: usize, sigma: f32) -> DVector<f32> {
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let v: f64 = StandardNormal.sample(rng);
        data.push(v as f32 * sigma);
    }
    DVector::from_vec(data)
}

/// `softplus(x) + GAMMA_EPS · relu(x)`: numerically stable `softplus`
/// (`max(x, 0) + ln_1p(exp(-|x|))`) with a linear-from-0 gradient floor
/// on the positive side so γ never gets stuck at saturation. Matches
/// `cell_activity_graph_embedding::gene_gating::softplus_floored` on
/// Candle tensors up to dtype.
#[inline]
fn softplus_floored_scalar(x: f32) -> f32 {
    let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
    softplus + if x > 0.0 { GAMMA_EPS * x } else { 0.0 }
}

pub fn softplus_floored(src: &DMatrix<f32>) -> DMatrix<f32> {
    src.map(softplus_floored_scalar)
}
