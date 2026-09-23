//! Peak rows folded in against frozen pseudobulk embeddings, all peaks at once.
//!
//! For peak `p` over units `u` with sizes `s_u` and embeddings `e_u`:
//!
//! ```text
//! n_pu ~ Poisson( s_u · exp(<e_u, φ_p> + b_p) )
//! ```
//!
//! One Newton (IRLS) step from `φ_p = 0`, `b̂_p = ln(Σ_u n_pu / Σ_u s_u)`. At
//! that point every peak's IRLS weights are `s_u · e^{b̂_p}`: the unit sizes
//! times a per-peak constant that cancels, so the weights are shared exactly.
//! Centring `e_u` by its `s`-weighted mean `ē` decouples `φ` from the
//! intercept, and the step is
//!
//! ```text
//! φ_p = (ẼᵀSẼ + λI)⁻¹ · Ẽᵀ n_p · e^{-b̂_p},      b_p = b̂_p − <ē, φ_p>
//! ```
//!
//! One sparse × dense pass over the counts plus one `H × H` solve, whatever the
//! number of peaks. The only error is stopping after one step: exact to first
//! order in `<e_u, φ_p>`, a direction estimate beyond it.

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::*;

/// Folded-in peak rows `[peaks × H]` and biases.
pub struct PeakFoldIn {
    pub phi: DMatrix<f32>,
    pub bias: Vec<f32>,
}

/// `counts` is `[peaks × units]`, `size` the unit sizes, `e` `[units × H]`.
/// `ridge` is relative to the mean diagonal of `ẼᵀSẼ`.
pub fn fold_in_peaks(
    counts: &CsrMatrix<f32>,
    size: &[f32],
    e: &DMatrix<f32>,
    ridge: f32,
) -> anyhow::Result<PeakFoldIn> {
    let (n_units, h) = e.shape();
    anyhow::ensure!(
        counts.ncols() == n_units,
        "counts have {} units, embeddings {n_units}",
        counts.ncols()
    );
    anyhow::ensure!(
        size.len() == n_units,
        "{} unit sizes for {n_units} units",
        size.len()
    );
    let e = e.map(f64::from);
    let s = DVector::from_iterator(n_units, size.iter().map(|&v| f64::from(v)));
    let total_s = s.sum();
    anyhow::ensure!(total_s > 0.0, "the units have no size");

    // Centre by the s-weighted mean; Ẽᵀ S Ẽ + λI, inverted once.
    let e_bar = e.transpose() * &s / total_s;
    let e_c = DMatrix::from_fn(n_units, h, |u, k| e[(u, k)] - e_bar[k]);
    let mut gram = e_c.transpose() * DMatrix::from_diagonal(&s) * &e_c;
    let lambda = f64::from(ridge) * gram.trace() / h as f64;
    for k in 0..h {
        gram[(k, k)] += lambda;
    }
    let inv = gram
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("the unit embeddings are rank-deficient"))?;

    let rows: Vec<(Vec<f32>, f32)> = (0..counts.nrows())
        .into_par_iter()
        .map(|p| {
            let row = counts.row(p);
            let total_n: f64 = row.values().iter().map(|&v| f64::from(v)).sum();
            if total_n <= 0.0 {
                return (vec![0.0; h], (0.5 / total_s).ln() as f32);
            }
            let b_hat = (total_n / total_s).ln();
            let scale = (-b_hat).exp();
            let mut v = DVector::<f64>::zeros(h);
            for (&u, &n) in row.col_indices().iter().zip(row.values()) {
                v += e_c.row(u).transpose() * (f64::from(n) * scale);
            }
            let phi = &inv * v;
            let b = b_hat - e_bar.dot(&phi);
            (phi.iter().map(|&x| x as f32).collect(), b as f32)
        })
        .collect();

    let phi = DMatrix::from_fn(rows.len(), h, |p, k| rows[p].0[k]);
    let bias = rows.into_iter().map(|(_, b)| b).collect();
    Ok(PeakFoldIn { phi, bias })
}
