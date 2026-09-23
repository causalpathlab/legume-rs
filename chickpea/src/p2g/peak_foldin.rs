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

/// The per-unit part of the step, shared by every peak: the `s`-weighted mean
/// `ē`, the centred embeddings `Ẽ`, and `(ẼᵀSẼ + λI)⁻¹`.
pub struct FoldInDesign {
    e_c: DMatrix<f64>,
    e_bar: DVector<f64>,
    inv: DMatrix<f64>,
    total_s: f64,
}

impl FoldInDesign {
    /// `e` is `[units × H]`, `size` the unit sizes; `ridge` is relative to the
    /// mean diagonal of `ẼᵀSẼ`.
    pub fn new(e: &DMatrix<f32>, size: &[f32], ridge: f32) -> anyhow::Result<Self> {
        let (n_units, h) = e.shape();
        anyhow::ensure!(
            size.len() == n_units,
            "{} unit sizes for {n_units} units",
            size.len()
        );
        let e = e.map(f64::from);
        let s = DVector::from_iterator(n_units, size.iter().map(|&v| f64::from(v)));
        let total_s = s.sum();
        anyhow::ensure!(total_s > 0.0, "the units have no size");
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
        Ok(Self {
            e_c,
            e_bar,
            inv,
            total_s,
        })
    }

    #[must_use]
    pub fn n_units(&self) -> usize {
        self.e_c.nrows()
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.e_c.ncols()
    }

    /// One peak's row and bias from `v = Σ_u n_pu ẽ_u` and `total_n = Σ_u n_pu`.
    fn solve(&self, v: &DVector<f64>, total_n: f64) -> (Vec<f32>, f32) {
        let h = self.dim();
        if total_n <= 0.0 {
            return (vec![0.0; h], (0.5 / self.total_s).ln() as f32);
        }
        let b_hat = (total_n / self.total_s).ln();
        let phi = &self.inv * v * (-b_hat).exp();
        let b = b_hat - self.e_bar.dot(&phi);
        (phi.iter().map(|&x| x as f32).collect(), b as f32)
    }

    /// Every peak's row and bias from accumulated moments.
    #[must_use]
    pub fn finish(&self, m: &PeakMoments) -> PeakFoldIn {
        let rows: Vec<(Vec<f32>, f32)> = (0..m.total.len())
            .into_par_iter()
            .map(|p| self.solve(&m.v.row(p).transpose(), m.total[p]))
            .collect();
        collect(rows, self.dim())
    }
}

/// Per-peak moments `Σ_u n_pu ẽ_u` `[peaks × H]` and `Σ_u n_pu`, accumulated
/// from cells: a unit's counts are the sums of its cells', so adding each
/// cell's counts at its unit's `ẽ_u` gives the same moments.
pub struct PeakMoments {
    v: DMatrix<f64>,
    total: Vec<f64>,
}

impl PeakMoments {
    #[must_use]
    pub fn new(n_peaks: usize, h: usize) -> Self {
        Self {
            v: DMatrix::zeros(n_peaks, h),
            total: vec![0.0; n_peaks],
        }
    }

    /// Add one cell of unit `u`, given as its `(peak, count)` entries.
    pub fn add_cell(&mut self, design: &FoldInDesign, u: usize, cell: &[(u32, f32)]) {
        let e_u = design.e_c.row(u);
        for &(p, x) in cell {
            let x = f64::from(x);
            let p = p as usize;
            for k in 0..e_u.len() {
                self.v[(p, k)] += x * e_u[k];
            }
            self.total[p] += x;
        }
    }
}

fn collect(rows: Vec<(Vec<f32>, f32)>, h: usize) -> PeakFoldIn {
    let phi = DMatrix::from_fn(rows.len(), h, |p, k| rows[p].0[k]);
    let bias = rows.into_iter().map(|(_, b)| b).collect();
    PeakFoldIn { phi, bias }
}

/// `counts` is `[peaks × units]`, `size` the unit sizes, `e` `[units × H]`.
/// `ridge` is relative to the mean diagonal of `ẼᵀSẼ`.
pub fn fold_in_peaks(
    counts: &CsrMatrix<f32>,
    size: &[f32],
    e: &DMatrix<f32>,
    ridge: f32,
) -> anyhow::Result<PeakFoldIn> {
    let design = FoldInDesign::new(e, size, ridge)?;
    anyhow::ensure!(
        counts.ncols() == design.n_units(),
        "counts have {} units, embeddings {}",
        counts.ncols(),
        design.n_units()
    );
    let h = design.dim();
    let rows: Vec<(Vec<f32>, f32)> = (0..counts.nrows())
        .into_par_iter()
        .map(|p| {
            let row = counts.row(p);
            let mut v = DVector::<f64>::zeros(h);
            let mut total_n = 0.0;
            for (&u, &n) in row.col_indices().iter().zip(row.values()) {
                v += design.e_c.row(u).transpose() * f64::from(n);
                total_n += f64::from(n);
            }
            design.solve(&v, total_n)
        })
        .collect();
    Ok(collect(rows, h))
}
