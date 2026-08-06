//! The component reference the Gibbs sampler allocates bulk counts against.
//!
//! Both reference modes reduce to the same object: `R` non-negative gene
//! profiles plus an `R×C` readout that maps components onto reported cell
//! types.
//!
//! * **low-rank** — `R = C`, one anchor per cell type, `μ_{g,c} = exp(ρ_g·t_c + a_g)`
//!   rebuilt every sweep as the anchors move, readout = identity.
//! * **archetype** — `R ≫ C` empirical profiles from collapsed annotated cells,
//!   fixed across sweeps, readout = the soft annotation posterior.

use crate::embed_common::Mat;

/// Clamp on the Poisson log-rate before `exp`.
pub const SCORE_CLAMP: f32 = 30.0;

pub struct Reference {
    /// Gene-major rates `mu_gm[g*n_comp + m] = μ_{g,m}`, so the inner per-component
    /// loop reads a contiguous, SIMD-friendly slice instead of a strided column.
    pub mu_gm: Vec<f32>,
    pub n_genes: usize,
    pub n_comp: usize,
    /// `R×C`, rows summing to 1: how much of component `m` counts as type `c`.
    pub readout: Mat,
    /// `R×H` component coordinates in the embedding, carried through to output.
    pub coords: Mat,
    /// Component row names (cell types, or archetype ids).
    pub comp_names: Vec<Box<str>>,
    pub celltype_names: Vec<Box<str>>,
}

impl Reference {
    #[must_use]
    pub fn n_types(&self) -> usize {
        self.readout.ncols()
    }

    /// `out[c] = Σ_m v[m]·A[m,c]` — component-indexed vector to cell types.
    pub fn contract(&self, v: &[f32], out: &mut [f32]) {
        out.fill(0.0);
        for (m, &vm) in v.iter().enumerate() {
            if vm == 0.0 {
                continue;
            }
            for (ct, o) in out.iter_mut().enumerate() {
                *o += vm * self.readout[(m, ct)];
            }
        }
    }

    /// Per-cell-type rate `μ_{g,c} = Σ_m μ_{g,m}·A[m,c] / Σ_m A[m,c]`, used only
    /// for the posterior-predictive residual report.
    #[must_use]
    pub fn type_rates(&self) -> Mat {
        let (d, r, c) = (self.n_genes, self.n_comp, self.n_types());
        let mut wsum = vec![0f32; c];
        for m in 0..r {
            for (ct, w) in wsum.iter_mut().enumerate() {
                *w += self.readout[(m, ct)];
            }
        }
        Mat::from_fn(d, c, |g, ct| {
            if wsum[ct] <= 0.0 {
                return 0.0;
            }
            let mut acc = 0f32;
            for m in 0..r {
                acc += self.mu_gm[g * r + m] * self.readout[(m, ct)];
            }
            acc / wsum[ct]
        })
    }
}

/// `C×C` identity readout — components already are the reported cell types.
#[must_use]
pub fn identity_readout(c: usize) -> Mat {
    Mat::from_fn(c, c, |i, j| if i == j { 1.0 } else { 0.0 })
}
