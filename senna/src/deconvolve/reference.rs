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

use super::anchors::AnchorPrior;
use super::source::EmbeddingSource;
use crate::embed_common::Mat;

/// Clamp on the Poisson log-rate before `exp`.
pub const SCORE_CLAMP: f32 = 30.0;

/// What the reported fractions are shares of.
///
/// This follows from how the component profiles are scaled, not from which mode
/// produced them: profiles normalised to sum to one over genes make an abundance
/// an mRNA mass, while a per-type exposure that carries the transcript load makes
/// it a cell count. Carrying it on the reference keeps the manifest honest if a
/// third reference is added.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FractionUnits {
    Cell,
    Mrna,
}

impl FractionUnits {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cell => "cell",
            Self::Mrna => "mrna",
        }
    }
}

/// What the rows of the component table are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ComponentAxis {
    CellType,
    Archetype,
}

impl ComponentAxis {
    /// Row-name column header for the component parquet.
    #[must_use]
    pub fn corner(self) -> &'static str {
        match self {
            Self::CellType => "celltype",
            Self::Archetype => "archetype",
        }
    }
}

#[derive(Clone)]
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
    pub units: FractionUnits,
    pub axis: ComponentAxis,
}

impl Reference {
    /// One reconstructed profile per cell type, at the prior anchor positions.
    ///
    /// Built complete rather than zero-filled: the anchor sampler refreshes the
    /// rates as it moves the anchors, but a `Reference` should never exist in a
    /// state where its rates are a lie waiting for someone to call `refresh`.
    #[must_use]
    pub fn low_rank(src: &EmbeddingSource, prior: &AnchorPrior, d: usize) -> Self {
        let c = prior.mean.nrows();
        let mut mu_gm = vec![0.0f32; d * c];
        refresh_low_rank(&mut mu_gm, &prior.mean, src, c);
        Self {
            mu_gm,
            n_genes: d,
            n_comp: c,
            readout: Mat::identity(c, c),
            coords: prior.mean.clone(),
            comp_names: prior.names.clone(),
            celltype_names: prior.names.clone(),
            units: FractionUnits::Cell,
            axis: ComponentAxis::CellType,
        }
    }

    #[must_use]
    pub fn n_types(&self) -> usize {
        self.readout.ncols()
    }

    /// `out[c] = Σ_m v[m]·A[m,c]` — component-indexed vector to cell types.
    ///
    /// The zero skip matters: with many archetypes most components hold no mass
    /// in a given draw, and the readout is a strided column read.
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
    ///
    /// `mu_gm` is already an `R×D` column-major matrix, so the contraction is one
    /// gemm rather than a `D·C·R` scan — at D = 36 000 and R = 600 the scan
    /// streams the whole 86 MB buffer once per cell type.
    #[must_use]
    pub fn type_rates(&self) -> Mat {
        let (d, r, c) = (self.n_genes, self.n_comp, self.n_types());
        let mu_rd = Mat::from_column_slice(r, d, &self.mu_gm);
        // (C×R)·(R×D) = C×D, then transpose to the D×C the residual expects.
        let mut out = (self.readout.transpose() * mu_rd).transpose();
        for ct in 0..c {
            let wsum: f32 = self.readout.column(ct).sum();
            if wsum > 0.0 {
                out.column_mut(ct).scale_mut(1.0 / wsum);
            }
        }
        out
    }
}

/// Rebuild `μ_{g,c} = exp(ρ_g·t_c + a_g)` from anchor rows `tmat` (`C×H`).
///
/// `tmat·ρᵀ` is `C×D` column-major, whose buffer index `g*C+ct` already equals
/// `ρ_g·t_c` — exactly the gene-major layout the sampler reads.
pub fn refresh_low_rank(mu_gm: &mut [f32], tmat: &Mat, src: &EmbeddingSource, c: usize) {
    let scd = tmat * src.rho.transpose();
    let scd_buf = scd.as_slice();
    let a = &src.gene_offset;
    for (i, mu) in mu_gm.iter_mut().enumerate() {
        *mu = (scd_buf[i] + a[i / c])
            .clamp(-SCORE_CLAMP, SCORE_CLAMP)
            .exp();
    }
}
