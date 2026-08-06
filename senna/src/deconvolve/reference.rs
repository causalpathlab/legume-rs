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

pub struct Reference {
    /// Gene-major rates `mu_gm[g*n_comp + m] = μ_{g,m}`, so the inner per-component
    /// loop reads a contiguous, SIMD-friendly slice instead of a strided column.
    pub mu_gm: Vec<f32>,
    /// `R×C`, rows summing to 1: how much of component `m` counts as type `c`.
    pub readout: Mat,
    /// `R×H` component coordinates in the embedding, carried through to output.
    pub coords: Mat,
    /// Component row names (cell types, or archetype ids).
    pub comp_names: Vec<Box<str>>,
    /// Cells behind each component, counted over the cells that actually formed
    /// its readout row.
    pub n_cells: Vec<f32>,
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
        refresh_low_rank(
            &mut mu_gm,
            &prior.mean,
            &src.rho.transpose(),
            &src.gene_offset,
            c,
        );
        Self {
            mu_gm,
            readout: Mat::identity(c, c),
            coords: prior.mean.clone(),
            comp_names: prior.names.clone(),
            n_cells: vec![0.0; c],
            celltype_names: prior.names.clone(),
            units: FractionUnits::Cell,
            axis: ComponentAxis::CellType,
        }
    }

    #[must_use]
    pub fn n_types(&self) -> usize {
        self.readout.ncols()
    }

    #[must_use]
    pub fn n_comp(&self) -> usize {
        self.readout.nrows()
    }

    #[must_use]
    pub fn n_genes(&self) -> usize {
        self.mu_gm.len() / self.n_comp().max(1)
    }

    /// Warm start for the abundances, `S×R`.
    ///
    /// How to start follows from how the profiles are scaled, which is the
    /// reference's business rather than the caller's: normalised profiles make
    /// an abundance a count, so an even split of the sample's total is the
    /// uniform-composition point; reconstructed rates carry their own exposure
    /// and start neutral, with the caller free to supply a better guess.
    #[must_use]
    pub fn init_abundances(&self, bulk: &Mat) -> Mat {
        let r = self.n_comp();
        match self.units {
            FractionUnits::Cell => Mat::from_element(bulk.ncols(), r, 1.0),
            FractionUnits::Mrna => {
                let totals: Vec<f32> = (0..bulk.ncols())
                    .map(|si| (bulk.column(si).iter().sum::<f32>() / r as f32).max(1e-6))
                    .collect();
                Mat::from_fn(bulk.ncols(), r, |si, _| totals[si])
            }
        }
    }

    /// Per-component prior shape when the user did not set one.
    ///
    /// Two standard deviations of the counts a component would hold under a
    /// uniform split. The gene allocation is winner-take-all among overlapping
    /// profiles, so a component that falls behind early is extinguished; holding
    /// it a couple of sampling-noise units off zero is what prevents that, and
    /// that scale is `sqrt(N/R)`. With few, well-separated components — the
    /// low-rank case — nothing is at risk and a weak prior is right.
    #[must_use]
    pub fn default_prior_shape(&self, bulk: &Mat) -> f32 {
        match self.units {
            FractionUnits::Cell => 1.0,
            FractionUnits::Mrna => {
                let mean_total = f64::from(bulk.sum()) / bulk.ncols().max(1) as f64;
                (2.0 * (mean_total / self.n_comp().max(1) as f64).sqrt()).max(1.0) as f32
            }
        }
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
}

/// Label for one component in the pooled output.
///
/// Pooled chains hold different partitions, so a component's label carries its
/// chain. This is the join key between the component tables and the abundance
/// columns, so it must be produced in exactly one place.
#[must_use]
pub fn comp_label(n_chains: usize, chain: usize, name: &str) -> Box<str> {
    if n_chains > 1 {
        format!("c{chain}_{name}").into_boxed_str()
    } else {
        name.into()
    }
}

/// Rebuild `μ_{g,c} = exp(ρ_g·t_c + a_g)` from anchor rows `tmat` (`C×H`).
///
/// `tmat·ρᵀ` is `C×D` column-major, whose buffer index `g*C+ct` already equals
/// `ρ_g·t_c` — exactly the gene-major layout the sampler reads.
pub fn refresh_low_rank(
    mu_gm: &mut [f32],
    tmat: &Mat,
    rho_t: &Mat,
    gene_offset: &crate::embed_common::DVec,
    c: usize,
) {
    let scd = tmat * rho_t;
    let scd_buf = scd.as_slice();
    let a = gene_offset;
    for (i, mu) in mu_gm.iter_mut().enumerate() {
        *mu = (scd_buf[i] + a[i / c])
            .clamp(-SCORE_CLAMP, SCORE_CLAMP)
            .exp();
    }
}
