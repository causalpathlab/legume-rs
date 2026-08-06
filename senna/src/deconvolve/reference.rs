//! The component reference the Gibbs sampler allocates bulk counts against.
//!
//! `R` non-negative gene profiles plus an `R×C` readout that maps components
//! onto reported cell types.
//!
//! Components are empirical profiles from collapsed annotated cells, fixed
//! across sweeps, with a readout given by the annotation posterior.

use crate::embed_common::Mat;

pub struct Reference {
    /// Gene-major rates `mu_gm[g*n_comp + m] = μ_{g,m}`, so the inner per-component
    /// loop reads a contiguous, SIMD-friendly slice instead of a strided column.
    pub mu_gm: Vec<f32>,
    /// `R×C`, rows summing to 1: how much of component `m` counts as type `c`.
    pub readout: Mat,
    /// `R×H` component coordinates in the embedding, carried through to output.
    pub coords: Mat,
    /// Component row names.
    pub comp_names: Vec<Box<str>>,
    /// Cells behind each component, counted over the cells that actually formed
    /// its readout row.
    pub n_cells: Vec<f32>,
    pub celltype_names: Vec<Box<str>>,
}

impl Reference {
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
    /// Profiles are normalised over genes, so an abundance is on the scale of
    /// allocated counts and an even split of the sample's total is the
    /// uniform-composition point.
    #[must_use]
    pub fn init_abundances(&self, bulk: &Mat) -> Mat {
        let r = self.n_comp();
        let totals: Vec<f32> = (0..bulk.ncols())
            .map(|si| (bulk.column(si).iter().sum::<f32>() / r as f32).max(1e-6))
            .collect();
        Mat::from_fn(bulk.ncols(), r, |si, _| totals[si])
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

/// Per-component prior shape when the user did not set one.
///
/// Two standard deviations of the counts a component would hold under a uniform
/// split. The gene allocation is winner-take-all among overlapping profiles, so
/// a component that falls behind early is extinguished; holding it a couple of
/// sampling-noise units off zero is what prevents that, and that scale is
/// `sqrt(N/R)`.
///
/// `n_comp` is the mean across the pooled chains, deliberately, so the prior is
/// **identical for every chain**. Deriving it from each chain's own component
/// count would give the chains different priors, and then they would target
/// different posteriors: pooling their draws would average three models rather
/// than one model over a partition, and the between-chain R̂ would be measuring
/// the prior as much as the partition.
#[must_use]
pub fn default_prior_shape(bulk: &Mat, n_comp: usize) -> f32 {
    let mean_total = f64::from(bulk.sum()) / bulk.ncols().max(1) as f64;
    (2.0 * (mean_total / n_comp.max(1) as f64).sqrt()).max(1.0) as f32
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
