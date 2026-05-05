//! Reference-conditioned copula structure used by `run_simulate` when a
//! `--reference` is supplied. The simulator's GLM (`β·θ·δ` + housekeeping)
//! is unchanged — the reference contributes only:
//!
//! 1. `r̂_g`, a per-gene NB dispersion (method-of-moments, pooled across
//!    all reference cells), and
//! 2. `Σ̂`, a single global Gaussian copula on PIT z-scores of HVGs,
//!    represented as a low-rank factor `F = U·diag(σ)/√N` plus an
//!    isotropic ridge `√λ I`. The dense `G × G` covariance is never
//!    materialized.
//!
//! Sampling at run time, per synthetic cell `j` with GLM mean `λ_{g,j}`:
//!
//! ```text
//! z* ∈ ℝ^G          = F · η + √λ · ε,  η ~ N(0, I_r),  ε ~ N(0, I_g)
//! For HVG genes g:  y_{g,j} = F⁻¹_NB(Φ(z*_g);  μ = λ_{g,j},  r = r̂_g)
//! For non-HVG g:    y_{g,j} ~ NB(λ_{g,j}, r̂_g)  (independent draw)
//! ```
//!
//! References:
//! - Li & Li 2019. *A statistical simulator scDesign for rational scRNA-seq
//!   experimental design.* Bioinformatics 35(14):i41–i50.
//! - Sun, Song, Li & Li 2021. *scDesign2.* Genome Biology 22:163.
//! - Song et al. 2024. *scDesign3.* Nature Biotechnology 42(2):247–252.

pub mod gaussian;
pub mod marginals;
pub mod reference;

use gaussian::CopulaCovariance;
use log::info;
use reference::SparseRef;

/// Output of [`fit_global_copula`]. Owns the fitted reference dispersions
/// and the low-rank Gaussian copula factor — everything needed to drive
/// the per-cell NB+copula sampler. The reference μ̂_g is intentionally
/// not exposed: synthetic cells get their μ from the GLM (`λ_{g,j}`),
/// not from the reference.
pub struct GlobalCopulaFit {
    pub gene_names: Vec<Box<str>>,
    pub n_genes: usize,
    /// Indices into `gene_names` chosen as HVGs for the copula structure.
    pub hvg_indices: Vec<usize>,
    /// Per-gene NB size `r̂_g`, length `n_genes`. `f32::INFINITY` signals
    /// the Poisson collapse from MoM (`σ² ≤ μ`).
    pub r_hat: Vec<f32>,
    /// `hvg_pos[g] = Some(h)` iff gene `g` is the `h`-th HVG; `None` for
    /// non-HVG genes. Length = `n_genes`.
    pub hvg_pos: Vec<Option<u32>>,
    pub copula: CopulaCovariance,
}

pub struct GlobalCopulaArgs<'a> {
    pub sc: &'a SparseRef,
    /// HVGs included in the copula dependence structure. Genes outside the
    /// HVG set are sampled independently from their NB fits at run time.
    pub n_hvg: usize,
    /// Maximum rank of the low-rank Σ̂ factor. Effective rank is
    /// `min(rank, n_hvg, n_cells)`.
    pub copula_rank: usize,
    /// Per-gene isotropic ridge variance added at sample time.
    pub regularization: f32,
    /// Lower bound on `r̂_g` to keep MoM stable for noisy genes.
    pub r_floor: f32,
}

/// Fit a single global copula on the entire reference: per-gene NB MoM
/// dispersion, then RSVD-based low-rank Σ̂ on the PIT z-scores of HVGs.
pub fn fit_global_copula(args: &GlobalCopulaArgs) -> anyhow::Result<GlobalCopulaFit> {
    let n_genes = args
        .sc
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("reference has no num_rows"))?;
    let n_cells = args
        .sc
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("reference has no num_columns"))?;
    if n_cells < 2 {
        anyhow::bail!("reference has only {} cells; need ≥2 to fit a copula", n_cells);
    }
    let gene_names = args.sc.row_names()?;
    info!("reference: {} genes × {} cells", n_genes, n_cells);

    let cells: Vec<usize> = (0..n_cells).collect();
    let (stats, marginals) =
        reference::per_gene_stats_and_marginals(args.sc, &cells, n_genes, args.r_floor)?;
    let hvg_indices = reference::select_hvg(&stats, args.n_hvg);
    let z = reference::build_z_matrix(args.sc, &cells, &hvg_indices, &marginals)?;
    let copula = CopulaCovariance::fit(&z, args.copula_rank, args.regularization)?;
    info!(
        "fit global copula: {} HVGs, rank {} (cap {}) + ridge sd {:.3}",
        hvg_indices.len(),
        copula.rank(),
        args.copula_rank,
        copula.ridge_sd
    );
    let r_hat: Vec<f32> = marginals.iter().map(|f| f.r).collect();
    let mut hvg_pos = vec![None; n_genes];
    for (h, &g) in hvg_indices.iter().enumerate() {
        hvg_pos[g] = Some(h as u32);
    }
    Ok(GlobalCopulaFit {
        gene_names,
        n_genes,
        hvg_indices,
        r_hat,
        hvg_pos,
        copula,
    })
}
