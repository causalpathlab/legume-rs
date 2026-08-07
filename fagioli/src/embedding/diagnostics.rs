//! The falsification test for the embedding, built before the model it judges.
//!
//! # What has to be true
//!
//! The embedding factors the *direct* effect matrix, `B = U V'`. Genetic
//! covariance between traits then follows from `β_t = U v_t`:
//!
//! ```text
//! Cov_g(t,t') = β_t' β_t' = v_t' (U'U) v_t'      →     Cov_g = V U'U V'
//! ```
//!
//! `V U'U V'` is invariant to the reparameterization `U→UA, V→VA⁻ᵀ` that leaves
//! `UV'` unchanged, whereas `VV'` is not — so `V U'U V'` is the only form of
//! this comparison that is even well posed.
//!
//! Genetic correlation is never an input to the fit. If the fitted geometry
//! reproduces `r_g` without having seen it, the geometry is real. If it
//! reproduces the sample-overlap matrix `Ω` instead, the overlap correction
//! failed and the "programs" are cohort structure.
//!
//! # Where `r_g` and `Ω` come from
//!
//! Both fall out of the same eigenspace regression used for null calibration,
//! applied to cross-products instead of squares:
//!
//! ```text
//! E[(V'z)_kt (V'z)_kt'] = √(N_t N_t') σ_β,tt' · d⁴_k  +  c_tt' · d²_k  +  τ_tt'
//! ```
//!
//! The `d⁴` coefficient is the genetic covariance and the intercept is the
//! sample overlap. One fit per trait pair yields both, so `r_g` and `Ω` are
//! never estimated by inconsistent machinery.
//!
//! # Reading the verdict
//!
//! `r_g` and `Ω` are frequently correlated with each other — traits measured in
//! overlapping cohorts are often genuinely related — so a marginal correlation
//! against either one proves little on its own. [`GeometryVerdict`] therefore
//! reports *partial* correlations, which is what distinguishes "tracks `r_g`"
//! from "tracks `Ω`" when the two are confounded.

use log::{info, warn};
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::summary_stats::calibration::{delete_a_group_se, ThreeTermDesign};
use crate::summary_stats::common::SumstatInput;
use crate::summary_stats::rss_svd::RssEigenBasis;

/// Minimum trait pairs before a correlation between matrices means anything.
const MIN_PAIRS_FOR_VERDICT: usize = 6;

/// Cross-trait moment components for one LD block, each `T x T`.
#[derive(Debug, Clone)]
pub struct CrossTraitMoments {
    /// `d⁴` coefficient: genetic covariance, up to the `√(N_t N_t')` scaling.
    pub genetic_cov: DMatrix<f32>,
    /// `d²` coefficient. Near the identity when the panel describes the noise.
    pub ld_coef: DMatrix<f32>,
    /// Intercept: sample overlap between the two traits' cohorts.
    pub overlap: DMatrix<f32>,
    /// SNPs in the block, for delete-a-group jackknife weighting.
    pub block_snps: usize,
}

/// Fit the three-term law to every trait pair in one block.
///
/// Only the upper triangle is solved; the result is symmetrized.
pub fn cross_trait_moments(
    d_sq: &[f32],
    vt_z: &DMatrix<f32>,
    block_snps: usize,
) -> Option<CrossTraitMoments> {
    let k = d_sq.len().min(vt_z.nrows());
    let t = vt_z.ncols();
    if t == 0 {
        return None;
    }
    let design = ThreeTermDesign::new(&d_sq[..k])?;

    let mut genetic_cov = DMatrix::<f32>::zeros(t, t);
    let mut ld_coef = DMatrix::<f32>::zeros(t, t);
    let mut overlap = DMatrix::<f32>::zeros(t, t);

    for a in 0..t {
        for b in a..t {
            let beta = design.solve(|ki| (vt_z[(ki, a)] as f64) * (vt_z[(ki, b)] as f64))?;
            // Diagonals are variances and cannot be negative; off-diagonals are
            // covariances and legitimately can be.
            let (g, c, o) = if a == b {
                (beta[0].max(0.0), beta[1].max(0.0), beta[2].max(0.0))
            } else {
                (beta[0], beta[1], beta[2])
            };
            genetic_cov[(a, b)] = g;
            genetic_cov[(b, a)] = g;
            ld_coef[(a, b)] = c;
            ld_coef[(b, a)] = c;
            overlap[(a, b)] = o;
            overlap[(b, a)] = o;
        }
    }

    Some(CrossTraitMoments {
        genetic_cov,
        ld_coef,
        overlap,
        block_snps,
    })
}

/// Genome-wide `r_g` and `Ω`, with per-block estimates retained for jackknifing.
#[derive(Debug, Clone)]
pub struct TraitGeometry {
    /// Genetic correlation, `T x T`, unit diagonal.
    pub genetic_correlation: DMatrix<f32>,
    /// Sample-overlap correlation, `T x T`, unit diagonal.
    pub overlap_correlation: DMatrix<f32>,
    /// Per-block moments, for the delete-a-group jackknife.
    pub per_block: Vec<CrossTraitMoments>,
}

/// Estimate `r_g` and `Ω` from summary statistics alone.
///
/// Blocks are pooled by summing the raw moment components before normalizing,
/// which weights each block by how much information it carries rather than
/// treating a 200-SNP block as equal to a 5000-SNP one.
pub fn estimate_trait_geometry(input: &SumstatInput) -> Option<TraitGeometry> {
    let t = input.zscores.ncols();
    if t == 0 {
        return None;
    }

    let per_block: Vec<CrossTraitMoments> = input
        .blocks
        .par_iter()
        .filter_map(|block| {
            let block_m = block.num_snps();
            let mut x_block = input
                .geno
                .genotypes
                .columns(block.snp_start, block_m)
                .clone_owned();
            x_block.scale_columns_inplace();

            let basis = RssEigenBasis::from_genotypes(&x_block, input.max_rank).ok()?;
            let z_block = input.zscores.rows(block.snp_start, block_m).clone_owned();
            let vt_z = basis.project_raw(&z_block);
            let d_sq = basis.singular_values_sq();

            cross_trait_moments(&d_sq, &vt_z, block_m)
        })
        .collect();

    if per_block.is_empty() {
        warn!("No LD block supported a cross-trait moment fit");
        return None;
    }

    let mut sum_gen = DMatrix::<f32>::zeros(t, t);
    let mut sum_ovl = DMatrix::<f32>::zeros(t, t);
    for m in &per_block {
        sum_gen += &m.genetic_cov;
        sum_ovl += &m.overlap;
    }

    let genetic_correlation = to_correlation(&sum_gen);
    let overlap_correlation = to_correlation(&sum_ovl);

    info!(
        "Trait geometry from {} blocks: {} traits, mean |r_g| {:.3}, mean |Ω| {:.3}",
        per_block.len(),
        t,
        mean_abs_offdiag(&genetic_correlation),
        mean_abs_offdiag(&overlap_correlation),
    );

    Some(TraitGeometry {
        genetic_correlation,
        overlap_correlation,
        per_block,
    })
}

/// Whether the fitted geometry reproduces `r_g` or merely `Ω`.
#[derive(Debug, Clone)]
pub struct GeometryVerdict {
    /// Marginal correlation of the fitted off-diagonals with `r_g`.
    pub corr_rg: f32,
    /// Marginal correlation with `Ω`.
    pub corr_overlap: f32,
    /// Correlation with `r_g` holding `Ω` fixed. This is the one that matters.
    pub partial_rg: f32,
    /// Correlation with `Ω` holding `r_g` fixed. Should be near zero.
    pub partial_overlap: f32,
    /// How much `r_g` and `Ω` overlap in this dataset; near 1 means the test
    /// has little power to separate them.
    pub corr_rg_overlap: f32,
    /// Delete-a-group jackknife SE of `partial_rg`.
    pub partial_rg_se: f32,
    /// Trait pairs compared.
    pub num_pairs: usize,
    /// `true` when the fitted geometry tracks `r_g` and not `Ω`.
    pub passes: bool,
}

/// Compare a fitted `V U'U V'` against `r_g` and `Ω`.
///
/// `fitted` must be the `A`-invariant form `V U'U V'`, not `VV'`.
pub fn compare_geometry(fitted: &DMatrix<f32>, geometry: &TraitGeometry) -> Option<GeometryVerdict> {
    let t = fitted.nrows();
    if t != fitted.ncols() || t != geometry.genetic_correlation.nrows() {
        warn!("Fitted geometry is not T x T, or trait counts disagree");
        return None;
    }

    let fit = offdiag(&to_correlation(fitted));
    let rg = offdiag(&geometry.genetic_correlation);
    let ovl = offdiag(&geometry.overlap_correlation);
    let num_pairs = fit.len();
    if num_pairs < MIN_PAIRS_FOR_VERDICT {
        warn!(
            "Only {} trait pairs; too few to judge the geometry",
            num_pairs
        );
        return None;
    }

    let corr_rg = pearson(&fit, &rg);
    let corr_overlap = pearson(&fit, &ovl);
    let corr_rg_overlap = pearson(&rg, &ovl);
    let partial_rg = partial_corr(corr_rg, corr_overlap, corr_rg_overlap);
    let partial_overlap = partial_corr(corr_overlap, corr_rg, corr_rg_overlap);

    // Jackknife the partial correlation by deleting one LD block at a time and
    // refitting r_g and Ω from the remaining blocks. The fitted matrix is held
    // fixed: this is the SE of the comparison, not of the embedding.
    let partial_rg_se = jackknife_partial_rg(&fit, geometry);

    let passes = partial_rg > 0.0 && partial_rg.abs() > partial_overlap.abs();

    info!(
        "Geometry verdict over {} pairs: corr(fit, r_g)={:.3}, corr(fit, Ω)={:.3}; \
         partial r_g={:.3} (SE {:.3}), partial Ω={:.3}; corr(r_g, Ω)={:.3}",
        num_pairs,
        corr_rg,
        corr_overlap,
        partial_rg,
        partial_rg_se,
        partial_overlap,
        corr_rg_overlap,
    );
    if corr_rg_overlap.abs() > 0.9 {
        warn!(
            "r_g and Ω are nearly collinear here (corr {:.3}); this dataset cannot \
             separate real geometry from sample overlap.",
            corr_rg_overlap,
        );
    }
    if !passes {
        warn!(
            "Fitted geometry does not track r_g above Ω (partial r_g {:.3} vs partial Ω {:.3}). \
             The overlap correction has not done its job.",
            partial_rg, partial_overlap,
        );
    }

    Some(GeometryVerdict {
        corr_rg,
        corr_overlap,
        partial_rg,
        partial_overlap,
        corr_rg_overlap,
        partial_rg_se,
        num_pairs,
        passes,
    })
}

fn jackknife_partial_rg(fit: &[f32], geometry: &TraitGeometry) -> f32 {
    let g = geometry.per_block.len();
    if g < 2 {
        return 0.0;
    }
    let t = geometry.genetic_correlation.nrows();

    let mut sum_gen = DMatrix::<f32>::zeros(t, t);
    let mut sum_ovl = DMatrix::<f32>::zeros(t, t);
    for m in &geometry.per_block {
        sum_gen += &m.genetic_cov;
        sum_ovl += &m.overlap;
    }

    let mut values = Vec::with_capacity(g);
    let mut sizes = Vec::with_capacity(g);
    for m in &geometry.per_block {
        let rg = offdiag(&to_correlation(&(&sum_gen - &m.genetic_cov)));
        let ovl = offdiag(&to_correlation(&(&sum_ovl - &m.overlap)));
        let c_rg = pearson(fit, &rg);
        let c_ovl = pearson(fit, &ovl);
        let c_cross = pearson(&rg, &ovl);
        values.push(partial_corr(c_rg, c_ovl, c_cross));
        sizes.push(m.block_snps);
    }

    delete_a_group_se(&values, &sizes)
}

/// Normalize a symmetric matrix to correlation form, unit diagonal.
pub fn to_correlation(m: &DMatrix<f32>) -> DMatrix<f32> {
    let t = m.nrows();
    let scale: Vec<f32> = (0..t)
        .map(|i| {
            let d = m[(i, i)];
            if d > 0.0 && d.is_finite() {
                d.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    DMatrix::from_fn(t, t, |i, j| {
        if i == j {
            1.0
        } else if scale[i] > 0.0 && scale[j] > 0.0 {
            (m[(i, j)] / (scale[i] * scale[j])).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    })
}

/// Strict upper-triangle entries, in a fixed order.
fn offdiag(m: &DMatrix<f32>) -> Vec<f32> {
    let t = m.nrows();
    let mut v = Vec::with_capacity(t * t.saturating_sub(1) / 2);
    for i in 0..t {
        for j in (i + 1)..t {
            v.push(m[(i, j)]);
        }
    }
    v
}

fn mean_abs_offdiag(m: &DMatrix<f32>) -> f32 {
    let v = offdiag(m);
    if v.is_empty() {
        0.0
    } else {
        v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32
    }
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma = a.iter().take(n).sum::<f32>() / n as f32;
    let mb = b.iter().take(n).sum::<f32>() / n as f32;
    let mut sab = 0.0f64;
    let mut saa = 0.0f64;
    let mut sbb = 0.0f64;
    for i in 0..n {
        let da = (a[i] - ma) as f64;
        let db = (b[i] - mb) as f64;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    ((sab / (saa * sbb).sqrt()) as f32).clamp(-1.0, 1.0)
}

/// First-order partial correlation of x with y, controlling for z.
fn partial_corr(r_xy: f32, r_xz: f32, r_yz: f32) -> f32 {
    let denom = ((1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)).max(1e-12).sqrt();
    ((r_xy - r_xz * r_yz) / denom).clamp(-1.0, 1.0)
}

#[cfg(test)]
#[path = "diagnostics_tests.rs"]
mod tests;
