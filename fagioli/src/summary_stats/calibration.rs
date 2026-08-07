//! Null calibration in the RSS eigenspace.
//!
//! # What is being fitted
//!
//! Write the RSS model per LD block as `z = √N R β + ε` with `R = V D² V'`.
//! Because `V' R = D² V'`, the eigenspace projection satisfies
//!
//! ```text
//! (V'z)_k = √N d²_k (V'β)_k + (V'ε)_k
//! ```
//!
//! Under an infinitesimal prior `β ~ N(0, σ²_β I)` and a noise covariance
//! `cR + τI` — LD-shaped inflation plus a nugget — the second moment is
//!
//! ```text
//! E[(V'z)²_k] = N σ²_β · d⁴_k  +  c · d²_k  +  τ
//! ```
//!
//! The `d⁴` basis is the eigenspace analogue of the LD score: in SNP space
//! `E[z²_j] = N σ²_β ℓ_j + 1` with `ℓ_j = (R²)_{jj}`, and `R²` has eigenvalues
//! `d⁴`. The `d²` term is the analogue of `R_jj = 1`, the null variance.
//!
//! # Why three terms and not two
//!
//! [`RssSvdNal::estimate_ldsc_intercept`] regresses on `d²` with an intercept,
//! which is the correct form for its documented model `z ~ N(0, hR + aI)` — a
//! model with **no signal term**. Applied to data carrying polygenic signal the
//! `d⁴` component has nowhere to go but the `d²` slope and the intercept, so
//! both come back biased. Recovering `σ²_β` needs the `d⁴` column.
//!
//! # What λ falls out
//!
//! Whitening divides by `D̃ = √(D² + λ)`, so
//!
//! ```text
//! E[z̃²_k] = (N σ²_β d⁴_k + c d²_k + τ) / (d²_k + λ)
//! ```
//!
//! Setting the *null* part flat at 1 requires `c = 1` and `λ = τ`. So λ is
//! closed-form rather than a search: **λ_white = τ**. `c` is then a residual
//! diagnostic — it should come back near 1, and a `c` far from 1 means the
//! reference panel's LD does not describe the noise, which is the panel
//! mismatch that no choice of λ can repair.

use log::{info, warn};
use matrix_util::traits::MatOps;
use nalgebra::{DMatrix, Matrix3, Vector3};
use rayon::prelude::*;

use crate::summary_stats::common::SumstatInput;
use crate::summary_stats::rss_svd::RssEigenBasis;

/// Minimum eigen-coordinates needed to identify three parameters with slack.
const MIN_RANK_FOR_FIT: usize = 8;

/// A whiteness deviation above this means no λ made the null white, so the
/// reference panel does not describe the noise.
const MISSPECIFICATION_ALARM: f32 = 0.5;

/// Eigenspace calibration for a single LD block, per trait.
#[derive(Debug, Clone)]
pub struct BlockCalibration {
    /// Coefficient on the eigen LD score `d⁴`, i.e. `N σ²_β`. This is the dense
    /// (polygenic) component, and the initializer for the BSLMM `Σ_d`.
    pub polygenic: Vec<f32>,
    /// Coefficient on `d²`. Near 1 when the panel describes the noise.
    pub c: Vec<f32>,
    /// Intercept τ — the nugget, and the λ that whitens this block's null.
    pub tau: Vec<f32>,
    /// Effective rank of the block's eigenbasis.
    pub rank: usize,
}

impl BlockCalibration {
    /// λ that whitens the null, pooled across traits by the median of τ.
    pub fn lambda_white(&self) -> f64 {
        median(&self.tau) as f64
    }
}

/// Fit `E[(V'z)²_k] = a₁ d⁴_k + a₂ d²_k + a₃` per trait by ordinary least squares.
///
/// `d_sq` holds `d²_k` (length K) and `vt_z` is the *raw* projection `V'z`,
/// shape (K, T) — that is [`crate::summary_stats::rss_svd::RssEigenBasis::project_raw`],
/// not the λ-scaled `project_zscores`.
///
/// Returns `None` when the block has too few eigen-coordinates to identify
/// three parameters.
pub fn calibrate_block(d_sq: &[f32], vt_z: &DMatrix<f32>) -> Option<BlockCalibration> {
    let k = d_sq.len().min(vt_z.nrows());
    let t = vt_z.ncols();
    if k < MIN_RANK_FOR_FIT || t == 0 {
        return None;
    }

    // Scale the two predictors to O(1) before forming normal equations; d⁴ and
    // d² otherwise differ by orders of magnitude and the 3x3 solve loses rank.
    let mean_d2 = mean_f64(d_sq.iter().take(k).map(|&v| v as f64));
    let mean_d4 = mean_f64(d_sq.iter().take(k).map(|&v| (v as f64) * (v as f64)));
    if mean_d2 <= 0.0 || mean_d4 <= 0.0 || !mean_d2.is_finite() || !mean_d4.is_finite() {
        return None;
    }

    // Gram matrix of [d⁴/mean_d4, d²/mean_d2, 1] — shared across traits.
    let mut gram = Matrix3::<f64>::zeros();
    let mut rows: Vec<[f64; 3]> = Vec::with_capacity(k);
    for &d2 in d_sq.iter().take(k) {
        let d2 = d2 as f64;
        let row = [d2 * d2 / mean_d4, d2 / mean_d2, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                gram[(i, j)] += row[i] * row[j];
            }
        }
        rows.push(row);
    }

    let decomp = gram.qr();

    let mut polygenic = Vec::with_capacity(t);
    let mut c = Vec::with_capacity(t);
    let mut tau = Vec::with_capacity(t);

    for tt in 0..t {
        let mut rhs = Vector3::<f64>::zeros();
        for (ki, row) in rows.iter().enumerate() {
            let y = vt_z[(ki, tt)] as f64;
            let y2 = y * y;
            for i in 0..3 {
                rhs[i] += row[i] * y2;
            }
        }

        let beta = decomp.solve(&rhs)?;

        // Undo the predictor scaling.
        polygenic.push((beta[0] / mean_d4).max(0.0) as f32);
        c.push((beta[1] / mean_d2).max(0.0) as f32);
        tau.push(beta[2].max(0.0) as f32);
    }

    Some(BlockCalibration {
        polygenic,
        c,
        tau,
        rank: k,
    })
}

/// Pool per-block calibrations into the genome-wide `(c, τ)` of the noise model.
///
/// `R` is a property of the reference panel, not of the trait, so `c` and `τ`
/// are pooled across both blocks and traits: a single matrix-normal
/// `Ω ⊗ (cR + τI)` needs scalar `c` and `τ`, with any per-trait inflation
/// carried by `diag(Ω)` instead. Medians are used so one pathological block
/// cannot move the noise model.
#[derive(Debug, Clone)]
pub struct NoiseCalibration {
    /// Pooled `c` — the LD coefficient of the noise.
    pub c: f32,
    /// Pooled `τ` — the nugget, and the whitening λ.
    pub tau: f32,
    /// Per-trait inflation relative to the pooled fit; feeds `diag(Ω)`.
    pub trait_inflation: Vec<f32>,
    /// Per-trait polygenic coefficient `N σ²_β`, averaged over blocks.
    pub polygenic: Vec<f32>,
    pub num_blocks: usize,
}

impl NoiseCalibration {
    pub fn lambda_white(&self) -> f64 {
        self.tau as f64
    }
}

pub fn pool_calibrations(blocks: &[BlockCalibration], num_traits: usize) -> Option<NoiseCalibration> {
    if blocks.is_empty() || num_traits == 0 {
        return None;
    }

    let all_c: Vec<f32> = blocks.iter().flat_map(|b| b.c.iter().copied()).collect();
    let all_tau: Vec<f32> = blocks.iter().flat_map(|b| b.tau.iter().copied()).collect();
    let c = median(&all_c);
    let tau = median(&all_tau);

    // Per-trait: how much more nugget does this trait carry than the pool?
    let mut trait_inflation = Vec::with_capacity(num_traits);
    let mut polygenic = Vec::with_capacity(num_traits);
    for tt in 0..num_traits {
        let taus: Vec<f32> = blocks.iter().filter_map(|b| b.tau.get(tt).copied()).collect();
        let polys: Vec<f32> = blocks
            .iter()
            .filter_map(|b| b.polygenic.get(tt).copied())
            .collect();
        let t_tau = if taus.is_empty() { tau } else { median(&taus) };
        trait_inflation.push(if tau > 0.0 { t_tau / tau } else { 1.0 });
        polygenic.push(if polys.is_empty() {
            0.0
        } else {
            polys.iter().sum::<f32>() / polys.len() as f32
        });
    }

    Some(NoiseCalibration {
        c,
        tau,
        trait_inflation,
        polygenic,
        num_blocks: blocks.len(),
    })
}

/// One point of the whiteness curve: a bin of eigen-coordinates by `d²`.
#[derive(Debug, Clone)]
pub struct WhitenessBin {
    /// Mean `d²` in the bin.
    pub d_sq: f32,
    /// Mean `z̃²` in the bin, averaged over traits. Should be 1 where the null
    /// dominates.
    pub mean_z_sq: f32,
    pub count: usize,
}

/// Mean `z̃²` per `d²` bin at a given λ, where `z̃²_k = (V'z)²_k / (d²_k + λ)`.
///
/// Both inputs are λ-independent, so sweeping λ costs `O(KT)` per grid point
/// and never re-decomposes anything.
pub fn whiteness_curve(
    d_sq: &[f32],
    vt_z: &DMatrix<f32>,
    lambda: f64,
    num_bins: usize,
) -> Vec<WhitenessBin> {
    let k = d_sq.len().min(vt_z.nrows());
    let t = vt_z.ncols();
    if k == 0 || t == 0 || num_bins == 0 {
        return Vec::new();
    }

    // Rank eigen-coordinates by d², then split into equal-count bins so the
    // small-d tail — where the nugget dominates and λ actually bites — is
    // resolved rather than swamped by the bulk.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| d_sq[a].partial_cmp(&d_sq[b]).unwrap_or(std::cmp::Ordering::Equal));

    let lambda_f = lambda as f32;
    let per_bin = k.div_ceil(num_bins);

    order
        .chunks(per_bin)
        .map(|chunk| {
            let mut sum_d2 = 0.0f64;
            let mut sum_z2 = 0.0f64;
            for &ki in chunk {
                sum_d2 += d_sq[ki] as f64;
                let denom = (d_sq[ki] + lambda_f).max(f32::MIN_POSITIVE) as f64;
                for tt in 0..t {
                    let v = vt_z[(ki, tt)] as f64;
                    sum_z2 += v * v / denom;
                }
            }
            let n = chunk.len();
            WhitenessBin {
                d_sq: (sum_d2 / n as f64) as f32,
                mean_z_sq: (sum_z2 / (n * t) as f64) as f32,
                count: n,
            }
        })
        .collect()
}

/// Worst absolute log-deviation of the whiteness curve from 1.
///
/// Zero means a perfectly white null. This is the scalar that flags panel
/// mismatch: if no λ brings it near zero, the reference LD does not describe
/// the noise and the deconvolution cannot be trusted.
pub fn whiteness_deviation(curve: &[WhitenessBin]) -> f32 {
    curve
        .iter()
        .filter(|b| b.mean_z_sq > 0.0)
        .map(|b| b.mean_z_sq.ln().abs())
        .fold(0.0f32, f32::max)
}

/// Genome-wide calibration: the noise model, the whitening λ, and the curve
/// that shows whether λ actually worked.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    pub noise: NoiseCalibration,
    /// Whiteness curve at the selected λ, pooled over blocks.
    pub curve: Vec<WhitenessBin>,
    /// Worst absolute log-deviation from 1 on `curve`.
    pub deviation: f32,
    /// Per-block jackknife SE of λ, using the Busing delete-a-group weighting
    /// for unequal block sizes.
    pub lambda_se: f32,
    /// True when no λ whitened the null — the panel-mismatch alarm.
    pub misspecified: bool,
}

/// Fit the eigenspace noise model across every LD block and select λ.
///
/// λ is not searched: the whitening ridge that flattens the null is the fitted
/// nugget τ (see the module docs). The λ grid still gets swept, but only to
/// *report* the whiteness curve — and because λ enters after the randomized
/// SVD, that sweep reuses one decomposition per block.
pub fn calibrate_input(input: &SumstatInput) -> Option<CalibrationReport> {
    let num_traits = input.zscores.ncols();

    // (block calibration, block d², block V'z) — the projections are
    // λ-independent, so they are computed once and reused for every λ.
    let per_block: Vec<(BlockCalibration, Vec<f32>, DMatrix<f32>, usize)> = input
        .blocks
        .par_iter()
        .filter_map(|block| {
            let block_m = block.num_snps();
            if block_m < MIN_RANK_FOR_FIT {
                return None;
            }
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

            let cal = calibrate_block(&d_sq, &vt_z)?;
            Some((cal, d_sq, vt_z, block_m))
        })
        .collect();

    if per_block.is_empty() {
        warn!("No LD block was large enough to calibrate");
        return None;
    }

    let cals: Vec<BlockCalibration> = per_block.iter().map(|(c, _, _, _)| c.clone()).collect();
    let noise = pool_calibrations(&cals, num_traits)?;
    let lambda = noise.lambda_white();

    // Whiteness at the selected λ, pooled across blocks.
    let curve = pool_curves(
        &per_block
            .iter()
            .map(|(_, d_sq, vt_z, _)| whiteness_curve(d_sq, vt_z, lambda, 12))
            .collect::<Vec<_>>(),
    );
    let deviation = whiteness_deviation(&curve);

    let block_sizes: Vec<usize> = per_block.iter().map(|(_, _, _, m)| *m).collect();
    let block_lambdas: Vec<f32> = cals.iter().map(|c| c.lambda_white() as f32).collect();
    let lambda_se = delete_a_group_se(&block_lambdas, &block_sizes);

    let misspecified = deviation > MISSPECIFICATION_ALARM;

    info!(
        "Eigenspace calibration over {} blocks: c={:.3}, τ={:.4} (λ_white, SE {:.4}), \
         whiteness deviation {:.3}",
        noise.num_blocks, noise.c, noise.tau, lambda_se, deviation,
    );
    if misspecified {
        warn!(
            "Null did not whiten (deviation {:.3} > {:.2}): the reference panel's LD \
             does not describe the noise. Fine-mapping and deconvolution from this \
             panel are not trustworthy.",
            deviation, MISSPECIFICATION_ALARM,
        );
    }
    if (noise.c - 1.0).abs() > 0.5 {
        warn!(
            "Noise LD coefficient c={:.3} is far from 1; residual LD structure remains \
             after whitening.",
            noise.c,
        );
    }

    Some(CalibrationReport {
        noise,
        curve,
        deviation,
        lambda_se,
        misspecified,
    })
}

/// Average per-block whiteness curves bin-by-bin, weighted by bin occupancy.
fn pool_curves(curves: &[Vec<WhitenessBin>]) -> Vec<WhitenessBin> {
    let n_bins = curves.iter().map(Vec::len).max().unwrap_or(0);
    (0..n_bins)
        .filter_map(|bi| {
            let mut w = 0.0f64;
            let mut sum_d2 = 0.0f64;
            let mut sum_z2 = 0.0f64;
            let mut count = 0usize;
            for c in curves {
                if let Some(b) = c.get(bi) {
                    let n = b.count as f64;
                    w += n;
                    sum_d2 += b.d_sq as f64 * n;
                    sum_z2 += b.mean_z_sq as f64 * n;
                    count += b.count;
                }
            }
            (w > 0.0).then(|| WhitenessBin {
                d_sq: (sum_d2 / w) as f32,
                mean_z_sq: (sum_z2 / w) as f32,
                count,
            })
        })
        .collect()
}

/// Delete-a-group jackknife SE for unequal group sizes.
///
/// Busing, Meijer & van der Leeden (1999), *Statistics and Computing* 9:3-8 —
/// the estimator LDSC uses. With `h_j = n/m_j` and pseudo-values
/// `θ̃_j = h_j θ̂ − (h_j−1) θ̂₍₋ⱼ₎`,
///
/// ```text
/// var = (1/g) Σ_j (θ̃_j − θ̃)² / (h_j − 1),     θ̃ = mean_j θ̃_j
/// ```
///
/// LD blocks differ in SNP count by an order of magnitude, and the unweighted
/// delete-one estimator is biased under unequal group sizes, so the weighting
/// is not optional here. At equal `m_j` this reduces to the ordinary delete-one
/// jackknife variance.
pub fn delete_a_group_se(values: &[f32], group_sizes: &[usize]) -> f32 {
    let g = values.len().min(group_sizes.len());
    if g < 2 {
        return 0.0;
    }
    let n: f64 = group_sizes.iter().take(g).map(|&m| m as f64).sum();
    if n <= 0.0 {
        return 0.0;
    }

    // Full-sample estimate as the size-weighted mean; each delete-one estimate
    // is the same mean with that group's contribution removed.
    let total: f64 = (0..g)
        .map(|j| values[j] as f64 * group_sizes[j] as f64)
        .sum();
    let theta = total / n;

    let mut pseudo = Vec::with_capacity(g);
    let mut h = Vec::with_capacity(g);
    for j in 0..g {
        let m_j = group_sizes[j] as f64;
        if m_j <= 0.0 || (n - m_j) <= 0.0 {
            continue;
        }
        let theta_minus = (total - values[j] as f64 * m_j) / (n - m_j);
        let h_j = n / m_j;
        pseudo.push(h_j * theta - (h_j - 1.0) * theta_minus);
        h.push(h_j);
    }
    if pseudo.len() < 2 {
        return 0.0;
    }

    let g_eff = pseudo.len() as f64;
    let pseudo_mean: f64 = pseudo.iter().sum::<f64>() / g_eff;
    let acc: f64 = pseudo
        .iter()
        .zip(&h)
        .map(|(&p, &h_j)| {
            let dev = p - pseudo_mean;
            dev * dev / (h_j - 1.0).max(f64::EPSILON)
        })
        .sum();

    ((acc / g_eff).max(0.0)).sqrt() as f32
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

fn mean_f64(iter: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for v in iter {
        sum += v;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;
