//! Per-cell-type anchor prior `N(t̂_c, Σ_c)` in the embedding space — the
//! channel through which annotate-by-projection uncertainty enters the
//! deconvolution.
//!
//! `t̂_c` is the IDF-weighted centroid of type `c`'s marker genes in the anchor
//! embedding (the same construction `annotate-by-projection` uses). `Σ_c` is
//! the within-type scatter of those marker coordinates: **isotropic** `σ_c²·I`
//! by default (confident, tight types → small `σ_c` → pinned fractions), or a
//! **shrunk full** `H×H` covariance. A type with too few markers falls back to
//! the global median spread.

use super::args::{AnchorConfig, AnchorCov};
use crate::embed_common::Mat;
use crate::marker_support::build_annotation_matrix;
use anyhow::Result;
use log::info;
use matrix_util::utils::median;

/// Minimum markers for a type to trust its own scatter estimate.
const MIN_MARKERS_FOR_SCATTER: usize = 2;
/// Shrinkage weight toward the isotropic target in `AnchorCov::Full`.
const SHRINK_ALPHA: f32 = 0.1;
/// Diagonal jitter keeping the shrunk covariance positive-definite.
const COV_JITTER: f32 = 1e-4;

pub struct AnchorPrior {
    /// `C×H` prior means `t̂_c` (rows).
    pub mean: Mat,
    /// `C` cell-type names.
    pub names: Vec<Box<str>>,
    /// `C` isotropic prior std per type (used directly in isotropic mode, and
    /// as the Cholesky fallback in full mode).
    pub sigma: Vec<f32>,
    /// `C` lower-triangular Cholesky factors `L_c` (`H×H`) for full-covariance
    /// mode; `None` for isotropic mode.
    pub chol: Option<Vec<Mat>>,
}

pub fn build_anchor_prior(
    anchor_emb: &Mat,
    gene_names: &[Box<str>],
    markers_path: &str,
    cfg: &AnchorConfig,
) -> Result<AnchorPrior> {
    let info = build_annotation_matrix(markers_path, gene_names)?;
    let membership = info.membership_ga; // G×C, IDF-weighted
    let names = info.annot_names;
    let (c, h, d) = (names.len(), anchor_emb.ncols(), anchor_emb.nrows());
    anyhow::ensure!(
        membership.nrows() == d,
        "anchor: marker membership genes ({}) != embedding genes ({d})",
        membership.nrows()
    );
    anyhow::ensure!(c >= 2, "anchor: need at least 2 cell types, got {c}");

    // Per-type IDF-weighted centroid + marker bookkeeping.
    let mut mean = Mat::zeros(c, h);
    let mut wsum = vec![0f32; c];
    let mut count = vec![0usize; c];
    for g in 0..d {
        for ct in 0..c {
            let w = membership[(g, ct)];
            if w > 0.0 {
                wsum[ct] += w;
                count[ct] += 1;
                for j in 0..h {
                    mean[(ct, j)] += w * anchor_emb[(g, j)];
                }
            }
        }
    }
    for ct in 0..c {
        if wsum[ct] > 0.0 {
            for j in 0..h {
                mean[(ct, j)] /= wsum[ct];
            }
        }
    }

    // Per-type average per-dimension scatter → isotropic σ_c².
    let mut var = vec![0f32; c];
    for g in 0..d {
        for ct in 0..c {
            let w = membership[(g, ct)];
            if w > 0.0 {
                let mut sq = 0f32;
                for j in 0..h {
                    let dlt = anchor_emb[(g, j)] - mean[(ct, j)];
                    sq += dlt * dlt;
                }
                var[ct] += w * sq;
            }
        }
    }
    let mut sigma = vec![0f32; c];
    let mut valid = Vec::new();
    for ct in 0..c {
        if wsum[ct] > 0.0 && count[ct] >= MIN_MARKERS_FOR_SCATTER {
            let s = (var[ct] / (h as f32 * wsum[ct])).max(0.0).sqrt();
            sigma[ct] = s;
            if s > 0.0 {
                valid.push(s);
            }
        }
    }
    // Global fallback spread for under-determined types.
    let global = if valid.is_empty() {
        embedding_std(anchor_emb).max(1e-3)
    } else {
        median(&valid)
    };
    let floor = 1e-3 * global;
    for s in &mut sigma {
        if *s <= 0.0 {
            *s = global;
        }
        *s = s.max(floor) * cfg.scale;
    }
    info!(
        "anchor prior: {c} types, σ range [{:.3e}, {:.3e}] (median {:.3e}, scale {:.2})",
        sigma.iter().cloned().fold(f32::INFINITY, f32::min),
        sigma.iter().cloned().fold(0.0, f32::max),
        global * cfg.scale,
        cfg.scale
    );

    let chol = match cfg.cov {
        AnchorCov::Isotropic => None,
        AnchorCov::Full => Some(full_cholesky(anchor_emb, &membership, &mean, &sigma, h)),
    };

    Ok(AnchorPrior {
        mean,
        names,
        sigma,
        chol,
    })
}

/// Global per-dimension std of the embedding (fallback spread).
fn embedding_std(emb: &Mat) -> f32 {
    let n = (emb.nrows() * emb.ncols()) as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean = emb.sum() / n;
    let var = emb.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    var.max(0.0).sqrt()
}

/// Shrunk full-covariance Cholesky factors, one per type. Shrinks the weighted
/// empirical covariance toward `σ_c²·I` and adds jitter; falls back to the
/// isotropic `σ_c·I` factor if the Cholesky is not positive-definite.
fn full_cholesky(emb: &Mat, membership: &Mat, mean: &Mat, sigma: &[f32], h: usize) -> Vec<Mat> {
    let c = mean.nrows();
    let d = emb.nrows();
    (0..c)
        .map(|ct| {
            let mut cov = Mat::zeros(h, h);
            let mut wsum = 0f32;
            for g in 0..d {
                let w = membership[(g, ct)];
                if w > 0.0 {
                    wsum += w;
                    let mut dv = vec![0f32; h];
                    for j in 0..h {
                        dv[j] = emb[(g, j)] - mean[(ct, j)];
                    }
                    for a in 0..h {
                        for b in 0..h {
                            cov[(a, b)] += w * dv[a] * dv[b];
                        }
                    }
                }
            }
            let target = sigma[ct] * sigma[ct]; // isotropic target variance
            if wsum > 0.0 {
                cov /= wsum;
            }
            for a in 0..h {
                for b in 0..h {
                    cov[(a, b)] *= 1.0 - SHRINK_ALPHA;
                }
                cov[(a, a)] += SHRINK_ALPHA * target + COV_JITTER;
            }
            cov.cholesky()
                .map_or_else(|| Mat::identity(h, h) * sigma[ct], |c| c.l())
        })
        .collect()
}
