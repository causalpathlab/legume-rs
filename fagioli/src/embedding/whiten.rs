//! Per-block whitening: the fixed, λ-dependent geometry the model trains on.
//!
//! # Two whitenings, one on each axis
//!
//! **Variant axis.** Per block `R = V_R D² V_R'`, and dividing by `D̃ = √(D²+λ)`
//! turns the RSS noise `N(0, R)` into something isotropic across eigen-
//! coordinates. The same operation deconvolves LD from the mean, because
//! `V_R' R β = D² V_R' β` — the whitening that handles correlated noise *is*
//! the deconvolution.
//!
//! **Trait axis.** Sample overlap `Ω` is estimated once and never learned, so
//! it can be divided out too: `ž = Ω^{-1/2} z̃`. Rather than carry `Ω`, `Λ_N`
//! and `V` separately, the model learns
//!
//! ```text
//! V̌ = Ω^{-1/2} Λ_N V
//! ```
//!
//! directly, and [`unwhiten_traits`] recovers `V`. The null is then standard
//! normal on **both** axes, so the matrix-variate law factorises completely and
//! the log Bayes factor reduces to a matched filter — see [`super::model`]. No
//! Kronecker product is ever formed; the matrix-variate normal is the
//! derivation, not the implementation.
//!
//! # `Ω = I` is the default
//!
//! Trait-side whitening is *optional* and off by default. It matters only for
//! **overlapping cohorts**: with disjoint cohorts `Ω = I` and the whole step is
//! the identity. Passing `None` skips a `T x T` matmul per block rather than
//! multiplying by an identity.
//!
//! When cohorts do overlap it cannot simply be ignored, and the reason is not
//! the null's likelihood — it is identifiability. Left in, `Ω` is absorbed by
//! `V̌`, and the learned "programs" become cohort structure. That is precisely
//! the failure [`super::diagnostics`] exists to detect, and the gate is a check,
//! not a fix. Supply `Ω^{-1/2}` from [`super::noise::omega_sqrt_pair`] in that
//! case.

use anyhow::Result;
use log::info;
use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::summary_stats::common::{BlockEigenBases, SumstatInput};

/// One LD block, whitened on both axes and ready for training.
pub struct WhitenedBlock {
    /// Index into the original block list.
    pub block_idx: usize,
    /// Offset of this block's first SNP in the global variant ordering.
    pub snp_start: usize,
    /// Number of SNPs in the block.
    pub num_snps: usize,
    /// `X̃ = D̃ V_R'`, shape (K, p). Left factor of the mean structure.
    pub x_design: DMatrix<f32>,
    /// `ž = Ω^{-1/2} D̃⁻¹ V_R' z`, shape (K, T). Standard normal under the null.
    pub z_white: DMatrix<f32>,
    /// `d²_k`, length K. Sets the negatives' per-coordinate variance.
    pub d_sq: Vec<f32>,
}

impl WhitenedBlock {
    pub fn rank(&self) -> usize {
        self.x_design.nrows()
    }

    pub fn num_traits(&self) -> usize {
        self.z_white.ncols()
    }
}

/// Whiten every decomposed block.
///
/// `bases` comes from [`decompose_blocks`](crate::summary_stats::common::decompose_blocks)
/// and is *consumed*: `D` and `V` are
/// λ-independent, so the only thing this stage adds is the ridge, and each
/// basis is turned into the design it implies rather than held alongside it.
/// Calibration borrows the same bases beforehand to derive the `lambda` passed
/// here, which is why the randomized SVD happens once for the two of them.
///
/// `omega_inv_sqrt` is `Ω^{-1/2}` (T x T), or `None` for `Ω = I` — the default,
/// correct whenever the cohorts behind the traits are disjoint. `lambda` should
/// be the calibrated `λ_white`.
pub fn whiten_blocks(
    input: &SumstatInput,
    bases: BlockEigenBases,
    omega_inv_sqrt: Option<&DMatrix<f32>>,
    lambda: f64,
) -> Result<Vec<WhitenedBlock>> {
    let num_decomposed = bases.len();

    let mut blocks: Vec<WhitenedBlock> = bases
        .into_blocks()
        .into_par_iter()
        .map(|b| {
            let d_sq = b.basis.singular_values_sq();
            let svd = b.basis.into_svd(lambda);

            let z_block = input.zscores.rows(b.snp_start, b.num_snps).clone_owned();
            // D̃⁻¹ V_R' z, then Ω^{-1/2} on the trait axis when overlap is modelled.
            let z_tilde = svd.project_zscores(&z_block);
            let z_white = match omega_inv_sqrt {
                Some(w) => z_tilde * w.transpose(),
                None => z_tilde,
            };

            WhitenedBlock {
                block_idx: b.block_idx,
                snp_start: b.snp_start,
                num_snps: b.num_snps,
                // Consumes the SVD, so V goes out of scope with it rather than
                // being held alongside a clone of the design.
                x_design: svd.into_x_design(),
                z_white,
                d_sq,
            }
        })
        .collect();

    blocks.sort_by_key(|b| b.block_idx);

    let total_rank: usize = blocks.iter().map(WhitenedBlock::rank).sum();
    info!(
        "Whitened {}/{} decomposed blocks (λ={:.4}): {} eigen-coordinates total",
        blocks.len(),
        num_decomposed,
        lambda,
        total_rank,
    );

    Ok(blocks)
}

/// Recover `V` from the learned `V̌ = Ω^{-1/2} Λ_N V`.
///
/// `omega_sqrt` is `Ω^{1/2}`, or `None` when `Ω = I`. `sqrt_n` holds `√N_t` per
/// trait. Per-trait scale is irrelevant to prediction — a PRS is scale-free
/// within a trait — but it matters for comparing `V U'U V'` against `r_g`.
pub fn unwhiten_traits(
    v_check: &DMatrix<f32>,
    omega_sqrt: Option<&DMatrix<f32>>,
    sqrt_n: &[f32],
) -> DMatrix<f32> {
    let mut v = match omega_sqrt {
        Some(s) => s * v_check,
        None => v_check.clone(),
    };
    for (t, &s) in sqrt_n.iter().enumerate().take(v.nrows()) {
        if s > 0.0 {
            v.row_mut(t).scale_mut(1.0 / s);
        }
    }
    v
}

#[cfg(test)]
#[path = "whiten_tests.rs"]
mod tests;
