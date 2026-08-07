//! The noise distribution the discriminator contrasts against.
//!
//! # Negatives are exactly characterised, and cheap
//!
//! The plan was to draw negatives in SNP space, `ε ~ N(0, cR + τI)`, and push
//! them through the same rSVD/whitening path as the real data. That turns out
//! to be unnecessary work, and the reason matters. For *any* `A` with
//! `AA' = cR + τI`, since `V_R'V_R = I` and `V_R'RV_R = D²`,
//!
//! ```text
//! V_R' ε  ~  N(0, diag(c d²_k + τ))
//! ```
//!
//! so after dividing by `D̃ = √(D²+λ)` the whitened negative is
//!
//! ```text
//! ž_k  ~  N(0, (c d²_k + τ) / (d²_k + λ))
//! ```
//!
//! per eigen-coordinate, independent of `A`. The p-dimensional draw carries no
//! information the per-`k` scalar does not, so sampling is `O(K)`, not `O(pK)`.
//!
//! At the calibrated `λ = τ` with `c = 1` this is exactly `N(0, 1)`: the
//! negatives are standard normal precisely when the null whitens.
//!
//! # What this costs the argument for NCE
//!
//! The case for contrastive estimation over maximum likelihood here was that a
//! defect shared by positives and negatives cancels — in particular that the
//! reference panel's LD is not the GWAS sample's LD. The algebra above says
//! negatives built from the *panel's own* `R` cannot carry that defect, because
//! the defect is precisely the difference between the panel's `R` and the true
//! one, which is unavailable by construction. So the cancellation does not
//! happen and NCE is, here, a higher-variance route to the same optimum.
//!
//! It is kept because the objective is what the model was specified against,
//! and because the score is exactly normalised (both densities are proper), so
//! the learned offset in [`super::train`] doubles as a calibration check. The
//! score is isolated so a likelihood objective is a flag, not a rewrite.
//!
//! # Sample overlap
//!
//! `Ω` never appears in the sampler. Drawing `E ~ MN(0, cR+τI, Ω)` and then
//! whitening the trait axis by `Ω^{-1/2}` cancels it exactly, which is why
//! [`super::whiten`] folds `Ω^{-1/2}` into the trait embedding instead.

use anyhow::{bail, Result};
use log::{info, warn};
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Ridge added before inverting `Ω`, as a fraction of its mean diagonal.
const OMEGA_RIDGE: f32 = 1e-4;

/// The eigenspace noise model: per-coordinate standard deviations for negatives.
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// `√((c d²_k + τ)/(d²_k + λ))` per block, matching `WhitenedBlock::d_sq`.
    pub scale: Vec<Vec<f32>>,
    pub c: f32,
    pub tau: f32,
    pub lambda: f64,
}

impl NoiseModel {
    /// Build the per-coordinate scales from the calibrated `(c, τ)` and λ.
    pub fn new(block_d_sq: &[Vec<f32>], c: f32, tau: f32, lambda: f64) -> Self {
        let lam = lambda as f32;
        let scale: Vec<Vec<f32>> = block_d_sq
            .iter()
            .map(|d_sq| {
                d_sq.iter()
                    .map(|&d2| {
                        let num = (c * d2 + tau).max(0.0);
                        let den = (d2 + lam).max(f32::MIN_POSITIVE);
                        (num / den).sqrt()
                    })
                    .collect()
            })
            .collect();

        let flat: Vec<f32> = scale.iter().flatten().copied().collect();
        let mean_scale = if flat.is_empty() {
            0.0
        } else {
            flat.iter().sum::<f32>() / flat.len() as f32
        };
        info!(
            "Noise model: c={:.3}, τ={:.4}, λ={:.4}; mean negative sd {:.3} \
             (1.000 means the null whitens exactly)",
            c, tau, lambda, mean_scale,
        );
        if (mean_scale - 1.0).abs() > 0.25 {
            warn!(
                "Negatives have mean sd {:.3}, not 1: the null is not whitening, so the \
                 discriminator can separate classes on scale alone.",
                mean_scale,
            );
        }

        Self {
            scale,
            c,
            tau,
            lambda,
        }
    }

    /// Draw one negative sample for a block: shape (K, T), whitened on both axes.
    pub fn sample_block(&self, block: usize, num_traits: usize, rng: &mut SmallRng) -> DMatrix<f32> {
        let scale = &self.scale[block];
        DMatrix::from_fn(scale.len(), num_traits, |ki, _| {
            let e: f64 = StandardNormal.sample(rng);
            scale[ki] * e as f32
        })
    }

    /// Deterministic per-(block, replicate) stream, so a run is reproducible
    /// regardless of how blocks are scheduled across threads.
    pub fn block_rng(seed: u64, block: usize, replicate: usize) -> SmallRng {
        SmallRng::seed_from_u64(
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((block as u64) << 20)
                .wrapping_add(replicate as u64),
        )
    }
}

/// Symmetric square root and inverse square root of `Ω`, via eigendecomposition.
///
/// # Why not Cholesky
///
/// `L L' = Ω` would also whiten, but `L` is *triangular*, so whitening by `L⁻¹`
/// depends on the order the traits happen to appear in the sumstat file — and
/// `V̌` is the object being interpreted and compared against `r_g`. The
/// symmetric root commutes with permutation. Cholesky also fails outright on a
/// singular `Ω`, which is the ordinary case when two traits are near-duplicates,
/// whereas an eigendecomposition can floor the spectrum and continue. Finally
/// both `Ω^{1/2}` and `Ω^{-1/2}` are needed — forward to un-whiten the fitted
/// embedding, inverse to whiten the data — and one decomposition yields both.
///
/// A small ridge is added first: `Ω` estimated from finite data is routinely
/// near-singular, and `Ω^{-1/2}` would then amplify one direction without bound.
///
/// # Where `Ω` comes from
///
/// From [`crate::embedding::diagnostics::estimate_trait_geometry`], as the
/// intercept of the three-term eigenspace fit. Estimating it instead as the
/// cross-trait correlation of low-χ² SNPs is biased: conditioning on small χ²
/// preferentially drops the concordant-large pairs that carry the correlation,
/// which measured ~16% low on simulated data with a known `Ω`.
pub fn omega_sqrt_pair(omega: &DMatrix<f32>) -> Result<(DMatrix<f32>, DMatrix<f32>)> {
    let t = omega.nrows();
    if t == 0 || omega.ncols() != t {
        bail!("Ω must be square and non-empty");
    }

    let mean_diag: f32 = (0..t).map(|i| omega[(i, i)]).sum::<f32>() / t as f32;
    let ridge = (mean_diag.abs() * OMEGA_RIDGE).max(1e-6);
    let mut m = omega.clone();
    for i in 0..t {
        m[(i, i)] += ridge;
    }

    let eig = m.symmetric_eigen();
    let floor = ridge;
    let mut n_floored = 0usize;

    let mut sqrt_vals = Vec::with_capacity(t);
    let mut inv_sqrt_vals = Vec::with_capacity(t);
    for i in 0..t {
        let mut ev = eig.eigenvalues[i];
        if ev < floor {
            ev = floor;
            n_floored += 1;
        }
        let s = ev.sqrt();
        sqrt_vals.push(s);
        inv_sqrt_vals.push(1.0 / s);
    }
    if n_floored > 0 {
        warn!(
            "Ω had {} eigenvalue(s) at or below the ridge; traits may be near-duplicates",
            n_floored
        );
    }

    let q = &eig.eigenvectors;
    let scale_cols = |vals: &[f32]| -> DMatrix<f32> {
        let scaled = DMatrix::from_fn(t, t, |i, j| q[(i, j)] * vals[j]);
        scaled * q.transpose()
    };

    Ok((scale_cols(&sqrt_vals), scale_cols(&inv_sqrt_vals)))
}

#[cfg(test)]
#[path = "noise_tests.rs"]
mod tests;
