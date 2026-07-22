//! RNA velocity: the cell-level delta `Δθ = θ^u − θ^s`, the common-mode gauge,
//! and the two tables they produce.
//!
//! # Why this is a post-hoc fit, not a training-time quantity
//!
//! The model has ONE latent by construction: both tracks are predicted from one
//! θ so that `δ` is the only thing that can distinguish them
//! ([`candle_util::vae::masked_gem`]). It therefore cannot express "this cell's
//! nascent state differs from its mature state" while training — deliberately,
//! because letting it do so lets the encoder absorb the contrast `δ` exists to
//! carry.
//!
//! The difference is recovered afterwards instead, by fitting θ to each track
//! separately against the FROZEN dictionaries
//! ([`candle_util::vae::masked_gem::fit_theta_to_track`]). Estimating `δ` first
//! and reading the latent delta out of it orders the two rather than letting
//! them compete.
//!
//! # What this replaces
//!
//! Two earlier estimators, both retired:
//!
//! - `Δz = z_u − z_s`, a difference of two ENCODER passes under different
//!   visibility. It measured the encoder's response to input sparsity as much as
//!   any biology — a between-pass offset was 55-95 % of its own sum-of-squares —
//!   and the passes it needed no longer exist.
//! - `v = P·θ`, a least-squares operator read off the dictionaries. It was a
//!   first-order LINEARIZATION (the decoder mixes dictionaries in probability
//!   space, not logit space) and `θ + v` was unconstrained, so it could leave
//!   the simplex. `graph_embedding_util::cell_projection::velocity_operator`
//!   still exists and is still used by `faba gem`; only this caller is gone.
//!
//! `Δθ` has neither problem: both θ are fitted compositions, so the difference
//! is a genuine movement between two reachable states, and because θ is a
//! composition it is depth-invariant — which matters when the nascent track is
//! several-fold sparser than the mature one.

use candle_util::candle_core::{Device, Tensor};
use log::{info, warn};

use crate::gem_encoder::infer::Inferred;
use crate::gem_encoder::write::write_cell_table;


/// Remove the per-axis population mean from the velocity, returning the removed
/// offset. Modifies `velocity` in place.
///
/// # Why this is necessary
///
/// `Δz = z_u − z_s` is a difference between two encoder passes run under
/// *different visibility patterns*, and those patterns are not symmetric: the
/// nascent track is far sparser than the mature one, so the `z_u` pass pools a
/// systematically thinner input than the `z_s` pass. The FC + BatchNorm trunk
/// turns that density difference into a roughly constant offset that is shared
/// by every cell and has nothing to do with any cell's dynamics. Left in, it
/// dominates: on a two-sample bulk run it accounted for ~62% of the velocity's
/// total sum of squares, with per-axis means larger than per-axis standard
/// deviations, which renders every arrow in a velocity field parallel.
///
/// Subtracting the population mean is the natural gauge for a *field* — the
/// quantity of interest is how cells differ from one another — and it is the
/// same choice `faba gem` makes for its operator velocity.
///
/// # What is given up
///
/// Unlike gem's operator, `Δz` does have a meaningful zero: `Δz = 0` means the
/// nascent and mature content imply the same state, i.e. steady state. So the
/// population mean is not pure nuisance in principle — a cohort genuinely
/// shifting in one direction would also produce one. In practice the two are
/// not separable here, and the magnitudes above are far more consistent with
/// the encoder asymmetry than with biology. The offset is therefore **returned
/// and recorded** in `{out}.model.json` rather than discarded, so anyone who
/// wants the uncentred field can add it back.
pub fn center_velocity(velocity: &mut [f32], n: usize, vdim: usize, label: &str) -> Vec<f32> {
    if n == 0 || vdim == 0 {
        return Vec::new();
    }
    let mut mean = vec![0f64; vdim];
    for c in 0..n {
        for k in 0..vdim {
            mean[k] += f64::from(velocity[c * vdim + k]);
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    let before: f64 = velocity.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    for c in 0..n {
        for k in 0..vdim {
            velocity[c * vdim + k] -= mean[k] as f32;
        }
    }
    let after: f64 = velocity.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let share = if before > 0.0 {
        100.0 * (before - after) / before
    } else {
        0.0
    };

    info!(
        "{label}: removed the per-axis common mode ({share:.1}% of total sum-of-squares); \n\
         the offset is recorded in {{out}}.model.json."
    );
    if share > 90.0 {
        warn!(
            "{label} was {share:.1}% common mode — almost nothing survives centring. For the \n\
             two-pass increment this is expected (it is why the operator is the headline); for \n\
             the operator velocity it means theta itself carries little per-cell spread."
        );
    }
    mean.into_iter().map(|x| x as f32).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn write_velocity_tables(
    inferred: &Inferred,
    alpha: &Tensor,
    cell_names: &[Box<str>],
    k: usize,
    out: &str,
    keep: Option<&[usize]>,
) -> anyhow::Result<Vec<f32>> {
    let n = cell_names.len();

    // Δθ = θ^u − θ^s, a difference of two SIMPLEX points. Both inputs are log θ,
    // so exponentiate first: the difference of probabilities is the movement,
    // whereas a difference of logs would be a ratio.
    let mut velocity: Vec<f32> = inferred
        .latent_nascent
        .iter()
        .zip(inferred.latent_mature.iter())
        .map(|(u, s)| u.exp() - s.exp())
        .collect();
    let common_mode = center_velocity(&mut velocity, n, k, "velocity");
    write_cell_table(
        &velocity,
        n,
        k,
        &format!("{out}.velocity_factor.parquet"),
        cell_names,
        "V",
        keep,
    )?;

    // Exact, not an approximation: θ^u·α − θ^s·α = (θ^u − θ^s)·α, so the H-space
    // field is the same movement expressed in the cell-embedding basis. Same
    // shape and space as before, so `faba lineage` is unaffected.
    let h = alpha.dim(1)?;
    let v_h: Vec<f32> = Tensor::from_vec(velocity, (n, k), &Device::Cpu)?
        .matmul(alpha)?
        .flatten_all()?
        .to_vec1()?;
    write_cell_table(
        &v_h,
        n,
        h,
        &format!("{out}.velocity.parquet"),
        cell_names,
        "H",
        keep,
    )?;

    Ok(common_mode)
}

#[cfg(test)]
#[path = "velocity_tests.rs"]
mod velocity_tests;
