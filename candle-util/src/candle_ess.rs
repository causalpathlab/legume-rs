#![allow(dead_code)]

use candle_core::{DType, Result, Tensor};
use std::f64::consts::PI;

/// One batched elliptical slice sampling transition.
///
/// All N samples are updated in parallel using masked tensor ops.
/// The prior is implicit: `prior_samples` should be drawn from N(0, I).
///
/// # Arguments
/// * `current` - [N, K] current parameter values
/// * `prior_samples` - [N, K] draws from the prior (e.g. N(0, I))
/// * `lnpdf` - batched log-likelihood: [N, K] → [N]
/// * `cur_lnpdf` - [N] cached log-likelihoods at current
/// * `max_shrink` - safety cap on slice iterations (e.g. 50)
///
/// # Returns
/// `(new_params [N, K], new_lnpdf [N])`
pub fn batched_ess_step(
    current: &Tensor,
    prior_samples: &Tensor,
    lnpdf: &impl Fn(&Tensor) -> Result<Tensor>,
    cur_lnpdf: &Tensor,
    max_shrink: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = current.device();
    let n = current.dim(0)?;

    // 1. Log-likelihood threshold: hh = ln(u) + cur_lnpdf
    let u = Tensor::rand(0f32, 1f32, &[n], dev)?;
    let hh = (u.log()? + cur_lnpdf)?;

    // 2. Initial angle and bracket
    let phi = Tensor::rand(0f32, 1f32, &[n], dev)?.affine(2.0 * PI, 0.0)?;
    let mut phi_min = (&phi - 2.0 * PI)?;
    let mut phi_max = phi.clone();
    let mut angle = phi;

    // 3. Result accumulators — start with current values
    // Use f32 mask throughout (0.0 = pending, 1.0 = accepted)
    let mut result_params = current.clone();
    let mut result_lnpdf = cur_lnpdf.clone();
    let mut done_mask = Tensor::zeros(&[n], DType::F32, dev)?;

    for iter in 0..max_shrink {
        // Proposal: current * cos(angle) + prior * sin(angle)
        let cos_a = angle.cos()?.unsqueeze(1)?; // [N, 1]
        let sin_a = angle.sin()?.unsqueeze(1)?; // [N, 1]
        let proposals = current
            .broadcast_mul(&cos_a)?
            .add(&prior_samples.broadcast_mul(&sin_a)?)?;

        // Evaluate log-likelihood for all proposals
        let new_lnpdf = lnpdf(&proposals)?; // [N]

        // Accept where new_lnpdf > hh AND not already done
        let above = new_lnpdf.gt(&hh)?.to_dtype(DType::F32)?; // [N] f32
        let pending = (1.0 - &done_mask)?;
        let newly_accepted = above.mul(&pending)?; // [N] f32, only pending ones

        // Update results for newly accepted
        let keep = (1.0 - &newly_accepted)?;
        let mask_2d = newly_accepted.unsqueeze(1)?; // [N, 1]
        let keep_2d = keep.unsqueeze(1)?; // [N, 1]

        result_params = result_params
            .broadcast_mul(&keep_2d)?
            .add(&proposals.broadcast_mul(&mask_2d)?)?;
        result_lnpdf = result_lnpdf
            .mul(&keep)?
            .add(&new_lnpdf.mul(&newly_accepted)?)?;

        // Mark as done
        done_mask = done_mask.add(&newly_accepted)?;

        // Check if all done (every 5 iterations to reduce GPU→CPU sync)
        if (iter + 1) % 5 == 0 {
            let n_done = done_mask.sum_all()?.to_scalar::<f32>()? as usize;
            if n_done >= n {
                break;
            }
        }

        // Shrink bracket for non-accepted samples
        let pending = (1.0 - &done_mask)?;
        let angle_neg = angle.lt(0f32)?.to_dtype(DType::F32)?; // [N]
        let angle_pos = (1.0 - &angle_neg)?;

        // phi_min = where(angle < 0 AND pending, angle, phi_min)
        let update_min = angle_neg.mul(&pending)?;
        let keep_min = (1.0 - &update_min)?;
        phi_min = phi_min.mul(&keep_min)?.add(&angle.mul(&update_min)?)?;

        // phi_max = where(angle >= 0 AND pending, angle, phi_max)
        let update_max = angle_pos.mul(&pending)?;
        let keep_max = (1.0 - &update_max)?;
        phi_max = phi_max.mul(&keep_max)?.add(&angle.mul(&update_max)?)?;

        // Redraw angle ~ Uniform(phi_min, phi_max) for pending samples
        let range = (&phi_max - &phi_min)?;
        let new_angle = Tensor::rand(0f32, 1f32, &[n], dev)?
            .mul(&range)?
            .add(&phi_min)?;
        let done_f = &done_mask;
        let pending_f = (1.0 - done_f)?;
        angle = new_angle.mul(&pending_f)?.add(&angle.mul(done_f)?)?;
    }

    Ok((result_params, result_lnpdf))
}

/// Run T elliptical slice sampling steps on a batch.
///
/// Each step draws fresh prior samples from N(0, I) and applies one
/// batched ESS transition. Returns the final state.
///
/// # Arguments
/// * `init` - [N, K] initial parameter values (e.g. from encoder)
/// * `lnpdf` - batched log-likelihood: [N, K] → [N]
/// * `n_steps` - number of ESS transitions
/// * `max_shrink` - safety cap on slice iterations per step (e.g. 50)
///
/// # Returns
/// `(final_params [N, K], final_lnpdf [N])`
pub fn batched_ess_steps(
    init: &Tensor,
    lnpdf: &impl Fn(&Tensor) -> Result<Tensor>,
    n_steps: usize,
    max_shrink: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = init.device();
    let shape = init.dims().to_vec();

    let mut current = init.clone();
    let mut cur_lnpdf = lnpdf(&current)?;

    for _ in 0..n_steps {
        let prior_samples = Tensor::randn(0f32, 1f32, shape.as_slice(), dev)?;
        let (new_params, new_lnpdf) =
            batched_ess_step(&current, &prior_samples, lnpdf, &cur_lnpdf, max_shrink)?;
        current = new_params;
        cur_lnpdf = new_lnpdf;
    }

    Ok((current, cur_lnpdf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Gaussian conjugate test (batched, D=5).
    /// Prior: N(0, I_5), Likelihood: N(y_obs | f, σ²·I_5)
    /// Posterior: N(μ_post, σ²_post·I) where σ²_post = 1/(1+1/σ²), μ_post = σ²_post·y/σ²
    #[test]
    fn test_batched_ess_gaussian_conjugate() {
        let dev = Device::Cpu;
        let n = 8; // batch size
        let d = 5;
        let sigma_sq = 2.0f32;
        let n_samples = 2000;
        let warmup = 500;

        let sigma_sq_post = 1.0 / (1.0 + 1.0 / sigma_sq);

        // Each sample in the batch has a different observation
        let y_vals: Vec<f32> = (0..n * d)
            .map(|i| ((i as f32) * 0.7 - 2.0).sin() * 2.0)
            .collect();
        let y_obs = Tensor::from_vec(y_vals.clone(), &[n, d], &dev).unwrap();

        let lnpdf = {
            let y = y_obs.clone();
            move |f: &Tensor| -> Result<Tensor> {
                let diff = (f - &y)?;
                let sq = (&diff * &diff)?.sum(1)?;
                sq * (-0.5f64 / sigma_sq as f64)
            }
        };

        // Run many single-step ESS transitions to collect samples
        let init = Tensor::zeros(&[n, d], DType::F32, &dev).unwrap();

        let mut current = init;
        let mut cur_ll = lnpdf(&current).unwrap();

        // Accumulate posterior mean
        let mut sum = Tensor::zeros(&[n, d], DType::F32, &dev).unwrap();
        let mut count = 0usize;

        for step in 0..(warmup + n_samples) {
            let prior = Tensor::randn(0f32, 1f32, &[n, d], &dev).unwrap();
            let (new_params, new_ll) =
                batched_ess_step(&current, &prior, &lnpdf, &cur_ll, 50).unwrap();
            current = new_params;
            cur_ll = new_ll;

            if step >= warmup {
                sum = (&sum + &current).unwrap();
                count += 1;
            }
        }

        let mean = (sum / count as f64).unwrap();
        let mean_vals: Vec<f32> = mean.flatten_all().unwrap().to_vec1().unwrap();
        let y_flat: Vec<f32> = y_obs.flatten_all().unwrap().to_vec1().unwrap();

        for i in 0..(n * d) {
            let analytic = sigma_sq_post * y_flat[i] / sigma_sq;
            assert!(
                (mean_vals[i] - analytic).abs() < 0.2,
                "index {}: mean={}, expected={}",
                i,
                mean_vals[i],
                analytic
            );
        }
    }

    /// Test batched_ess_steps convenience wrapper
    #[test]
    fn test_batched_ess_steps_wrapper() {
        let dev = Device::Cpu;
        let n = 4;
        let d = 3;
        let sigma_sq = 1.0f32;

        let y_obs = Tensor::from_vec(
            vec![
                1.0f32, -1.0, 2.0, 0.5, -0.5, 1.5, -2.0, 0.0, 1.0, 0.3, -0.3, 0.8,
            ],
            &[n, d],
            &dev,
        )
        .unwrap();

        let lnpdf = {
            let y = y_obs.clone();
            move |f: &Tensor| -> Result<Tensor> {
                let diff = (f - &y)?;
                (&diff * &diff)?.sum(1)? * (-0.5f64 / sigma_sq as f64)
            }
        };

        let init = Tensor::zeros(&[n, d], DType::F32, &dev).unwrap();

        // Check it runs without error and produces reasonable output
        let (result, llik) = batched_ess_steps(&init, &lnpdf, 100, 50).unwrap();
        assert_eq!(result.dims(), &[n, d]);
        assert_eq!(llik.dims(), &[n]);

        // After 100 steps from zero, should have moved away from init
        let r: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let norm_sq: f32 = r.iter().map(|x| x * x).sum();
        assert!(
            norm_sq > 0.1,
            "expected non-trivial movement from zero, got norm²={}",
            norm_sq
        );
    }
}
