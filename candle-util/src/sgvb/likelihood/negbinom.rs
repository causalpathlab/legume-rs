//! Negative Binomial likelihood for overdispersed count data.

use candle_core::{Result, Tensor};

use crate::sgvb::BlackBoxLikelihood;

/// Fast lgamma approximation for tensors (Paul Mineiro's fastlgamma).
///
/// ```text
/// lgamma(x) ≈ -2.081061466 - x + 0.0833333/(x+3) - log(x*(1+x)*(2+x)) + (2.5+x)*log(x+3)
/// ```
pub fn lgamma_approx(x: &Tensor) -> Result<Tensor> {
    // Clamp x to avoid log(0) issues
    let x_safe = x.clamp(1e-6f32, f32::MAX)?;

    // logterm = log(x * (1 + x) * (2 + x))
    let x_plus_1 = (&x_safe + 1.0)?;
    let x_plus_2 = (&x_safe + 2.0)?;
    let product = ((&x_safe * &x_plus_1)? * &x_plus_2)?;
    let logterm = product.log()?;

    // xp3 = x + 3
    let xp3 = (&x_safe + 3.0)?;
    let log_xp3 = xp3.log()?;

    // lgamma(x) = -2.081061466 - x + 0.0833333/xp3 - logterm + (2.5 + x)*log(xp3)
    let recip_term = (xp3.recip()? * 0.0833333)?;
    let mult_term = ((&x_safe + 2.5)? * &log_xp3)?;

    (((recip_term - 2.081061466)? - &x_safe)? - &logterm)? + &mult_term
}

/// Negative Binomial likelihood: y ~ NB(exp(η₁), exp(η₂))
///
/// # Model
/// Parameterization: μ = exp(η₁) is mean, r = exp(η₂) is dispersion.
/// Variance: Var(y) = μ + μ²/r
///
/// ```text
/// log P(y | μ, r) = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
///                 + r*log(r/(r+μ)) + y*log(μ/(r+μ))
/// ```
///
/// Requires two etas:
/// - etas[0]: log-mean (log μ)
/// - etas[1]: log-dispersion (log r)
pub struct NegativeBinomialLikelihood {
    y: Tensor,
}

impl NegativeBinomialLikelihood {
    pub fn new(y: Tensor) -> Self {
        Self { y }
    }
}

impl BlackBoxLikelihood for NegativeBinomialLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        assert!(
            etas.len() >= 2,
            "NegativeBinomialLikelihood requires 2 etas (log_mean, log_dispersion)"
        );

        let log_mu_raw = etas[0]; // log-mean: (S, n, k)
        let log_r_raw = etas[1]; // log-dispersion: (S, n, k)

        // Clamp to avoid numerical issues
        let log_mu = log_mu_raw.clamp(-10.0, 10.0)?;
        let log_r = log_r_raw.clamp(-10.0, 10.0)?;

        let mu = log_mu.exp()?;
        let r = log_r.exp()?;

        // NB log-likelihood:
        // log P(y | μ, r) = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
        //                 + r*log(r) + y*log(μ) - (r+y)*log(r+μ)

        let y_plus_r = self.y.broadcast_add(&r)?;
        let r_plus_mu = (&r + &mu)?;

        // Gamma terms - use broadcast for correct shape handling
        let lgamma_y_plus_r = lgamma_approx(&y_plus_r)?;
        let lgamma_r = lgamma_approx(&r)?;
        let y_plus_1 = (&self.y + 1.0)?;
        let lgamma_y_plus_1 = lgamma_approx(&y_plus_1)?;

        // Log terms
        let r_log_r = (&r * &log_r)?;
        let y_log_mu = self.y.broadcast_mul(&log_mu)?;
        let r_plus_y_log_r_plus_mu = (&y_plus_r * r_plus_mu.log()?)?;

        // Combine using broadcast operations
        let log_prob = lgamma_y_plus_r
            .broadcast_sub(&lgamma_r)?
            .broadcast_sub(&lgamma_y_plus_1)?
            .broadcast_add(&r_log_r)?
            .broadcast_add(&y_log_mu)?
            .broadcast_sub(&r_plus_y_log_r_plus_mu)?;

        // Sum over (n, k) dimensions
        log_prob.sum(2)?.sum(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_negbinom_likelihood() -> Result<()> {
        let device = Device::Cpu;

        // y shape: (n, k) = (3, 1), etas shape: (S, n, k) = (1, 3, 1)
        let y = Tensor::from_vec(vec![1.0f32, 2.0, 5.0], (3, 1), &device)?;
        let log_mu = Tensor::from_vec(vec![0.0f32, 0.5, 1.5], (1, 3, 1), &device)?;
        let log_r = Tensor::ones((1, 3, 1), candle_core::DType::F32, &device)?;

        let likelihood = NegativeBinomialLikelihood::new(y.clone());
        let log_lik = likelihood.log_likelihood(&[&log_mu, &log_r])?;

        let val: f32 = log_lik.get(0)?.to_scalar()?;
        assert!(val.is_finite());
        println!("NegBinom log_lik: {}", val);

        Ok(())
    }

    #[test]
    fn test_lgamma_approx() -> Result<()> {
        let device = Device::Cpu;

        // Test lgamma at some known values
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 5.0, 10.0], (4,), &device)?;
        let lg = lgamma_approx(&x)?;
        let vals: Vec<f32> = lg.to_vec1()?;

        // lgamma(1) = 0, lgamma(2) = 0, lgamma(5) ≈ 3.178, lgamma(10) ≈ 12.802
        println!("lgamma approx: {:?}", vals);
        assert!((vals[0] - 0.0).abs() < 0.1);
        assert!((vals[1] - 0.0).abs() < 0.1);
        assert!((vals[2] - 3.178).abs() < 0.2);
        assert!((vals[3] - 12.802).abs() < 0.5);

        Ok(())
    }
}
