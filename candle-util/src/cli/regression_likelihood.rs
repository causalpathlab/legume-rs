use candle_core::{Result, Tensor};

use crate::sgvb::BlackBoxLikelihood;

/// Fast lgamma approximation based on YPARK/fqtl fastgamma.h
///
/// Uses the formula:
///   lgamma(x) ≈ -2.081061466 - x + 0.0833333/xp3 - log(x*(1+x)*(2+x)) + (2.5+x)*log(xp3)
/// where xp3 = x + 3
fn lgamma_approx(x: &Tensor) -> Result<Tensor> {
    // Clamp x to avoid log(0) issues
    let x_safe = x.clamp(1e-6, f64::MAX)?;

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

/// Poisson likelihood for count data: y ~ Poisson(exp(eta))
pub struct PoissonLikelihood {
    y: Tensor,
}

impl PoissonLikelihood {
    pub fn new(y: Tensor) -> Self {
        Self { y }
    }
}

impl BlackBoxLikelihood for PoissonLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> candle_core::Result<Tensor> {
        let eta = etas[0];
        // log p(y | lambda) = y * log(lambda) - lambda - log(y!)
        // with lambda = exp(eta):
        // log p(y | eta) = y * eta - exp(eta) - log(y!)
        // We ignore log(y!) as it's constant w.r.t. eta
        let y_eta = eta.broadcast_mul(&self.y)?;
        let exp_eta = eta.exp()?;
        let log_prob = (y_eta - exp_eta)?;

        log_prob.sum(2)?.sum(1)
    }
}

/// Gaussian likelihood: y ~ N(η₁, exp(η₂))
///
/// Requires two etas:
/// - etas[0]: mean
/// - etas[1]: log-variance
pub struct GaussianLikelihood {
    y: Tensor,
}

impl GaussianLikelihood {
    pub fn new(y: Tensor) -> Self {
        Self { y }
    }
}

impl BlackBoxLikelihood for GaussianLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> candle_core::Result<Tensor> {
        assert!(
            etas.len() >= 2,
            "GaussianLikelihood requires 2 etas (mean, log_var)"
        );
        let mu = etas[0]; // mean: (S, n, k)
        let log_var_raw = etas[1]; // log-variance: (S, n, k)

        // Clamp log_var to avoid numerical issues with exp()
        // Range [-10, 10] gives variance in [4.5e-5, 22026]
        let log_var = log_var_raw.clamp(-10.0, 10.0)?;

        // log N(y; μ, exp(log_var)) = -0.5 * [log(2π) + log_var + (y-μ)²/exp(log_var)]
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();

        let diff = mu.broadcast_sub(&self.y)?;
        let diff_sq = diff.powf(2.0)?;
        let var = log_var.exp()?;
        let scaled_diff_sq = (diff_sq / &var)?;

        let log_prob = ((scaled_diff_sq + &log_var)? + ln_2pi)? * (-0.5);

        // Sum over (n, k) dimensions
        log_prob?.sum(2)?.sum(1)
    }
}

/// Negative Binomial likelihood: y ~ NB(exp(η₁), exp(η₂))
///
/// Parameterization: μ = exp(η₁) is mean, r = exp(η₂) is dispersion.
/// Var(y) = μ + μ²/r
///
/// Requires two etas:
/// - etas[0]: log-mean
/// - etas[1]: log-dispersion
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
        //                 + r*log(r/(r+μ)) + y*log(μ/(r+μ))
        //
        // Rearranging:
        // = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
        //   + r*log(r) - r*log(r+μ) + y*log(μ) - y*log(r+μ)
        // = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
        //   + r*log(r) + y*log(μ) - (r+y)*log(r+μ)

        let y_plus_r = self.y.broadcast_add(&r)?;
        let r_plus_mu = (&r + &mu)?;

        // Gamma terms
        let lgamma_y_plus_r = lgamma_approx(&y_plus_r)?;
        let lgamma_r = lgamma_approx(&r)?;
        let lgamma_y_plus_1 = lgamma_approx(&(&self.y + 1.0)?)?;

        // Log terms
        let r_log_r = (&r * &log_r)?;
        let y_log_mu = self.y.broadcast_mul(&log_mu)?;
        let r_plus_y_log_r_plus_mu = (&y_plus_r * r_plus_mu.log()?)?;

        // Combine
        let log_prob = (((((&lgamma_y_plus_r - &lgamma_r)? - &lgamma_y_plus_1)?
            + &r_log_r)?
            + &y_log_mu)?
            - &r_plus_y_log_r_plus_mu)?;

        // Sum over (n, k) dimensions
        log_prob.sum(2)?.sum(1)
    }
}
