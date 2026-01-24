use candle_core::Tensor;

use crate::sgvb::BlackBoxLikelihood;

/// Gaussian likelihood for regression: y ~ N(eta, sigma^2)
pub struct GaussianLikelihood {
    y: Tensor,
    ln_sigma: f64,
}

impl GaussianLikelihood {
    pub fn new(y: Tensor, sigma: f64) -> Self {
        Self { y, ln_sigma: sigma.ln() }
    }
}

impl BlackBoxLikelihood for GaussianLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> candle_core::Result<Tensor> {
        let eta = etas[0];
        let sigma_sq = (2.0 * self.ln_sigma).exp();
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();
        let const_term = 2.0 * self.ln_sigma + ln_2pi;

        let diff_sq = eta.broadcast_sub(&self.y)?.powf(2.0)?;
        let log_prob = (((diff_sq / sigma_sq)? + const_term)? * (-0.5))?;

        log_prob.sum(2)?.sum(1)
    }
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
