#![allow(dead_code)]

use candle_core::{Result, Tensor};

/// KL divergence loss between two Gaussian distributions
///
/// -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
///
/// * `z_mean` - mean of Gaussian distribution
/// * `z_lnvar` - log variance of Gaussian distribution
///
pub fn gaussian_kl_loss(z_mean: &Tensor, z_lnvar: &Tensor) -> Result<Tensor> {
    let z_var = z_lnvar.exp()?;
    (z_var - 1. + z_mean.powf(2.)? - z_lnvar)?.sum(1)? * 0.5
}

/// Topic model log-likelihood of multinomial data
///
/// llik(i) = sum_w x(i,w) * log pr(i,w)
///
/// * `x_nd` - data tensor (observed data)
/// * `logits_nd` - logit tensor (reconstruction)
///
pub fn topic_likelihood(x_nd: &Tensor, logits_nd: &Tensor) -> Result<Tensor> {
    x_nd.mul(logits_nd)?.sum(1)
}

/// Poisson log-likelihood of count-ish data
///
/// llik(i) = sum_w x(i,w) * log(rate(i,w)) - rate(i,w)
///
/// * `x_nd` - data tensor (observed data)
/// * `rate_nd` - rate tensor (reconstruction)
///
pub fn poisson_likelihood(x_nd: &Tensor, rate_nd: &Tensor) -> Result<Tensor> {
    x_nd.mul(&rate_nd.log()?)?.sub(rate_nd)?.sum(1)
}

/// Gaussian log-likelihood of count-ish data
///
/// llik(i) = -0.5 * sum_w [ x(i,w) - xhat(i,w) ]^2
///
/// * `x_nd` - data tensor (observed data)
/// * `rate_nd` - rate tensor (reconstruction)
///
pub fn gaussian_likelihood(x_nd: &Tensor, hat_nd: &Tensor) -> Result<Tensor> {
    x_nd.sub(hat_nd)?.powf(2.)?.sum(1)? * (-0.5)
}
