#![allow(dead_code)]

use core::f64;

use candle_core::{Result, Tensor};
use candle_nn::ops;

/// KL divergence loss between two Gaussian distributions
///
/// -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
///
/// * `z_mean` - mean of Gaussian distribution
/// * `z_lnvar` - log variance of Gaussian distribution
///
pub fn gaussian_kl_loss(z_mean: &Tensor, z_lnvar: &Tensor) -> Result<Tensor> {
    let z_var = z_lnvar.exp()?;
    (z_var - 1. + z_mean.powf(2.)? - z_lnvar)?.sum(z_mean.rank() - 1)? * 0.5
}

/// Topic model log-likelihood of multinomial data
///
/// llik(i) = sum_w x(i,w) * log pr(i,w)
///
/// * `x_nd` - data tensor (observed data)
/// * `recon_nd` - probability tensor (reconstruction)
///
pub fn topic_likelihood(x_nd: &Tensor, recon_nd: &Tensor) -> Result<Tensor> {
    let eps = 1e-8;
    let log_recon_nd = (recon_nd + eps)?.log()?;
    // .gt(0.0)?
    // .where_cond(&recon_nd.log()?, &Tensor::zeros_like(&recon_nd)?)?;

    x_nd.clamp(0.0, f64::INFINITY)?
        .mul(&log_recon_nd)?
        .sum(x_nd.rank() - 1)
}

/// Dirichlet-Multinomial log-likelihood (pretty slow...)
///
/// α(i,w) = x(i,w) + mass(i,w)
/// llik(i) = sum_w lgamma( α(i,w) ) - lgamma( sum_w α(i,w) )
///           - sum_w lgamma( mass(i,w) ) + lgamma( sum_w mass(i,w) )
///
/// * `x_nd` - data tensor (observed data)
/// * `mass_nd` - mass tensor (reconstruction)
///
pub fn dirichlet_likelihood(x_nd: &Tensor, mass_nd: &Tensor) -> Result<Tensor> {
    let a_nd = x_nd.add(mass_nd)?;

    let term1 = approx_lgamma(&a_nd)?
        .sub(&approx_lgamma(mass_nd)?)?
        .sum(a_nd.rank() - 1)?;

    let term2 = approx_lgamma(&mass_nd.sum(mass_nd.rank() - 1)?)?
        .sub(&approx_lgamma(&a_nd.sum(a_nd.rank() - 1)?)?)?;

    term1.add(&term2)
}

/// -0.0810614667f - x - log(x) + (0.5f + x) * log(1.0f + x);
fn approx_lgamma(x: &Tensor) -> Result<Tensor> {
    // let x = (x + 1e-8)?;
    let term1 = (x.neg()? - 0.0810614667)?;
    let term2 = x.log()?.neg()?;
    let term3 = (x + 0.5)?.mul(&(x + 1.0)?.log()?)?;
    term1.add(&term2)?.add(&term3)
}

/// Poisson log-likelihood of count-ish data
///
/// llik(i) = sum_w x(i,w) * log(rate(i,w)) - rate(i,w)
///
/// * `x_nd` - data tensor (observed data)
/// * `rate_nd` - rate tensor (reconstruction)
///
pub fn poisson_likelihood(x_nd: &Tensor, rate_nd: &Tensor) -> Result<Tensor> {
    x_nd.mul(&rate_nd.log()?)?
        .sub(rate_nd)?
        .sum(x_nd.rank() - 1)
}

/// Zero-inflated topic model log-likelihood
///
/// log p(x_nd) =
///   log(π_d + (1 - π_d) · μ_nd)       if x_nd = 0
///   log(1 - π_d) + x_nd · log(μ_nd)   if x_nd > 0
///
/// where π_d = sigmoid(dropout_logit_d) is a per-feature dropout probability
///
/// * `x_nd` - data tensor (observed data) [n, d]
/// * `recon_nd` - probability tensor (reconstruction) [n, d]
/// * `dropout_logit_1d` - dropout logits [1, d]
///
/// Returns: log-likelihood per sample [n]
///
pub fn zi_topic_likelihood(
    x_nd: &Tensor,
    recon_nd: &Tensor,
    dropout_logit_1d: &Tensor,
) -> Result<Tensor> {
    let eps = 1e-8;

    // π_d = sigmoid(dropout_logit_1d), broadcast over rows
    let pi = ops::sigmoid(dropout_logit_1d)?;
    let one_minus_pi = (1.0 - &pi)?;

    // Zero mask: x_nd == 0
    let is_zero = x_nd.eq(0.0f64)?;

    // Zero path: log(π + (1-π) * recon + eps)
    // pi is [1,D], recon is [N,D] → need broadcast_add
    let zero_llik = (pi.broadcast_add(&one_minus_pi.broadcast_mul(recon_nd)?)? + eps)?.log()?;

    // Nonzero path: log(1-π+eps) + x * log(recon+eps)
    // one_minus_pi is [1,D], x*log(recon) is [N,D] → need broadcast_add
    let nonzero_llik = (&one_minus_pi + eps)?
        .log()?
        .broadcast_add(&x_nd.mul(&(recon_nd + eps)?.log()?)?)?;

    // Combine via where_cond, sum over features
    is_zero
        .where_cond(&zero_llik, &nonzero_llik)?
        .sum(x_nd.rank() - 1)
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
