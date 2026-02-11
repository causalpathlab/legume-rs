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

/// Topic model log-likelihood of multinomial data (probability-scale input)
///
/// llik(i) = sum_w x(i,w) * log(pr(i,w) + eps)
///
/// * `x_nd` - data tensor (observed data)
/// * `recon_nd` - probability tensor (reconstruction)
///
/// Prefer `topic_log_likelihood` when log-reconstructions are available.
pub fn topic_likelihood(x_nd: &Tensor, recon_nd: &Tensor) -> Result<Tensor> {
    let eps = 1e-8;
    let log_recon_nd = (recon_nd + eps)?.log()?;

    x_nd.clamp(0.0, f64::INFINITY)?
        .mul(&log_recon_nd)?
        .sum(x_nd.rank() - 1)
}

/// Topic model log-likelihood of multinomial data (log-scale input)
///
/// llik(i) = sum_w x(i,w) * log_recon(i,w)
///
/// * `x_nd` - data tensor (observed data)
/// * `log_recon_nd` - log-probability tensor (log-reconstruction)
///
/// This avoids the exp→log roundtrip and is numerically superior.
pub fn topic_log_likelihood(x_nd: &Tensor, log_recon_nd: &Tensor) -> Result<Tensor> {
    x_nd.clamp(0.0, f64::INFINITY)?
        .mul(log_recon_nd)?
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

/// Zero-inflated topic model log-likelihood (log-scale input)
///
/// log p(x_nd) =
///   log(π_d + (1 - π_d) · exp(log_μ_nd))  if x_nd = 0
///   log(1 - π_d) + x_nd · log_μ_nd         if x_nd > 0
///
/// where π_d = sigmoid(dropout_logit_d) is a per-feature dropout probability
///
/// * `x_nd` - data tensor (observed data) [n, d]
/// * `log_recon_nd` - log-probability tensor (log-reconstruction) [n, d]
/// * `dropout_logit_1d` - dropout logits [1, d]
///
/// Returns: log-likelihood per sample [n]
///
pub fn zi_topic_log_likelihood(
    x_nd: &Tensor,
    log_recon_nd: &Tensor,
    dropout_logit_1d: &Tensor,
) -> Result<Tensor> {
    // π_d = sigmoid(dropout_logit_1d), broadcast over rows
    let pi = ops::sigmoid(dropout_logit_1d)?;
    let one_minus_pi = (1.0 - &pi)?;

    // Zero mask: x_nd == 0
    let is_zero = x_nd.eq(0.0f64)?;

    // Zero path: log(π + (1-π) * exp(log_recon))
    // Use logsumexp for numerical stability:
    // log(π + (1-π) * μ) = log(exp(log π) + exp(log(1-π) + log_μ))
    let eps = 1e-20;
    let log_pi = (&pi + eps)?.log()?;
    let log_one_minus_pi = (&one_minus_pi + eps)?.log()?;
    let log_term2 = log_one_minus_pi.broadcast_add(log_recon_nd)?; // [N, D]
    // logsumexp of log_pi [1,D] and log_term2 [N,D]
    let max_val = log_pi.broadcast_maximum(&log_term2)?;
    let sum_exp = log_pi
        .broadcast_sub(&max_val)?
        .exp()?
        .add(&log_term2.broadcast_sub(&max_val)?.exp()?)?;
    let zero_llik = sum_exp.log()?.add(&max_val)?;

    // Nonzero path: log(1-π) + x * log_recon
    let nonzero_llik = log_one_minus_pi.broadcast_add(&x_nd.mul(log_recon_nd)?)?;

    // Combine via where_cond, sum over features
    is_zero
        .where_cond(&zero_llik, &nonzero_llik)?
        .sum(x_nd.rank() - 1)
}

/// Zero-inflated topic model log-likelihood (probability-scale input, legacy)
///
/// * `x_nd` - data tensor (observed data) [n, d]
/// * `recon_nd` - probability tensor (reconstruction) [n, d]
/// * `dropout_logit_1d` - dropout logits [1, d]
pub fn zi_topic_likelihood(
    x_nd: &Tensor,
    recon_nd: &Tensor,
    dropout_logit_1d: &Tensor,
) -> Result<Tensor> {
    let eps = 1e-20;
    let log_recon_nd = (recon_nd + eps)?.log()?;
    zi_topic_log_likelihood(x_nd, &log_recon_nd, dropout_logit_1d)
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
