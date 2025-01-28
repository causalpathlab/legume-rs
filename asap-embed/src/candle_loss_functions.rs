use candle_core::{Result, Tensor};

/// KL divergence loss between two Gaussian distributions
///
/// -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
///
/// * `z_mean` - mean of Gaussian distribution
/// * `z_lnvar` - log variance of Gaussian distribution
///
#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn topic_likelihood(x_nd: &Tensor, logits_nd: &Tensor) -> Result<Tensor> {
    x_nd.mul(&logits_nd)?.sum(1)
}
