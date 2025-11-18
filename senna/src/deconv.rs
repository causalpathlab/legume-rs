use crate::embed_common::*;
use candle_core::{Result, Tensor};
use candle_nn::ops;
use core::f64;

pub struct PoissonDeconv {
    logit_theta_at: Tensor, // annotation x topic/type logits
    log_gamma_a: Tensor,    // annotation x 1
    log_bias_g: Tensor,     // gene x 1
    eps: f64,               // ε to avoid zeros
}

pub fn poission_deconv(
    in_dim: usize,
    out_dim: usize,
    var_dim: usize,
    vb: candle_nn::VarBuilder,
) -> anyhow::Result<PoissonDeconv> {
    let init_normal = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let init_zero = candle_nn::init::ZERO;
    let logit_theta_at = vb.get_with_hints((var_dim, out_dim), "logits", init_normal)?;
    let log_gamma_a = vb.get_with_hints((var_dim, 1), "single.effect", init_zero)?;
    let log_bias_g = vb.get_with_hints((in_dim, 1), "bias", init_zero)?;
    let eps = 1e-4;

    Ok(PoissonDeconv {
        logit_theta_at,
        log_gamma_a,
        log_bias_g,
        eps,
    })
}

impl PoissonDeconv {
    pub fn log_pip(&self) -> Result<Tensor> {
        ops::log_softmax(&self.logit_theta_at, 0)
    }

    /// Poisson regression
    ///
    /// y[g,t] ~ Σ_a x[g,a] λ[a,t] + δ[g]
    /// where λ[a,t] = θ[a,t] γ[a]
    ///
    pub fn log_likelihood(&self, y_gt: &Tensor, x_ga: &Tensor) -> Result<Tensor> {
        let eps = self.eps;
        let theta_at = self.log_pip()?.exp()?;
        let gamma_a = softplus(&self.log_gamma_a)?;
        let bias_g = softplus(&self.log_bias_g)?;

        let yy_gt = y_gt.clamp(eps, f64::INFINITY)?;
        let xx_ga = x_ga.clamp(eps, f64::INFINITY)?;
        let lambda_at = theta_at.broadcast_mul(&gamma_a)?;
        let yhat_gt = (xx_ga.matmul(&lambda_at)?.broadcast_add(&bias_g)? + eps)?;
        let log_yhat_gt = yhat_gt.log()?;

        (yy_gt.mul(&log_yhat_gt) - yhat_gt)?.sum(yy_gt.rank() - 1)
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let zero = Tensor::zeros_like(x)?;
    let mask = x.lt(&zero)?; // mask: x < 0

    let x_neg = x.clamp(-f64::INFINITY, 0.0)?;
    let x_pos = x.clamp(0.0, f64::INFINITY)?;

    let softplus_neg = (x_neg.exp()? + 1.0)?.log()?; // log(1 + exp(x))
    let softplus_pos = (x_pos.neg()?.exp()? + 1.0)?.log()?.add(&x_pos)?;
    Tensor::where_cond(&mask, &softplus_neg, &softplus_pos)
}
