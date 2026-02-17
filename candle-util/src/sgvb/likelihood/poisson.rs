//! Poisson likelihood for count data.

use candle_core::{Result, Tensor};

use crate::sgvb::BlackBoxLikelihood;

/// Poisson likelihood for count data: y ~ Poisson(exp(η))
///
/// # Model
/// ```text
/// log p(y | η) = y * η - exp(η) - log(y!)
/// ```
/// The log(y!) term is constant w.r.t. η and is omitted.
pub struct PoissonLikelihood {
    y: Tensor,
}

impl PoissonLikelihood {
    pub fn new(y: Tensor) -> Self {
        Self { y }
    }
}

impl BlackBoxLikelihood for PoissonLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        let eta = etas[0];
        // log p(y | eta) = y * eta - exp(eta) - log(y!)
        // We ignore log(y!) as it's constant w.r.t. eta
        let y_eta = eta.broadcast_mul(&self.y)?;
        let exp_eta = eta.exp()?;
        let log_prob = (y_eta - exp_eta)?;

        log_prob.sum(2)?.sum(1)
    }
}

/// Poisson likelihood with a fixed offset: y ~ Poisson(exp(offset + η))
///
/// This is useful when the linear predictor η = Xβ has no intercept.
/// The offset absorbs the baseline log-rate so that β = 0 → μ = exp(offset).
///
/// ```text
/// log p(y | η) = y * (offset + η) - exp(offset + η) - log(y!)
/// ```
pub struct OffsetPoissonLikelihood {
    y: Tensor,
    offset: Tensor,
}

impl OffsetPoissonLikelihood {
    /// Create with a scalar offset broadcast to all observations.
    pub fn new(y: Tensor, offset: f32) -> candle_core::Result<Self> {
        let device = y.device().clone();
        let (n, k) = (y.dim(0)?, y.dim(1)?);
        let offset = Tensor::full(offset, (n, k), &device)?;
        Ok(Self { y, offset })
    }
}

impl BlackBoxLikelihood for OffsetPoissonLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        let eta = etas[0];
        // η_full = offset + η (broadcast offset from (n,k) to (S,n,k))
        let eta_full = eta.broadcast_add(&self.offset)?;
        let y_eta = eta_full.broadcast_mul(&self.y)?;
        let exp_eta = eta_full.exp()?;
        let log_prob = (y_eta - exp_eta)?;
        log_prob.sum(2)?.sum(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_poisson_likelihood() -> Result<()> {
        let device = Device::Cpu;

        // y shape: (n, k) = (3, 1), eta shape: (S, n, k) = (1, 3, 1)
        let y = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3, 1), &device)?;
        let eta = Tensor::from_vec(vec![0.0f32, 0.5, 1.0], (1, 3, 1), &device)?;

        let likelihood = PoissonLikelihood::new(y);
        let log_lik = likelihood.log_likelihood(&[&eta])?;

        let val: f32 = log_lik.get(0)?.to_scalar()?;
        assert!(val.is_finite());
        println!("Poisson log_lik: {}", val);

        Ok(())
    }
}
