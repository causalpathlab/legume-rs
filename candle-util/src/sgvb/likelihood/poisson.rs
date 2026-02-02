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
