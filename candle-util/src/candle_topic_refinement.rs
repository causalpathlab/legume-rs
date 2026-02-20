use crate::candle_model_traits::DecoderModuleT;
use candle_core::{Result, Tensor, Var};
use candle_nn::ops;

/// Configuration for per-cell topic proportion refinement.
pub struct TopicRefinementConfig {
    pub num_steps: usize,
    pub learning_rate: f64,
    pub regularization: f64,
}

/// Refine per-cell topic proportions by gradient descent against the frozen decoder.
///
/// Starting from the encoder's log-softmax output, optimize per-cell logits
/// to maximize decoder likelihood with L2 regularization anchoring to the
/// encoder's initial estimate.
///
/// # Arguments
/// * `log_z_nk` - `[N, K]` log-probabilities from encoder (detached)
/// * `x_nd` - `[N, D]` observed counts
/// * `decoder` - frozen decoder module
/// * `config` - refinement hyperparameters
///
/// # Returns
/// `[N, K]` refined log-probabilities
pub fn refine_topic_proportions<Dec: DecoderModuleT>(
    log_z_nk: &Tensor,
    x_nd: &Tensor,
    decoder: &Dec,
    config: &TopicRefinementConfig,
) -> Result<Tensor> {
    let z_logits_init = log_z_nk.detach();
    let z_var = Var::from_tensor(&z_logits_init)?;

    let dummy_llik = |_x: &Tensor, _y: &Tensor| -> Result<Tensor> { unreachable!() };

    for _step in 0..config.num_steps {
        let log_z = ops::log_softmax(z_var.as_tensor(), 1)?;
        let (_, llik) = decoder.forward_with_llik(&log_z, x_nd, &dummy_llik)?;

        // L2 regularization: ||z_var - z_init||²
        let diff = (z_var.as_tensor() - &z_logits_init)?;
        let reg = (&diff * &diff)?.sum_all()?;

        // loss = α·reg - mean(llik)
        let loss = ((reg * config.regularization)? - llik.mean_all()?)?;

        let grad = loss.backward()?;
        let z_grad = grad.get(z_var.as_tensor()).unwrap();

        // SGD step
        let updated = (z_var.as_tensor() - (z_grad * config.learning_rate)?)?;
        z_var.set(&updated)?;
    }

    ops::log_softmax(z_var.as_tensor(), 1)
}
