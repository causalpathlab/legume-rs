use crate::candle_aux_linear::logsumexp_forward;
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
    // Use bias-free dictionary: softmax(logits) without the shared baseline.
    // The bias makes all topics look identical on sparse data.
    // The raw logits capture topic-specific signal that differentiates topics.
    let log_dict_dk = decoder.get_dictionary()?.detach(); // [D, K], no bias
    let log_dict_kd = log_dict_dk.t()?.contiguous()?; // [K, D]

    let z_logits_init = log_z_nk.detach();
    let z_var = Var::from_tensor(&z_logits_init)?;

    let x_pos = x_nd.clamp(0.0, f64::INFINITY)?;

    for _step in 0..config.num_steps {
        let log_z = ops::log_softmax(z_var.as_tensor(), 1)?;

        // Multinomial llik against bias-free dictionary
        let log_recon_nd = logsumexp_forward(&log_z, &log_dict_kd)?;
        let llik = x_pos.mul(&log_recon_nd)?.sum(1)?;

        // L2 regularization: ||z_var - z_init||²
        let diff = (z_var.as_tensor() - &z_logits_init)?;
        let reg = (&diff * &diff)?.sum_all()?;

        let loss = ((reg * config.regularization)? - llik.mean_all()?)?;

        let grad = loss.backward()?;
        let z_grad = grad.get(z_var.as_tensor()).unwrap();

        let updated = (z_var.as_tensor() - (z_grad * config.learning_rate)?)?;
        z_var.set(&updated)?;
    }

    ops::log_softmax(z_var.as_tensor(), 1)
}
