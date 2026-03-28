use candle_core::{Result, Tensor};

/// KL(softmax(logits, dim=1) || Uniform(d)) summed over all distributions.
///
/// For logits shape (L, d, ...), computes `Σ_l Σ_i q[l,i,...] * (log q[l,i,...] - log(α/d))`
/// where `q = softmax(logits, dim=1)` and `α = prior_alpha`.
///
/// Shared across SusieVar and BiSusieVar categorical KL computations.
pub fn kl_categorical_uniform(logits: &Tensor, prior_alpha: f64) -> Result<Tensor> {
    let log_sm = candle_nn::ops::log_softmax(logits, 1)?;
    let sm = log_sm.exp()?;
    let d = sm.dim(1)? as f64;
    let log_prior = (prior_alpha / d).ln();
    let neg_entropy = sm.broadcast_mul(&log_sm)?.sum_all()?;
    let total_distributions = sm.elem_count() as f64 / d;
    neg_entropy - log_prior * total_distributions
}

/// Compute posterior inclusion probability from per component selection weights.
///
/// PIP_j = 1 - prod_l (1 - alpha_{l,j})
///
/// Computed in log space for numerical stability.
/// Shared across SuSiE-like variational distributions and the multilevel model.
///
/// alpha shape (L, p, k), returns (p, k).
pub fn pip_from_alpha(alpha: &Tensor) -> Result<Tensor> {
    let one_minus = (1.0 - alpha)?.clamp(1e-10, 1.0)?;
    let log_one_minus = one_minus.log()?;
    let sum_log = log_one_minus.sum(0)?;
    1.0 - sum_log.exp()?
}
