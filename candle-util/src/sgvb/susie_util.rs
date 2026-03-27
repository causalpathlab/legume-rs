use candle_core::{Result, Tensor};

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
