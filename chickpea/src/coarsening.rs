use crate::common::*;
use candle_util::candle_core::Tensor;

/// Compute per-level feature coarsenings with log-spaced targets.
pub fn log_spaced_coarsenings(
    sketch: &nalgebra::DMatrix<f32>,
    num_levels: usize,
    max_features: usize,
) -> anyhow::Result<Vec<Option<FeatureCoarsening>>> {
    if max_features == 0 || sketch.nrows() <= max_features {
        return Ok(vec![None; num_levels]);
    }
    let min_target = (max_features / num_levels).max(50);
    let log_min = (min_target as f64).ln();
    let log_max = (max_features as f64).ln();

    (0..num_levels)
        .map(|i| {
            let frac = if num_levels > 1 {
                i as f64 / (num_levels - 1) as f64
            } else {
                1.0
            };
            let target = (log_min + frac * (log_max - log_min)).exp().round() as usize;
            let target = target.clamp(min_target, max_features);
            Ok(Some(compute_feature_coarsening(sketch, target)?))
        })
        .collect()
}

/// Coarsen a tensor [N, D] → [N, d] using feature coarsening.
/// If coarsening is None, returns the tensor unchanged.
pub fn coarsen_tensor(
    x: &Tensor,
    fc: Option<&FeatureCoarsening>,
) -> candle_util::candle_core::Result<Tensor> {
    match fc {
        Some(fc) => {
            let n = x.dim(0)?;
            let d = x.dim(1)?;
            let data: Vec<f32> = x.flatten_all()?.to_vec1()?;

            let mut out = vec![0.0f32; n * fc.num_coarse];
            for row in 0..n {
                for col in 0..d {
                    let coarse = fc.fine_to_coarse[col];
                    out[row * fc.num_coarse + coarse] += data[row * d + col];
                }
            }

            Tensor::from_vec(out, (n, fc.num_coarse), x.device())
        }
        None => Ok(x.clone()),
    }
}
