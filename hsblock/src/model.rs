//! Poisson (Gamma-Poisson conjugate) score functions.
//!
//! Provides both CPU-native functions (for the hot Gibbs sampling loop)
//! and candle tensor functions (for the differentiable M-step).
//!
//! # Score formula (log marginal likelihood at one tree node)
//!
//! ```text
//! score(a0, b0, edge, total) = a0 * ln(b0) + lgamma(a0 + edge)
//!                              - lgamma(a0) - (a0 + edge) * ln(b0 + total)
//! ```

use candle_core::{Result, Tensor};
use candle_util::sgvb::lgamma_approx;
use special::Gamma as SpecialGamma;

/// CPU-native Poisson-Gamma conjugate score at a single tree node.
///
/// This is the hot path in the Gibbs sampler — called O(n * K) times per sweep.
///
/// * `a0` - Gamma shape parameter (> 0)
/// * `b0` - Gamma rate parameter (> 0)
/// * `edge` - Observed edge count (sum of weights) for this block
/// * `total` - Total possible edges (depends on degree correction)
#[inline]
pub fn poisson_score_cpu(a0: f64, b0: f64, edge: f64, total: f64) -> f64 {
    a0 * b0.ln() + SpecialGamma::ln_gamma(a0 + edge).0
        - SpecialGamma::ln_gamma(a0).0
        - (a0 + edge) * (b0 + total).ln()
}

/// Compute the full tree score by aggregating per-node scores.
///
/// For each pair of clusters (k, l), the interaction is governed by the
/// LCA node in the tree. This function takes pre-aggregated per-node
/// statistics and computes the total score.
///
/// * `a0` - Shape parameters, one per tree node
/// * `b0` - Rate parameters, one per tree node
/// * `edge_stats` - Aggregated edge counts per tree node
/// * `total_stats` - Aggregated total counts per tree node
pub fn tree_score_cpu(a0: &[f64], b0: &[f64], edge_stats: &[f64], total_stats: &[f64]) -> f64 {
    let mut score = 0.0;
    for i in 0..a0.len() {
        score += poisson_score_cpu(a0[i], b0[i], edge_stats[i], total_stats[i]);
    }
    score
}

/// Candle-based differentiable tree score for the variational M-step.
///
/// Uses `candle_util::sgvb::lgamma_approx` for autodiff-compatible lgamma.
///
/// * `ln_a0` - Learnable log(shape) parameters, shape `(num_nodes,)`
/// * `ln_b0` - Learnable log(rate) parameters, shape `(num_nodes,)`
/// * `edge_stats` - Aggregated edge counts per tree node, shape `(num_nodes,)`
/// * `total_stats` - Aggregated totals per tree node, shape `(num_nodes,)`
///
/// Returns the scalar total score (sum over all tree nodes).
pub fn tree_score_candle(
    ln_a0: &Tensor,
    ln_b0: &Tensor,
    edge_stats: &Tensor,
    total_stats: &Tensor,
) -> Result<Tensor> {
    let a0 = ln_a0.exp()?;
    let b0 = ln_b0.exp()?;

    // score = a0 * ln(b0) + lgamma(a0 + edge) - lgamma(a0)
    //       - (a0 + edge) * ln(b0 + total)
    let a0_plus_edge = (&a0 + edge_stats)?;
    let b0_plus_total = (&b0 + total_stats)?;

    let term1 = (&a0 * ln_b0)?;
    let term2 = lgamma_approx(&a0_plus_edge)?;
    let term3 = lgamma_approx(&a0)?;
    let term4 = (&a0_plus_edge * &b0_plus_total.log()?)?;

    let per_node_score = ((term1 + term2)? - term3)?.sub(&term4)?;

    // Sum over all nodes to get scalar score
    per_node_score.sum_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_poisson_score_basic() {
        // lgamma(1) = 0, lgamma(2) = 0, ln(1) = 0
        let score = poisson_score_cpu(1.0, 1.0, 0.0, 0.0);
        // a0*ln(b0) + lgamma(a0+0) - lgamma(a0) - (a0+0)*ln(b0+0)
        // = 1*0 + lgamma(1) - lgamma(1) - 1*0 = 0
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_score_with_data() {
        let a0 = 1.0;
        let b0 = 1.0;
        let edge = 5.0;
        let total = 10.0;

        let score = poisson_score_cpu(a0, b0, edge, total);

        // Manual computation:
        // 1*ln(1) + lgamma(6) - lgamma(1) - 6*ln(11)
        // = 0 + ln(120) - 0 - 6*ln(11)
        let expected = 0.0 + 120.0_f64.ln() - 0.0 - 6.0 * 11.0_f64.ln();
        assert!(
            (score - expected).abs() < 1e-10,
            "score={}, expected={}",
            score,
            expected
        );
    }

    #[test]
    fn test_poisson_score_positive_params() {
        // Score should be finite for reasonable parameters
        let score = poisson_score_cpu(2.0, 3.0, 10.0, 100.0);
        assert!(score.is_finite());
    }

    #[test]
    fn test_candle_score_matches_cpu() -> Result<()> {
        let device = Device::Cpu;

        let a0_vals = vec![1.0f32, 2.0, 3.0];
        let b0_vals = vec![1.0f32, 1.5, 2.0];
        let edge_vals = vec![5.0f32, 10.0, 2.0];
        let total_vals = vec![20.0f32, 50.0, 8.0];

        // CPU reference
        let cpu_score: f64 = a0_vals
            .iter()
            .zip(&b0_vals)
            .zip(&edge_vals)
            .zip(&total_vals)
            .map(|(((a, b), e), t)| poisson_score_cpu(*a as f64, *b as f64, *e as f64, *t as f64))
            .sum();

        // Candle version
        let ln_a0 = Tensor::from_vec(
            a0_vals.iter().map(|x| x.ln()).collect::<Vec<f32>>(),
            (3,),
            &device,
        )?;
        let ln_b0 = Tensor::from_vec(
            b0_vals.iter().map(|x| x.ln()).collect::<Vec<f32>>(),
            (3,),
            &device,
        )?;
        let edge_t = Tensor::from_vec(edge_vals.clone(), (3,), &device)?;
        let total_t = Tensor::from_vec(total_vals.clone(), (3,), &device)?;

        let candle_score: f32 = tree_score_candle(&ln_a0, &ln_b0, &edge_t, &total_t)?
            .to_dtype(DType::F32)?
            .to_scalar()?;

        // Allow tolerance for f32 approx lgamma
        assert!(
            (candle_score as f64 - cpu_score).abs() < 1.0,
            "candle={}, cpu={}",
            candle_score,
            cpu_score
        );

        Ok(())
    }
}
