//! Poisson (Gamma-Poisson conjugate) score functions.
//!
//! # Score formula (log marginal likelihood at one tree node)
//!
//! ```text
//! score(a0, b0, edge, total) = a0 * ln(b0) + lgamma(a0 + edge)
//!                              - lgamma(a0) - (a0 + edge) * ln(b0 + total)
//! ```

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
