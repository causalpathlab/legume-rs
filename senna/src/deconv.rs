//! Topic annotation via cosine similarity with vMF softmax.
//!
//! Computes probability of each cell type for each topic using directional
//! similarity on the unit hypersphere.

use crate::embed_common::Mat;

// ============================================================================
// vMF log normalizer (Bessel function computation)
// ============================================================================

/// Fast lgamma approximation (Paul Mineiro).
#[inline]
fn fast_lgamma(x: f64) -> f64 {
    let logterm = (x * (1.0 + x) * (2.0 + x)).ln();
    let xp3 = 3.0 + x;
    -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * xp3.ln()
}

/// Numerically stable log(1 + exp(x)).
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 10.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Numerically stable log(exp(a) + exp(b)).
#[inline]
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let (max, min) = if a > b { (a, b) } else { (b, a) };
    max + softplus(min - max)
}

/// Log of modified Bessel function I_p(x).
fn log_bessel_i(p: f64, x: f64) -> f64 {
    if x < 1e-10 {
        return f64::NEG_INFINITY;
    }
    let log_half_x = (x * 0.5).ln();
    let n_terms = (3.0 * p).clamp(30.0, 200.0) as usize;

    let mut result = -fast_lgamma(p + 1.0);
    for j in 1..n_terms {
        let jf = j as f64;
        let term = 2.0 * jf * log_half_x - fast_lgamma(jf + 1.0) - fast_lgamma(p + jf + 1.0);
        result = log_sum_exp(result, term);
    }
    result + p * log_half_x
}

/// vMF log normalizer: log C_d(κ) = (d/2-1)·log(κ) - (d/2)·log(2π) - log I_{d/2-1}(κ)
fn vmf_log_normalizer(dim: usize, kappa: f64) -> f64 {
    let d = dim as f64;
    let v = d / 2.0 - 1.0;

    if kappa < 1e-10 {
        // Uniform on sphere
        return -(d / 2.0) * std::f64::consts::TAU.ln() + fast_lgamma(d / 2.0);
    }
    v * kappa.ln() - (d / 2.0) * std::f64::consts::TAU.ln() - log_bessel_i(v, kappa)
}

// ============================================================================
// Public API
// ============================================================================

/// Compute topic-to-celltype probabilities using vMF softmax.
///
/// Returns Topic × CellType matrix where rows sum to 1.
pub fn vmf_assign(topic_profiles: &Mat, marker_profiles: &Mat, kappa: f32) -> Mat {
    let y = l2_normalize_columns(topic_profiles);
    let x = l2_normalize_columns(marker_profiles);
    let similarity = y.transpose() * x;
    softmax_rows(&similarity, kappa)
}

/// Bayesian model averaging over κ values, weighted by marginal likelihood.
pub fn vmf_assign_averaged(topic_profiles: &Mat, marker_profiles: &Mat, kappas: &[f32]) -> Mat {
    let (n_topics, n_celltypes, n_genes) = (
        topic_profiles.ncols(),
        marker_profiles.ncols(),
        topic_profiles.nrows(),
    );

    let y = l2_normalize_columns(topic_profiles);
    let x = l2_normalize_columns(marker_profiles);
    let sim = y.transpose() * x;

    // Compute log marginal likelihood and probability matrix for each κ
    let mut log_marginals = Vec::with_capacity(kappas.len());
    let mut prob_matrices = Vec::with_capacity(kappas.len());

    for &kappa in kappas {
        let k = kappa as f64;
        let log_c = vmf_log_normalizer(n_genes, k);

        // Sum log p(topic|κ) = log Σ_c exp(κ·sim + log_c) over all topics
        let total: f64 = (0..n_topics)
            .map(|t| {
                let scores: Vec<f64> = (0..n_celltypes)
                    .map(|c| k * (sim[(t, c)] as f64) + log_c)
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                max + scores.iter().map(|&s| (s - max).exp()).sum::<f64>().ln()
            })
            .sum();

        log_marginals.push(total);
        prob_matrices.push(softmax_rows(&sim, kappa));
    }

    // Softmax over log marginals to get weights
    let max_log = log_marginals
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_marginals.iter().map(|&x| (x - max_log).exp()).collect();
    let sum: f64 = weights.iter().sum();
    let weights: Vec<f32> = weights.iter().map(|&w| (w / sum) as f32).collect();

    log::debug!("vMF model averaging:");
    for ((&k, &lm), &w) in kappas.iter().zip(&log_marginals).zip(&weights) {
        log::debug!("  κ={:.0}: log_marginal={:.1}, weight={:.4}", k, lm, w);
    }

    // Weighted average
    let mut result = Mat::zeros(n_topics, n_celltypes);
    for (mat, &w) in prob_matrices.iter().zip(&weights) {
        result += &(mat * w);
    }
    result
}

// ============================================================================
// Helpers
// ============================================================================

fn l2_normalize_columns(mat: &Mat) -> Mat {
    let mut out = mat.clone();
    for j in 0..mat.ncols() {
        let norm: f32 = mat.column(j).iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for i in 0..mat.nrows() {
                out[(i, j)] /= norm;
            }
        }
    }
    out
}

fn softmax_rows(mat: &Mat, kappa: f32) -> Mat {
    let mut out = Mat::zeros(mat.nrows(), mat.ncols());
    for i in 0..mat.nrows() {
        let max = mat.row(i).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..mat.ncols() {
            let v = (kappa * (mat[(i, j)] - max)).exp();
            out[(i, j)] = v;
            sum += v;
        }
        for j in 0..mat.ncols() {
            out[(i, j)] /= sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmf_assign_diagonal() {
        let (n_genes, n_topics, n_celltypes) = (12, 3, 3);

        let mut topics = Mat::zeros(n_genes, n_topics);
        let mut markers = Mat::zeros(n_genes, n_celltypes);
        for k in 0..3 {
            for g in (k * 4)..((k + 1) * 4) {
                topics[(g, k)] = 1.0;
                markers[(g, k)] = 1.0;
            }
        }

        let probs = vmf_assign(&topics, &markers, 10.0);

        // Diagonal should dominate
        for t in 0..n_topics {
            for c in 0..n_celltypes {
                if c != t {
                    assert!(probs[(t, t)] > probs[(t, c)]);
                }
            }
            // Rows sum to 1
            let sum: f32 = (0..n_celltypes).map(|c| probs[(t, c)]).sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_kappa_effect() {
        let mut topics = Mat::zeros(8, 2);
        let mut markers = Mat::zeros(8, 2);

        for g in 0..5 {
            topics[(g, 0)] = 1.0;
        }
        for g in 4..8 {
            topics[(g, 1)] = 1.0;
        }
        for g in 0..4 {
            markers[(g, 0)] = 1.0;
        }
        for g in 4..8 {
            markers[(g, 1)] = 1.0;
        }

        let low = vmf_assign(&topics, &markers, 1.0);
        let high = vmf_assign(&topics, &markers, 20.0);

        // Higher κ → lower entropy (more peaked)
        let entropy = |p: &Mat, t: usize| -> f32 {
            (0..2)
                .map(|c| {
                    let v = p[(t, c)];
                    if v > 0.0 {
                        -v * v.ln()
                    } else {
                        0.0
                    }
                })
                .sum()
        };
        assert!(entropy(&high, 0) < entropy(&low, 0));
    }
}
