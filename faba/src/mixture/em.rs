/// EM algorithm parameters.
pub struct EmParams {
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Convergence tolerance for log-likelihood change
    pub tol: f64,
    /// Minimum component weight before pruning
    pub min_weight: f64,
}

impl Default for EmParams {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-4,
            min_weight: 0.01,
        }
    }
}

/// Result of fixed-parameter EM (weights only).
pub struct FixedEmResult {
    /// Estimated mixing weights pi_0..pi_K (pi_0 = noise)
    pub weights: Vec<f64>,
    /// Posterior assignment probabilities gamma[n][k] (k=0 is noise)
    pub gamma: Vec<Vec<f64>>,
    /// Final log-likelihood
    pub log_lik: f64,
    /// BIC
    pub bic: f64,
    /// Number of iterations
    pub n_iter: usize,
}

/// Result of full Gaussian mixture EM.
pub struct GmmResult {
    /// Mixing weights pi_0..pi_K (pi_0 = noise/uniform)
    pub weights: Vec<f64>,
    /// Learned component means
    pub mus: Vec<f64>,
    /// Learned component standard deviations
    pub sigmas: Vec<f64>,
    /// Posterior assignment probabilities gamma[n][k]
    pub gamma: Vec<Vec<f64>>,
    /// BIC
    pub bic: f64,
}

/// Numerically stable log-sum-exp: log(exp(a) + exp(b))
pub fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Fixed-parameter EM: given precomputed per-observation component log-likelihoods,
/// estimate only the mixing weights.
///
/// * `component_log_liks` - `[n_obs][n_components]` log-likelihood matrix
/// * `n_free_params` - number of free parameters for BIC computation
/// * `params` - EM parameters
pub fn fixed_em(
    component_log_liks: &[Vec<f64>],
    n_free_params: usize,
    params: &EmParams,
) -> FixedEmResult {
    let n_obs = component_log_liks.len();
    if n_obs == 0 {
        return FixedEmResult {
            weights: vec![],
            gamma: vec![],
            log_lik: 0.0,
            bic: 0.0,
            n_iter: 0,
        };
    }
    let n_total = component_log_liks[0].len();

    let mut weights = vec![1.0 / n_total as f64; n_total];
    let mut gamma = vec![vec![0.0; n_total]; n_obs];
    let mut prev_ll = f64::NEG_INFINITY;

    let mut iter = 0;
    loop {
        // E-step
        let mut total_ll = 0.0;

        for n in 0..n_obs {
            let mut log_probs = vec![f64::NEG_INFINITY; n_total];
            for k in 0..n_total {
                if weights[k] > 0.0 {
                    log_probs[k] = weights[k].ln() + component_log_liks[n][k];
                }
            }

            let log_norm = log_probs
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &x| log_sum_exp(acc, x));

            total_ll += log_norm;

            for (k, lp) in log_probs.iter().enumerate().take(n_total) {
                gamma[n][k] = (lp - log_norm).exp();
            }
        }

        iter += 1;

        let ll_change = (total_ll - prev_ll).abs();
        if iter > 1 && (ll_change < params.tol || iter >= params.max_iter) {
            let bic = -2.0 * total_ll + n_free_params as f64 * (n_obs as f64).ln();

            return FixedEmResult {
                weights,
                gamma,
                log_lik: total_ll,
                bic,
                n_iter: iter,
            };
        }

        prev_ll = total_ll;

        // M-step: update weights only
        for k in 0..n_total {
            let sum_gamma: f64 = gamma.iter().map(|g| g[k]).sum();
            weights[k] = sum_gamma / n_obs as f64;
        }

        // Prune low-weight components (skip noise at k=0)
        for w in weights.iter_mut().skip(1) {
            if *w < params.min_weight {
                *w = 0.0;
            }
        }

        // Renormalize
        let w_sum: f64 = weights.iter().sum();
        if w_sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= w_sum;
            }
        }
    }
}

/// Log PDF of a Gaussian distribution
fn gaussian_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * std::f64::consts::TAU.ln()
}

/// Full 1D Gaussian mixture EM with a uniform noise component.
///
/// * `observations` - 1D data points
/// * `initial_mus` - initial means for K Gaussian components
/// * `initial_sigma` - shared initial sigma for all components
/// * `domain_length` - length of domain for uniform noise component
/// * `params` - EM parameters
///
/// Component 0 is always uniform noise over `[0, domain_length]`.
/// Components 1..K are Gaussians.
pub fn gaussian_mixture_em(
    observations: &[f64],
    initial_mus: &[f64],
    initial_sigma: f64,
    domain_length: f64,
    params: &EmParams,
) -> GmmResult {
    let n = observations.len();
    let k = initial_mus.len();
    let n_total = k + 1; // +1 for noise

    if n == 0 || k == 0 {
        return GmmResult {
            weights: vec![],
            mus: vec![],
            sigmas: vec![],
            gamma: vec![],
            bic: 0.0,
        };
    }

    let mut weights = vec![1.0 / n_total as f64; n_total];
    let mut mus = initial_mus.to_vec();
    let mut sigmas = vec![initial_sigma; k];
    let mut gamma = vec![vec![0.0; n_total]; n];
    let mut prev_ll = f64::NEG_INFINITY;

    let noise_log_lik = if domain_length > 0.0 {
        -(domain_length.ln())
    } else {
        f64::NEG_INFINITY
    };

    let sigma_floor = domain_length / (100.0 * k as f64);

    let mut iter = 0;
    loop {
        // E-step
        let mut total_ll = 0.0;

        for i in 0..n {
            let x = observations[i];
            let mut log_probs = vec![f64::NEG_INFINITY; n_total];

            // Noise component
            if weights[0] > 0.0 {
                log_probs[0] = weights[0].ln() + noise_log_lik;
            }

            // Gaussian components
            for j in 0..k {
                if weights[j + 1] > 0.0 {
                    log_probs[j + 1] = weights[j + 1].ln() + gaussian_log_pdf(x, mus[j], sigmas[j]);
                }
            }

            let log_norm = log_probs
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &lp| log_sum_exp(acc, lp));

            total_ll += log_norm;

            for (kk, lp) in log_probs.iter().enumerate() {
                gamma[i][kk] = (lp - log_norm).exp();
            }
        }

        iter += 1;

        let ll_change = (total_ll - prev_ll).abs();
        if iter > 1 && (ll_change < params.tol || iter >= params.max_iter) {
            // BIC: 3K (mu, sigma, weight per Gaussian) + 1 (noise weight) - 1 (constraint)
            let n_params = 3 * k;
            let bic = -2.0 * total_ll + n_params as f64 * (n as f64).ln();

            return GmmResult {
                weights,
                mus,
                sigmas,
                gamma,
                bic,
            };
        }

        prev_ll = total_ll;

        // M-step: update weights, mus, sigmas
        for j in 0..k {
            let kk = j + 1; // skip noise
            let sum_gamma: f64 = gamma.iter().map(|g| g[kk]).sum();

            if sum_gamma < 1e-10 {
                weights[kk] = 0.0;
                continue;
            }

            weights[kk] = sum_gamma / n as f64;

            // Update mu
            let mu_new: f64 = gamma
                .iter()
                .zip(observations.iter())
                .map(|(g, &x)| g[kk] * x)
                .sum::<f64>()
                / sum_gamma;
            mus[j] = mu_new;

            // Update sigma
            let var_new: f64 = gamma
                .iter()
                .zip(observations.iter())
                .map(|(g, &x)| g[kk] * (x - mu_new).powi(2))
                .sum::<f64>()
                / sum_gamma;
            sigmas[j] = var_new.sqrt().max(sigma_floor);
        }

        // Update noise weight
        let noise_sum: f64 = gamma.iter().map(|g| g[0]).sum();
        weights[0] = noise_sum / n as f64;

        // Prune low-weight Gaussian components
        for w in weights.iter_mut().skip(1) {
            if *w < params.min_weight {
                *w = 0.0;
            }
        }

        // Renormalize
        let w_sum: f64 = weights.iter().sum();
        if w_sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= w_sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp() {
        let a = 2.0_f64.ln();
        let b = 3.0_f64.ln();
        let result = log_sum_exp(a, b);
        assert!((result - 5.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_neg_inf() {
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(log_sum_exp(1.0, f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn test_gmm_single_cluster() {
        // Generate 100 points around mu=50
        let mut obs = Vec::new();
        let mut rng_state = 42u64;
        for _ in 0..100 {
            // Simple LCG for deterministic test
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
            // Box-Muller (simplified, using one uniform)
            let x = 50.0 + 5.0 * (2.0 * std::f64::consts::PI * u).cos();
            obs.push(x);
        }

        let params = EmParams {
            max_iter: 100,
            tol: 1e-6,
            min_weight: 0.01,
        };

        let result = gaussian_mixture_em(&obs, &[50.0], 10.0, 100.0, &params);

        assert_eq!(result.mus.len(), 1);
        assert!(
            (result.mus[0] - 50.0).abs() < 10.0,
            "mu should be near 50, got {}",
            result.mus[0]
        );
        assert!(
            result.weights[1] > 0.5,
            "Gaussian weight should dominate, got {}",
            result.weights[1]
        );
    }

    #[test]
    fn test_gmm_two_well_separated_clusters() {
        // Two groups: 50 points around 20, 50 around 80
        let mut obs = Vec::new();
        for i in 0..50 {
            obs.push(20.0 + (i as f64 - 25.0) * 0.5);
        }
        for i in 0..50 {
            obs.push(80.0 + (i as f64 - 25.0) * 0.5);
        }

        let params = EmParams {
            max_iter: 200,
            tol: 1e-6,
            min_weight: 0.01,
        };

        let result = gaussian_mixture_em(&obs, &[20.0, 80.0], 15.0, 100.0, &params);

        assert_eq!(result.mus.len(), 2);
        // Check means are near their true values
        let mut sorted_mus = result.mus.clone();
        sorted_mus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (sorted_mus[0] - 20.0).abs() < 5.0,
            "first mu should be near 20, got {}",
            sorted_mus[0]
        );
        assert!(
            (sorted_mus[1] - 80.0).abs() < 5.0,
            "second mu should be near 80, got {}",
            sorted_mus[1]
        );
    }

    #[test]
    fn test_fixed_em_basic() {
        // 2 components, 10 observations
        // Component 0 has high likelihood for first 5, component 1 for last 5
        let mut component_log_liks = Vec::new();
        for i in 0..10 {
            if i < 5 {
                component_log_liks.push(vec![-1.0, -10.0]);
            } else {
                component_log_liks.push(vec![-10.0, -1.0]);
            }
        }

        let params = EmParams {
            max_iter: 100,
            tol: 1e-6,
            min_weight: 0.01,
        };

        let result = fixed_em(&component_log_liks, 1, &params);

        assert!(
            (result.weights[0] - 0.5).abs() < 0.1,
            "weight 0 should be ~0.5, got {}",
            result.weights[0]
        );
        assert!(
            (result.weights[1] - 0.5).abs() < 0.1,
            "weight 1 should be ~0.5, got {}",
            result.weights[1]
        );
    }
}
